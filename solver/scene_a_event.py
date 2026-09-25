"""场景 A 的局部图、逐事件搬运和轻量时间精评；尚不作为默认搜索代理。"""
from collections import defaultdict
import math
from pathlib import Path
import sys

from spill_events import count_spill_events

CODE = Path(__file__).resolve().parents[1] / '通用神经网络处理器下的多核调度问题附件/code'
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))
from schedule_step1 import step1_schedule
from stub_multicore_cut_and_schedule import derive_multicore_plan
from evaluation_validation import validate_task_order


class SceneAEventModel:
    def __init__(self, graph, bandwidth=60.0, capacity=None):
        self.graph = graph
        self.bandwidth = bandwidth
        self.capacity = capacity or {'L1': 524288, 'UB': 131072}
        self.ops = {o['id']: o for o in graph['ops']}
        self.tensors = {t['id']: t for t in graph['tensors']}
        self.producers, self.consumers = defaultdict(set), defaultdict(set)
        self.direct = []
        for e in graph['edges']:
            src, dst = e['source'], e['target']
            if src in self.ops and dst in self.ops:
                if src != dst:
                    self.direct.append(e)
            elif src in self.ops:
                self.producers[dst].add(src)
            elif dst in self.ops:
                self.consumers[src].add(dst)
        self.original_bytes = self.copy_bytes(graph)

    @staticmethod
    def copy_bytes(graph):
        ops = {o['id']: o for o in graph['ops']}
        tensors = {t['id']: t for t in graph['tensors']}
        total = 0
        for e in graph['edges']:
            a, b = e['source'], e['target']
            if a in ops and ops[a]['op'] == 'COPY_IN' and b in tensors:
                total += tensors[b]['size']
            elif b in ops and ops[b]['op'] == 'COPY_OUT' and a in tensors:
                total += tensors[a]['size']
        return total

    def local_graphs(self, plan):
        view = derive_multicore_plan(self.graph, plan)
        validate_task_order(view)
        mapping = view['mapping']
        used = set(self.ops) | set(self.tensors)
        next_op = max([*self.ops, 0]) + 1
        next_tensor = max([*self.tensors, 10000]) + 1
        def ids():
            nonlocal next_op, next_tensor
            while next_tensor in used:
                next_tensor += 1
            tid = next_tensor
            used.add(tid)
            next_tensor += 1
            while next_op in used:
                next_op += 1
            oid = next_op
            used.add(oid)
            next_op += 1
            return tid, oid
        touched = defaultdict(set)
        for tid in self.tensors:
            for oid in self.producers[tid] | self.consumers[tid]:
                if oid in mapping:
                    touched[mapping[oid]].add(tid)
        graphs = []
        for task in view['subgraph_ids']:
            members = set(view['nodes_by_subgraph'][task])
            ops = [dict(self.ops[o]) for o in sorted(members)]
            tensors, edges = [], []
            for tid in sorted(touched[task]):
                tensor = dict(self.tensors[tid])
                prods = self.producers[tid] & members
                cons = self.consumers[tid] & members
                eligible_cons = self.consumers[tid] & mapping.keys()
                has_copy_out = any(self.ops[o]['op'] == 'COPY_OUT' for o in self.consumers[tid])
                incoming = bool(cons) and not prods
                outgoing = bool(prods) and (has_copy_out or not eligible_cons or bool(eligible_cons - members))
                if tensor['pos'] == 'DDR':
                    tensor['pos'] = 'UB'
                tensors.append(tensor)
                edges.extend({'source': p, 'target': tid} for p in sorted(prods))
                edges.extend({'source': tid, 'target': c} for c in sorted(cons))
                for boundary, kind, pipe in ((incoming, 'COPY_IN', 'PIPE_MTE2'),
                                             (outgoing, 'COPY_OUT', 'PIPE_MTE3')):
                    if not boundary:
                        continue
                    external, oid = ids()
                    tensors.append({'id': external, 'pos': 'DDR', 'size': tensor['size']})
                    ops.append({'id': oid, 'op': kind, 'pipe': pipe,
                                'cycles': max(1, math.ceil(tensor['size'] / self.bandwidth))})
                    if kind == 'COPY_IN':
                        edges.extend([{'source': external, 'target': oid}, {'source': oid, 'target': tid}])
                    else:
                        edges.extend([{'source': tid, 'target': oid}, {'source': oid, 'target': external}])
            edges.extend(dict(e) for e in self.direct if e['source'] in members and e['target'] in members)
            graphs.append((task, {'ops': ops, 'tensors': tensors, 'edges': edges}))
        return graphs

    @staticmethod
    def pipe_time(graph, sequence):
        ops = {op['id']: op for op in graph['ops']}
        producers, consumers, preds = defaultdict(set), defaultdict(set), defaultdict(set)
        for e in graph['edges']:
            a, b = e['source'], e['target']
            if a in ops and b in ops:
                preds[b].add(a)
            elif a in ops:
                producers[b].add(a)
            elif b in ops:
                consumers[a].add(b)
        for tid, cs in consumers.items():
            for c in cs:
                preds[c].update(producers[tid])
        free, finish = defaultdict(float), {}
        for oid in sequence:
            op = ops[oid]
            start = max(free[op['pipe']], max((finish[p] for p in preds[oid]), default=0))
            finish[oid] = start + max(1, op.get('cycles', 1))
            free[op['pipe']] = finish[oid]
        return max(finish.values(), default=0)

    def evaluate(self, plan):
        view = derive_multicore_plan(self.graph, plan)
        graphs = self.local_graphs(plan)
        tasks = {}
        raw_copy = spill = 0
        for sid, graph in graphs:
            seq = step1_schedule(graph)
            stats = count_spill_events(graph, seq, self.capacity)
            boundary = self.copy_bytes(graph)
            base = self.pipe_time(graph, seq)
            duration = max(base + stats['spill_bytes'] / self.bandwidth,
                           (boundary + stats['spill_bytes']) / self.bandwidth)
            tasks[sid] = {'duration': duration, 'pipe_time': base,
                          'boundary_bytes': boundary, **stats}
            raw_copy += boundary
            spill += stats['spill_bytes']
        orders = plan['core_schedules']
        core_of = {sid: c for c, order in enumerate(orders) for sid in order}
        free, ptr, finish = [0.0] * len(orders), [0] * len(orders), {}
        while len(finish) < len(tasks):
            progressed = False
            for c, order in enumerate(orders):
                while ptr[c] < len(order):
                    sid = order[ptr[c]]
                    ps = view['subgraph_preds'][sid]
                    if not all(p in finish for p in ps):
                        break
                    start = max(free[c] + (100 if ptr[c] else 0),
                                max((finish[p] + (1000 if core_of[p] != c else 0) for p in ps), default=0))
                    finish[sid] = start + tasks[sid]['duration']
                    free[c] = finish[sid]
                    ptr[c] += 1
                    progressed = True
            if not progressed:
                raise ValueError('子图顺序与依赖不兼容')
        return {'makespan': max(max(finish.values(), default=0), (raw_copy + spill) / self.bandwidth),
                'partition_raw_bytes': raw_copy,
                'partition_added_bytes': raw_copy - self.original_bytes,
                'spill_bytes': spill, 'total_added_bytes': raw_copy - self.original_bytes + spill,
                'total_copy_bytes': raw_copy + spill, 'tasks': tasks}
