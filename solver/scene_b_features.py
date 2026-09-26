"""场景B/C的廉价、可解释特征。

这里不调用官方展开器；寿命峰值是按候选核序的离散序位置估计，不能替代真实流水执行。
"""
import math
import heapq
from collections import defaultdict, deque

from tensor_index import COPY, TensorIndex, _plan_parts


class SceneBFeatures:
    def __init__(self, graph, capacity=None, bandwidth=60):
        self.graph = graph
        self.capacity = {'L1': 524288, 'UB': 131072} if capacity is None else dict(capacity)
        self.bandwidth = bandwidth
        self.index = TensorIndex(graph)
        self.ops = {int(o['id']): o for o in graph.get('ops', [])}
        self.op_ids = {i for i, o in self.ops.items() if o.get('op') not in COPY}
        self.tensor_by_id = self.index.tensors
        self.in_tensors, self.out_tensors = defaultdict(list), defaultdict(list)
        for edge in graph.get('edges', []):
            s, d = int(edge['source']), int(edge['target'])
            if d in self.ops and s in self.tensor_by_id: self.in_tensors[d].append(s)
            if s in self.ops and d in self.tensor_by_id: self.out_tensors[s].append(d)
        from scenario_contract import operation_dependencies
        self.predecessors, successors = operation_dependencies(graph)
        degree = {op: len(pred) for op, pred in self.predecessors.items()}
        ready = [op for op, count in degree.items() if count == 0]
        heapq.heapify(ready)
        self.topological = []
        while ready:
            op = heapq.heappop(ready)
            self.topological.append(op)
            for child in sorted(successors[op]):
                degree[child] -= 1
                if degree[child] == 0:
                    heapq.heappush(ready, child)
        if len(self.topological) != len(self.op_ids):
            raise ValueError('原算子依赖成环或覆盖不完整')
        self.topological_rank = {op: i for i, op in enumerate(self.topological)}

    def _orders(self, plan):
        mapping, core_of, rows = _plan_parts(plan)
        members = defaultdict(list)
        for op, sg in mapping.items():
            if op in self.op_ids: members[sg].append(op)
        orders = {}
        for core, row in enumerate(rows):
            seq = []
            for sg in row:
                seq.extend(sorted(members.get(sg, ()), key=self.topological_rank.__getitem__))
            orders[core] = seq
        # tolerate a plan whose schedule omitted an otherwise valid group
        present = {core: set(seq) for core, seq in orders.items()}
        for op in sorted(self.op_ids):
            sg = mapping.get(op)
            if sg in core_of:
                core = core_of[sg]
                if op not in present.setdefault(core, set()):
                    orders.setdefault(core, []).append(op)
                    present[core].add(op)
        return mapping, core_of, orders

    def _boundary_metrics(self, plan):
        mapping, core_of, _ = self._orders(plan)
        boundary = original = service = writes = 0
        for tid, tensor in self.tensor_by_id.items():
            prod = {core_of[mapping[o]] for o in self.index.producers.get(tid, ()) if o in mapping and mapping[o] in core_of}
            cons = {core_of[mapping[o]] for o in self.index.consumers.get(tid, ()) if o in mapping and mapping[o] in core_of}
            size = max(0, int(tensor.get('size', 0)))
            for op in self.index.copy_consumers.get(tid, ()):
                original += size
            if not prod and not cons: continue
            has_out = bool(self.index.copy_consumers.get(tid))
            x = len(prod) * len(cons) - len(prod & cons)
            n_in = (len(cons) if not prod and cons else 0) + x
            n_out = (len(prod) if prod and (has_out or not cons) else 0) + x
            n = n_in + n_out
            boundary += size * n
            service += n * max(1, math.ceil(size / self.bandwidth))
            writes += n_out * max(1, math.ceil(size / self.bandwidth))
        for edge in self.index.direct_edges:
            s, d = int(edge['source']), int(edge['target'])
            if s in mapping and d in mapping and mapping[s] in core_of and mapping[d] in core_of and core_of[mapping[s]] != core_of[mapping[d]]:
                size = max(0, int(edge.get('data_size', 0)))
                boundary += 2 * size
                service += 2 * max(1, math.ceil(size / self.bandwidth))
                writes += max(1, math.ceil(size / self.bandwidth))
        # COPY_IN records are input copies, and COPY_OUT records are writebacks.
        for edge in self.graph.get('edges', ()):
            s, d = int(edge['source']), int(edge['target'])
            if s in self.ops and self.ops[s].get('op') == 'COPY_IN' and d in self.tensor_by_id:
                original += max(0, int(self.tensor_by_id[d].get('size', 0)))
        return boundary, original, service, writes

    def _memory(self, orders):
        result = {'UB': {'peak': 0, 'overage': 0, 'area': 0}, 'L1': {'peak': 0, 'overage': 0, 'area': 0}}
        per_core = {}
        for core, seq in orders.items():
            positions = {op: i for i, op in enumerate(seq)}
            uses = defaultdict(list)
            for op in seq:
                for tid in self.in_tensors[op] + self.out_tensors[op]: uses[tid].append(positions[op])
            active = {'UB': {}, 'L1': {}}
            core_stats = {'UB': {'peak': 0, 'overage': 0, 'area': 0}, 'L1': {'peak': 0, 'overage': 0, 'area': 0}}
            for pos, op in enumerate(seq):
                requested = set(self.in_tensors[op] + self.out_tensors[op])
                for tid in requested:
                    t = self.tensor_by_id.get(tid, {})
                    pool = 'UB' if t.get('pos', 'UB') == 'DDR' else t.get('pos', 'UB')
                    if pool not in active: pool = 'UB'
                    active[pool].setdefault(tid, max(0, int(t.get('size', 0))))
                for pool in ('UB', 'L1'):
                    current = sum(active[pool].values())
                    cap = int(self.capacity.get(pool, 0)); over = max(0, current - cap)
                    st = core_stats[pool]; st['peak'] = max(st['peak'], current); st['overage'] = max(st['overage'], over); st['area'] += over
                for tid in requested:
                    if max(uses.get(tid, [pos])) <= pos:
                        for pool in ('UB', 'L1'): active[pool].pop(tid, None)
            for pool in ('UB', 'L1'):
                for key in core_stats[pool]: result[pool][key] = max(result[pool][key], core_stats[pool][key])
            per_core[core] = core_stats
        return result, per_core

    def _compute_bound(self, orders, service, writes, scene):
        loads = defaultdict(lambda: {'M': 0, 'V': 0})
        for core, seq in orders.items():
            for op in seq:
                pipe = self.ops[op].get('pipe', '')
                bucket = 'M' if pipe == 'PIPE_M' else 'V'
                loads[core][bucket] += max(1, int(self.ops[op].get('cycles', 1)))
        compute = max([1] + [max(v.values()) for v in loads.values()])
        # DAG critical path over original operation dependencies.
        dist = {}
        for op in self.topological:
            dist[op] = max((dist[p] for p in self.predecessors[op]), default=0) + max(1, int(self.ops[op].get('cycles', 1)))
        critical = max(dist.values(), default=0)
        return compute, max(compute, critical, service if scene == 'B' else compute)

    def evaluate(self, plan, scene='B'):
        if scene not in ('B', 'C'): raise ValueError('场景必须为B或C')
        mapping, core_of, orders = self._orders(plan)
        boundary, original, service, writes = self._boundary_metrics(plan)
        memory, _ = self._memory(orders)
        compute, lower = self._compute_bound(orders, service, writes, scene)
        if scene == 'B':
            # B's cheap bound includes required graph-input reads; C deliberately excludes all reads.
            input_reads = 0
            for tid, tensor in self.tensor_by_id.items():
                if self.index.producers.get(tid):
                    continue
                cores = {core_of[mapping[o]] for o in self.index.consumers.get(tid, ()) if o in mapping and mapping[o] in core_of}
                input_reads += len(cores) * max(1, math.ceil(max(0, int(tensor.get('size', 0))) / self.bandwidth))
            lower = max(lower, input_reads)
        local = {int(c): [int(o) for o in seq] for c, seq in orders.items()}
        hot = []
        for core, seq in orders.items():
            if not seq: continue
            hot.append({'core': int(core), 'ops': [int(o) for o in seq[:32]], 'pressure': memory['UB']['overage']})
        return {'boundary_bytes': int(boundary), 'original_copy_bytes': int(original),
                'partition_added_bytes': int(boundary - original), 'compute_lower_bound': int(compute),
                'lower_bound': int(lower), 'mandatory_write_cycles': int(writes),
                'boundary_service_cycles': int(service), 'per_pool': memory,
                'local_orders': local, 'hotspot_windows': hot[:2], 'area_unit': '字节×序位置'}
