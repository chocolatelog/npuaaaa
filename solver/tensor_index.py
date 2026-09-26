"""原始图张量/边界索引；仅处理非复制原算子，不展开官方任务图。"""
from collections import defaultdict


COPY = {'COPY_IN', 'COPY_OUT'}


def _plan_parts(plan):
    if hasattr(plan, 'mapping'):
        return dict(plan.mapping), dict(plan.core_of), plan.orders
    mapping = {int(k): int(v) for k, v in plan.get('node_to_subgraph', {}).items()}
    rows = tuple(tuple(int(x) for x in row) for row in plan.get('core_schedules', ()))
    return mapping, {sg: c for c, row in enumerate(rows) for sg in row}, rows


class TensorIndex:
    """保留每核生产/消费计数，支持可靠的整量差分迁移。"""

    def __init__(self, graph):
        self.graph = graph
        self.ops = {int(o['id']): o for o in graph.get('ops', [])}
        self.tensors = {int(t['id']): dict(t) for t in graph.get('tensors', [])}
        self.producers = defaultdict(list)
        self.consumers = defaultdict(list)
        self.copy_consumers = defaultdict(list)
        self.direct_edges = []
        opids = set(self.ops)
        for edge in graph.get('edges', []):
            s, d = int(edge['source']), int(edge['target'])
            if s in opids and d not in opids:
                if self.ops[s].get('op') not in COPY:
                    self.producers[d].append(s)
            elif s not in opids and d in opids:
                if self.ops[d].get('op') == 'COPY_OUT':
                    self.copy_consumers[s].append(d)
                elif self.ops[d].get('op') not in COPY:
                    self.consumers[s].append(d)
            elif s in opids and d in opids and self.ops[s].get('op') not in COPY and self.ops[d].get('op') not in COPY:
                self.direct_edges.append(dict(edge))

    def core_counts(self, plan):
        mapping, core_of, _ = _plan_parts(plan)
        result = {}
        for tid in self.tensors:
            prod, cons = defaultdict(int), defaultdict(int)
            for op in self.producers.get(tid, ()):
                if op in mapping and mapping[op] in core_of: prod[core_of[mapping[op]]] += 1
            for op in self.consumers.get(tid, ()):
                if op in mapping and mapping[op] in core_of: cons[core_of[mapping[op]]] += 1
            result[tid] = {'producers': dict(prod), 'consumers': dict(cons)}
        return result

    def migration_delta(self, plan, ops, target_core):
        """按一次组迁移返回边界量差分；采用前后全量公式确保多消费者计数不丢失。"""
        mapping, core_of, rows = _plan_parts(plan)
        before = self.boundary(plan)
        moved = set(int(o) for o in ops)
        changed = dict(mapping)
        target_sg = next((sg for sg, c in core_of.items() if c == target_core), None)
        if target_sg is None:
            target_sg = max(core_of, default=-1) + 1
        for op in moved:
            if op in changed: changed[op] = target_sg
        new_rows = [list(r) for r in rows]
        if target_sg not in {x for r in new_rows for x in r}:
            while len(new_rows) <= target_core: new_rows.append([])
            new_rows[target_core].append(target_sg)
        after = self.boundary({'node_to_subgraph': {str(k): v for k, v in changed.items()}, 'core_schedules': new_rows})
        return {'boundary_bytes_delta': after - before, 'before': before, 'after': after}

    def boundary(self, plan):
        mapping, core_of, _ = _plan_parts(plan)
        total = 0
        for tid, tensor in self.tensors.items():
            prod = {core_of[mapping[o]] for o in self.producers.get(tid, ()) if o in mapping and mapping[o] in core_of}
            cons = {core_of[mapping[o]] for o in self.consumers.get(tid, ()) if o in mapping and mapping[o] in core_of}
            if not prod and not cons:
                continue
            size = max(0, int(tensor.get('size', 0)))
            has_out = bool(self.copy_consumers.get(tid))
            x = len(prod) * len(cons) - len(prod & cons)
            total += size * ((len(cons) if not prod and cons else 0) +
                             (len(prod) if prod and (has_out or not cons) else 0) + 2 * x)
        for edge in self.direct_edges:
            s, d = int(edge['source']), int(edge['target'])
            if s in mapping and d in mapping and mapping[s] in core_of and mapping[d] in core_of and core_of[mapping[s]] != core_of[mapping[d]]:
                total += 2 * max(0, int(edge.get('data_size', 0)))
        return total
