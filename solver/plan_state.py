"""原算子方案物化；任何局部操作都保留完整映射与真实核心列表。"""
import heapq

from contracts import PlanState, content_digest
from scenario_contract import validate_plan, operation_dependencies


def from_plan(graph, plan, config_digest='', scenario='B'):
    view = validate_plan(graph, plan, scenario)
    return PlanState(content_digest(graph), config_digest,
                     tuple(sorted(view['mapping'].items())),
                     tuple(tuple(row) for row in plan['core_schedules']))


def _materialize(graph, parent, mapping, orders, scenario='B'):
    if parent.graph_digest != content_digest(graph):
        raise ValueError('方案与输入图身份不一致')
    return from_plan(graph, {'node_to_subgraph': {str(o): s for o, s in mapping.items()},
                            'core_schedules': [list(row) for row in orders]},
                     parent.config_digest, scenario)


def topological_ops(graph):
    preds, succs = operation_dependencies(graph)
    degree = {o: len(ps) for o, ps in preds.items()}
    ready = [o for o, d in degree.items() if not d]; heapq.heapify(ready)
    result = []
    while ready:
        op = heapq.heappop(ready); result.append(op)
        for child in sorted(succs[op]):
            degree[child] -= 1
            if not degree[child]: heapq.heappush(ready, child)
    if len(result) != len(preds): raise ValueError('原操作依赖成环')
    return result


def refine_window(graph, parent, hot_ops, granularity=16, local_orders=None, scenario='B'):
    if type(granularity) is not int or granularity < 1:
        raise ValueError('微块粒度必须为正整数')
    hot = set(hot_ops); mapping = parent.mapping
    if not hot <= mapping.keys(): raise ValueError('热点包含未知原算子')
    if not hot: return parent
    order = topological_ops(graph)
    by_core = local_orders or {c: order for c in range(len(parent.orders))}
    touched = {mapping[o] for o in hot}
    members = parent.sg_members; core_of = parent.core_of
    new_mapping = dict(mapping); replacements = {}; next_id = max(members, default=-1)+1
    for sg in sorted(touched):
        seq = [o for o in by_core.get(core_of[sg], ()) if mapping.get(o) == sg]
        if len(seq) != len(members[sg]) or set(seq) != set(members[sg]):
            raise ValueError('局部原算子序遗漏或重复')
        segments = []; current = []; inside = None
        for op in seq:
            is_hot = op in hot
            if current and (is_hot != inside or (is_hot and len(current) >= granularity)):
                segments.append(current); current=[]
            current.append(op); inside=is_hot
        if current: segments.append(current)
        ids = [sg]
        for _ in segments[1:]: ids.append(next_id); next_id += 1
        replacements[sg] = ids
        for sid, ops in zip(ids, segments):
            for op in ops: new_mapping[op] = sid
    orders = [[new for sg in row for new in replacements.get(sg, [sg])] for row in parent.orders]
    return _materialize(graph, parent, new_mapping, orders, scenario)


def _closure(starts, adjacency):
    found = set(); stack = list(starts)
    while stack:
        op = stack.pop()
        for nxt in adjacency.get(op, ()):
            if nxt not in found: found.add(nxt); stack.append(nxt)
    return found


def legal_slots(graph, parent, group, target_core):
    return group_slots(graph, parent, (group,), target_core)


def group_slots(graph, parent, groups, target_core):
    if type(target_core) is not int or not 0 <= target_core < len(parent.orders):
        raise ValueError('目的核心非法')
    selected = set(groups); members = parent.sg_members
    if not selected or not selected <= members.keys(): raise ValueError('待移动子图不存在')
    ops = {o for s in selected for o in members[s]}
    preds, succs = operation_dependencies(graph)
    before = _closure(ops, preds)-ops; after = _closure(ops, succs)-ops
    row = [s for s in parent.orders[target_core] if s not in selected]
    lo = 1 + max((i for i, s in enumerate(row) if set(members[s]) & before), default=-1)
    hi = min((i for i, s in enumerate(row) if set(members[s]) & after), default=len(row))
    return tuple(range(lo, hi+1))


def move_group(graph, parent, groups, target_core, slot, scenario='B'):
    groups = tuple(groups)
    if len(groups) != len(set(groups)): raise ValueError('重复移动子图')
    if type(slot) is not int or slot not in group_slots(graph, parent, groups, target_core):
        raise ValueError('插入位置不满足传递依赖')
    chosen = set(groups)
    orders = [[sg for sg in row if sg not in chosen] for row in parent.orders]
    orders[target_core][slot:slot] = groups
    return _materialize(graph, parent, parent.mapping, orders, scenario)


def merge_groups(graph, parent, groups, scenario='B'):
    groups=tuple(groups); selected=set(groups); owners=parent.core_of
    if len(selected)<2 or not selected <= owners.keys(): raise ValueError('合并至少两个已存在子图')
    if len({owners[s] for s in selected})!=1: raise ValueError('先明确迁核，再合并同核子图')
    first=next(s for s in parent.orders[owners[groups[0]]] if s in selected)
    mapping={o:first if s in selected else s for o,s in parent.assignments}
    orders=[[s for s in row if s not in selected or s==first] for row in parent.orders]
    return _materialize(graph,parent,mapping,orders,scenario)

