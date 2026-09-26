"""A同核相邻块回并：父轨迹筛选、路径匹配、完整合法性检查。

时间余量与边界字节只用于生成候选；不预测合并后的竞争、溢出或总时间。
"""
from collections import Counter, defaultdict
import copy
import math

from scenario_contract import validate_plan
from task_order_search import replay_tasks


def path_matching(order, weights):
    """路径上的最大权不重叠匹配，线性时间；平分时保留较早方案。"""
    values = [0.0] * (len(order) + 1)
    take = [False] * (len(order) + 1)
    for i in range(2, len(order) + 1):
        pair = (order[i-2], order[i-1])
        w = weights.get(pair, 0)
        if not math.isfinite(w):
            raise ValueError('匹配权重必须有限')
        values[i] = values[i-1]
        if w > 0 and values[i-2] + w > values[i]:
            values[i], take[i] = values[i-2] + w, True
    pairs, i = [], len(order)
    while i >= 2:
        if take[i]:
            pairs.append((order[i-2], order[i-1])); i -= 2
        else:
            i -= 1
    return pairs[::-1]


def _boundary_index(graph, mapping):
    ops = {o['id']: o for o in graph['ops']}
    tensors = {t['id']: t for t in graph.get('tensors', [])}
    prod, cons, original_out = defaultdict(set), defaultdict(set), set()
    for e in graph.get('edges', []):
        u, v = e['source'], e['target']
        if u in mapping and v in tensors:
            prod[v].add(mapping[u])
        if u in tensors and v in mapping:
            cons[u].add(mapping[v])
        if u in tensors and v in ops and ops[v].get('op') == 'COPY_OUT':
            original_out.add(u)
    adjacent = defaultdict(set)
    for t in tensors:
        for g in prod[t] | cons[t]:
            adjacent[g].add(t)

    def saving(g, h):
        groups, result = {g, h}, 0
        for t in adjacent[g] | adjacent[h]:
            p, c = prod[t], cons[t]
            old = len((c-p) & groups)
            old += sum(t in original_out or not c or bool(c-{a}) for a in p & groups)
            new = int(bool(c & groups) and not bool(p & groups))
            new += int(bool(p & groups) and (t in original_out or not c or bool(c-groups)))
            result += tensors[t]['size'] * (old-new)
        return result
    return saving


def generate_observed_merge_candidates(graph, parent, raw, max_candidates=3, max_ops=240):
    """返回最多三份完整候选和筛选审计；不修改输入、不迁核。"""
    if type(max_candidates) is not int or not 1 <= max_candidates <= 3:
        raise ValueError('回并候选预算必须在1～3')
    if type(max_ops) is not int or max_ops < 2:
        raise ValueError('合并算子上限至少为2')
    validate_plan(graph, parent, 'A')
    mapping = {int(k): v for k, v in parent['node_to_subgraph'].items()}
    members = Counter(mapping.values())
    tasks, by_group = {}, {}
    for c in raw['per_core_timeline']:
        for t in c['tasks']:
            if t['task_id'] in tasks or t['subgraph_id'] in by_group:
                raise ValueError('A父轨迹任务或子图重复')
            tasks[t['task_id']] = dict(t, core=c['core_id'])
            by_group[t['subgraph_id']] = t['task_id']
    if set(by_group) != set(members):
        raise ValueError('父轨迹与方案子图不一致')
    orders = [[by_group[g] for g in row] for row in parent['core_schedules']]
    for core, row in enumerate(orders):
        if any(tasks[t]['core'] != core for t in row):
            raise ValueError('父轨迹与方案分核不一致')
    preds, succ = {t: set() for t in tasks}, {t: set() for t in tasks}
    for e in raw['task_dependencies']:
        preds[e['target']].add(e['source']); succ[e['source']].add(e['target'])
    same, cross = raw['task_same_core_wait_cycles'], raw['task_cross_core_wait_cycles']
    bandwidth = raw['bandwidth_bytes_per_cycle']
    if any(not math.isfinite(v) or v < 0 for v in (same, cross)) or not math.isfinite(bandwidth) or bandwidth <= 0:
        raise ValueError('父轨迹硬件参数非法')
    replay = replay_tasks({t: r['duration'] for t, r in tasks.items()}, preds, orders, same, cross)
    if replay is None or replay['makespan'] != raw['makespan'] or any(
            replay['finish'][t] != r['end'] or r['end']-r['duration'] != r['start'] for t, r in tasks.items()):
        raise ValueError('父轨迹不能按原时长和等待精确重建')
    critical = set(replay['critical'])
    saved_bytes = _boundary_index(graph, mapping)
    eligible, rejects = {}, Counter()
    def lag(a, b):
        return cross if tasks[a]['core'] != tasks[b]['core'] else 0
    for row in orders:
        for g, h in zip(row, row[1:]):
            a, b = tasks[g], tasks[h]
            if members[a['subgraph_id']] + members[b['subgraph_id']] > max_ops:
                rejects['size'] += 1; continue
            if any(tasks[p]['end'] + lag(p, g) > a['start'] for p in preds[h]-{g}):
                rejects['new_input_wait'] += 1; continue
            if any(tasks[s]['start'] < b['end'] + lag(h, s) for s in succ[g]-{h}):
                rejects['delayed_output'] += 1; continue
            saving = saved_bytes(a['subgraph_id'], b['subgraph_id'])
            eligible[g, h] = dict(boundary_bytes_saved_estimate=saving,
                score=max(0, saving)/bandwidth + same,
                critical=g in critical and h in critical)
    audit = dict(eligible_pairs=len(eligible), rejections=dict(rejects),
                 max_ops=max_ops, max_candidates=max_candidates, duplicate_candidates=0,
                 invalid_candidates=[], scope='父实测轨迹筛选；边界估计不含溢出与竞争，最终必须官方评估')
    rows, seen = [], set()
    for menu in ('boundary_and_wait', 'critical_boundary_and_wait', 'pair_count'):
        weights = {pair: (1 if menu == 'pair_count' else info['score'])
                   for pair, info in eligible.items()
                   if menu != 'critical_boundary_and_wait' or info['critical']}
        pairs = [pair for order in orders for pair in path_matching(order, weights)]
        signature = tuple(pairs)
        if not pairs:
            continue
        if signature in seen:
            audit['duplicate_candidates'] += 1; continue
        seen.add(signature)
        replace = {tasks[h]['subgraph_id']: tasks[g]['subgraph_id'] for g, h in pairs}
        plan = copy.deepcopy(parent)
        plan['node_to_subgraph'] = {k: replace.get(v, v) for k, v in parent['node_to_subgraph'].items()}
        plan['core_schedules'] = [[g for g in order if g not in replace] for order in parent['core_schedules']]
        try:
            validate_plan(graph, plan, 'A')
        except (ValueError, KeyError, TypeError) as exc:
            audit['invalid_candidates'].append(dict(menu=menu, error=str(exc))); continue
        rows.append(dict(plan=plan, source=dict(module='observed_task_merge', menu=menu,
            merged_pairs=len(pairs), boundary_bytes_saved_estimate=sum(eligible[p]['boundary_bytes_saved_estimate'] for p in pairs),
            removed_same_core_wait_estimate=len(pairs)*same,
            pairs=[[tasks[g]['subgraph_id'], tasks[h]['subgraph_id']] for g, h in pairs])))
        if len(rows) >= max_candidates:
            break
    return rows, audit
