"""末端精评只推荐挑战者；最终优劣必须由官方评估裁决。"""
import hashlib
import json
import math
import time

from scene_a_event import SceneAEventModel


def plan_signature(plan):
    return json.dumps({'n': plan['node_to_subgraph'],
                       'c': plan['core_schedules']}, sort_keys=True)


def diverse_shortlist(rows, max_precise=12):
    by_time = sorted(rows, key=lambda r: (r['mk'], r['ad'], r['plan_id']))
    by_traffic = sorted(rows, key=lambda r: (r['ad'], r['mk'], r['plan_id']))
    shortlist, used = [], set()
    groups = {}
    for row in by_time:
        if not row['source'].startswith('reservoir:'):
            continue
        _, paradigm, granularity, level = row['source'].split(':')
        key = paradigm, granularity
        # 同一范式粒度先保留原始构造，精化候选进入后面的按分数补齐。
        if key not in groups or (level == 'R0' and not groups[key]['source'].endswith(':R0')):
            groups[key] = row
    queues = {p: sorted([r for (paradigm, _), r in groups.items() if paradigm == p],
                        key=lambda r: (r['mk'], r['ad'], r['plan_id']))
              for p in ('heft', 'strip', 'chain', 'netbenefit')}
    reserve = min(8, max_precise)
    for depth in range(3):
        for queue in queues.values():
            if depth < len(queue) and len(shortlist) < reserve:
                row = queue[depth]
                shortlist.append(row)
                used.add(row['plan_id'])
    for pair in zip(by_time, by_traffic):
        for row in pair:
            if row['plan_id'] not in used and len(shortlist) < max_precise:
                shortlist.append(row)
                used.add(row['plan_id'])
    return shortlist


def diverse_finalists(ranked, limit, verified_partitions=None):
    verified_partitions = verified_partitions or set()
    # 排序稳定：预测并列按生成顺序；优先未官方评估过的切分。
    ranked = sorted(ranked, key=lambda row:
                    tuple(sorted(row['plan']['node_to_subgraph'].items())) in verified_partitions)
    selected, partitions = [], set()
    for row in ranked:
        # 不合并子图重编号，只避免同一映射的分核/顺序变体占满两个名额。
        partition = tuple(sorted(row['plan']['node_to_subgraph'].items()))
        if partition in partitions:
            continue
        partitions.add(partition)
        selected.append(row)
        if len(selected) >= limit:
            return selected
    # 只有一种切分时仍允许原有顺序变体进入，不浪费剩余名额。
    selected_ids = {row['plan_id'] for row in selected}
    for row in ranked:
        if len(selected) >= limit:
            break
        if row['plan_id'] not in selected_ids:
            selected.append(row)
            selected_ids.add(row['plan_id'])
    return selected


def recommend_challengers(graph, ctx, candidates, seen, max_precise=12,
                          max_official=2, seconds=3.0, diversity=False,
                          excluded_plan_ids=None):
    started = time.perf_counter()
    audit = {'errors': [], 'evaluated': [], 'precise_count': 0,
             'official_count': 0, 'max_precise': max_precise,
             'max_official': max_official, 'soft_budget_seconds': seconds}
    unique = {}
    for source, candidate in candidates:
        if time.perf_counter() - started >= seconds:
            break
        try:
            sol = candidate.clone()
            sol.compact()
            mk, ad, info = ctx.evaluate(sol)
            plan = ctx.model.plan_from(sol.sg_of_block, info['orders'])
            sig = plan_signature(plan)
            plan_id = hashlib.sha256(sig.encode('utf-8')).hexdigest()
            if sig in seen or sig in unique or plan_id in (excluded_plan_ids or ()):
                continue
            unique[sig] = {'plan': plan, 'mk': mk, 'ad': ad, 'info': info,
                           'source': source, 'sig': sig,
                           'plan_id': plan_id}
        except Exception as exc:
            audit['errors'].append({'source': source, 'stage': 'decode', 'error': repr(exc)})
    audit['unique_available'] = len(unique)
    shortlist = diverse_shortlist(list(unique.values()), max_precise)
    evaluator = SceneAEventModel(graph) if shortlist else None
    evaluated = []
    for row in shortlist:
        if time.perf_counter() - started >= seconds:
            break
        audit['precise_count'] += 1
        try:
            result = evaluator.evaluate(row['plan'])
            if not math.isfinite(result['makespan']):
                raise ValueError('精评耗时不是有限数值')
            row['precise_makespan'] = result['makespan']
            row['precise_added'] = result['total_added_bytes']
            audit['evaluated'].append({k: row[k] for k in
                ('plan_id', 'source', 'mk', 'ad', 'precise_makespan', 'precise_added')})
            evaluated.append(row)
        except Exception as exc:
            audit['errors'].append({'plan_id': row['plan_id'], 'stage': 'precise', 'error': repr(exc)})
    evaluated.sort(key=lambda r: (r['precise_makespan'], r['precise_added'],
                                  '' if diversity else r['plan_id']))
    audit['precise_seconds'] = time.perf_counter() - started
    audit['soft_budget_exhausted'] = audit['precise_seconds'] >= seconds
    verified_partitions = {tuple(sorted(json.loads(sig)['n'].items())) for sig in seen}
    finalists = (diverse_finalists(evaluated, max_official, verified_partitions)
                 if diversity else evaluated[:max_official])
    return finalists, audit


def refine_reservoir_branch(graph, ctx, plan, baseline, seen, excluded_ids,
                            official_evaluator):
    """原路径已完成后，才验证构造旁路，且只精修真正胜出的挑战者。"""
    from task_order_search import refine_plan_orders
    started = time.perf_counter()
    sources = [(f'reservoir:{item.paradigm}:{item.granularity}:{item.refine_level}', item.sol)
               for item in getattr(ctx, 'construct_reservoir', ())]
    challengers, ranking = recommend_challengers(
        graph, ctx, sources, seen, diversity=True, excluded_plan_ids=excluded_ids)
    selected, best, records = plan, baseline, []
    for row in challengers:
        truth = official_evaluator(graph, row['plan'], 'A')
        records.append({'plan': row['plan'], 'estimate': row['precise_makespan'],
                        'source': row['source'], 'official': truth})
        if truth is None or truth.get('error'):
            continue
        if (truth['makespan'], truth['added_copy_bytes']) < (best['makespan'], best['added_copy_bytes']):
            selected, best = row['plan'], truth
    refinement = None
    if selected is not plan:
        selected, best, refinement = refine_plan_orders(graph, selected, best, official_evaluator)
        for candidate in refinement['candidates']:
            records.append({'plan': {**selected, 'core_schedules': candidate['orders']},
                            'estimate': candidate['estimate'],
                            'source': 'reservoir_order', 'official': candidate['official']})
    audit = {'baseline_official': baseline, 'final_official': best,
             'reservoir_count': len(sources), 'ranking': ranking,
             'order_refinement': refinement, 'candidates': records,
             'official_count': len(records), 'errors': ranking['errors'] +
             (refinement['errors'] if refinement else []),
             'extra_seconds': time.perf_counter() - started}
    return selected, best, audit
