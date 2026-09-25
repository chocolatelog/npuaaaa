"""边界字节引导的合并与多粒度拆分；只在末端用精评和官方值裁决。"""
import itertools
import math
import time

from model import Model
from run_all import block_cap_for
from scene_a_event import SceneAEventModel, derive_multicore_plan
from solution import Sol
from task_order_search import search_orders
from partition_identity import select_distinct_partitions


def split_solution(model, parent, task, fraction, target_core, num_cores):
    blocks = sorted(parent.blocks_in_sg[task], key=lambda b: model.block_pos[b])
    if len(blocks) < 2 or not 0 < fraction < 1:
        return None
    weights = [max(1.0, model.block_work_m[b], model.block_work_v[b]) for b in blocks]
    cumulative = list(itertools.accumulate(weights))
    cut = min(range(1, len(blocks)), key=lambda i: abs(cumulative[i-1] - cumulative[-1]*fraction))
    child = parent.clone()
    new_id = len(child.core_of_sg)
    child.core_of_sg.append(target_core)
    child.blocks_in_sg.append(set())
    for block in blocks[cut:]:
        child.move_block(block, new_id)
    child.compact()
    return child if child.validate(model, num_cores) else None


def merge_solution(model, parent, first, second, num_cores):
    child = parent.clone()
    child.merge_sg(first, second)
    child.compact()
    return child if child.validate(model, num_cores) else None


def boundary_savings(evaluator, mapping, first, second):
    """逐张量判断合并前后复制次数，保留其他消费者需要的输出副本。"""
    total = 0
    for tid, tensor in evaluator.tensors.items():
        producers = {mapping[o] for o in evaluator.producers[tid] if o in mapping}
        consumers = {mapping[o] for o in evaluator.consumers[tid] if o in mapping}
        if not ({first, second} & (producers | consumers)):
            continue
        original_output = any(evaluator.ops[o]['op'] == 'COPY_OUT'
                              for o in evaluator.consumers[tid])
        def copies(members):
            produced = bool(producers & members)
            incoming = bool(consumers & members) and not produced
            outgoing = produced and (original_output or not consumers or bool(consumers - members))
            return int(incoming) + int(outgoing)
        total += tensor['size'] * (copies({first}) + copies({second}) - copies({first, second}))
    return total


def polish_partition(graph, plan, baseline, official_evaluator, seconds=4.0, max_precise=12,
                     model=None, local_search_seconds=0.06):
    started = time.perf_counter()
    def exhausted():
        return seconds is not None and time.perf_counter() - started >= seconds
    n = len(plan['core_schedules'])
    eligible = sum(o['op'] not in ('COPY_IN', 'COPY_OUT') for o in graph['ops'])
    if model is None:
        model = Model(graph, block_ops_cap=block_cap_for(eligible))
    sg = [plan['node_to_subgraph'][str(block[0])] for block in model.blocks]
    cores = [0] * (max(sg) + 1)
    for c, order in enumerate(plan['core_schedules']):
        for s in order:
            cores[s] = c
    if model.plan_from(sg, plan['core_schedules']) != plan:
        raise ValueError('输入切分不能无损映射到块模型')
    parent = Sol(sg, cores)
    if not parent.validate(model, n):
        raise ValueError('输入方案块覆盖或子图依赖非法')
    evaluator = SceneAEventModel(graph)
    estimate = evaluator.evaluate(plan)
    view = derive_multicore_plan(graph, plan)
    model_seconds = time.perf_counter() - started
    stats = {'precise_count': 0, 'invalid_variants': 0, 'duplicates': 0,
             'base_makespan': estimate['makespan'], 'evaluated': [],
             'soft_budget_seconds': seconds, 'max_precise': max_precise}
    merge_rows = []
    for a, b in view['dependency_pairs']:
        if exhausted():
            break
        saved = boundary_savings(evaluator, view['mapping'], a, b)
        if saved > 0:
            merge_rows.append((saved, a, b))
    merges = []
    for saved, a, b in sorted(merge_rows, key=lambda r: (-r[0], r[1], r[2]))[:6]:
        merges.append(({'kind': 'merge', 'tasks': [a, b], 'boundary_saved_bytes': saved},
                       merge_solution(model, parent, a, b, n)))
    loads = [sum(estimate['tasks'][s]['duration'] for s in order) for order in plan['core_schedules']]
    heavy = sorted((s for s in estimate['tasks'] if len(parent.blocks_in_sg[s]) >= 2),
                   key=lambda s: (-estimate['tasks'][s]['duration'], s))[:3]
    splits = []
    for fraction in (1/3, 1/2, 2/3):
        for task in heavy:
            removed_work = estimate['tasks'][task]['duration'] * (1-fraction)
            target = min(range(n), key=lambda c: (loads[c] - (removed_work if c == cores[task] else 0), c))
            splits.append(({'kind': 'split', 'task': task, 'fraction': fraction},
                           split_solution(model, parent, task, fraction, target, n)))
    variants = []
    for pair in itertools.zip_longest(merges, splits):
        variants.extend(row for row in pair if row is not None)
    seen, errors, evaluated = set(), [], []
    for source, candidate in variants:
        if stats['precise_count'] >= max_precise or exhausted():
            break
        if candidate is None:
            stats['invalid_variants'] += 1
            continue
        sig = tuple(candidate.sg_of_block), tuple(candidate.core_of_sg)
        if sig in seen:
            stats['duplicates'] += 1
            continue
        seen.add(sig)
        stats['precise_count'] += 1
        try:
            _, _, info = model.evaluate(candidate.sg_of_block, candidate.core_of_sg, 'A', n)
            proposal = model.plan_from(candidate.sg_of_block, info['orders'])
            precise = evaluator.evaluate(proposal)
            if not math.isfinite(precise['makespan']):
                raise ValueError('切分候选精评耗时不是有限数值')
            candidate_view = derive_multicore_plan(graph, proposal)
            local, local_stats = search_orders(
                {s: t['duration'] for s, t in precise['tasks'].items()},
                candidate_view['subgraph_preds'], proposal['core_schedules'],
                bandwidth_floor=precise['total_copy_bytes']/60,
                max_evals=400, rounds=2, seconds=local_search_seconds, keep=1)
            predicted = precise['makespan']
            if local and local[0]['makespan'] < predicted:
                proposal['core_schedules'] = local[0]['orders']
                predicted = local[0]['makespan']
            row = {'source': source, 'estimate': predicted,
                   'partition_added': precise['partition_added_bytes'],
                   'spill_added': precise['spill_bytes'], 'total_added': precise['total_added_bytes'],
                   'local_evaluations': local_stats['evaluations'], 'plan': proposal}
            stats['evaluated'].append({k: v for k, v in row.items() if k != 'plan'})
            evaluated.append(row)
        except Exception as exc:
            errors.append({'stage': 'precise', 'source': source, 'error': repr(exc)})
    stats['seconds'] = time.perf_counter() - started
    stats['soft_budget_exhausted'] = exhausted()
    ranked = sorted(evaluated, key=lambda r: (r['estimate'], r['total_added']))
    # 每种不同切分只复核一个顺序，避免两次复核浪费在同一切分的微小变体。
    selected_rows = select_distinct_partitions(
        (row for row in ranked if (row['estimate'], row['total_added']) <
         (estimate['makespan'], estimate['total_added_bytes'])), lambda row: row['plan'], 2)
    best_plan, best, records = plan, baseline, []
    for row in selected_rows:
        truth = official_evaluator(graph, row['plan'], 'A')
        record = {**row, 'official': truth}
        records.append(record)
        if truth is None or truth.get('error'):
            errors.append({'stage': 'official', 'result': truth})
            continue
        if (truth['makespan'], truth['added_copy_bytes']) < (best['makespan'], best['added_copy_bytes']):
            best_plan, best = row['plan'], truth
    return best_plan, best, {'baseline_official': baseline, 'final_official': best,
                            'search': stats, 'model_seconds': model_seconds,
                            'candidates': records, 'official_count': len(records),
                            'errors': errors, 'extra_seconds': time.perf_counter() - started}
