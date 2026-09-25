"""E02：固定图/粒度/分核流程，原净收益构造与可选修正逐对官方比较。"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import statistics
import time

from run_all import (ensure_run_manifest, run_input_files, parse_cases, block_cap_for,
                     append_run_row, plan_digest, valid_official)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / '通用神经网络处理器下的多核调度问题附件/data'
DEVELOPMENT = '1,5,7,12,15,23,24,35,42,44,45,49,55,63,65,67,75,78,82,83,95,97'


def evaluate_pair(task):
    from model import Model
    from solution import Sol, Context
    from construct_v2 import netbenefit_construct
    from construct_refine import resolve_granularity, _coarsen_solution
    from pipeline import real_evaluate
    case, n, granularity = task['case'], task['N'], task['granularity']
    graph = json.loads((DATA / f'{case}.json').read_text(encoding='utf-8'))
    eligible = sum(o['op'] not in ('COPY_IN', 'COPY_OUT') for o in graph['ops'])
    model = Model(graph, block_ops_cap=block_cap_for(eligible))
    profile = resolve_granularity(model, n, granularity)
    result = {'case': case, 'N': n, 'granularity': granularity, 'id': task['id'],
              'fingerprint': task['fingerprint'], 'attempt': task['attempt'],
              'n_ops': len(graph['ops']), 'n_eligible': eligible,
              'production_granularity': eligible <= 2000 or granularity == 'balanced',
              'target_ops': profile.target_ops, 'target_strips': profile.target_strips,
              'matrix_work': sum(model.block_work_m), 'vector_work': sum(model.block_work_v),
              'variants': {}}
    ctx = Context(model, n, 'A')
    modes = (False, True) if task['index'] % 2 == 0 else (True, False)
    # 交替执行先后，两个模式使用相同块模型、参数、收缩和后续派生顺序。
    for corrected in modes:
        label = 'corrected' if corrected else 'legacy'
        started = time.perf_counter()
        raw_sg, raw_core = netbenefit_construct(model, n, 'A',
            max_sg_ops=profile.target_ops, mem_budget=0.85, corrected=corrected)
        raw = Sol(raw_sg, raw_core)
        raw_counts = Counter()
        for b, s in enumerate(raw.sg_of_block):
            raw_counts[s] += len(model.blocks[b])
        sol = _coarsen_solution(model, raw, profile.target_strips)
        valid_before_repair = sol.validate(model, n)
        proxy_mk, proxy_added, info = ctx.evaluate(sol, use_cache=False)
        if not sol.validate(model, n):
            raise ValueError(f'{label} 按正式流程修复后仍非法')
        plan = model.plan_from(sol.sg_of_block, info['orders'])
        construct_seconds = time.perf_counter() - started
        pp = Path(task['plans']) / f'{case}_A_N{n}_{granularity}_{label}.json'
        pp.write_text(json.dumps(plan, ensure_ascii=False), encoding='utf-8')
        before = time.perf_counter()
        truth = real_evaluate(graph, plan, 'A')
        official_seconds = time.perf_counter() - before
        if not valid_official(truth):
            raise ValueError(f'{label} 官方复核失败: {truth}')
        counts = Counter(plan['node_to_subgraph'].values())
        result['variants'][label] = {
            'real': truth, 'plan_path': str(pp), 'plan_id': plan_digest(plan),
            'plan_sha256': hashlib.sha256(pp.read_bytes()).hexdigest(),
            'raw_task_count': len(raw_counts), 'raw_max_ops': max(raw_counts.values()),
            'raw_over_target': sum(v > profile.target_ops for v in raw_counts.values()),
            'task_count': len(counts), 'max_ops': max(counts.values()),
            'construct_seconds': construct_seconds, 'official_seconds': official_seconds,
            'valid_before_repair': valid_before_repair,
            'proxy_makespan': proxy_mk, 'proxy_added': proxy_added,
            'proxy_spill_bytes': info.get('spill_bytes'),
            'over_target_after_coarsen': sum(v > profile.target_ops for v in counts.values())}
        core_m, core_v = [0.0]*n, [0.0]*n
        for b, sg in enumerate(sol.sg_of_block):
            c = sol.core_of_sg[sg]
            core_m[c] += model.block_work_m[b]
            core_v[c] += model.block_work_v[b]
        result['variants'][label].update(core_matrix_work=core_m, core_vector_work=core_v,
            compute_load_bound=max(max(core_m), max(core_v)))
        if task['audit_raw']:
            raw_copy = raw.clone()
            raw_valid = raw_copy.validate(model, n)
            rm, ra, ri = ctx.evaluate(raw_copy, use_cache=False)
            raw_plan = model.plan_from(raw_copy.sg_of_block, ri['orders'])
            rp = pp.with_name(pp.stem+'_raw.json')
            rp.write_text(json.dumps(raw_plan,ensure_ascii=False),encoding='utf-8')
            raw_real = real_evaluate(graph, raw_plan, 'A')
            if not valid_official(raw_real):
                raise ValueError(f'{label} 原始构造诊断官方失败: {raw_real}')
            result['variants'][label]['raw_audit'] = {'real':raw_real,'proxy_makespan':rm,
                'valid_before_repair':raw_valid,'plan_path':str(rp),'plan_sha256':hashlib.sha256(rp.read_bytes()).hexdigest(),
                'task_count':len(set(raw_plan['node_to_subgraph'].values()))}

    a, b = result['variants']['legacy'], result['variants']['corrected']
    result.update(status='official_success', relative_change=b['real']['makespan']/a['real']['makespan']-1,
                  same_plan=a['plan_id'] == b['plan_id'])
    result['binding'] = hashlib.sha256(json.dumps(result['variants'], sort_keys=True).encode()).hexdigest()
    return result


def valid_pair(row, fingerprint):
    if row.get('status') != 'official_success' or row.get('fingerprint') != fingerprint:
        return False
    variants = row.get('variants', {})
    if set(variants) != {'legacy', 'corrected'}:
        return False
    if row.get('binding') != hashlib.sha256(json.dumps(variants, sort_keys=True).encode()).hexdigest():
        return False
    for v in variants.values():
        p = Path(v['plan_path'])
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != v['plan_sha256']:
            return False
        if not valid_official(v['real']):
            return False
        raw = v.get('raw_audit')
        if raw and (not Path(raw['plan_path']).is_file() or hashlib.sha256(Path(raw['plan_path']).read_bytes()).hexdigest() != raw['plan_sha256']):
            return False
    return True


def summarize(rows):
    if not rows:
        return {'count': 0}
    return {'count': len(rows), 'wins': sum(r['relative_change'] < 0 for r in rows),
            'losses': sum(r['relative_change'] > 0 for r in rows),
            'ties': sum(r['relative_change'] == 0 for r in rows),
            'same_plans': sum(r['same_plan'] for r in rows),
            'mean_relative_change': statistics.mean(r['relative_change'] for r in rows),
            'total_time_ratio_change': sum(r['variants']['corrected']['real']['makespan'] for r in rows)/
                                      sum(r['variants']['legacy']['real']['makespan'] for r in rows)-1,
            'regressions_over_5pct': sum(r['relative_change'] > .05 for r in rows)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cases', default=DEVELOPMENT)
    parser.add_argument('--cores', default='2,3,4,5')
    parser.add_argument('--granularities', default='coarse,balanced,fine')
    parser.add_argument('--raw-audit-cases', default='44,49,65,82', help='额外评估收缩前构造，定位候选潜力被后处理掩盖的情况')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--max-pairs', type=int, default=0, help='本次处理上限，0 为全部待完成配对')
    args = parser.parse_args()
    if args.workers < 1 or args.max_pairs < 0:
        parser.error('并发必须为正，配对上限不能为负')
    cases = parse_cases(args.cases)
    raw_cases = parse_cases(args.raw_audit_cases) if args.raw_audit_cases else []
    cores = list(dict.fromkeys(map(int, args.cores.split(','))))
    grains = list(dict.fromkeys(args.granularities.split(',')))
    if any(n not in (2,3,4,5) for n in cores) or any(g not in ('coarse','balanced','fine') for g in grains):
        parser.error('核数或粒度非法')
    folder = Path(args.output_dir).resolve()
    plans = folder / 'plans'
    plans.mkdir(parents=True, exist_ok=True)
    out = folder / 'pairs.jsonl'
    fingerprint = ensure_run_manifest(out, run_input_files(cases), {
        'cases': cases, 'cores': cores, 'granularities': grains, 'workers': args.workers,
        'method': 'netbenefit_pair_with_common_coarsening_and_production_repair', 'mem_budget': .85,
        'raw_audit_cases':raw_cases,
        'output_dir': str(folder)})
    old, malformed = {}, []
    if out.exists():
        for i, line in enumerate(out.read_text(encoding='utf-8').splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                old[row['id']] = row
            except (ValueError, KeyError):
                malformed.append(i)
    tasks = []
    for c in cases:
        for n in cores:
            for g in grains:
                key = f'{c}_N{n}_{g}'
                tasks.append({'id': key, 'case': c, 'N': n, 'granularity': g,
                    'index': len(tasks), 'fingerprint': fingerprint, 'plans': str(plans),
                    'audit_raw': c in raw_cases,
                    'attempt': old.get(key, {}).get('attempt', 0)+1})
    done = {key for key, row in old.items() if valid_pair(row, fingerprint)}
    pending = [t for t in tasks if t['id'] not in done]
    if args.max_pairs:
        pending = pending[:args.max_pairs]
    print(f'[{len(done)}/{len(tasks)}] E02 构造配对；本批 {len(pending)}，损坏行 {malformed}', flush=True)
    started = time.perf_counter()
    if pending:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(evaluate_pair, task): task for task in pending}
            for future in as_completed(futures):
                task = futures[future]
                try:
                    row = future.result()
                    done.add(task['id'])
                except Exception as exc:
                    row = {'id': task['id'], 'status': 'failed', 'error': repr(exc),
                           'fingerprint': fingerprint, 'attempt': task['attempt']}
                append_run_row(out, row)
                old[task['id']] = row
                print(f'[{len(done)}/{len(tasks)}] {task["id"]}: {row["status"]}, '
                      f'耗时变化={row.get("relative_change")}', flush=True)
    complete = [old[t['id']] for t in tasks if t['id'] in done]
    summary = {'expected_pairs': len(tasks), 'completed_pairs': len(complete),
               'fingerprint': fingerprint, 'wall_seconds_this_batch': time.perf_counter()-started,
               'all': summarize(complete),
               'production_granularities': summarize([r for r in complete if r['production_granularity']]),
               'by_granularity': {g:summarize([r for r in complete if r['granularity']==g]) for g in grains},
               'by_core': {str(n):summarize([r for r in complete if r['N']==n]) for n in cores},
               'worst_regressions': sorted(complete, key=lambda r:-r['relative_change'])[:10],
               'scope': '固定构造及相同收缩/顺序的配对；非完整求解器性能，不据此替换默认种子。'}
    (folder/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:summary[k] for k in ('expected_pairs','completed_pairs','all','production_granularities')},ensure_ascii=False))
    if any(r.get('status') == 'failed' for r in old.values()):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
