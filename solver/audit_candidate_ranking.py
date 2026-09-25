"""在同一算例/核数的多版本保存方案中比较代理排序，避免跨算例相关性假象。"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import statistics
import time

from model import Model
from run_all import block_cap_for
from calibrate_spill import decode_plan
from pipeline import real_evaluate
from scene_a_event import SceneAEventModel

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIRS = ('plans_fixed_v1', 'plans_control_base_v1',
                'plans_control_calibrated_v1', 'plans_control_calibrated_gated_v1',
                'plans_control_calibrated_gated_naware_v1', 'plans_control_lru_v1')


def signature(plan):
    return hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()


def run_group(task):
    case, n, directories = task
    g = json.loads((ROOT / '通用神经网络处理器下的多核调度问题附件/data' / f'{case}.json').read_text(encoding='utf-8'))
    eligible = sum(o['op'] not in ('COPY_IN', 'COPY_OUT') for o in g['ops'])
    model = Model(g, block_ops_cap=block_cap_for(eligible))
    precise = SceneAEventModel(g)
    plans, rows = {}, []
    for directory in directories:
        path = Path(directory) / f'{case}_A_N{n}.json'
        if not path.exists():
            continue
        plan = json.loads(path.read_text(encoding='utf-8'))
        sig = signature(plan)
        if sig not in plans:
            plans[sig] = (plan, [])
        plans[sig][1].append(str(path))
    for sig, (plan, origins) in plans.items():
        sg, cores = decode_plan(model, plan)
        if model.plan_from(sg, plan['core_schedules']) != plan:
            raise ValueError('方案不能无损映射到块模型')
        old_mk, old_added, _ = model.evaluate(sg, cores, 'A', n,
            use_cache=False, orders_override=plan['core_schedules'])
        started = time.perf_counter()
        new = precise.evaluate(plan)
        seconds = time.perf_counter() - started
        truth = real_evaluate(g, plan, 'A')
        if not truth or truth.get('error'):
            raise ValueError(f'官方评估失败: {case}/N{n}: {truth}')
        rows.append({'plan_id': sig, 'origins': origins, 'old_makespan': old_mk,
                     'new_makespan': new['makespan'], 'official': truth,
                     'new_spill_bytes': new['spill_bytes'], 'old_added': old_added,
                     'new_added': new['total_added_bytes'], 'new_seconds': seconds})
    if not rows:
        raise ValueError('没有可比较方案')
    optimum = min(r['official']['makespan'] for r in rows)
    old_best = min(rows, key=lambda r: (r['old_makespan'], r['old_added'], r['plan_id']))
    new_best = min(rows, key=lambda r: (r['new_makespan'], r['new_added'], r['plan_id']))
    pairs = old_right = new_right = 0
    for i, a in enumerate(rows):
        for b in rows[i + 1:]:
            diff = a['official']['makespan'] - b['official']['makespan']
            if abs(diff) <= .003 * min(a['official']['makespan'], b['official']['makespan']):
                continue
            pairs += 1
            for field in ('old_makespan', 'new_makespan'):
                delta = a[field] - b[field]
                credit = 0.5 if delta == 0 else float(delta * diff > 0)
                if field == 'old_makespan':
                    old_right += credit
                else:
                    new_right += credit
    return {'case': case, 'N': n, 'rows': rows, 'candidate_count': len(rows),
            'old_selected_official': old_best['official']['makespan'],
            'new_selected_official': new_best['official']['makespan'],
            'oracle_official': optimum,
            'old_regret_ratio': old_best['official']['makespan'] / optimum - 1,
            'new_regret_ratio': new_best['official']['makespan'] / optimum - 1,
            'comparable_pairs': pairs, 'old_correct_pairs': old_right,
            'new_correct_pairs': new_right}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cases', default='1,7,15,23,42,44,49,63,67,78,82,97')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--output', default='results/candidate_ranking_event_v1.jsonl')
    args = p.parse_args()
    dirs = [str(ROOT / 'results' / d) for d in DEFAULT_DIRS]
    source = hashlib.sha256(''.join(hashlib.sha256(Path(__file__).with_name(p).read_bytes()).hexdigest()
        for p in ('audit_candidate_ranking.py', 'scene_a_event.py', 'spill_events.py', 'model.py', 'pipeline.py')).encode()).hexdigest()
    tasks = [(f'case_{int(c):03}', n, dirs) for c in args.cases.split(',') for n in (2, 3, 4, 5)]
    # 固定输入指纹，防止同一输出文件混入不同版本方案。
    input_hash = hashlib.sha256()
    for case, n, directories in tasks:
        for d in directories:
            file = Path(d) / f'{case}_A_N{n}.json'
            input_hash.update(str(file).encode('utf-8'))
            input_hash.update(file.read_bytes() if file.exists() else b'missing')
    provenance = source + ':' + input_hash.hexdigest()
    out = Path(args.output)
    rows = [json.loads(s) for s in out.read_text(encoding='utf-8').splitlines()] if out.exists() else []
    if any(r['provenance'] != provenance for r in rows):
        raise ValueError('代码或输入已变化，请使用新输出文件')
    done = {(r['case'], r['N']) for r in rows}
    todo = [t for t in tasks if t[:2] not in done]
    with ProcessPoolExecutor(max_workers=args.workers) as executor, out.open('a', encoding='utf-8') as f:
        futures = [executor.submit(run_group, task) for task in todo]
        for future in as_completed(futures):
            row = future.result()
            row['provenance'] = provenance
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
            f.flush()
            rows.append(row)
            print(f'[{len(rows)}/{len(tasks)}] {row["case"]}/N{row["N"]}: '
                  f'候选={row["candidate_count"]}, 最优差距={row["old_regret_ratio"]:.2%}->{row["new_regret_ratio"]:.2%}', flush=True)
    pairs = sum(r['comparable_pairs'] for r in rows)
    summary = {'groups': len(rows), 'candidates': sum(r['candidate_count'] for r in rows),
               'comparable_pairs': pairs,
               'old_pair_accuracy': sum(r['old_correct_pairs'] for r in rows)/pairs if pairs else None,
               'new_pair_accuracy': sum(r['new_correct_pairs'] for r in rows)/pairs if pairs else None,
               'old_mean_regret': statistics.mean(r['old_regret_ratio'] for r in rows),
               'new_mean_regret': statistics.mean(r['new_regret_ratio'] for r in rows),
               'new_selection_wins': sum(r['new_selected_official'] < r['old_selected_official'] for r in rows),
               'new_selection_losses': sum(r['new_selected_official'] > r['old_selected_official'] for r in rows),
               'scope': '历史多版本方案池，固定候选重排序；不能直接当作新求解器性能'}
    out.with_suffix('.summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
