"""固定保存方案的迁核与顺序搜索配对实验，逐任务落盘并校验续跑指纹。"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import statistics
import time

from pipeline import real_evaluate
from run_all import ensure_run_manifest, parse_cases
from task_order_search import refine_plan_orders

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / '通用神经网络处理器下的多核调度问题附件/data'


def run_one(task):
    case, n, source, output, partition_polish = task
    graph = json.loads((DATA / f'{case}.json').read_text(encoding='utf-8'))
    source_path = Path(source) / f'{case}_A_N{n}.json'
    plan = json.loads(source_path.read_text(encoding='utf-8'))
    baseline = real_evaluate(graph, plan, 'A')
    if not baseline or baseline.get('error'):
        raise ValueError(f'基线官方评估失败: {case}/{n}: {baseline}')
    if partition_polish:
        from partition_polish import polish_partition
        selected, best, audit = polish_partition(graph, plan, baseline, real_evaluate)
    else:
        selected, best, audit = refine_plan_orders(graph, plan, baseline, real_evaluate)
    if audit['errors']:
        raise ValueError(f'后处理候选审计失败: {case}/{n}: {audit["errors"]}')
    records, stats = audit['candidates'], audit['search']
    out_path = Path(output) / f'{case}_A_N{n}.json'
    out_path.write_text(json.dumps(selected), encoding='utf-8')
    return {'case': case, 'scene': 'A', 'N': n, 'baseline': baseline,
            'real': best, 'source_path': str(source_path),
            'source_sha256': hashlib.sha256(source_path.read_bytes()).hexdigest(),
            'output_path': str(out_path), 'candidate_count': len(records),
            'candidates': records, 'search': stats, 'model_seconds': audit['model_seconds'],
            'elapsed': audit['extra_seconds'], 'change': best['makespan'] / baseline['makespan'] - 1}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cases', default='1,7,15,23,42,44,49,63,67,78,82,97')
    parser.add_argument('--cores', default='2,3,4,5')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--source-dir', default='results/plans_event_terminal_v1')
    parser.add_argument('--output-dir', default='results/plans_task_order_v1')
    parser.add_argument('--log-file', default='results/task_order_audit_v1.jsonl')
    parser.add_argument('--partition-polish', action='store_true', help='改为多粒度拆分与边界合并精化')
    args = parser.parse_args()
    source, output, log_path = (Path(args.source_dir).resolve(),
                                Path(args.output_dir).resolve(), Path(args.log_file).resolve())
    if source == output:
        raise ValueError('输出目录不能覆盖基线方案目录')
    output.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cases = parse_cases(args.cases)
    cores = [int(n) for n in args.cores.split(',')]
    tasks = [(c, n, str(source), str(output), args.partition_polish) for c in cases for n in cores]
    files = list((ROOT / 'solver').glob('*.py'))
    files += list((ROOT / '通用神经网络处理器下的多核调度问题附件/code').glob('*.py'))
    files += [DATA / f'{c}.json' for c in cases]
    files += [source / f'{c}_A_N{n}.json' for c in cases for n in cores]
    fingerprint = ensure_run_manifest(log_path, files, vars(args))
    rows = [json.loads(s) for s in log_path.read_text(encoding='utf-8').splitlines()] if log_path.exists() else []
    done = {(r['case'], r['N']) for r in rows}
    todo = [t for t in tasks if t[:2] not in done]
    method = '切分精化' if args.partition_polish else '固定切分迁核与顺序'
    print(f'[{len(done)}/{len(tasks)}] {method}实验，进程数={args.workers}', flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as executor, log_path.open('a', encoding='utf-8') as f:
        futures = [executor.submit(run_one, t) for t in todo]
        for future in as_completed(futures):
            row = future.result()
            row['fingerprint'] = fingerprint
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
            f.flush()
            rows.append(row)
            print(f'[{len(rows)}/{len(tasks)}] {row["case"]}/N{row["N"]}: '
                  f'{row["baseline"]["makespan"]}->{row["real"]["makespan"]} '
                  f'({row["change"]:.2%})，额外 {row["elapsed"]:.2f} 秒', flush=True)
    summary = {'count': len(rows), 'wins': sum(r['change'] < 0 for r in rows),
               'losses': sum(r['change'] > 0 for r in rows),
               'mean_change': statistics.mean(r['change'] for r in rows),
               'total_change': sum(r['real']['makespan'] for r in rows) /
                               sum(r['baseline']['makespan'] for r in rows) - 1,
               'extra_seconds': sum(r['elapsed'] for r in rows),
               'candidate_errors': sum(bool((c['official'] or {}).get('error'))
                                       for r in rows for c in r['candidates']),
               'method': method,
               'scope': '固定保存方案的后处理配对实验；不等同端到端求解器性能'}
    log_path.with_suffix('.summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
