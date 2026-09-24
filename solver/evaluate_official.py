"""官方评估器全量评估：单核基线 + P1/P2/P3 × 核数，结果写入 results/official/。"""
import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
ATTACH = os.path.normpath(os.path.join(
    HERE, '..', '通用神经网络处理器下的多核调度问题附件'))
DATA = os.path.join(ATTACH, 'data')
OUT = os.path.normpath(os.path.join(HERE, '..', 'results', 'official'))
PY = sys.executable


def run_one(job):
    kind, case, n = job
    graph = os.path.join(DATA, f'{case}.json')
    stem = os.path.join(OUT, f'{case}_{kind}_N{n}' if n else f'{case}_{kind}')
    if os.path.exists(stem + '_res.json'):
        return (job, 'cached', None)
    if kind == 'singlecore':
        cmd = [PY, '-X', 'utf8', 'code/singlecore_evaluate.py', graph,
               '--config', 'data/config.txt',
               '-o', stem + '_res.json',
               '--trace-output', stem + '_trace.json',
               '--log-output', stem + '_log.txt']
    else:
        plan = os.path.normpath(os.path.join(
            HERE, '..', 'results', 'plans',
            f'{case}_{"A" if kind == "problem_1" else "B"}_N{n}.json'))
        if not os.path.exists(plan):
            return (job, 'missing_plan', plan)
        cmd = [PY, '-X', 'utf8', f'code/multicore_cut_evaluate_{kind}.py',
               graph, plan, '--config', 'data/config.txt',
               '-o', stem + '_res.json',
               '--trace-output', stem + '_trace.json',
               '--log-output', stem + '_log.txt']
    try:
        r = subprocess.run(cmd, cwd=ATTACH, capture_output=True, text=True,
                           encoding='utf-8', errors='replace', timeout=3600)
        ok = r.returncode == 0 and os.path.exists(stem + '_res.json')
        return (job, 'ok' if ok else 'fail',
                None if ok else (r.stderr or r.stdout)[-500:])
    except subprocess.TimeoutExpired:
        return (job, 'timeout', None)


def parse_cases(spec):
    out = []
    for part in spec.split(','):
        if '-' in part:
            lo, hi = part.split('-')
            out.extend(f'case_{i:03d}' for i in range(int(lo), int(hi) + 1))
        else:
            out.append(f'case_{int(part):03d}')
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cases', default='1-100')
    parser.add_argument('--cores', default='2,3,4,5')
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--skip-singlecore', action='store_true')
    args = parser.parse_args()

    os.makedirs(OUT, exist_ok=True)
    cases = parse_cases(args.cases)
    cores = [int(x) for x in args.cores.split(',')]
    jobs = []
    if not args.skip_singlecore:
        for case in cases:
            jobs.append(('singlecore', case, 0))
    for case in cases:
        for n in cores:
            jobs.append(('problem_1', case, n))
            jobs.append(('problem_2', case, n))
            jobs.append(('problem_3', case, n))
    print(f'{len(jobs)} official evaluations, workers={args.workers}')
    stats = {}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_one, j): j for j in jobs}
        done = 0
        for fut in as_completed(futs):
            j = futs[fut]
            try:
                job, status, err = fut.result()
            except Exception as e:
                job, status, err = futs[fut], 'exception', repr(e)
            stats[status] = stats.get(status, 0) + 1
            done += 1
            if status not in ('ok', 'cached'):
                print(f'[{done}/{len(jobs)}] {job} -> {status}: {err}',
                      flush=True)
            elif done % 50 == 0:
                print(f'[{done}/{len(jobs)}] running... {stats}', flush=True)
    print('done:', stats)


if __name__ == '__main__':
    main()
