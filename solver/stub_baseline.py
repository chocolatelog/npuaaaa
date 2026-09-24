"""原始代码基线：附件 stub 生成器（随机合法方案）× 官方评估器。

生成：stub_multicore_cut_and_schedule.py -n N（默认参数、seed 0）
评估：P1/P2/P3 × 100 用例 × N∈{2,3,4,5}，输出 results/official_stub/
对比：ours(results/official) vs stub 按三指标统计。
"""
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
ATTACH = os.path.normpath(os.path.join(
    HERE, '..', '通用神经网络处理器下的多核调度问题附件'))
RESULTS = os.path.normpath(os.path.join(HERE, '..', 'results'))
PY = sys.executable
TIMEOUT = 3600


def gen_and_eval(job):
    case, n, prob = job
    out_dir = os.path.join(RESULTS, 'official_stub')
    os.makedirs(out_dir, exist_ok=True)
    res_path = os.path.join(out_dir, f'{case}_{prob}_N{n}_res.json')
    if os.path.exists(res_path):
        return (job, 'cached', '')
    plan_path = os.path.join(RESULTS, 'plans_stub', f'{case}_N{n}.json')
    # 1) 生成 stub 方案（写进 data/，随后搬走）
    if not os.path.exists(plan_path):
        tmp = os.path.join(ATTACH, 'data', f'{case}_multicore_res.json')
        r = subprocess.run(
            [PY, '-X', 'utf8', 'code/stub_multicore_cut_and_schedule.py',
             f'data/{case}.json', '-n', str(n)],
            cwd=ATTACH, capture_output=True, text=True,
            encoding='utf-8', errors='replace', timeout=1800)
        if r.returncode != 0 or not os.path.exists(tmp):
            return (job, 'gen_fail', (r.stderr or '')[-200:])
        os.makedirs(os.path.dirname(plan_path), exist_ok=True)
        shutil.move(tmp, plan_path)
    # 2) 官方评估
    stem = os.path.join(out_dir, f'{case}_{prob}_N{n}')
    cmd = [PY, '-X', 'utf8', f'code/multicore_cut_evaluate_{prob}.py',
           os.path.join(ATTACH, 'data', f'{case}.json'), plan_path,
           '--config', 'data/config.txt', '-o', stem + '_res.json',
           '--trace-output', 'NUL', '--log-output', 'NUL']
    try:
        r = subprocess.run(cmd, cwd=ATTACH, capture_output=True, text=True,
                           encoding='utf-8', errors='replace', timeout=TIMEOUT)
    except subprocess.TimeoutExpired:
        return (job, 'timeout', '')
    if r.returncode != 0 or not os.path.exists(stem + '_res.json'):
        return (job, 'eval_fail', (r.stderr or r.stdout)[-300:])
    return (job, 'ok', '')


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cores', default='2,3,4,5')
    parser.add_argument('--workers', type=int, default=10)
    parser.add_argument('--problems', default='problem_1,problem_2,problem_3')
    args = parser.parse_args()
    cores = [int(x) for x in args.cores.split(',')]
    probs = args.problems.split(',')
    cases = sorted(f'case_{i:03d}' for i in range(1, 101))
    jobs = [(c, n, p) for c in cases for n in cores for p in probs]
    # 先串行生成方案（每核数一份），再并行评估
    print(f'{len(jobs)} evaluations (stub baseline)')
    stats = {}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(gen_and_eval, j): j for j in jobs}
        nd = 0
        for fut in as_completed(futs):
            j = futs[fut]
            try:
                job, status, err = fut.result()
            except Exception as e:
                job, status, err = j, 'exception', repr(e)[:200]
            stats[status] = stats.get(status, 0) + 1
            nd += 1
            if status not in ('ok', 'cached'):
                print(f'[{nd}/{len(jobs)}] {job} -> {status}: {err}',
                      flush=True)
            elif nd % 60 == 0:
                print(f'[{nd}/{len(jobs)}] {stats}', flush=True)
    print('done:', stats)


if __name__ == '__main__':
    main()
