"""全量求解入口：并行处理 (用例 × 场景 × 核数)，输出方案 JSON 与求解日志。

用法：
  python run_all.py [--cases 1-100] [--cores 2,3,4,5] [--scenes A,B]
                    [--workers 8] [--time-budget auto]
"""
import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
ATTACH = os.path.normpath(os.path.join(
    HERE, '..', '通用神经网络处理器下的多核调度问题附件'))
DATA = os.path.join(ATTACH, 'data')
OUT_PLANS = os.path.normpath(os.path.join(HERE, '..', 'results', 'plans'))
OUT_LOG = os.path.normpath(os.path.join(HERE, '..', 'results', 'solve_log.jsonl'))


def budget_for(n_ops):
    if n_ops <= 1500:
        return 12.0
    if n_ops <= 5000:
        return 15.0
    if n_ops <= 12000:
        return 20.0
    return 28.0


def block_cap_for(n_eligible):
    """块粒度：约 400 op 一块的上限并夹在 [24, 240]，保证子图可贴合缓存。"""
    return int(max(24, min(240, n_eligible // 400)))


def stable_seed(case, scene, n, base_seed=0):
    """Return the same search seed across Python processes and runs."""
    payload = f'{base_seed}:{case}:{scene}:{n}'.encode('utf-8')
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(),
                          'little')


def solve_task(task):
    from model import load_graph
    from pipeline import solve_case
    case, scene, n = task['case'], task['scene'], task['n']
    graph_path = os.path.join(DATA, f'{case}.json')
    graph = load_graph(graph_path)
    n_ops = len(graph['ops'])
    n_eligible = sum(1 for o in graph['ops']
                     if o['op'] not in ('COPY_IN', 'COPY_OUT'))
    warm_plan = None
    if task.get('warm_start_dir'):
        warm_path = os.path.join(task['warm_start_dir'],
                                 f'{case}_{scene}_N{n}.json')
        if os.path.isfile(warm_path):
            with open(warm_path, encoding='utf-8') as f:
                warm_plan = json.load(f)
    res = solve_case(graph, N=n, scene=scene,
                     time_budget=task.get('budget') or budget_for(n_ops),
                     seed=task.get('seed', stable_seed(case, scene, n)),
                     verify_k=task.get('verify_k', 16),
                     block_ops_cap=block_cap_for(n_eligible),
                     warm_plan=warm_plan,
                     use_mcts=task.get('use_mcts', False))
    out_plan = os.path.join(task.get('out_plans', OUT_PLANS),
                            f'{case}_{scene}_N{n}.json')
    with open(out_plan, 'w', encoding='utf-8') as f:
        json.dump(res['plan'], f)
    entry = {'case': case, 'scene': scene, 'N': n,
             'seed': task.get('seed', stable_seed(case, scene, n)),
             'est_mk': res['est'][0], 'est_added': res['est'][1],
             'real': res['real'], 'log': res['log'],
             'elapsed': round(res['elapsed'], 1), 'n_ops': n_ops}
    return entry


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
    global OUT_PLANS, OUT_LOG
    parser = argparse.ArgumentParser()
    parser.add_argument('--cases', default='1-100')
    parser.add_argument('--cores', default='2,3,4,5')
    parser.add_argument('--scenes', default='A,B')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--budget', type=float, default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--verify-k', type=int, default=16,
                        help='最多送官方评估器核验的候选数（小图 16-32）')
    parser.add_argument('--mcts', action='store_true',
                        help='启用有界 PUCT 调度树搜索实验')
    parser.add_argument('--warm-start-dir', default=None)
    parser.add_argument('--output-dir', default=OUT_PLANS)
    parser.add_argument('--log-file', default=OUT_LOG)
    args = parser.parse_args()

    OUT_PLANS = os.path.abspath(args.output_dir)
    OUT_LOG = os.path.abspath(args.log_file)
    warm_start_dir = (os.path.abspath(args.warm_start_dir)
                      if args.warm_start_dir else None)
    if (warm_start_dir and
            os.path.normcase(warm_start_dir) ==
            os.path.normcase(OUT_PLANS)):
        parser.error('--warm-start-dir must differ from --output-dir')
    if warm_start_dir and not os.path.isdir(warm_start_dir):
        parser.error('--warm-start-dir does not exist')
    os.makedirs(OUT_PLANS, exist_ok=True)
    cases = parse_cases(args.cases)
    cores = [int(x) for x in args.cores.split(',')]
    scenes = list(args.scenes.split(','))

    done = set()
    if os.path.exists(OUT_LOG):
        with open(OUT_LOG, encoding='utf-8') as f:
            for line in f:
                try:
                    e = json.loads(line)
                    done.add((e['case'], e['scene'], e['N']))
                except Exception:
                    pass

    tasks = []
    for case in cases:
        for scene in scenes:
            for n in cores:
                if (case, scene, n) not in done:
                    tasks.append({'case': case, 'scene': scene, 'n': n,
                                  'budget': args.budget,
                                  'seed': stable_seed(case, scene, n,
                                                      args.seed),
                                  'verify_k': args.verify_k,
                                  'use_mcts': args.mcts,
                                  'warm_start_dir': warm_start_dir,
                                  'out_plans': OUT_PLANS})
    print(f'{len(tasks)} tasks to solve '
          f'({len(done)} already done), workers={args.workers}')
    t0 = time.time()
    with open(OUT_LOG, 'a', encoding='utf-8') as logf:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(solve_task, t): t for t in tasks}
            n_done = 0
            for fut in as_completed(futs):
                t = futs[fut]
                try:
                    entry = fut.result()
                    logf.write(json.dumps(entry, ensure_ascii=False) + '\n')
                    logf.flush()
                    n_done += 1
                    real = entry['real']
                    real_mk = real['makespan'] if real else '-'
                    print(f'[{n_done}/{len(tasks)}] {entry["case"]} '
                          f'{entry["scene"]} N={entry["N"]} '
                          f'est={round(entry["est_mk"])} real={real_mk} '
                          f'({entry["elapsed"]:.0f}s)', flush=True)
                except Exception as e:
                    print(f'FAIL {t}: {e!r}', flush=True)
    print(f'all done in {time.time() - t0:.0f}s')


if __name__ == '__main__':
    sys.path.insert(0, HERE)
    main()
