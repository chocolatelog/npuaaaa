"""子集验证：升级后的求解器 vs 现有官方结果（keep-best 对比）。"""
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
RESULTS = os.path.normpath(os.path.join(HERE, '..', 'results'))
ATTACH = os.path.normpath(os.path.join(
    HERE, '..', '通用神经网络处理器下的多核调度问题附件'))

CASES = ['case_001', 'case_002', 'case_003', 'case_004', 'case_011',
         'case_016', 'case_020', 'case_033', 'case_044', 'case_047',
         'case_054', 'case_064', 'case_076', 'case_091', 'case_093',
         'case_100']
N = 4


def solve_one(task):
    from model import load_graph
    from pipeline import solve_case
    from run_all import budget_for, block_cap_for, stable_seed
    case, scene = task
    g = load_graph(os.path.join(ATTACH, 'data', f'{case}.json'))
    n_ops = len(g['ops'])
    n_el = sum(1 for o in g['ops']
               if o['op'] not in ('COPY_IN', 'COPY_OUT'))
    r = solve_case(g, N=N, scene=scene, time_budget=budget_for(n_ops),
                   seed=stable_seed(case, scene, N, seed_base=3),
                   block_ops_cap=block_cap_for(n_el))
    return {'case': case, 'scene': scene,
            'new_mk': r['real']['makespan'] if r['real'] else None,
            'est_mk': r['est'][0]}


def old_official(case, scene):
    prob = 'problem_1' if scene == 'A' else 'problem_2'
    p = os.path.join(RESULTS, 'official', f'{case}_{prob}_N{N}_res.json')
    if not os.path.exists(p):
        return None
    return json.load(open(p, encoding='utf-8'))['makespan']


def legacy_disabled():
    raise SystemExit('旧实验入口已停用：请使用 run_all.py 的新日志/方案目录及 evaluate_official.py；历史结果保留。')


def main():
    legacy_disabled()
    tasks = [(c, s) for s in ('A', 'B') for c in CASES]
    res = []
    with ProcessPoolExecutor(max_workers=10) as ex:
        futs = {ex.submit(solve_one, t): t for t in tasks}
        for fut in as_completed(futs):
            res.append(fut.result())
    print(f'\n=== 子集验证 (N={N}, 新求解器 vs 现有官方结果) ===')
    wins, losses, ties = 0, 0, 0
    deltas = []
    for r in sorted(res, key=lambda x: (x['scene'], x['case'])):
        old = old_official(r['case'], r['scene'])
        if r['new_mk'] is None or old is None:
            print(f"  {r['case']}/{r['scene']}: new_est={r['est_mk']:.0f} "
                  f'(无真值可比)')
            continue
        d = (r['new_mk'] - old) / old * 100
        deltas.append(d)
        mark = 'WIN ' if d < -0.5 else ('LOSS' if d > 0.5 else 'TIE ')
        if d < -0.5:
            wins += 1
        elif d > 0.5:
            losses += 1
        else:
            ties += 1
        print(f"  {r['case']}/{r['scene']}: 旧={old} 新={r['new_mk']} "
              f"({d:+.1f}%) {mark}")
    import statistics as st
    print(f'\n胜/平/负 = {wins}/{ties}/{losses} | 均值 {st.mean(deltas):+.2f}% '
          f'中位 {st.median(deltas):+.2f}%')


if __name__ == '__main__':
    main()
