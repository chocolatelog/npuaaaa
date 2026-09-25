"""活体验证：校准后模型重求解子集 -> 官方评估 -> 与现有官方结果对比。

子集含大用例（>1.2万op，无真值兜底、纯靠代理，校准收益应最明显）。
"""
import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ATTACH = os.path.normpath(os.path.join(
    HERE, '..', '通用神经网络处理器下的多核调度问题附件'))
RESULTS = os.path.normpath(os.path.join(HERE, '..', 'results'))
PY = sys.executable

CASES = ['case_003', 'case_014', 'case_016', 'case_054', 'case_076',
         'case_091',                      # >1.2万 op 大用例
         'case_002', 'case_033', 'case_044', 'case_100']   # 中小对照
N = 4


def solve_one(task):
    from model import load_graph
    from pipeline import solve_case
    from run_all import budget_for, block_cap_for, stable_seed
    case, scene = task
    g = load_graph(os.path.join(ATTACH, 'data', f'{case}.json'))
    n_ops = len(g['ops'])
    n_el = sum(1 for o in g['ops'] if o['op'] not in ('COPY_IN', 'COPY_OUT'))
    r = solve_case(g, N=N, scene=scene, time_budget=budget_for(n_ops),
                   seed=stable_seed(case, scene, N, seed_base=2),
                   block_ops_cap=block_cap_for(n_el))
    out = os.path.join(RESULTS, 'plans_v2', f'{case}_{scene}_N{N}.json')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(r['plan'], f)
    return {'case': case, 'scene': scene, 'est_mk': r['est'][0],
            'real_mk': r['real']['makespan'] if r['real'] else None,
            'elapsed': r['elapsed']}


def eval_one(task):
    case, scene = task
    prob = 'problem_1' if scene == 'A' else 'problem_2'
    stem = os.path.join(RESULTS, 'official_v2', f'{case}_{prob}_N{N}')
    os.makedirs(os.path.dirname(stem), exist_ok=True)
    plan = os.path.join(RESULTS, 'plans_v2', f'{case}_{scene}_N{N}.json')
    cmd = [PY, '-X', 'utf8', f'code/multicore_cut_evaluate_{prob}.py',
           os.path.join(ATTACH, 'data', f'{case}.json'), plan,
           '--config', 'data/config.txt',
           '-o', stem + '_res.json',
           '--trace-output', stem + '_trace.json',
           '--log-output', stem + '_log.txt']
    r = subprocess.run(cmd, cwd=ATTACH, capture_output=True, text=True,
                       encoding='utf-8', errors='replace', timeout=3600)
    return case, scene, (r.returncode == 0), (r.stderr or '')[-200:]


def legacy_disabled():
    raise SystemExit('旧实验入口已停用：请使用 run_all.py 的新日志/方案目录及 evaluate_official.py；历史结果保留。')


def main():
    legacy_disabled()
    tasks = [(c, s) for s in ('A', 'B') for c in CASES]
    print('=== 重求解（校准后模型） ===')
    solved = []
    with ProcessPoolExecutor(max_workers=10) as ex:
        futs = {ex.submit(solve_one, t): t for t in tasks}
        for fut in as_completed(futs):
            r = fut.result()
            solved.append(r)
            print(f"  {r['case']}/{r['scene']}: est={r['est_mk']:.0f} "
                  f"inrun_real={r['real_mk']} ({r['elapsed']:.0f}s)", flush=True)
    print('=== 官方评估新方案 ===')
    with ProcessPoolExecutor(max_workers=10) as ex:
        futs = {ex.submit(eval_one, t): t for t in tasks}
        for fut in as_completed(futs):
            case, scene, ok, err = fut.result()
            if not ok:
                print(f'  EVAL FAIL {case}/{scene}: {err}', flush=True)
    print('=== 新旧官方 Makespan 对比 ===')
    import statistics as st
    d_all, d_big = [], []
    for case, scene in tasks:
        prob = 'problem_1' if scene == 'A' else 'problem_2'
        old_p = os.path.join(RESULTS, 'official', f'{case}_{prob}_N{N}_res.json')
        new_p = os.path.join(RESULTS, 'official_v2', f'{case}_{prob}_N{N}_res.json')
        if not (os.path.exists(old_p) and os.path.exists(new_p)):
            print(f'  {case}/{scene}: missing')
            continue
        old = json.load(open(old_p, encoding='utf-8'))['makespan']
        new = json.load(open(new_p, encoding='utf-8'))['makespan']
        d = (new - old) / old * 100
        (d_big if case in CASES[:6] else d_all).append(d)
        print(f'  {case}/{scene}: 旧={old} 新={new} ({d:+.1f}%)')
    print(f'\n大用例均值 {st.mean(d_big):+.2f}% | 中小用例均值 {st.mean(d_all):+.2f}% '
          f'| 全部 {st.mean(d_all + d_big):+.2f}%')


if __name__ == '__main__':
    main()
