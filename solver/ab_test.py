"""A/B 对照（等完整仿真预算）：Mamba 式 SSM 粗筛 开/关。

隔离实验：构造池 -> 禁忌搜索 -> 模拟退火，两臂使用完全相同的
构造解、随机种子与【完整仿真次数上限】；筛选臂在预算内可尝试
多 ~4-12 倍的邻域扰动（被跳过的扰动不消耗完整仿真预算）。
指标：最终代理 Makespan（同预算下越低越好）、扰动尝试次数、命中率。
"""
import json
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ATTACH = os.path.normpath(os.path.join(
    HERE, '..', '通用神经网络处理器下的多核调度问题附件'))

CASES = ['case_001', 'case_002', 'case_004', 'case_011', 'case_016',
         'case_020', 'case_033', 'case_044', 'case_047', 'case_064',
         'case_093', 'case_100']
EVAL_BUDGET = 1500     # TS+SA 阶段的完整仿真次数上限（两臂相同）
N_CORES = 4


def run_arm(task):
    from model import load_graph, Model
    from solution import Sol, Context
    from pipeline import build_construct_pool
    from tabu import tabu_search
    from sa import sa_search
    case, scene, use_screen = task
    g = load_graph(os.path.join(ATTACH, 'data', f'{case}.json'))
    n_el = sum(1 for o in g['ops'] if o['op'] not in ('COPY_IN', 'COPY_OUT'))
    cap = int(max(24, min(240, n_el // 400)))
    model = Model(g, block_ops_cap=cap)
    ctx = Context(model, N_CORES, scene, screen_enabled=use_screen, seed=99)
    rng = random.Random(99)
    pool = build_construct_pool(model, N_CORES, scene)
    scored = []
    for sol in pool:
        sol.compact()
        mk, ad, _ = ctx.evaluate(sol)
        scored.append((ctx.fitness(mk, ad), sol, mk, ad))
    scored.sort(key=lambda x: x[0])
    best = scored[0][1].clone()
    evals_construct = ctx.n_evals
    t0 = time.time()
    tabu_search(ctx, best.clone(), 1e9, rng=rng,
                max_evals=EVAL_BUDGET // 2)
    sa_search(ctx, best.clone(), 1e9, rng=rng,
              max_evals=EVAL_BUDGET // 2)
    elapsed = time.time() - t0
    # 统计扰动尝试数（TS 12/代 + SA 4/步 或 1/步）
    tried = ctx.screen.stats['admitted'] + ctx.screen.stats['skipped'] if use_screen else None
    ctx.best_sol = best
    # 重评当前最优
    best.compact()
    mk, ad, _ = ctx.evaluate(best, use_cache=False)
    # 用归档外最后已知最优重算——best 可能不是最终解；直接用 TS/SA 返回
    return {'case': case, 'scene': scene, 'screen': use_screen,
            'est_mk': mk, 'est_added': ad, 'evals': ctx.n_evals - evals_construct,
            'tried': tried, 'elapsed': elapsed,
            'screen_stats': (dict(ctx.screen.stats) if use_screen else None)}


def run_arm_full(task):
    """返回 TS/SA 各自的最终解中较优者的指标。"""
    from model import load_graph, Model
    from solution import Sol, Context
    from pipeline import build_construct_pool
    from tabu import tabu_search
    from sa import sa_search
    case, scene, use_screen = task
    g = load_graph(os.path.join(ATTACH, 'data', f'{case}.json'))
    n_el = sum(1 for o in g['ops'] if o['op'] not in ('COPY_IN', 'COPY_OUT'))
    cap = int(max(24, min(240, n_el // 400)))
    model = Model(g, block_ops_cap=cap)
    ctx = Context(model, N_CORES, scene, screen_enabled=use_screen, seed=99)
    rng = random.Random(99)
    pool = build_construct_pool(model, N_CORES, scene)
    scored = []
    for sol in pool:
        sol.compact()
        mk, ad, _ = ctx.evaluate(sol)
        scored.append((ctx.fitness(mk, ad), sol, mk, ad))
    scored.sort(key=lambda x: x[0])
    best = scored[0][1].clone()
    evals_construct = ctx.n_evals
    t0 = time.time()
    s1, f1, mk1, ad1 = tabu_search(ctx, best.clone(), 1e9, rng=rng,
                                   max_evals=EVAL_BUDGET // 2)
    s2, f2, mk2, ad2 = sa_search(ctx, s1.clone(), 1e9, rng=rng,
                                 max_evals=EVAL_BUDGET // 2)
    elapsed = time.time() - t0
    if f2 is not None and f2 < f1:
        mk, ad = mk2, ad2
    else:
        mk, ad = mk1, ad1
    tried = None
    if use_screen:
        st = ctx.screen.stats
        tried = st['admitted'] + st['skipped']
    return {'case': case, 'scene': scene, 'screen': use_screen,
            'est_mk': mk, 'est_added': ad,
            'evals': ctx.n_evals - evals_construct, 'tried': tried,
            'elapsed': elapsed,
            'screen_stats': (dict(ctx.screen.stats) if use_screen else None)}


def main():
    tasks = []
    for scene in ('A', 'B'):
        for case in CASES:
            tasks.append((case, scene, False))
            tasks.append((case, scene, True))
    results = []
    with ProcessPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(run_arm_full, t): t for t in tasks}
        for fut in as_completed(futs):
            t = futs[fut]
            try:
                results.append(fut.result())
            except Exception as e:
                print('FAIL', t, repr(e), flush=True)
    with open(os.path.join(HERE, '..', 'results', 'ab_test.json'), 'w',
              encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    import statistics as st
    print(f'\n=== 等完整仿真预算 A/B (TS+SA 共 {EVAL_BUDGET} 次, N=4) ===')
    print(f'{"case/scene":16s} {"基线mk":>10s} {"筛选mk":>10s} {"mkΔ":>7s} '
          f'{"基线evals":>9s} {"筛选evals":>9s} {"尝试倍数":>7s} {"命中":>6s}')
    d_mk, ratios = [], []
    for scene in ('A', 'B'):
        for case in CASES:
            b = next(r for r in results if r['case'] == case
                     and r['scene'] == scene and not r['screen'])
            s = next(r for r in results if r['case'] == case
                     and r['scene'] == scene and r['screen'])
            d_mk.append((s['est_mk'] - b['est_mk']) / b['est_mk'] * 100)
            ratio = (s['tried'] / s['evals']) if s['tried'] else 1.0
            ratios.append(ratio)
            prec = (s['screen_stats']['admit_improved'] /
                    max(1, s['screen_stats']['admitted'])) if s['screen_stats'] else float('nan')
            print(f'{case+"/"+scene:16s} {b["est_mk"]:10.0f} {s["est_mk"]:10.0f} '
                  f'{(s["est_mk"]-b["est_mk"])/b["est_mk"]*100:+6.1f}% '
                  f'{b["evals"]:9d} {s["evals"]:9d} {ratio:6.1f}x {prec:6.1%}')
    wins = sum(1 for d in d_mk if d < 0)
    print(f'\n均值 makespan 变化 {st.mean(d_mk):+.2f}% | 筛选臂更优 {wins}/{len(d_mk)} '
          f'| 平均扰动尝试倍数 {st.mean(ratios):.1f}x')


if __name__ == '__main__':
    main()
