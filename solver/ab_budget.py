"""补充实验：筛选臂以 60% 完整仿真预算（900 次）对比基线全预算（1500 次）。"""
import json
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ab_test import CASES, N_CORES, ATTACH  # noqa: E402


def run_arm(task):
    from model import load_graph, Model
    from solution import Context
    from pipeline import build_construct_pool
    from tabu import tabu_search
    from sa import sa_search
    case, scene, budget = task
    g = load_graph(os.path.join(ATTACH, 'data', f'{case}.json'))
    n_el = sum(1 for o in g['ops'] if o['op'] not in ('COPY_IN', 'COPY_OUT'))
    cap = int(max(24, min(240, n_el // 400)))
    model = Model(g, block_ops_cap=cap)
    ctx = Context(model, N_CORES, scene, screen_enabled=True, seed=99)
    rng = random.Random(99)
    pool = build_construct_pool(model, N_CORES, scene)
    scored = []
    for sol in pool:
        sol.compact()
        mk, ad, _ = ctx.evaluate(sol)
        scored.append((ctx.fitness(mk, ad), sol))
    scored.sort(key=lambda x: x[0])
    best = scored[0][1].clone()
    e0 = ctx.n_evals
    s1, f1, mk1, ad1 = tabu_search(ctx, best.clone(), 1e9, rng=rng,
                                   max_evals=budget // 2)
    s2, f2, mk2, ad2 = sa_search(ctx, s1.clone(), 1e9, rng=rng,
                                 max_evals=budget // 2)
    return {'case': case, 'scene': scene, 'budget': budget,
            'est_mk': mk2 if (f2 is not None and f2 < f1) else mk1,
            'evals': ctx.n_evals - e0}


def main():
    tasks = [(c, s, 900) for s in ('A', 'B') for c in CASES]
    res = []
    with ProcessPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(run_arm, t): t for t in tasks}
        for fut in as_completed(futs):
            res.append(fut.result())
    with open(os.path.join(HERE, '..', 'results', 'ab_test_900.json'), 'w',
              encoding='utf-8') as f:
        json.dump(res, f)
    base = json.load(open(os.path.join(HERE, '..', 'results', 'ab_test.json'),
                          encoding='utf-8'))
    import statistics as st
    print('筛选臂预算900 vs 基线预算1500:')
    d = []
    for r in sorted(res, key=lambda x: (x['scene'], x['case'])):
        b = next(x for x in base if x['case'] == r['case']
                 and x['scene'] == r['scene'] and not x['screen'])
        delta = (r['est_mk'] - b['est_mk']) / b['est_mk'] * 100
        d.append(delta)
        print(f'  {r["case"]}/{r["scene"]}: 基线1500次={b["est_mk"]:.0f} '
              f'筛选900次={r["est_mk"]:.0f} ({delta:+.1f}%)')
    print(f'均值: {st.mean(d):+.2f}% | 退化>1%的用例: '
          f'{sum(1 for x in d if x > 1)}/{len(d)}')


if __name__ == '__main__':
    main()
