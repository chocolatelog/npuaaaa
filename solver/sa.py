"""模拟退火：几何降温 + 混合邻域（随机移动 + 通信边界定向移动）。"""
import math
import random


def sa_search(ctx, sol, time_budget, deadline=None, T0_frac=0.15,
              cool=0.995, restarts=2, archive=None, rng=None, max_evals=None):
    """返回 (best_sol, best_fitness, best_mk, best_added)。"""
    import time
    rng = rng or random.Random()
    t_end = deadline if deadline is not None else time.time() + time_budget
    evals0 = ctx.n_evals
    cur = sol.clone()
    cur.compact()
    f_cur, mk_cur, ad_cur = ctx.eval_fitness(cur)
    ctx.update_snapshot(cur, f_cur, phase=0.0)
    best, f_best = cur.clone(), f_cur
    best_mk, best_ad = mk_cur, ad_cur
    if archive is not None:
        archive.insert(mk_cur, ad_cur, cur)
    while True:
        # 温度按初始适应度比例设定
        T = max(1.0, f_cur * T0_frac)
        while T > max(1.0, f_best * 1e-4):
            if time.time() >= t_end or (
                    max_evals is not None and ctx.n_evals - evals0 >= max_evals):
                return best, f_best, best_mk, best_ad
            # 锦标赛生成：4 个候选动作，SSM 预测后仅完整仿真预测最优者
            n_try = 4
            cands = []
            for _ in range(n_try):
                cand = cur.clone()
                mv, ok = (ctx.targeted_move if rng.random() < 0.5
                          else ctx.random_move)(cand, rng)
                if ok:
                    cands.append((cand, mv))
            if not cands:
                continue
            if ctx.screen_enabled:
                scored = []
                for cand, mv in cands:
                    y, phi = ctx.predict_move(cand, mv)
                    scored.append((y, phi, cand))
                scored.sort(key=lambda t: -t[0])
                pick = scored[0]
                probe = False
                if len(scored) > 1 and rng.random() < ctx.screen.eps:
                    pick = scored[rng.randrange(1, len(scored))]
                    probe = True
                ctx.screen_skip(len(scored) - 1)
                cand = pick[2]
                if probe:
                    # ε 探索者：真值评估 + 回传漏筛统计，本步即用其结果
                    f_new, mk_new, ad_new = ctx.screen_probe(pick[1], f_cur, cand)
                else:
                    f_new, mk_new, ad_new = ctx.eval_and_learn(cand, pick[1], f_cur)
            else:
                cand = cands[0][0]
                f_new, mk_new, ad_new = ctx.eval_fitness(cand)
            if archive is not None:
                archive.insert(mk_new, ad_new, cand)
            d = f_new - f_cur
            if d <= 0 or rng.random() < math.exp(-d / T):
                cur, f_cur, mk_cur, ad_cur = cand, f_new, mk_new, ad_new
                ctx.update_snapshot(cur, f_cur, phase=T)
                if f_new < f_best:
                    best, f_best = cand.clone(), f_new
                    best_mk, best_ad = mk_new, ad_new
            T *= cool
        # 重启：从最优解扰动出发
        cur = best.clone()
        for _ in range(max(2, ctx.nb // 50)):
            ctx.random_move(cur, rng)
        f_cur, mk_cur, ad_cur = ctx.eval_fitness(cur)
        if time.time() >= t_end:
            return best, f_best, best_mk, best_ad
