"""禁忌搜索：邻域采样 + 属性禁忌表 + 渴望准则。"""
import random
import time


def tabu_search(ctx, sol, time_budget, deadline=None, tenure_factor=0.6,
                n_samples=12, archive=None, rng=None, max_evals=None, max_rounds=None):
    """返回 (best_sol, best_fitness, best_mk, best_added)。
    max_evals 给定时按完整仿真次数（而非墙钟）停止。"""
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
    tabu = {}
    it = 0
    tenure = max(4, int(tenure_factor * (ctx.nb ** 0.5)))
    while (it < max_rounds if max_rounds is not None else time.time() < t_end):
        if max_evals is not None and ctx.n_evals - evals0 >= max_evals:
            break
        it += 1
        best_cand = None
        f_bc, mk_bc, ad_bc, key_bc = None, None, None, None
        cands = []
        for _ in range(n_samples):
            cand = cur.clone()
            mv, ok = (ctx.targeted_move if rng.random() < 0.5
                      else ctx.random_move)(cand, rng)
            if ok:
                cands.append((cand, mv))
        # TS 保持全量评估（锦标赛会改变禁忌搜索动态）；
        # 粗筛层仅作用于 SA 的邻域扰动。
        for cand, mv in cands:
            f_new, mk_new, ad_new = ctx.eval_fitness(cand)
            if archive is not None:
                archive.insert(mk_new, ad_new, cand)
            key = _attr(mv)
            if tabu.get(key, 0) > it and f_new >= f_best:
                continue                     # 禁忌且不满足渴望准则
            if f_bc is None or f_new < f_bc:
                best_cand, f_bc = cand, f_new
                mk_bc, ad_bc, key_bc = mk_new, ad_new, key
        if best_cand is None:
            continue
        tabu[key_bc] = it + tenure
        cur, f_cur, mk_cur, ad_cur = best_cand, f_bc, mk_bc, ad_bc
        ctx.update_snapshot(cur, f_cur, phase=float(it % 100))
        if f_bc < f_best:
            best, f_best, best_mk, best_ad = best_cand.clone(), f_bc, mk_bc, ad_bc
    return best, f_best, best_mk, best_ad


def _attr(mv):
    """禁忌属性：动作作用的对象（块/子图）维度。"""
    if mv is None:
        return ('none',)
    if mv[0] in ('move_block', 'merge_edge'):
        return (mv[0], mv[1])
    if mv[0] == 'swap':
        return (mv[0], mv[1])
    if mv[0] in ('merge',):
        return (mv[0], mv[1], mv[2])
    return (mv[0], mv[1])
