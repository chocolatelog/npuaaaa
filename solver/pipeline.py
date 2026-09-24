"""求解流水线：构造 -> 禁忌搜索 -> 模拟退火 -> 蚁群 -> MOPSO -> 精选。

多目标优化框架：所有算法向共享 ParetoArchive 供解（目标：
估计 Makespan、估计新增搬运量），最终按 makespan + w*added/BW
标量化选解；小图用官方评估器对 top-k 候选做真值校验。
"""
import json
import os
import random
import sys
import time

from model import Model, load_graph, BW, MEM_CAP
from solution import Sol, Context
from archive import ParetoArchive
import construct
from construct_refine import build_construct_candidates
from tabu import tabu_search
from sa import sa_search
from aco import aco_search
from mopso import mopso_search

ATTACH_CODE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           '..', '通用神经网络处理器下的多核调度问题附件', 'code')
SPILL_COEFS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'spill_coefs.json')


def _apply_spill_coefs(model):
    """若存在 NNLS 校准系数则启用（否则用默认启发式）。"""
    if os.path.exists(SPILL_COEFS):
        try:
            with open(SPILL_COEFS, encoding='utf-8') as f:
                coefs = json.load(f)['coefs']
            model.set_spill_coefs(tuple(float(x) for x in coefs))
        except Exception:
            pass


def real_evaluate(graph_json, plan, scene, verify_cap=12000):
    """调用官方评估器（in-process）。scene: 'A'|'B'|'C'。"""
    n_ops = len(graph_json['ops'])
    if n_ops > verify_cap:
        return None
    if ATTACH_CODE not in sys.path:
        sys.path.insert(0, os.path.abspath(ATTACH_CODE))
    cap = {'L1': 524288, 'UB': 131072}
    try:
        if scene == 'A':
            from multicore_cut_evaluate_problem_1 import evaluate_scene_a
            r = evaluate_scene_a(graph_json, plan, 60.0, cap, 1000, 100)
        elif scene == 'B':
            from multicore_cut_evaluate_problem_2 import evaluate_scene_b
            r = evaluate_scene_b(graph_json, plan, 60.0, cap, 500)
        else:
            from multicore_cut_evaluate_problem_3 import evaluate_problem_3
            r = evaluate_problem_3(graph_json, plan, 60.0, cap, 500,
                                   1048576, 250.0)
    except Exception as e:
        return {'error': repr(e)}
    dm = r['data_movement_bytes']
    out = {'makespan': r['makespan'],
           'added_copy_bytes': dm['added_copy_bytes'],
           'scheduled_copy_bytes': dm['scheduled_copy_bytes'],
           'partition_added': dm.get('partition_added_copy_bytes'),
           'spill_added': dm.get('spill_added_copy_bytes')}
    if scene == 'C':
        cs = r.get('cache_stats', {})
        total = cs.get('hit_bytes', 0) + cs.get('miss_bytes', 0)
        out['cache_hit_rate'] = cs.get('hit_bytes', 0) / total if total else 0.0
    return out


def build_construct_pool(model, N, scene, ctx=None, return_candidates=False):
    """兼容入口：返回新构造池中的方案对象。"""
    items = build_construct_candidates(model, N, scene, ctx=ctx)
    return items if return_candidates else [item.sol for item in items]


def _select_seed_solutions(scored, limit=6):
    """按范式、粒度和结构签名保留多样化种子。"""
    if not scored:
        return []
    ranked = sorted(scored, key=lambda x: x[0])
    selected = []
    seen_sig = set()
    # 第一轮保证范式覆盖，第二轮保证粒度覆盖，最后按代理适应度补齐。
    for key_index in ("paradigm", "granularity"):
        keys = []
        for row in ranked:
            meta = row[4] if len(row) > 4 else None
            key = getattr(meta, key_index, None)
            if key is not None and key not in keys:
                keys.append(key)
        for key in keys:
            row = next((r for r in ranked
                        if getattr(r[4], key_index, None) == key
                        and id(r[1]) not in seen_sig), None)
            if row is not None and len(selected) < limit:
                selected.append(row)
                seen_sig.add(id(row[1]))
    for row in ranked:
        if len(selected) >= limit:
            break
        if id(row[1]) not in seen_sig:
            selected.append(row)
            seen_sig.add(id(row[1]))
    return selected[:limit]


def solve_case(graph_json, N, scene, time_budget=10.0, seed=0,
               verify_k=3, block_ops_cap=120, traffic_weight=0.2,
               use_screen=False, spill_calibrated=False):
    """求解一个 (用例, 核数, 场景)。use_screen=True 时 SA/TS 启用
    Mamba 式 SSM 粗筛层；spill_calibrated=True 时启用 NNLS 校准的
    spill 系数（默认启发式）。返回 dict。"""
    t0 = time.time()
    model = Model(graph_json, block_ops_cap=block_ops_cap)
    if spill_calibrated:
        _apply_spill_coefs(model)
    ctx = Context(model, N, scene, traffic_weight=traffic_weight,
                  screen_enabled=use_screen, seed=seed)
    rng = random.Random(seed)
    archive = ParetoArchive(cap=48)

    # 1. 构造池
    # 大图的官方真值评估本身受 verify_cap 限制，构造池按方案中的
    # 时间预算降级规则只保留 R0，避免精化层吞掉主搜索预算。
    if len(graph_json['ops']) > 12000:
        # 超大图的边界流量矩阵构造会超过总预算；按方案的超时兜底规则
        # 使用单个 HEFT 代表进入搜索，专项构造池验证仍覆盖完整 R0 矩阵。
        sg, core = construct.heft_construct(model, N, scene,
                                             max_sg_ops=1280)
        pool = [Sol(sg, core)]
    else:
        items = build_construct_pool(model, N, scene, ctx=ctx,
                                     return_candidates=True)
        pool = [item.sol for item in items]
    scored = []
    for index, sol in enumerate(pool):
        sol.compact()
        mk, ad, _ = ctx.evaluate(sol)
        archive.insert(mk, ad, sol)
        meta = (items[index] if len(graph_json['ops']) <= 12000 and
                'items' in locals() else None)
        scored.append((ctx.fitness(mk, ad), sol, mk, ad, meta))
    scored.sort(key=lambda x: x[0])
    seed_rows = _select_seed_solutions(scored, min(6, len(scored)))
    seed_solutions = [row[1].clone() for row in seed_rows]
    best_row = scored[0]
    best_sol, f_best, mk_best, ad_best = (best_row[1].clone(), best_row[0],
                                          best_row[2], best_row[3])
    log = {'construct_best': [mk_best, ad_best],
           'construct_seed_count': len(seed_solutions),
           'construct_seed_paradigms': [getattr(row[4], 'paradigm', None)
                                        for row in seed_rows]}

    # 超大图的邻域搜索会使单个任务远超总预算；保留构造解代理结果，
    # 由全量统计单独标记为时间预算兜底。官方真值评估对这类图也受限。
    if len(graph_json['ops']) > 12000:
        best_sol.compact()
        mk_final, ad_final, info = ctx.evaluate(best_sol)
        plan = model.plan_from(best_sol.sg_of_block, info['orders'])
        log['large_case_fallback'] = True
        log['final_est'] = [mk_final, ad_final]
        log['n_evals'] = ctx.n_evals
        return {'plan': plan, 'est': (mk_final, ad_final), 'real': None,
                'log': log, 'elapsed': time.time() - t0}

    # 2. 元启发式串联（共享归档）
    deadline = t0 + max(0.5, time_budget)
    t_left = max(0.0, deadline - time.time())
    budget = min(t_left * 0.25, max(0.05, t_left))
    per_seed = budget / max(1, len(seed_solutions))
    for seed_sol in seed_solutions:
        if time.time() >= deadline:
            break
        sol, f, mk, ad = tabu_search(ctx, seed_sol.clone(), per_seed,
                                     archive=archive, rng=rng,
                                     deadline=deadline)
        if f is not None and f < f_best:
            best_sol, f_best, mk_best, ad_best = sol.clone(), f, mk, ad
    log['after_tabu'] = [mk_best, ad_best]

    t_left = max(0.0, deadline - time.time())
    budget = min(t_left * 0.40, max(0.05, t_left))
    per_seed = budget / max(1, len(seed_solutions))
    for seed_sol in seed_solutions:
        if time.time() >= deadline:
            break
        start_sol = seed_sol.clone()
        sol, f, mk, ad = sa_search(ctx, start_sol, per_seed,
                                   archive=archive, rng=rng,
                                   deadline=deadline)
        if f is not None and f < f_best:
            best_sol, f_best, mk_best, ad_best = sol.clone(), f, mk, ad
    log['after_sa'] = [mk_best, ad_best]

    t_left = max(0.0, deadline - time.time())
    budget = min(t_left * 0.25, max(0.05, t_left))
    sol, f, mk, ad = aco_search(ctx, budget, archive=archive,
                                rng=rng, deadline=deadline)
    if f is not None and f < f_best:
        best_sol, f_best, mk_best, ad_best = sol.clone(), f, mk, ad
    log['after_aco'] = [mk_best, ad_best]

    t_left = max(0.0, deadline - time.time())
    budget = min(t_left * 0.25, max(0.05, t_left))
    sol, f, mk, ad = mopso_search(ctx, budget, archive=archive,
                                  rng=rng, deadline=deadline)
    if f is not None and f < f_best:
        best_sol, f_best, mk_best, ad_best = sol.clone(), f, mk, ad
    log['after_mopso'] = [mk_best, ad_best]

    # 2.5 大用例专属：逐案真值在线校正 + 歧义裁决。
    # 小用例最终会全量真值校验，无需此处；大用例（>1.2万op）搜索完全
    # 依赖代理，用 (est, real) 比值做逐案乘法校正后再精搜一轮。
    per_case_ratio = None
    n_ops = len(graph_json['ops'])
    if n_ops > 12000 and time.time() - t0 < time_budget * 0.55:
        top2 = []
        if archive.items:
            for _mk, _ad, s in sorted(
                    archive.items,
                    key=lambda it: it[0] + traffic_weight * it[1] / BW)[:2]:
                top2.append(s)
        ratios = []
        for s in top2:
            s.compact()
            mkx, adx, infox = ctx.evaluate(s)
            planx = model.plan_from(s.sg_of_block, infox['orders'])
            rx = real_evaluate(graph_json, planx, scene)
            if rx and not rx.get('error') and mkx > 0:
                ratios.append(rx['makespan'] / mkx)
                if real is None or rx['makespan'] < real['makespan']:
                    real, plan = rx, planx
        if ratios:
            per_case_ratio = max(0.2, min(5.0, sum(ratios) / len(ratios)))
            ctx.fitness_bias = per_case_ratio   # mk 分量加权
            log['per_case_ratio'] = per_case_ratio
            # 用校正后的适应度再精搜一轮（剩余时间的一半）
            t_left = time_budget - (time.time() - t0)
            if t_left > 3.0:
                sol, f, mk, ad = tabu_search(
                    ctx, best_sol.clone(), t_left * 0.5,
                    archive=archive, rng=rng)
                if f is not None and f < f_best:
                    best_sol, f_best, mk_best, ad_best = (sol.clone(), f,
                                                          mk, ad)
                log['after_recail'] = [mk_best, ad_best]

    # 3. 最终选择：候选 = 搜索最优 + 归档标量最优 + 构造池前几名；
    #    小图用官方评估器真值校验选优，大图按代理值选优。
    best_sol.compact()
    mk_final, ad_final, info = ctx.evaluate(best_sol)
    # Tessel 式顺序精修：固定切图/分核，交换相邻独立子图（代理择优）
    refined_orders = None
    try:
        from order_refine import refine_orders
        t_left = time_budget - (time.time() - t0)
        cand_orders, base_mk, ref_mk = refine_orders(
            ctx, best_sol, info['orders'], rounds=3,
            deadline=time.time() + max(1.0, t_left * 0.2))
        if ref_mk < base_mk - 1e-9:
            refined_orders = cand_orders
            info = dict(info)
            info['orders'] = cand_orders
            mk_final = ref_mk
            log['order_refine'] = [base_mk, ref_mk]
    except Exception:
        pass
    plan = model.plan_from(best_sol.sg_of_block, info['orders'])
    real = None
    cands = [best_sol]
    if archive.items:
        # 歧义裁决：归档中标量分接近 best_sol（<5%）的候选也送真值校验
        # （仅小图，大图由 2.5 的逐案校正覆盖）
        _mk2, _ad2, sol2 = archive.best_by_scalar(traffic_weight)
        if sol2 is not None:
            sol2.compact()
            cands.append(sol2)
    for entry in scored[:max(3, len(seed_rows))]:
        if entry[1] is not None and all(
                entry[1].sg_of_block != c.sg_of_block for c in cands):
            cands.append(entry[1])
    n_ops = len(graph_json['ops'])
    VERIFY_CAP = 12000
    if n_ops > VERIFY_CAP:
        cands = cands[:1]
    elif n_ops > 6000:
        cands = cands[:3]
    if n_ops <= VERIFY_CAP:
        best_key, best_plan, real = None, plan, None
        seen = set()

        def lexi_better(mk_a, ad_a, mk_b, ad_b, tol=0.003):
            """词典序：Makespan 严格优先（tol 内并列比新增搬运）。"""
            if mk_a < mk_b * (1 - tol):
                return True
            if mk_a > mk_b * (1 + tol):
                return False
            return (ad_a or 0) < (ad_b or 0)

        for solx in cands:
            solx.compact()
            mkx, adx, infox = ctx.evaluate(solx)
            # best_sol 用精修后的顺序
            planx = model.plan_from(
                solx.sg_of_block,
                refined_orders if (solx is best_sol and refined_orders)
                else infox['orders'])
            sig = json.dumps({'n': planx['node_to_subgraph'],
                              'c': planx['core_schedules']}, sort_keys=True)
            if sig in seen:
                continue
            seen.add(sig)
            rx = real_evaluate(graph_json, planx, scene)
            if rx is None or rx.get('error'):
                continue
            if best_key is None or lexi_better(
                    rx['makespan'], rx['added_copy_bytes'],
                    best_key[0], best_key[1]):
                best_key = (rx['makespan'], rx['added_copy_bytes'])
                best_plan, real = planx, rx
        plan = best_plan
    log['final_est'] = [mk_final, ad_final]
    log['n_evals'] = ctx.n_evals
    log['pareto_size'] = len(archive.items)
    if ctx.screen is not None:
        s = ctx.screen.stats
        log['screen'] = {
            'skipped': s['skipped'], 'admitted': s['admitted'],
            'full_eval_reduction': ctx.screen.skip_ratio,
            'precision': ctx.screen.precision,
            'miss_rate': ctx.screen.miss_rate,
            'tau': ctx.screen.tau, 'n_learn': s['learn'],
        }
    return {'plan': plan, 'est': (mk_final, ad_final), 'real': real,
            'log': log, 'elapsed': time.time() - t0}
