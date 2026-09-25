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

from model import Model, load_graph, BW, MEM_CAP, _tarjan_scc
from solution import Sol, Context
from archive import ParetoArchive
import construct
from construct_refine import build_construct_candidates, ConstructCandidate
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


def _repair_quotient_cycles(model, sol):
    """Merge cyclic quotient SCCs while preserving the representative core."""
    repaired = sol.clone()
    repaired.compact()
    for _ in range(16):
        succs = [set() for _ in repaired.core_of_sg]
        for edge in model.block_edges:
            if len(edge) == 2 and isinstance(edge[0], (tuple, list)):
                u, v = edge[0]
            else:
                u, v = edge[:2]
            source = repaired.sg_of_block[u]
            target = repaired.sg_of_block[v]
            if source != target:
                succs[source].add(target)
        cyclic = [component for component in _tarjan_scc(succs)
                  if len(component) > 1]
        if not cyclic:
            return repaired
        remap = list(range(len(repaired.core_of_sg)))
        for component in cyclic:
            representative = min(component)
            for subgraph in component:
                remap[subgraph] = representative
        repaired = Sol([remap[subgraph]
                        for subgraph in repaired.sg_of_block],
                       list(repaired.core_of_sg))
        repaired.compact()
    return repaired


def _legacy_n5_constructs(model, N, scene):
    """Recover two high-value main-branch structures omitted by coarsening."""
    if N != 5:
        return []
    builders = (
        ('legacy_strip10', lambda: construct.strip_construct(
            model, N, scene, num_strips=2*N), False),
        ('legacy_strip40', lambda: construct.strip_construct(
            model, N, scene, num_strips=8*N), False),
        ('legacy_chain_uncapped', lambda: construct.chain_construct(
            model, N, scene, max_sg_ops=None, max_subgraphs=None), True),
    )
    rows = []
    for name, builder, repair_cycles in builders:
        try:
            sg, cores = builder()
            sol = Sol(sg, cores)
            sol.compact()
            if repair_cycles:
                sol = _repair_quotient_cycles(model, sol)
            if sol.validate(model, N):
                rows.append(ConstructCandidate(
                    paradigm=name, granularity='legacy', scene=scene,
                    refine_level='R0', sol=sol))
        except Exception:
            continue
    return rows


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


def _lru_rerank(ctx, options, limit=16):
    """Rerank a small diverse pool with the slower lifetime spill model."""
    unique = []
    seen = set()
    for sol in options:
        cand = sol.clone()
        cand.compact()
        sig = (tuple(cand.sg_of_block), tuple(cand.core_of_sg))
        if sig not in seen:
            seen.add(sig)
            unique.append(cand)
        if len(unique) >= limit:
            break
    if not unique:
        return [], {'candidates': 0}
    old_mode = ctx.model.use_lru_spill
    ranked = []
    try:
        ctx.model.use_lru_spill = True
        for cand in unique:
            mk, added, _info = ctx.evaluate(cand, use_cache=False)
            ranked.append((ctx.fitness(mk, added), cand))
    finally:
        ctx.model.use_lru_spill = old_mode
    ranked.sort(key=lambda item: item[0])
    return [item[1] for item in ranked], {
        'candidates': len(ranked), 'best_fitness': ranked[0][0]}


def _record_diagnostics(model, plan, scene, num_cores, real, log):
    try:
        from diagnostics import schedule_diagnostics
        official_mk = real.get('makespan') if real else None
        log['diagnostics'] = schedule_diagnostics(
            model, plan, scene, num_cores, official_mk)
    except Exception as exc:
        log['diagnostics_error'] = repr(exc)


def _solution_from_plan(model, plan, num_cores):
    """Decode a persisted plan into the block/subgraph representation."""
    schedules = plan.get('core_schedules')
    mapping = plan.get('node_to_subgraph')
    if not isinstance(schedules, list) or len(schedules) != num_cores:
        raise ValueError('invalid core_schedules')
    if not isinstance(mapping, dict):
        raise ValueError('invalid node_to_subgraph')
    block_to_sg = []
    for block in range(len(model.blocks)):
        op_ids = model.blocks[block]
        if not op_ids:
            block_to_sg.append(0)
            continue
        sg = mapping.get(str(op_ids[0]), mapping.get(op_ids[0]))
        if sg is None:
            raise ValueError('plan does not cover block')
        block_to_sg.append(int(sg))
    max_sg = max(block_to_sg, default=-1)
    core_of_sg = [0] * (max_sg + 1)
    for core, row in enumerate(schedules):
        for sg in row:
            if int(sg) >= len(core_of_sg):
                core_of_sg.extend([0] * (int(sg) + 1 - len(core_of_sg)))
            core_of_sg[int(sg)] = core
    from solution import Sol
    sol = Sol(block_to_sg, core_of_sg)
    sol.compact()
    if not sol.validate(model, num_cores):
        raise ValueError('decoded plan is not valid')
    return sol


def _bounded_deadline(global_deadline, budget, now=None):
    """Give one search call its own budget without exceeding the task limit."""
    now = time.time() if now is None else now
    return min(global_deadline, now + max(0.0, budget))


def solve_case(graph_json, N, scene, time_budget=10.0, seed=0,
               verify_k=16, block_ops_cap=120, traffic_weight=0.2,
               use_screen=False, spill_calibrated=False, warm_plan=None,
               warm_plans=None, use_mcts=False, use_beam=False):
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
        legacy_items = _legacy_n5_constructs(model, N, scene)
        items.extend(legacy_items)
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
    # 高官方核验预算同时扩大构造种子，避免核验上限提高后仍被少量
    # 同质化候选填满；默认模式保持 6 个种子以控制运行时间。
    seed_limit = 8 if int(verify_k) >= 24 else 6
    seed_rows = _select_seed_solutions(scored,
                                       min(seed_limit, len(scored)))
    seed_solutions = [row[1].clone() for row in seed_rows]
    best_row = scored[0]
    best_sol, f_best, mk_best, ad_best = (best_row[1].clone(), best_row[0],
                                          best_row[2], best_row[3])
    log = {'construct_best': [mk_best, ad_best],
           'construct_seed_count': len(seed_solutions),
           'construct_seed_paradigms': [getattr(row[4], 'paradigm', None)
                                        for row in seed_rows],
           'legacy_construct_count': len(legacy_items)
           if 'legacy_items' in locals() else 0}

    # 超大图的邻域搜索会使单个任务远超总预算；保留构造解代理结果，
    # 由全量统计单独标记为时间预算兜底。官方真值评估对这类图也受限。
    if len(graph_json['ops']) > 12000:
        best_sol.compact()
        mk_final, ad_final, info = ctx.evaluate(best_sol)
        plan = model.plan_from(best_sol.sg_of_block, info['orders'])
        log['large_case_fallback'] = True
        log['final_est'] = [mk_final, ad_final]
        log['n_evals'] = ctx.n_evals
        _record_diagnostics(model, plan, scene, N, None, log)
        return {'plan': plan, 'est': (mk_final, ad_final), 'real': None,
                'log': log, 'elapsed': time.time() - t0}

    # 2. 元启发式串联（共享归档）
    total_budget = max(0.5, time_budget)
    deadline = t0 + total_budget
    t_left = max(0.0, deadline - time.time())
    budget = min(t_left * 0.25, max(0.05, t_left))
    per_seed = budget / max(1, len(seed_solutions))
    stage_start, stage_evals = time.time(), ctx.n_evals
    for seed_sol in seed_solutions:
        if time.time() >= deadline:
            break
        sol, f, mk, ad = tabu_search(ctx, seed_sol.clone(), per_seed,
                                     archive=archive, rng=rng,
                                     deadline=_bounded_deadline(
                                         deadline, per_seed))
        if f is not None and f < f_best:
            best_sol, f_best, mk_best, ad_best = sol.clone(), f, mk, ad
    log['after_tabu'] = [mk_best, ad_best]
    log.setdefault('search_stage_evals', {})['tabu'] = ctx.n_evals-stage_evals
    log.setdefault('search_stage_seconds', {})['tabu'] = time.time()-stage_start

    t_left = max(0.0, deadline - time.time())
    budget = min(t_left * 0.40, max(0.05, t_left))
    per_seed = budget / max(1, len(seed_solutions))
    stage_start, stage_evals = time.time(), ctx.n_evals
    for seed_sol in seed_solutions:
        if time.time() >= deadline:
            break
        start_sol = seed_sol.clone()
        sol, f, mk, ad = sa_search(ctx, start_sol, per_seed,
                                   archive=archive, rng=rng,
                                   deadline=_bounded_deadline(
                                       deadline, per_seed))
        if f is not None and f < f_best:
            best_sol, f_best, mk_best, ad_best = sol.clone(), f, mk, ad
    log['after_sa'] = [mk_best, ad_best]
    log['search_stage_evals']['sa'] = ctx.n_evals-stage_evals
    log['search_stage_seconds']['sa'] = time.time()-stage_start

    t_left = max(0.0, deadline - time.time())
    budget = min(t_left * 0.25, max(0.05, t_left))
    stage_start, stage_evals = time.time(), ctx.n_evals
    sol, f, mk, ad = aco_search(ctx, budget, archive=archive,
                                rng=rng,
                                deadline=_bounded_deadline(deadline, budget))
    if f is not None and f < f_best:
        best_sol, f_best, mk_best, ad_best = sol.clone(), f, mk, ad
    log['after_aco'] = [mk_best, ad_best]
    log['search_stage_evals']['aco'] = ctx.n_evals-stage_evals
    log['search_stage_seconds']['aco'] = time.time()-stage_start

    t_left = max(0.0, deadline - time.time())
    budget = min(t_left * 0.25, max(0.05, t_left))
    stage_start, stage_evals = time.time(), ctx.n_evals
    sol, f, mk, ad = mopso_search(ctx, budget, archive=archive,
                                  rng=rng,
                                  deadline=_bounded_deadline(deadline, budget))
    if f is not None and f < f_best:
        best_sol, f_best, mk_best, ad_best = sol.clone(), f, mk, ad
    log['after_mopso'] = [mk_best, ad_best]
    log['search_stage_evals']['mopso'] = ctx.n_evals-stage_evals
    log['search_stage_seconds']['mopso'] = time.time()-stage_start

    # MCTS is an additive candidate generator.  It gets a small extra budget
    # and a separate archive, so it cannot consume the original search time or
    # evict candidates produced by the established pipeline.
    mcts_candidates = []
    beam_candidates = []
    wavefront_candidates = []
    guided_high_yield = (scene, N) in {
        ('A', 3), ('A', 4), ('A', 5), ('B', 4), ('B', 5)}
    guided_evals = 192 if guided_high_yield else 64
    if use_mcts:
        try:
            from mcts_schedule import mcts_search
            mcts_budget = min(4.0, max(2.0, total_budget * 0.20))
            mcts_deadline = time.time() + mcts_budget
            mcts_archive = ParetoArchive(cap=24)
            mcts_ranked = []
            mcts_sol, mcts_f, mcts_mk, mcts_ad, mcts_stats = mcts_search(
                ctx, best_sol, mcts_budget, deadline=mcts_deadline, rng=rng,
                archive=mcts_archive, max_depth=3, branching=20,
                max_evals=guided_evals, stall_limit=256,
                candidate_pool=mcts_ranked, candidate_cap=32)
            log['mcts'] = mcts_stats
            log['mcts']['budget'] = mcts_budget
            log['mcts']['proxy_improved'] = mcts_f < f_best
            log['after_mcts'] = [mcts_mk, mcts_ad]
            mcts_candidates.append(mcts_sol.clone())
            mcts_candidates.extend(
                item[2].clone() for item in sorted(
                    mcts_archive.items,
                    key=lambda it: ctx.fitness(it[0], it[1])))
            mcts_candidates.extend(item[3].clone() for item in mcts_ranked)
        except Exception as exc:
            log['mcts_error'] = repr(exc)

    if use_beam:
        try:
            from beam_schedule import beam_search
            beam_budget = min(4.0, max(2.0, total_budget * 0.20))
            beam_deadline = time.time() + beam_budget
            beam_archive = ParetoArchive(cap=24)
            beam_ranked = []
            beam_sol, beam_f, beam_mk, beam_ad, beam_stats = beam_search(
                ctx, best_sol, beam_budget, deadline=beam_deadline,
                archive=beam_archive, max_depth=4, width=8, branching=24,
                max_evals=guided_evals, candidate_pool=beam_ranked,
                candidate_cap=32, structural_actions=True)
            log['beam'] = beam_stats
            log['beam']['budget'] = beam_budget
            log['beam']['proxy_improved'] = beam_f < f_best
            log['after_beam'] = [beam_mk, beam_ad]
            beam_candidates.append(beam_sol.clone())
            beam_candidates.extend(
                item[2].clone() for item in sorted(
                    beam_archive.items,
                    key=lambda it: ctx.fitness(it[0], it[1])))
            beam_candidates.extend(item[3].clone() for item in beam_ranked)
        except Exception as exc:
            log['beam_error'] = repr(exc)

    if N == 5:
        try:
            from wavefront_schedule import generate_candidates
            wavefront_rows, wavefront_stats = generate_candidates(
                ctx, best_sol, max_actions=24)
            log['wavefront'] = wavefront_stats
            wavefront_candidates.extend(row[2].clone() for row in wavefront_rows)
        except Exception as exc:
            log['wavefront_error'] = repr(exc)

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

    # 3. 慢速生命周期模型只复核少量候选，不拖慢整个邻域搜索。
    cheap_best_sol = best_sol.clone()
    lru_ranked = []
    if n_ops <= 12000:
        pool_for_rerank = [best_sol]
        pool_for_rerank.extend(
            item[2] for item in sorted(
                archive.items,
                key=lambda it: ctx.fitness(it[0], it[1]))[:12])
        pool_for_rerank.extend(row[1] for row in scored[:6])
        # 让慢速生命周期模型覆盖与官方核验相同量级的候选；默认值仍为
        # 16，实验可通过 verify_k 提升到 24/32。
        rerank_limit = max(16, min(32, int(verify_k)))
        lru_ranked, lru_stats = _lru_rerank(
            ctx, pool_for_rerank, limit=rerank_limit)
        log['lru_rerank'] = lru_stats
        if lru_ranked:
            best_sol = lru_ranked[0].clone()

    # 4. 最终选择：候选 = 搜索最优 + 归档标量最优 + 构造池前几名；
    #    小图用官方评估器真值校验选优，大图按代理值选优。
    best_sol.compact()
    unrefined_best_sol = best_sol.clone()
    mk_final, ad_final, info = ctx.evaluate(best_sol)
    # 固定切图/分核：先相邻交换，再对关键路径窗口做精确重排。
    refined_orders = None
    try:
        from order_refine import refine_orders, refine_orders_exact
        t_left = time_budget - (time.time() - t0)
        cand_orders, base_mk, ref_mk = refine_orders(
            ctx, best_sol, info['orders'], rounds=3,
            deadline=time.time() + max(1.0, t_left * 0.2))
        exact_orders, _exact_base, exact_mk, exact_stats = refine_orders_exact(
            ctx, best_sol, cand_orders)
        log['exact_refine'] = exact_stats
        if exact_mk is not None and exact_mk < ref_mk - 1e-9:
            cand_orders, ref_mk = exact_orders, exact_mk
        if ref_mk < base_mk - 1e-9:
            refined_orders = cand_orders
            info = dict(info)
            info['orders'] = cand_orders
            mk_final = ref_mk
            log['order_refine'] = [base_mk, ref_mk]
    except Exception as exc:
        log['order_refine_error'] = repr(exc)
    plan = model.plan_from(best_sol.sg_of_block, info['orders'])
    real = None
    base_cands = [best_sol, unrefined_best_sol, cheap_best_sol]
    if archive.items:
        # 歧义裁决：归档中标量分接近 best_sol（<5%）的候选也送真值校验
        # （仅小图，大图由 2.5 的逐案校正覆盖）
        _mk2, _ad2, sol2 = archive.best_by_scalar(traffic_weight)
        if sol2 is not None:
            sol2.compact()
            base_cands.append(sol2)
    base_cands.extend(entry[1] for entry in scored[:max(3, len(seed_rows))]
                      if entry[1] is not None)
    base_cands.extend(lru_ranked)
    legacy_cands = ([item.sol for item in legacy_items]
                    if 'legacy_items' in locals() else [])
    n_ops = len(graph_json['ops'])
    VERIFY_CAP = 12000
    if n_ops > VERIFY_CAP:
        max_verify = 1
    elif n_ops > 6000:
        # 中大图官方评估较慢，保守限制候选数；调用方仍可用 verify_k
        # 在预算允许时提高上限。
        max_verify = min(max(8, int(verify_k)), 24)
    else:
        max_verify = min(max(16, int(verify_k)), 32)
    if n_ops <= VERIFY_CAP:
        best_key, best_plan, real = None, plan, None
        seen = set()

        def lexi_better(mk_a, ad_a, mk_b, ad_b):
            """Speedup uses makespan, so compare it before copy traffic."""
            return (mk_a, ad_a or 0) < (mk_b, ad_b or 0)

        best_source = None
        best_record = None
        official_trace = []

        def verify_group(candidates, limit, source):
            nonlocal best_key, best_plan, real, best_source, best_record
            verified = 0
            for solx in candidates:
                if verified >= limit:
                    break
                solx.compact()
                _mkx, _adx, infox = ctx.evaluate(solx)
                planx = model.plan_from(
                    solx.sg_of_block,
                    refined_orders if (solx is best_sol and refined_orders)
                    else infox['orders'])
                sig = json.dumps({'n': planx['node_to_subgraph'],
                                  'c': planx['core_schedules']},
                                 sort_keys=True)
                if sig in seen:
                    continue
                seen.add(sig)
                verified += 1
                rx = real_evaluate(graph_json, planx, scene)
                record = {
                    'source': source,
                    'proxy_makespan': _mkx,
                    'proxy_added': _adx,
                    'official_makespan': (
                        rx.get('makespan') if rx and not rx.get('error')
                        else None),
                    'official_added': (
                        rx.get('added_copy_bytes')
                        if rx and not rx.get('error') else None),
                }
                official_trace.append(record)
                if rx is None or rx.get('error'):
                    continue
                if best_key is None or lexi_better(
                        rx['makespan'], rx['added_copy_bytes'],
                        best_key[0], best_key[1]):
                    best_key = (rx['makespan'], rx['added_copy_bytes'])
                    best_plan, real, best_source = planx, rx, source
                    best_record = record
            return verified

        base_verified = verify_group(base_cands, max_verify, 'base')
        legacy_verified = verify_group(legacy_cands, 3, 'legacy_main')
        guided_limit = 8 if guided_high_yield and n_ops <= 6000 else 4
        if use_mcts and use_beam:
            mcts_limit = (guided_limit + 1) // 2
            beam_limit = guided_limit // 2
        else:
            mcts_limit = guided_limit if use_mcts else 0
            beam_limit = guided_limit if use_beam else 0
        mcts_verified = verify_group(
            mcts_candidates, mcts_limit, 'mcts') if mcts_limit else 0
        beam_verified = verify_group(
            beam_candidates, beam_limit, 'beam') if beam_limit else 0
        warm_candidates = list(warm_plans or ())
        if warm_plan is not None:
            warm_candidates.insert(0, {'plan': warm_plan,
                                       'source': 'legacy_argument'})
        warm_verified = 0
        warm_errors = []
        if N == 5 and warm_candidates:
            try:
                from wavefront_schedule import generate_candidates
                for warm_candidate in warm_candidates[:24]:
                    warm_plan0 = (warm_candidate.get('plan')
                                  if isinstance(warm_candidate, dict)
                                  else warm_candidate)
                    try:
                        warm_sol = _solution_from_plan(model, warm_plan0, N)
                    except Exception:
                        continue
                    wf_rows, _wf_stats = generate_candidates(
                        ctx, warm_sol, max_actions=8)
                    wavefront_candidates.extend(row[2].clone() for row in wf_rows)
                if wavefront_candidates:
                    log.setdefault('wavefront', {})['warm_candidates'] = len(wavefront_candidates)
            except Exception as exc:
                log['wavefront_warm_error'] = repr(exc)
        wavefront_limit = 16 if wavefront_candidates and guided_high_yield else (4 if wavefront_candidates else 0)
        wavefront_verified = verify_group(
            wavefront_candidates, wavefront_limit, 'wavefront') if wavefront_limit else 0
        for warm_index, candidate in enumerate(warm_candidates):
            candidate_plan = (candidate.get('plan')
                              if isinstance(candidate, dict) else candidate)
            candidate_source = (candidate.get('source', str(warm_index))
                                if isinstance(candidate, dict)
                                else str(warm_index))
            warm_real = real_evaluate(graph_json, candidate_plan, scene)
            if warm_real is not None and not warm_real.get('error'):
                warm_verified += 1
                warm_record = {
                    'source': 'warm_portfolio',
                    'warm_source': candidate_source,
                    'proxy_makespan': None,
                    'proxy_added': None,
                    'official_makespan': warm_real['makespan'],
                    'official_added': warm_real['added_copy_bytes'],
                }
                official_trace.append(warm_record)
                log.setdefault('warm_start_makespans', []).append({
                    'source': candidate_source,
                    'makespan': warm_real['makespan'],
                })
                if best_key is None or lexi_better(
                        warm_real['makespan'], warm_real['added_copy_bytes'],
                        best_key[0], best_key[1]):
                    best_key = (warm_real['makespan'],
                                warm_real['added_copy_bytes'])
                    best_plan, real = candidate_plan, warm_real
                    best_source = 'warm_portfolio'
                    best_record = warm_record
                    log['warm_start_used'] = True
            else:
                warm_errors.append({
                    'source': candidate_source,
                    'error': (warm_real.get('error') if warm_real
                              else 'unavailable'),
                })
        plan = best_plan
        log['official_verified_candidates'] = len(seen)
        log['official_verified_base_candidates'] = base_verified
        log['official_verified_legacy_candidates'] = legacy_verified
        log['official_verified_mcts_candidates'] = mcts_verified
        log['official_verified_beam_candidates'] = beam_verified
        log['official_verified_wavefront_candidates'] = wavefront_verified
        log['official_verified_warm_candidates'] = warm_verified
        if warm_errors:
            log['warm_start_errors'] = warm_errors
        log['official_selected_source'] = best_source
        for record in official_trace:
            record['selected'] = record is best_record
        log['official_candidate_trace'] = official_trace
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
    _record_diagnostics(model, plan, scene, N, real, log)
    return {'plan': plan, 'est': (mk_final, ad_final), 'real': real,
            'log': log, 'elapsed': time.time() - t0}
