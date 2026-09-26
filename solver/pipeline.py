"""求解流水线：构造 -> 禁忌搜索 -> 模拟退火 -> 蚁群 -> MOPSO -> 精选。

多目标优化框架：所有算法向共享 ParetoArchive 供解（目标：
估计 Makespan、估计新增搬运量），最终按 makespan + w*added/BW
标量化选解；小图用官方评估器对 top-k 候选做真值校验。
"""
import json
import copy
import hashlib
import os
import random
import sys
import time

from model import Model, load_graph, BW, MEM_CAP
from solution import Sol, Context
from archive import ParetoArchive
import construct
from construct_refine import (build_construct_candidates,
                              build_large_construct_candidates)
from tabu import tabu_search
from sa import sa_search
from aco import aco_search
from mopso import mopso_search

ATTACH_CODE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           '..', '通用神经网络处理器下的多核调度问题附件', 'code')
SPILL_COEFS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'spill_coefs.json')


def _apply_spill_coefs(model, num_cores):
    """仅用 NNLS 校准系数修正原 spill 启发式的高估部分。"""
    if os.path.exists(SPILL_COEFS):
        try:
            with open(SPILL_COEFS, encoding='utf-8') as f:
                coefs = json.load(f)['coefs']
            # 2 核样本的 spill/并行度关系与多核不同，保持原启发式；
            # 3 核以上只下调高估，避免不稳定的放大项改变候选排序。
            blend = 0.0 if int(num_cores) <= 2 else 0.25
            model.set_spill_coefs(tuple(float(x) for x in coefs), blend=blend)
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


def fast_evaluate(graph_json, plan, scene, verify_cap=12000):
    """仅供候选复核的计数式副本；最终结果仍需原官方确认。"""
    if scene != 'A' or len(graph_json['ops']) > verify_cap:
        return real_evaluate(graph_json, plan, scene, verify_cap)
    try:
        from scene_a_fast import evaluate_scene_a_fast
        r = evaluate_scene_a_fast(graph_json, plan, 60.0,
                                  {'L1': 524288, 'UB': 131072}, 1000, 100)
        dm = r['data_movement_bytes']
        return {'makespan': r['makespan'], 'added_copy_bytes': dm['added_copy_bytes'],
                'scheduled_copy_bytes': dm['scheduled_copy_bytes'],
                'partition_added': dm.get('partition_added_copy_bytes'),
                'spill_added': dm.get('spill_added_copy_bytes')}
    except Exception as exc:
        return {'error': repr(exc)}


def _cached_candidate_evaluator():
    """每次末端精化独占局部模板；最终确认仍调用 real_evaluate。"""
    from local_template_cache import TemplateCache
    cache = TemplateCache(max_entries=128, max_bytes=64 * 1024 * 1024)
    evaluate = None

    def candidate(graph_json, plan, scene):
        nonlocal evaluate
        if scene != 'A' or len(graph_json['ops']) > 12000:
            return {'error': '末端模板仅用于场景 A 的官方可评规模'}
        try:
            if evaluate is None:
                from scene_a_fast import build_counter_evaluator
                evaluate = build_counter_evaluator(template_cache=cache)
            result = evaluate(graph_json, plan, 60.0,
                              {'L1': 524288, 'UB': 131072}, 1000, 100)
            dm = result['data_movement_bytes']
            return {'makespan': result['makespan'],
                    'added_copy_bytes': dm['added_copy_bytes'],
                    'scheduled_copy_bytes': dm['scheduled_copy_bytes'],
                    'partition_added': dm.get('partition_added_copy_bytes'),
                    'spill_added': dm.get('spill_added_copy_bytes')}
        except Exception as exc:
            return {'error': repr(exc)}

    return candidate, cache


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


def _project_plan_to_blocks(model, plan):
    """将原算子方案投影到当前块模型；块内拆分时返回 None。

    B/C 原算子候选可以表达比搜索块更细的切分，但当前快速代理只接受
    一个块对应一个子图。保留 None 作为明确的可表达性信号，避免把无损
    映射失败伪装成搜索失败或让全量任务直接报错。
    """
    mapping = {int(k): int(v) for k, v in plan.get('node_to_subgraph', {}).items()}
    sg = []
    for block in model.blocks:
        groups = {mapping.get(int(op)) for op in block}
        if None in groups or len(groups) != 1:
            return None
        sg.append(next(iter(groups)))
    cores = [0] * (max(sg) + 1)
    for c, order in enumerate(plan['core_schedules']):
        for s in order:
            cores[s] = c
    try:
        projected = model.plan_from(sg, plan['core_schedules'])
    except (KeyError, IndexError, TypeError, ValueError):
        return None
    if projected != plan:
        return None
    return projected


def _proxy_for_plan(model, plan, scene, num_cores, graph=None):
    projected = _project_plan_to_blocks(model, plan)
    if projected is None:
        if scene in ('B', 'C') and graph is not None:
            from scene_b_features import SceneBFeatures
            features = SceneBFeatures(graph).evaluate(plan, scene)
            added = max(0, features['partition_added_bytes'])
            return features['lower_bound'], added, {
                'orders': plan['core_schedules'], 'partition_raw_bytes': features['boundary_bytes'],
                'partition_added_bytes': added, 'total_added_bytes': added,
                'proxy_representation': 'original_operations',
                'proxy_scope': 'compute_and_boundary_without_spill_prediction',
                'spill_prediction_available': False,
                'memory_overage': {p: features['per_pool'][p]['overage'] for p in ('L1', 'UB')},
                'peak_memory_bytes': {p: features['per_pool'][p]['peak'] for p in ('L1', 'UB')}}
        raise ValueError('后处理方案不能无损映射回块模型')
    mapping = {int(k): int(v) for k, v in plan['node_to_subgraph'].items()}
    sg = [mapping[int(block[0])] for block in model.blocks]
    cores = [0] * (max(sg) + 1)
    for c, order in enumerate(plan['core_schedules']):
        for s in order:
            cores[s] = c
    return model.evaluate(sg, cores, scene, num_cores, use_cache=False,
                          orders_override=plan['core_schedules'])


def _record_postprocess(model, proposed, audit, source, scene, num_cores, log):
    """统一记录固定切分和变切分后处理，确保最终摘要能追溯到候选。"""
    from terminal_refine import plan_signature
    final_metrics = _proxy_for_plan(model, proposed, scene, num_cores)
    for candidate in audit['candidates']:
        candidate_plan = candidate.get('plan')
        if candidate_plan is None:
            candidate_plan = {**proposed, 'core_schedules': candidate['orders']}
        mk, ad, info = _proxy_for_plan(model, candidate_plan, scene, num_cores)
        candidate_id = hashlib.sha256(plan_signature(candidate_plan).encode('utf-8')).hexdigest()
        candidate['plan_id'] = candidate_id
        log['official_candidates'].append({
            'plan_id': candidate_id, 'source': source,
            'proxy_makespan': mk, 'proxy_total_added_bytes': ad,
            'proxy_partition_added_bytes': info.get('partition_added_bytes'),
            'proxy_spill_added_bytes': info.get('spill_bytes'),
            'precise_makespan': candidate['estimate'], 'official': candidate['official']})
    log['selected_plan_id'] = hashlib.sha256(plan_signature(proposed).encode('utf-8')).hexdigest()
    return final_metrics


def solve_case(graph_json, N, scene, time_budget=10.0, seed=0,
               verify_k=3, block_ops_cap=120, traffic_weight=0.2,
               use_screen=False, spill_calibrated=False,
               use_lru_spill=False, event_rerank=False, task_order_refine=False,
               partition_polish=False, construct_reservoir=False,
               fast_candidate_eval=False, search_rounds=None,
               terminal_swaps=False, terminal_window=False, terminal_cache=False,
               bandwidth_proxy=False, contention_weight=0.0,
               large_case_reservoir=False, bc_refine=False,
               c_cache_search=False):
    """求解一个 (用例, 核数, 场景)。use_screen=True 时 SA/TS 启用
    Mamba 式 SSM 粗筛层；spill_calibrated=True 时启用 NNLS 校准的
    spill 系数（默认启发式）。返回 dict。"""
    if terminal_cache and not terminal_swaps:
        raise ValueError('terminal_cache 必须与 terminal_swaps 同时启用')
    if search_rounds is not None:
        if isinstance(search_rounds, bool) or not isinstance(search_rounds, int) or search_rounds < 1:
            raise ValueError('固定轮数必须为正整数')
        if any((use_screen, event_rerank, task_order_refine, partition_polish,
                construct_reservoir, fast_candidate_eval, terminal_window)):
            raise ValueError('固定轮数模式暂不支持额外限时后处理或学习筛选；请独立进行固定候选消融')
    t0 = time.time()
    n_ops = len(graph_json.get('ops', ()))
    model = Model(graph_json, block_ops_cap=block_ops_cap)
    model.use_lru_spill = bool(use_lru_spill)
    model.use_bandwidth_proxy = bool(bandwidth_proxy)
    if spill_calibrated and scene in ('A', 'B'):
        _apply_spill_coefs(model, N)
    ctx = Context(model, N, scene, traffic_weight=traffic_weight,
                  screen_enabled=use_screen, seed=seed,
                  contention_weight=(contention_weight if bandwidth_proxy else 0.0))
    ctx.retain_constructs = bool(construct_reservoir and event_rerank and scene == 'A')
    rng = random.Random(seed)
    archive = ParetoArchive(cap=48)

    # 1. 构造池
    # 大图的官方真值评估本身受 verify_cap 限制，构造池按方案中的
    # 时间预算降级规则只保留 R0，避免精化层吞掉主搜索预算。
    items = []
    if len(graph_json['ops']) > 12000:
        # 默认仍使用单个 HEFT 保底；大图专项实验显式打开候选储备池后，
        # 才比较四类低成本构造来源，避免改变历史实验的搜索路径。
        if large_case_reservoir:
            items = build_large_construct_candidates(
                model, N, scene, max_sg_ops=1280, max_candidates=4)
            pool = [item.sol for item in items]
        else:
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
        meta = (items[index] if index < len(items) else None)
        scored.append((ctx.fitness(mk, ad), sol, mk, ad, meta))
    scored.sort(key=lambda x: x[0])
    seed_rows = _select_seed_solutions(scored, min(6, len(scored)))
    seed_solutions = [row[1].clone() for row in seed_rows]
    best_row = scored[0]
    best_sol, f_best, mk_best, ad_best = (best_row[1].clone(), best_row[0],
                                          best_row[2], best_row[3])
    log = {'search_mode': 'fixed_rounds' if search_rounds is not None else 'wall_clock',
           'search_rounds': search_rounds, 'construct_best': [mk_best, ad_best],
           'construct_seed_count': len(seed_solutions),
           'construct_seed_paradigms': [getattr(row[4], 'paradigm', None)
                                        for row in seed_rows],
           'large_case_reservoir': bool(large_case_reservoir and
                                        len(graph_json['ops']) > 12000)}
    if len(graph_json['ops']) > 12000 and large_case_reservoir:
        log['large_case_candidates'] = [
            {'paradigm': item.paradigm, 'reason': item.reason,
             'signature': item.signature,
             'proxy_makespan': row[2],
             'proxy_added_copy_bytes': row[3]}
            for row in scored for item in ([row[4]] if row[4] is not None else [])]

    # 超大图的邻域搜索会使单个任务远超总预算；保留构造解代理结果，
    # 由全量统计单独标记为时间预算兜底。官方真值评估对这类图也受限。
    if len(graph_json['ops']) > 12000:
        best_sol.compact()
        mk_final, ad_final, info = ctx.evaluate(best_sol)
        plan = model.plan_from(best_sol.sg_of_block, info['orders'])
        log['large_case_fallback'] = True
        log['large_case_selected_source'] = (
            getattr(best_row[4], 'paradigm', 'heft')
            if best_row[4] is not None else 'heft')
        if terminal_swaps or terminal_window:
            log['terminal_swaps' if terminal_swaps else 'terminal_window'] = {
                'skipped': '大图尚无原官方保底，末端顺序精化未执行'}
        log['final_est'] = [mk_final, ad_final]
        log['n_evals'] = ctx.n_evals
        return {'plan': plan, 'est': (mk_final, ad_final), 'real': None,
                'log': log, 'elapsed': time.time() - t0}

    # 2. 元启发式串联（共享归档）
    if search_rounds is not None:
        # 每个构造种子固定禁忌/退火轮数；蚁群和粒子群固定代数。
        # 各算法每轮候选数不同，分别记评估数，不能宣称等评估次数预算。
        log['fixed_stage_evals'] = {}
        for name, search in (('tabu', tabu_search), ('sa', sa_search)):
            count_before = ctx.n_evals
            for seed_sol in seed_solutions:
                sol, f, mk, ad = search(ctx, seed_sol.clone(), 0, archive=archive,
                                       rng=rng, max_rounds=search_rounds)
                if f is not None and f < f_best:
                    best_sol, f_best, mk_best, ad_best = sol.clone(), f, mk, ad
            log['after_'+name] = [mk_best, ad_best]
            log['fixed_stage_evals'][name] = ctx.n_evals-count_before
        for name, search in (('aco', aco_search), ('mopso', mopso_search)):
            count_before = ctx.n_evals
            sol, f, mk, ad = search(ctx, 0, archive=archive, rng=rng, max_rounds=search_rounds)
            if f is not None and f < f_best:
                best_sol, f_best, mk_best, ad_best = sol.clone(), f, mk, ad
            log['after_'+name] = [mk_best, ad_best]
            log['fixed_stage_evals'][name] = ctx.n_evals-count_before
    else:
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
            deadline=float("inf") if search_rounds is not None else time.time() + max(1.0, t_left * 0.2))
        if ref_mk < base_mk - 1e-9:
            refined_orders = cand_orders
            mk_final, ad_final, info = model.evaluate(
                best_sol.sg_of_block, best_sol.core_of_sg, scene, N,
                use_cache=False, orders_override=cand_orders)
            log['order_refine'] = [base_mk, ref_mk]
    except Exception:
        pass
    plan = model.plan_from(best_sol.sg_of_block, info['orders'])
    log['search_best_est'] = [mk_final, ad_final]
    log['official_candidates'] = []
    bc_candidates = []
    bc_audit = None
    # B/C 的原算子精化只生成合法方案；真值仍统一走下面的原官方循环。
    # 大图没有可接受的官方保底时不生成候选，避免代理结果冒充正式选择。
    if scene in ('B', 'C') and n_ops <= 12000 and (bc_refine or c_cache_search):
        try:
            if scene == 'C' and c_cache_search:
                from bc_search import beam_search_candidates
                bc_candidates, bc_audit = beam_search_candidates(
                    graph_json, plan, scene, depth=2, width=4,
                    max_states=64, max_candidates=6, seed=seed)
                bc_audit['search'] = 'beam'
            else:
                from bc_refine import generate_bc_candidates
                bc_candidates, bc_audit = generate_bc_candidates(
                    graph_json, plan, scene, max_candidates=6)
                bc_audit['search'] = 'one_step'
            log['bc_refine'] = bc_audit
        except Exception as exc:
            log['bc_refine'] = {'scene': scene, 'error': repr(exc),
                                'generated': 0, 'returned': 0}
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
            if solx is best_sol and refined_orders:
                mkx, adx, infox = model.evaluate(
                    solx.sg_of_block, solx.core_of_sg, scene, N,
                    use_cache=False, orders_override=refined_orders)
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
            plan_id = hashlib.sha256(sig.encode('utf-8')).hexdigest()
            log['official_candidates'].append({
                'plan_id': plan_id, 'proxy_makespan': mkx,
                'proxy_total_added_bytes': adx,
                'proxy_partition_added_bytes': infox.get('partition_added_bytes'),
                'proxy_spill_added_bytes': infox.get('spill_bytes'),
                'official': rx})
            if rx is None or rx.get('error'):
                continue
            if best_key is None or lexi_better(
                    rx['makespan'], rx['added_copy_bytes'],
                    best_key[0], best_key[1]):
                best_key = (rx['makespan'], rx['added_copy_bytes'])
                best_plan, real = planx, rx
                mk_final, ad_final, info = mkx, adx, infox
                log['selected_plan_id'] = plan_id
        # 新 B/C 候选与原流程共享同一官方确认门槛；任何异常均回退父方案。
        for bc_plan in bc_candidates:
            sig = json.dumps({'n': bc_plan['node_to_subgraph'],
                              'c': bc_plan['core_schedules']}, sort_keys=True)
            if sig in seen:
                continue
            seen.add(sig)
            metrics = next((row for row in (bc_audit or {}).get('candidate_metrics', [])
                            if row.get('plan_id') == hashlib.sha256(
                                json.dumps(bc_plan, sort_keys=True,
                                           separators=(',', ':')).encode('utf-8')).hexdigest()), {})
            rx = real_evaluate(graph_json, bc_plan, scene)
            plan_id = hashlib.sha256(sig.encode('utf-8')).hexdigest()
            block_proxy_plan = _project_plan_to_blocks(model, bc_plan)
            block_proxy_eligible = block_proxy_plan is not None
            log['official_candidates'].append({
                'plan_id': plan_id, 'source': 'bc_refine',
                'proxy_makespan': metrics.get('proxy_lower_bound'),
                'proxy_boundary_cycles': metrics.get('proxy_boundary_cycles'),
                'proxy_total_added_bytes': None,
                'proxy_lifetime_pressure': metrics.get('lifetime_pressure'),
                'proxy_cache_reuse_bytes': metrics.get('cache_reuse_bytes'),
                'proxy_cache_hit_potential_bytes': metrics.get('cache_hit_potential_bytes'),
                'proxy_cache_miss_bytes': metrics.get('cache_miss_bytes'),
                'proxy_l1_peak_bytes': metrics.get('l1_peak_bytes'),
                'proxy_ub_peak_bytes': metrics.get('ub_peak_bytes'),
                'block_proxy_eligible': block_proxy_eligible,
                'selection_reject_reason': None,
                'official': rx})
            if rx is None or rx.get('error'):
                continue
            # B/C 新模块使用严格官方字典序；A 的历史 0.3% 容差不外溢。
            bc_better = (best_key is None or
                         (rx['makespan'], rx['added_copy_bytes']) < best_key)
            if bc_better:
                best_key = (rx['makespan'], rx['added_copy_bytes'])
                best_plan, real = bc_plan, rx
                mk_final, ad_final, info = _proxy_for_plan(
                    model, bc_plan, scene, N, graph=graph_json)
                log['selected_plan_id'] = plan_id
                if bc_audit is not None:
                    bc_audit['selected_plan_id'] = plan_id
        plan = best_plan
        baseline = copy.deepcopy((plan, real, mk_final, ad_final, info,
                                  log.get('selected_plan_id')))
        accelerated = bool(fast_candidate_eval and scene == 'A' and
                           real is not None and not real.get('error'))
        base_candidate_count = len(log['official_candidates'])
        baseline_seconds = time.perf_counter()
        for candidate in log['official_candidates']:
            candidate['evaluation_source'] = 'original_official'
        candidate_evaluate = fast_evaluate if accelerated else real_evaluate
        if event_rerank and scene == 'A':
            from terminal_refine import recommend_challengers
            extra_started = time.perf_counter()
            sources = [('construct', row[1]) for row in scored]
            sources.extend(('archive', row[2]) for row in archive.items)
            challengers, audit = recommend_challengers(graph_json, ctx, sources, seen)
            audit['baseline_official'] = real
            audit['baseline_plan_id'] = log.get('selected_plan_id')
            for row in challengers:
                rx = candidate_evaluate(graph_json, row['plan'], scene)
                audit['official_count'] += 1
                log['official_candidates'].append({
                    'plan_id': row['plan_id'], 'source': 'event_challenger',
                    'proxy_makespan': row['mk'], 'proxy_total_added_bytes': row['ad'],
                    'precise_makespan': row['precise_makespan'],
                    'precise_added': row['precise_added'], 'official': rx})
                if rx is None or rx.get('error'):
                    continue
                # 挑战者必须严格不降低耗时；原流程的容差比较不向扩展传播。
                if real is None or (rx['makespan'], rx['added_copy_bytes']) < (
                        real['makespan'], real['added_copy_bytes']):
                    plan, real = row['plan'], rx
                    mk_final, ad_final, info = row['mk'], row['ad'], row['info']
                    log['selected_plan_id'] = row['plan_id']
            audit['final_official'] = real
            audit['selected_plan_id'] = log.get('selected_plan_id')
            audit['extra_seconds'] = time.perf_counter() - extra_started
            log['event_rerank'] = audit
        if task_order_refine and scene == 'A' and real is not None:
            from task_order_search import refine_plan_orders
            extra_started = time.perf_counter()
            try:
                proposed, proposed_real, audit = refine_plan_orders(
                    graph_json, plan, real, candidate_evaluate)
                metrics = _record_postprocess(model, proposed, audit, 'task_order_challenger', scene, N, log)
                plan, real = proposed, proposed_real
                mk_final, ad_final, info = metrics
                audit['extra_seconds'] = time.perf_counter() - extra_started
                log['task_order_refine'] = audit
            except Exception as exc:
                log['task_order_refine'] = {'error': repr(exc)}
        if partition_polish and scene == 'A' and real is not None:
            from partition_polish import polish_partition
            extra_started = time.perf_counter()
            try:
                proposed, proposed_real, audit = polish_partition(
                    graph_json, plan, real, candidate_evaluate, model=model)
                metrics = _record_postprocess(model, proposed, audit, 'partition_challenger', scene, N, log)
                plan, real = proposed, proposed_real
                mk_final, ad_final, info = metrics
                audit['extra_seconds'] = time.perf_counter() - extra_started
                log['partition_polish'] = audit
            except Exception as exc:
                log['partition_polish'] = {'error': repr(exc)}
        if construct_reservoir and event_rerank and scene == 'A' and real is not None:
            from terminal_refine import refine_reservoir_branch, plan_signature
            extra_started = time.perf_counter()
            try:
                excluded_ids = {c['plan_id'] for c in log['official_candidates']}
                proposed, proposed_real, audit = refine_reservoir_branch(
                    graph_json, ctx, plan, real, seen | {plan_signature(plan)},
                    excluded_ids, candidate_evaluate)
                audit['baseline_plan_id'] = log['selected_plan_id']
                metrics = _record_postprocess(model, proposed, audit, 'reservoir_challenger', scene, N, log)
                plan, real = proposed, proposed_real
                mk_final, ad_final, info = metrics
                audit['extra_seconds'] = time.perf_counter() - extra_started
                log['construct_reservoir'] = audit
            except Exception as exc:
                log['construct_reservoir'] = {'error': repr(exc)}
        source = 'counter_replica' if accelerated else 'original_official'
        for candidate in log['official_candidates'][base_candidate_count:]:
            candidate['evaluation_source'] = source
        for stage in ('event_rerank', 'task_order_refine', 'partition_polish', 'construct_reservoir'):
            if stage in log:
                log[stage]['candidate_evaluation_source'] = source
        if accelerated:
            audit = {'enabled': True, 'fallback_used': False, 'final_verified': False,
                     'baseline_plan_id': baseline[5], 'baseline_official': baseline[1],
                     'candidate_evaluation_source': source,
                     'final_validation_seconds': 0.0}
            if plan != baseline[0]:
                started = time.perf_counter()
                validated = real_evaluate(graph_json, plan, scene)
                audit['final_validation_seconds'] = time.perf_counter() - started
                audit['replica_final'] = real
                audit['original_final'] = validated
                audit['attempted_plan_id'] = log.get('selected_plan_id')
                fields = ('makespan', 'added_copy_bytes', 'scheduled_copy_bytes',
                          'partition_added', 'spill_added')
                matches = (validated is not None and not validated.get('error') and
                           all(k in validated and k in real and validated[k] == real[k]
                               for k in fields))
                audit['final_verified'] = bool(matches)
                log['official_candidates'].append({
                    'plan_id': log.get('selected_plan_id'), 'source': 'final_validation',
                    'evaluation_source': 'original_official', 'official': validated,
                    'proxy_makespan': mk_final, 'proxy_total_added_bytes': ad_final})
                if matches:
                    real = validated
                else:
                    plan, real, mk_final, ad_final, info, selected_id = baseline
                    log['selected_plan_id'] = selected_id
                    audit['fallback_used'] = True
                    audit['reason'] = '最终原官方结果与副本不一致或验证失败'
            else:
                real = baseline[1]
                audit['final_verified'] = True
                audit['validation_reused_baseline'] = True
            audit['total_postprocess_seconds'] = time.perf_counter() - baseline_seconds
            log['fast_candidate_eval'] = audit
        # E03/E04 的输入是完整旧流程最终方案，必须在上面的原官方守卫之后。
        if terminal_swaps or terminal_window:
            from run_all import valid_official
            if scene != 'A' or not valid_official(real):
                log['terminal_swaps' if terminal_swaps else 'terminal_window'] = {
                    'skipped': '只处理具有有效原官方保底的场景 A'}
            else:
                from task_order_search import refine_plan_orders
                extra_started = time.perf_counter()
                cache = None
                search_seconds = None if search_rounds is not None else 2.0
                try:
                    screen_evaluate = fast_evaluate
                    if terminal_cache:
                        screen_evaluate, cache = _cached_candidate_evaluator()
                    proposed, proposed_real, audit = refine_plan_orders(
                        graph_json, plan, real, real_evaluate,
                        enable_swaps=terminal_swaps,
                        enable_window=terminal_window,
                        search_seconds=search_seconds, max_evals=4000,
                        candidate_evaluator=screen_evaluate, rerank_keep=8)
                    # 指标与候选记录先写入临时容器，异常不能留下半次选中记录。
                    stage_log = {'official_candidates': []}
                    stage_name = ('terminal_swap_challenger'
                                  if terminal_swaps else 'terminal_window_challenger')
                    metrics = _record_postprocess(model, proposed, audit,
                        stage_name, scene, N, stage_log)
                    for row in stage_log['official_candidates']:
                        row['evaluation_source'] = 'original_official'
                    audit['baseline_plan_id'] = log.get('selected_plan_id')
                    audit['selected_plan_id'] = stage_log['selected_plan_id']
                    audit['candidate_evaluation_source'] = 'original_official'
                    audit['screening_source'] = ('cached_counter_replica' if terminal_cache
                                                 else 'counter_replica')
                    audit['search_seconds_limit'] = search_seconds
                    audit['cache_stats'] = dict(cache.stats) if cache is not None else None
                    audit['extra_seconds'] = time.perf_counter() - extra_started
                    plan, real = proposed, proposed_real
                    mk_final, ad_final, info = metrics
                    log['official_candidates'].extend(stage_log['official_candidates'])
                    log['selected_plan_id'] = stage_log['selected_plan_id']
                    log_key = 'terminal_swaps' if terminal_swaps else 'terminal_window'
                    log[log_key] = audit
                except Exception as exc:
                    log_key = 'terminal_swaps' if terminal_swaps else 'terminal_window'
                    log[log_key] = {'error': repr(exc), 'fallback_used': True,
                        'baseline_plan_id': log.get('selected_plan_id'),
                        'search_seconds_limit': search_seconds,
                        'cache_stats': dict(cache.stats) if cache is not None else None,
                        'extra_seconds': time.perf_counter() - extra_started}
    log['final_est'] = [mk_final, ad_final]
    log['spill_calibration_requested'] = bool(spill_calibrated)
    log['spill_calibrated'] = bool(model.spill_coefs is not None
                                  and model.spill_calibration_blend > 0
                                  and not use_lru_spill)
    log['spill_mode'] = ('lru' if use_lru_spill else
                         'calibrated_gated' if log['spill_calibrated'] else 'heuristic')
    log['spill_calibration_blend'] = (float(model.spill_calibration_blend)
                                     if log['spill_calibrated'] else 0.0)
    log['proxy_representation'] = info.get('proxy_representation', 'blocks')
    log['proxy_scope'] = info.get('proxy_scope', 'legacy_block_proxy')
    log['spill_prediction_available'] = info.get('spill_prediction_available', True)
    log['proxy_partition_raw_bytes'] = float(
        info.get('partition_raw_bytes', 0.0))
    log['proxy_partition_added_bytes'] = float(
        info.get('partition_added_bytes', 0.0))
    log['proxy_spill_added_bytes'] = (float(info.get('spill_bytes', 0.0))
                                    if log['spill_prediction_available'] else None)
    log['proxy_total_added_bytes'] = float(
        info.get('total_added_bytes', ad_final))
    log['proxy_memory_overage'] = {
        key: float(value) for key, value in
        (info.get('memory_overage') or {}).items()
        if key in ('L1p', 'L1w', 'UBp', 'UBw', 'L1l', 'UBl')
    }
    log['proxy_peak_memory_bytes'] = {
        key: float(value) for key, value in
        (info.get('peak_memory_bytes') or {}).items()
        if key in ('L1', 'UB')
    }
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
