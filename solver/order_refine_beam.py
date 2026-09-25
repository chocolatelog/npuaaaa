"""融合 W 分支的束搜索顺序精化：task_order_search.search_orders +
本地代理时长（融合层）。

与 W 原版差异：时长来源不是 SceneAEventModel（需官方局部图重建，
大图慢），而是我们的块级代理时长映射到子图粒度；束搜索算法本体
（beam+关键源插入+跨核交换）原样保留。
"""
import time
from model import BW
from task_order_search import search_orders


def refine_orders_beam(ctx, sol, orders, rounds=2, beam_width=4,
                       max_evals=300, deadline=None):
    """返回 (best_orders, base_mk, refined_mk)。

    时长模型：子图时长 = max(wm, wv, in/60, out/60, cp)（与代理一致），
    依赖 = 子图商图；replay_tasks 处理同核/跨核等待与合法性。
    """
    model = ctx.model
    K = len(sol.core_of_sg)
    # 子图时长（与 model.evaluate 的 durs 同构）
    sg_wm = [0.0] * K
    sg_wv = [0.0] * K
    sg_in = [0.0] * K
    sg_out = [0.0] * K
    for b in range(len(model.blocks)):
        s = sol.sg_of_block[b]
        sg_wm[s] += model.block_work_m[b]
        sg_wv[s] += model.block_work_v[b]
    for pb, cb, has_out, tpos, size in model.tg:
        for b in cb:
            sg_in[sol.sg_of_block[b]] += size
        for b in pb:
            sg_out[sol.sg_of_block[b]] += size
    # 子图内部 CP（简化：块级 CP 取最大）
    sg_cp = [0.0] * K
    blk_pred = {}
    for (bi, bj), _w in model.block_edges:
        if sol.sg_of_block[bi] == sol.sg_of_block[bj]:
            blk_pred.setdefault(bj, []).append(bi)
    for bi in model.block_topo:
        s = sol.sg_of_block[bi]
        v = model.block_cp_weight[bi] + max(
            (model.block_cp_weight[p] for p in blk_pred.get(bi, ())),
            default=0.0)
        if v > sg_cp[s]:
            sg_cp[s] = v
    durations = {}
    for s in range(K):
        durations[s] = max(sg_wm[s], sg_wv[s], sg_in[s] / BW,
                           sg_out[s] / BW, sg_cp[s])
    # 商图依赖
    preds = {}
    for (bi, bj), _w in model.block_edges:
        si, sj = sol.sg_of_block[bi], sol.sg_of_block[bj]
        if si != sj:
            preds.setdefault(sj, set()).add(si)
    for s in range(K):
        preds.setdefault(s, set())
    # 原始字节下界（带宽地板）
    floor = model.original_copy_bytes / BW

    try:
        ranked, stats = search_orders(
            durations, preds, [list(o) for o in orders],
            same_wait=100, cross_wait=1000,
            bandwidth_floor=floor, beam_width=beam_width, rounds=rounds,
            max_evals=max_evals, seconds=None, max_sources=12, keep=2,
            enable_swaps=True)
    except Exception:
        return [list(o) for o in orders], None, None
    if not ranked:
        return [list(o) for o in orders], None, None
    best_orders = ranked[0]['orders']
    # 用我们的代理完整评估（含带宽下界与 spill）核对改进
    mk_new = model.evaluate(sol.sg_of_block, sol.core_of_sg, ctx.scene,
                            ctx.num_cores, use_cache=False,
                            orders_override=best_orders)[0]
    mk_base = model.evaluate(sol.sg_of_block, sol.core_of_sg, ctx.scene,
                             ctx.num_cores, use_cache=False,
                             orders_override=[list(o) for o in orders])[0]
    return best_orders, mk_base, mk_new
