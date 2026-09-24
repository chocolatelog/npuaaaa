"""Tessel 式核内顺序精修：固定切图与分核，交换每核顺序中相邻且
相互独立（无直接依赖边）的子图对，代理评估择优，多轮爬山。"""
import time


def refine_orders(ctx, sol, orders, rounds=3, deadline=None):
    """返回 (best_orders, base_mk, refined_mk)。

    合法性：官方校验只检查同核子图的**直接**依赖序（contracted DAG），
    相邻对无直接边则交换合法；跨核传递路径不影响同核顺序合法性。
    """
    model = ctx.model
    K = len(sol.core_of_sg)
    sg_succs = [set() for _ in range(K)]
    for (bi, bj), _w in model.block_edges:
        si, sj = sol.sg_of_block[bi], sol.sg_of_block[bj]
        if si != sj:
            sg_succs[si].add(sj)

    def swappable(a, b):
        return a not in sg_succs[b] and b not in sg_succs[a]

    best_orders = [list(o) for o in orders]
    t0 = time.time()
    base = model.evaluate(sol.sg_of_block, sol.core_of_sg, ctx.scene,
                          ctx.num_cores, use_cache=False,
                          orders_override=best_orders)[0]
    mk_best = base
    for rnd in range(rounds):
        improved = False
        for c in range(ctx.num_cores):
            order = best_orders[c]
            i = 0
            while i < len(order) - 1:
                if deadline is not None and time.time() > deadline:
                    return best_orders, base, mk_best
                a, b = order[i], order[i + 1]
                if not swappable(a, b):
                    i += 1
                    continue
                cand = [list(o) for o in best_orders]
                cand[c][i], cand[c][i + 1] = b, a
                mk = model.evaluate(sol.sg_of_block, sol.core_of_sg,
                                    ctx.scene, ctx.num_cores,
                                    use_cache=False,
                                    orders_override=cand)[0]
                if mk < mk_best - 1e-9:
                    mk_best = mk
                    best_orders = cand
                    improved = True
                    continue        # 新相邻对 (b, next) 可能仍可换
                i += 1
        if not improved:
            break
    return best_orders, base, mk_best
