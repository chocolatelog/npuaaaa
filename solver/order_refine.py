"""Tessel 式核内顺序精修：固定切图与分核，交换每核顺序中相邻且
相互独立（无直接依赖边）的子图对，代理评估择优，多轮爬山。"""
import time
import math

from model import WAIT_CROSS, WAIT_SAME, DELAY_B


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


def _critical_chain(info, core_of_sg, scene):
    """Trace binding dependency/core-order edges in a proxy schedule."""
    orders = info['orders']
    end = info['end_time']
    durations = info['durs']
    previous = {}
    for row in orders:
        for i, sg in enumerate(row):
            previous[sg] = row[i - 1] if i else None
    parent = {}
    for sg in range(len(end)):
        start = end[sg] - durations[sg]
        causes = []
        prev = previous[sg]
        if prev is not None:
            causes.append((end[prev] +
                           (WAIT_SAME if scene == 'A' else 0), prev))
        for pred in info['sg_preds'][sg]:
            wait = (0 if core_of_sg[pred] == core_of_sg[sg] else
                    (WAIT_CROSS if scene == 'A' else DELAY_B))
            causes.append((end[pred] + wait, pred))
        if causes:
            bound, pred = max(causes)
            if bound >= start - 1e-6:
                parent[sg] = pred
    if not end:
        return []
    chain = []
    sg = max(range(len(end)), key=lambda s: end[s])
    while sg not in chain:
        chain.append(sg)
        if sg not in parent:
            break
        sg = parent[sg]
    return chain


def _solve_window(ctx, sol, orders, core, left, right, seconds):
    """Optimize one contiguous order window with fixed subgraph assignment."""
    from ortools.sat.python import cp_model

    model = ctx.model
    _base, _added, info = model.evaluate(
        sol.sg_of_block, sol.core_of_sg, ctx.scene, ctx.num_cores,
        use_cache=False, orders_override=orders, diagnostics=True)
    k = len(info['durs'])
    durations = [max(1, int(math.ceil(x))) for x in info['durs']]
    horizon = sum(durations) + (WAIT_CROSS + WAIT_SAME) * k + 1
    cp = cp_model.CpModel()
    starts = [cp.new_int_var(0, horizon, f'start_{s}') for s in range(k)]
    ends = [cp.new_int_var(0, horizon, f'end_{s}') for s in range(k)]
    for s in range(k):
        cp.add(ends[s] == starts[s] + durations[s])
        cp.add_hint(starts[s], max(0, int(info['end_time'][s] -
                                          info['durs'][s])))
    for child, parents in enumerate(info['sg_preds']):
        for pred in parents:
            wait = (0 if sol.core_of_sg[pred] == sol.core_of_sg[child]
                    else (WAIT_CROSS if ctx.scene == 'A' else DELAY_B))
            cp.add(starts[child] >= ends[pred] + wait)

    movable = set(orders[core][left:right])
    same_gap = WAIT_SAME if ctx.scene == 'A' else 0
    for row in orders:
        for i, a in enumerate(row):
            for b in row[i + 1:]:
                if a in movable and b in movable:
                    before = cp.new_bool_var(f'{a}_before_{b}')
                    cp.add(starts[b] >= ends[a] + same_gap).only_enforce_if(
                        before)
                    cp.add(starts[a] >= ends[b] + same_gap).only_enforce_if(
                        before.Not())
                else:
                    cp.add(starts[b] >= ends[a] + same_gap)
    finish = cp.new_int_var(0, horizon, 'finish')
    cp.add_max_equality(finish, ends)
    cp.minimize(finish)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = max(0.05, seconds)
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = 1
    status = solver.solve(cp)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return orders, 'no_solution'
    candidate = [list(row) for row in orders]
    candidate[core][left:right] = sorted(
        candidate[core][left:right], key=lambda s: (solver.value(starts[s]), s))
    return candidate, ('optimal' if status == cp_model.OPTIMAL else 'feasible')


def refine_orders_exact(ctx, sol, orders, window=8, max_windows=3,
                        seconds_per_window=0.35):
    """Use CP-SAT on critical-path windows; accept only proxy improvements.

    The partition and core assignment stay fixed. Without OR-Tools this is a
    no-op, so the existing adjacent-swap refinement remains available.
    """
    try:
        from ortools.sat.python import cp_model  # noqa: F401
    except ImportError:
        return orders, None, None, {'status': 'unavailable', 'windows': 0}
    model = ctx.model
    best_orders = [list(row) for row in orders]
    base, _added, info = model.evaluate(
        sol.sg_of_block, sol.core_of_sg, ctx.scene, ctx.num_cores,
        use_cache=False, orders_override=best_orders, diagnostics=True)
    if not math.isfinite(base):
        return best_orders, base, base, {'status': 'invalid', 'windows': 0}
    best = base
    chosen = set()
    solved = 0
    improved = 0
    for sg in _critical_chain(info, sol.core_of_sg, ctx.scene):
        core = sol.core_of_sg[sg]
        row = best_orders[core]
        if len(row) < 3:
            continue
        pos = row.index(sg)
        left = max(0, min(pos - window // 2, len(row) - window))
        right = min(len(row), left + window)
        key = core, left, right
        if key in chosen:
            continue
        chosen.add(key)
        candidate, _status = _solve_window(
            ctx, sol, best_orders, core, left, right, seconds_per_window)
        solved += 1
        mk = model.evaluate(sol.sg_of_block, sol.core_of_sg, ctx.scene,
                            ctx.num_cores, use_cache=False,
                            orders_override=candidate)[0]
        if mk < best - 1e-6:
            best_orders, best = candidate, mk
            improved += 1
        if solved >= max_windows:
            break
    return best_orders, base, best, {'status': 'ok', 'windows': solved,
                                     'improved_windows': improved}
