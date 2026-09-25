"""多目标粒子群（MOPSO）：随机键编码 + Pareto 归档 + 拥挤距离领袖选择。

粒子维度：每块两个连续基因 —— cut[b]（切分得分）与 pref[b]（核偏好），
外加一个全局基因 g 控制目标子图数。解码：
  1. 按拓扑序取 cut 得分最大的 K-1 个位置作为切点 -> K 个凸子图；
  2. 子图沿拓扑序逐个分核：最早完成时间 + 核偏好票数偏置。
领袖在基因空间维护局部 Pareto 前沿，用拥挤距离轮盘赌选取。
"""
import random
import time

from solution import Sol


def _dominates(a, b):
    return (a[0] <= b[0] and a[1] <= b[1]) and (a[0] < b[0] or a[1] < b[1])


def decode(ctx, cut, pref, g, kmin, kmax):
    nb, N = ctx.nb, ctx.num_cores
    topo_order = list(ctx.model.block_topo)
    K = max(N, min(kmax, kmin + int(round(g * (kmax - kmin)))))
    if nb <= K:
        strips = [[b] for b in topo_order]
    else:
        positions = sorted(range(1, nb),
                           key=lambda i: -cut[topo_order[i]])[:K - 1]
        cuts = set(positions)
        strips = []
        cur = []
        for i, b in enumerate(topo_order):
            cur.append(b)
            if i + 1 in cuts:
                strips.append(cur)
                cur = []
        if cur:
            strips.append(cur)
    sdur = [sum(ctx.bdur[b] for b in st) for st in strips]
    sg_of_block = [0] * nb
    core_of_sg = []
    core_free = [0.0] * N
    sg = -1
    for si, st in enumerate(strips):
        votes = [0] * N
        for b in st:
            votes[min(N - 1, int(pref[b] * N))] += 1
        best_c, best_cost = None, None
        for c in range(N):
            fin = core_free[c] + sdur[si]
            fin -= 0.15 * sdur[si] * votes[c] / max(1, len(st))
            if best_cost is None or fin < best_cost:
                best_cost, best_c = fin, c
        sg += 1
        core_of_sg.append(best_c)
        for b in st:
            sg_of_block[b] = sg
        core_free[best_c] = max(best_cost, core_free[best_c] + sdur[si])
    return Sol(sg_of_block, core_of_sg)


def _crowd_weights(front):
    """前沿解的拥挤距离权重（边界解权重无穷大）。"""
    n = len(front)
    weights = []
    mks = sorted(x[0] for x in front)
    ads = sorted(x[1] for x in front)
    for (mk, ad, _) in front:
        def crow(vals, v):
            if len(vals) < 2 or vals[-1] - vals[0] < 1e-12:
                return 1.0
            j = vals.index(v)
            if j in (0, len(vals) - 1):
                return 1e6
            return (vals[j + 1] - vals[j - 1]) / (vals[-1] - vals[0])
        weights.append(crow(mks, mk) + crow(ads, ad))
    return weights


def _update_local_front(front, item, cap=24):
    for f in front:
        if _dominates(f, item):
            return front
    front = [f for f in front if not _dominates(item, f)]
    front.append(item)
    if len(front) > cap:
        w = _crowd_weights(front)
        order = sorted(range(len(front)), key=lambda i: -w[i])
        front = [front[i] for i in order[:cap]]
    return front


def mopso_search(ctx, time_budget, n_particles=12, kmin=None, kmax=None,
                 w=0.5, c1=1.2, c2=1.2, archive=None, deadline=None, rng=None, max_rounds=None):
    """返回 (best_sol, best_fitness, best_mk, best_added)，并向 archive 供解。"""
    rng = rng or random.Random()
    t_end = deadline if deadline is not None else time.time() + time_budget
    nb, N = ctx.nb, ctx.num_cores
    kmin = kmin if kmin is not None else max(N, 2)
    kmax = kmax if kmax is not None else min(nb, max(4 * N, 16))
    dims = 2 * nb + 1
    X = [[rng.random() for _ in range(dims)] for _ in range(n_particles)]
    V = [[0.0] * dims for _ in range(n_particles)]
    pbest = [None] * n_particles      # (mk, added, x)
    front = []                        # 基因空间局部 Pareto 前沿

    def scalar(mk, ad):
        return mk + 0.2 * ad / 60.0

    rounds = 0
    while (rounds < max_rounds if max_rounds is not None else time.time() < t_end):
        rounds += 1
        leaders = None
        for i in range(n_particles):
            x = X[i]
            sol = decode(ctx, x[:nb], x[nb:2 * nb], x[2 * nb], kmin, kmax)
            mk, ad, _ = ctx.evaluate(sol)
            if archive is not None:
                archive.insert(mk, ad, sol)
            front = _update_local_front(front, (mk, ad, list(x)))
            if pbest[i] is None or scalar(mk, ad) < scalar(pbest[i][0],
                                                           pbest[i][1]):
                pbest[i] = (mk, ad, list(x))
        if front:
            weights = _crowd_weights(front)
            total = sum(weights)
            leaders = []
            for _ in range(n_particles):
                r = rng.random() * total
                acc = 0.0
                pick = front[-1]
                for j, f in enumerate(front):
                    acc += weights[j]
                    if acc >= r:
                        pick = f
                        break
                leaders.append(pick[2])
        for i in range(n_particles):
            lb = leaders[i] if leaders else X[i]
            for d in range(dims):
                V[i][d] = (w * V[i][d]
                           + c1 * rng.random() * (pbest[i][2][d] - X[i][d])
                           + c2 * rng.random() * (lb[d] - X[i][d]))
                V[i][d] = max(-0.25, min(0.25, V[i][d]))
                X[i][d] = max(0.0, min(1.0, X[i][d] + V[i][d]))
    if front:
        mk, ad, x = min(front, key=lambda f: scalar(f[0], f[1]))
        sol = decode(ctx, x[:nb], x[nb:2 * nb], x[2 * nb], kmin, kmax)
        return sol, scalar(mk, ad), mk, ad
    return None, None, None, None
