"""蚁群算法：信息素挂在 (块, 归属决策) 上，蚂蚁沿拓扑序构造切分+分核方案。

决策空间：对每个块，选择并入最近创建的 m 个簇之一（0 表示新建簇）；
新建簇时再选核。信息素按精英蚂蚁沉积，启发式因子由通信亲和度与
负载均衡决定。
"""
import math
import random
import time
from collections import defaultdict

from model import BW
from solution import Sol


def aco_search(ctx, time_budget, n_ants=6, m_open=6, rho=0.15,
               alpha=1.0, beta=2.0, archive=None, deadline=None,
               polish_moves=30, rng=None):
    """返回 (best_sol, best_fitness, best_mk, best_added)。"""
    rng = rng or random.Random()
    t_end = deadline if deadline is not None else time.time() + time_budget
    nb, N = ctx.nb, ctx.num_cores
    scene = ctx.scene
    # 信息素：tau_attach[b][0..m_open]（0=新建），tau_core[b][core]
    tau_attach = [[1.0] * (m_open + 1) for _ in range(nb)]
    tau_core = [[1.0] * N for _ in range(nb)]
    best, f_best, best_mk, best_ad = None, None, None, None
    topo_order = list(ctx.model.block_topo)
    while time.time() < t_end:
        iter_solutions = []
        for _ in range(n_ants):
            sol = _construct(ctx, topo_order, tau_attach, tau_core,
                             m_open, alpha, beta, rng)
            if sol is None:
                continue
            f, mk, ad = ctx.eval_fitness(sol)
            iter_solutions.append((f, mk, ad, sol, sol.decisions))
            if archive is not None:
                archive.insert(mk, ad, sol)
            if f_best is None or f < f_best:
                best, f_best, best_mk, best_ad = sol.clone(), f, mk, ad
        if not iter_solutions:
            break
        # 信息素更新：蒸发 + 精英沉积
        for b in range(nb):
            for d in range(m_open + 1):
                tau_attach[b][d] *= (1 - rho)
            for c in range(N):
                tau_core[b][c] *= (1 - rho)
        iter_solutions.sort(key=lambda x: x[0])
        for rank_i, (f, mk, ad, sol, decisions) in enumerate(
                iter_solutions[:max(1, n_ants // 3)]):
            w = 1.0 / (1 + rank_i)
            for b, d, c in decisions:
                tau_attach[b][d] += w
                tau_core[b][c] += w
        for b in range(nb):
            for d in range(m_open + 1):
                tau_attach[b][d] = max(0.05, tau_attach[b][d])
            for c in range(N):
                tau_core[b][c] = max(0.05, tau_core[b][c])
    return best, f_best, best_mk, best_ad


def _construct(ctx, topo_order, tau_attach, tau_core, m_open, alpha, beta, rng):
    """一只蚂蚁：沿拓扑序决定块的簇归属与核。"""
    nb, N, scene = ctx.nb, ctx.num_cores, ctx.scene
    sg_of_block = [-1] * nb
    core_of_sg = []
    blocks_in_sg = []
    open_clusters = []           # 最近创建的簇 id（新的在后）
    cluster_bytes_in = defaultdict(float)
    cluster_work = []
    core_load = [0.0] * N
    decisions = []
    for b in topo_order:
        # 候选：0=新建；1..m_open=并入对应开簇
        cands = [0]
        for d, s in enumerate(reversed(open_clusters), start=1):
            if d > m_open:
                break
            cands.append((d, s))
        scores = []
        for cand in cands:
            if cand == 0:
                # 启发式：新建簇的收益（并行度 + 均衡）
                est_work = ctx.bdur[b]
                h = 1.0 / (1.0 + min(core_load) / max(1.0, sum(core_load) / N))
                tau = tau_attach[b][0]
            else:
                d, s = cand
                # 与簇内块的通信亲和度
                aff = 0.0
                for v, w in ctx.block_core_affinity.get(b, {}).items():
                    if sg_of_block[v] == s:
                        aff += w
                size_out = max(1.0, ctx._block_out_traffic(b))
                h = (1.0 + aff / size_out)
                tau = tau_attach[b][d]
            scores.append((cand, (tau ** alpha) * (h ** beta)))
        total = sum(x[1] for x in scores)
        r = rng.random() * total
        acc = 0.0
        chosen = scores[-1][0]
        for cand, sc in scores:
            acc += sc
            if acc >= r:
                chosen = cand
                break
        if chosen == 0:
            # 新建簇：选核（信息素 + 负载均衡 + 与前驱局部性）
            cs = []
            for c in range(N):
                locality = 0.0
                for p in ctx.bpreds.get(b, ()):
                    if sg_of_block[p] >= 0 and core_of_sg[sg_of_block[p]] == c:
                        locality += ctx.btraffic.get((p, b), 0.0)
                h = 1.0 + locality / max(1.0, ctx._block_out_traffic(b))
                cs.append((c, (tau_core[b][c] ** alpha) * (h ** beta)))
            tot = sum(x[1] for x in cs)
            r = rng.random() * tot
            acc = 0.0
            core = cs[-1][0]
            for c, sc in cs:
                acc += sc
                if acc >= r:
                    core = c
                    break
            s = len(core_of_sg)
            core_of_sg.append(core)
            blocks_in_sg.append({b})
            cluster_work.append(ctx.bdur[b])
            core_load[core] += ctx.bdur[b]
            open_clusters.append(s)
            if len(open_clusters) > m_open * 2:
                open_clusters.pop(0)
            decisions.append((b, 0, core))
        else:
            d, s = chosen
            blocks_in_sg[s].add(b)
            cluster_work[s] += ctx.bdur[b]
            core_of_sg_stay = core_of_sg[s]
            core_load[core_of_sg_stay] += ctx.bdur[b]
            decisions.append((b, d, core_of_sg_stay))
        sg_of_block[b] = (len(core_of_sg) - 1) if chosen == 0 else chosen[1]
    if any(s < 0 for s in sg_of_block):
        return None
    sol = Sol(sg_of_block, core_of_sg, blocks_in_sg)
    sol.decisions = decisions
    return sol
