"""场景 B/A 通用构造器 v2：NetBenefit 凝聚切分 + 容量装箱 EFT 分核。

流程（综述 LATTICE/HAMPS 路线的多核切图版）：
1. 凝聚：每块初始一簇，反复合并边际收益最高的拓扑相邻簇对；
   收益 = 省边界搬运/60 + 消除跨核等待 + M/V 互补 − 并行度损失；
   簇保持拓扑凸（首尾相接合并），块数上限控制粒度。
2. SCC 修复：簇商图成环则合并同 SCC 簇（保证方案合法的关键步骤）。
3. 分核：EFT 列表调度（bottom-level 优先）+ 每核 L1/UB 驻留预算
   + M/V 互补打分（核内双管道平衡）。
"""
import heapq
from collections import defaultdict

from model import L1_CAP, UB_CAP, BW, _tarjan_scc
from solution import Sol

MEM = (L1_CAP, UB_CAP)


def _block_pool_traffic(model):
    """块对的边界搬运字节：tg 组字节均分到组内跨块 (生产,消费) 有向对，
    保证同组总字节守恒。"""
    traffic = defaultdict(float)
    for pb, cb, _ho, _pos, size in model.tg:
        cross = [(b, c) for b in pb for c in cb if b != c]
        if not cross:
            continue
        share = size / len(cross)
        for b, c in cross:
            traffic[(b, c)] += share
    return traffic


def _mismatch(work_m, work_v):
    tot = work_m + work_v
    if tot <= 0:
        return 0.0
    return abs(work_m - work_v) / tot


def netbenefit_construct(model, num_cores, scene, max_sg_ops=240,
                         mem_budget=0.85):
    """返回 (sg_of_block, core_of_sg)。scene: 'A'|'B'（等待成本不同）。"""
    nb = len(model.blocks)
    traffic = _block_pool_traffic(model)
    bpos = model.block_pos
    total_work = sum(model.block_work_m[b] + model.block_work_v[b]
                     for b in range(nb))
    share = total_work / max(1, num_cores)
    wait = 1000.0 if scene == 'A' else 500.0
    work_m = model.block_work_m
    work_v = model.block_work_v

    # ---- 阶段1：凝聚（簇 = (frozenset 成员, 首/尾块号)） ----
    cl_members = [frozenset([b]) for b in range(nb)]
    cl_head = list(range(nb))
    cl_tail = list(range(nb))
    cl_alive = set(range(nb))
    cid_of = list(range(nb))

    def stats(cid):
        wm = sum(work_m[b] for b in cl_members[cid])
        wv = sum(work_v[b] for b in cl_members[cid])
        return wm, wv, len(cl_members[cid])

    def ext_traffic(a, b):
        return sum(traffic.get((u, v), 0.0) + traffic.get((v, u), 0.0)
                   for u in cl_members[a] for v in cl_members[b])

    def has_edge(a, b):
        for u in cl_members[a]:
            for v in cl_members[b]:
                if traffic.get((u, v), 0) > 0 or traffic.get((v, u), 0) > 0:
                    return True
        return False

    def neighbors(cid):
        res = set()
        for op in model.blocks[cl_tail[cid]]:
            for v in model.eligible_succs.get(op, ()):
                cj = cid_of[model.block_of_op[v]]
                if cj != cid and cj in cl_alive:
                    res.add(cj)
        for op in model.blocks[cl_head[cid]]:
            for v in model.eligible_preds.get(op, ()):
                cj = cid_of[model.block_of_op[v]]
                if cj != cid and cj in cl_alive:
                    res.add(cj)
        return res

    def merge_gain(a, b):
        wma, wva, sza = stats(a)
        wmb, wvb, szb = stats(b)
        # max_sg_ops 的单位是算子数；不能用 block 数与 HEFT/CHAIN 混用。
        ops_a = sum(len(model.blocks[x]) for x in cl_members[a])
        ops_b = sum(len(model.blocks[x]) for x in cl_members[b])
        if ops_a + ops_b > max_sg_ops:
            return None
        saved = ext_traffic(a, b) / BW
        wg = wait if has_edge(a, b) else 0.0
        mis_a = _mismatch(wma, wva)
        mis_b = _mismatch(wmb, wvb)
        mis_ab = _mismatch(wma + wmb, wva + wvb)
        tot = max(1.0, wma + wva + wmb + wvb)
        mv = (mis_a * (wma + wva) + mis_b * (wmb + wvb)
              - mis_ab * tot) / tot * 100.0
        wa, wb = wma + wva, wmb + wvb
        par = max(0.0, min(wa, share) + min(wb, share)
                  - min(wa + wb, share)) * 0.5
        val = saved + wg + mv - par
        return val if val > 0 else None

    heap = []
    for cid in list(cl_alive):
        for cj in neighbors(cid):
            g = merge_gain(cid, cj)
            if g is not None:
                heapq.heappush(heap, (-g, cid, cj))
    while heap:
        _neg, a, b = heapq.heappop(heap)
        if a not in cl_alive or b not in cl_alive:
            continue
        if cid_of[a] != a or cid_of[b] != b:
            continue
        if bpos[cl_head[a]] > bpos[cl_head[b]]:
            a, b = b, a
        cl_members[a] = cl_members[a] | cl_members[b]
        # 按 bpos 判断 a/b 前后：a 在前则吸收 b 的尾，反之吸收 b 的头
        if bpos[cl_tail[a]] < bpos[cl_tail[b]]:
            cl_tail[a] = cl_tail[b]
        else:
            cl_head[a] = cl_head[b]
        for blk in cl_members[b]:
            cid_of[blk] = a
        cl_alive.discard(b)
        for cj in neighbors(a):
            g = merge_gain(a, cj)
            if g is not None:
                heapq.heappush(heap, (-g, a, cj))

    # ---- 阶段2：SCC 修复（簇商图成环则合并） ----
    def build_cluster_graph():
        cmap = {}
        clusters = sorted(cl_alive, key=lambda c: bpos[cl_head[c]])
        for ci, c in enumerate(clusters):
            for blk in cl_members[c]:
                cmap[blk] = ci
        succs = [set() for _ in clusters]
        for (bi, bj), _w in model.block_edges:
            ci, cj = cmap[bi], cmap[bj]
            if ci != cj:
                succs[ci].add(cj)
        return clusters, succs

    for _ in range(8):
        clusters, succs = build_cluster_graph()
        sccs = _tarjan_scc(succs)
        if all(len(s) == 1 for s in sccs):
            break
        # sccs 里的 ci 是 clusters 列表的下标；合并对应成员集合
        new_members = []
        for scc in sccs:
            blk = set()
            for ci in scc:
                blk |= cl_members[clusters[ci]]
            new_members.append(frozenset(blk))
        cl_members = new_members
        cl_alive = set(range(len(new_members)))
        cl_head = [min(bpos[b] for b in blk) for blk in cl_members]
        cl_tail = [max(bpos[b] for b in blk) for blk in cl_members]
        pos_to_blk = {bpos[b]: b for b in range(nb)}
        cl_head = [pos_to_blk[p] for p in cl_head]
        cl_tail = [pos_to_blk[p] for p in cl_tail]
        for cid, blk in enumerate(cl_members):
            for b in blk:
                cid_of[b] = cid
    clusters, cl_succs = build_cluster_graph()
    K = len(clusters)

    # ---- 阶段3：EFT 分核（容量预算 + M/V 互补 + bottom-level 优先） ----
    cmap = {}
    for ci, c in enumerate(clusters):
        for blk in cl_members[c]:
            cmap[blk] = ci
    internal = [[0.0, 0.0] for _ in range(K)]
    pool_idx = {'L1': 0, 'UB': 1}
    for pb, cb, _ho, tpos, size in model.tg:
        cs_p = {cmap[b] for b in pb}
        cs_c = {cmap[b] for b in cb}
        if len(cs_p) == 1 and cs_p == cs_c:
            internal[cs_p.pop()][pool_idx.get(tpos, 0)] += size
    cl_preds = defaultdict(set)
    for (bi, bj), _w in model.block_edges:
        ci, cj = cmap[bi], cmap[bj]
        if ci != cj:
            cl_preds[cj].add(ci)
    cl_in_traffic = defaultdict(float)
    for pb, cb, _ho, _pos, size in model.tg:
        cs_p = {cmap[b] for b in pb}
        cs_c = {cmap[b] for b in cb}
        for cs in cs_c - cs_p:
            cl_in_traffic[cs] += size
    cl_work = [sum(work_m[b] + work_v[b] for b in cl_members[ci])
               for ci in range(K)]
    cl_m = [sum(work_m[b] for b in cl_members[ci]) for ci in range(K)]
    cl_v = [sum(work_v[b] for b in cl_members[ci]) for ci in range(K)]
    # bottom-level（K 很小，直接 relax）
    bl = [0.0] * K
    for ci in reversed(range(K)):
        best = 0.0
        for cj in range(K):
            if ci in cl_preds[cj] and bl[cj] > best:
                best = bl[cj]
        bl[ci] = cl_work[ci] + best
    indeg = [len(cl_preds.get(ci, ())) for ci in range(K)]
    ready = [(-bl[ci], ci) for ci in range(K) if indeg[ci] == 0]
    heapq.heapify(ready)
    sg_of_block = [0] * nb
    core_of_sg = []
    core_live = [[0.0, 0.0] for _ in range(num_cores)]
    core_free = [0.0] * num_cores
    core_m = [0.0] * num_cores
    core_v = [0.0] * num_cores
    while ready:
        _, ci = heapq.heappop(ready)
        best_c, best_key = None, None
        for c in range(num_cores):
            over = sum(max(0.0, core_live[c][pi] + internal[ci][pi]
                           - mem_budget * MEM[pi]) for pi in (0, 1))
            nm = core_m[c] + cl_m[ci]
            nv = core_v[c] + cl_v[ci]
            imbalance = abs(nm - nv) / max(1.0, nm + nv)
            est = core_free[c] + cl_in_traffic.get(ci, 0.0) / BW
            key = (over > 0,
                   over / BW + imbalance * max(1.0, cl_work[ci]) * 0.3,
                   est)
            if best_key is None or key < best_key:
                best_key, best_c = key, c
        sg = len(core_of_sg)
        core_of_sg.append(best_c)
        for pi in (0, 1):
            core_live[best_c][pi] += internal[ci][pi]
        core_free[best_c] += cl_work[ci] + cl_in_traffic.get(ci, 0.0) / BW
        core_m[best_c] += cl_m[ci]
        core_v[best_c] += cl_v[ci]
        for blk in cl_members[ci]:
            sg_of_block[blk] = sg
        for cj in range(K):
            if ci in cl_preds[cj]:
                indeg[cj] -= 1
                if indeg[cj] == 0:
                    heapq.heappush(ready, (-bl[cj], cj))
    return sg_of_block, core_of_sg


def eft_place_core(model, sg_of_block, num_cores, scene,
                   mem_budget=0.85, mv_weight=0.3):
    """EFT 容量装箱分核：对任意凝聚结果 (sg_of_block) 重新分核。

    - 每核 L1/UB 驻留预算（超预算重罚，控制场景B spill）；
    - M/V 互补打分（核内双管道平衡，场景B 合并 Task 的吞吐关键）；
    - 输入通信量计入 EFT。
    返回 (core_of_sg, sg_of_block)（sg 已做环修复与重编号）。
    """
    BW_ = BW
    # 先做商图环修复（heft 等凝聚的商图可能有环）
    sg_of_block = list(sg_of_block)
    blocks_in_sg = [set() for _ in range(max(sg_of_block) + 1)]
    for b, s in enumerate(sg_of_block):
        blocks_in_sg[s].add(b)
    succs = [set() for _ in blocks_in_sg]
    for (bi, bj), _w in model.block_edges:
        si, sj = sg_of_block[bi], sg_of_block[bj]
        if si != sj:
            succs[si].add(sj)
    for scc in _tarjan_scc(succs):
        if len(scc) < 2:
            continue
        tgt = min(scc)
        for s in scc:
            if s == tgt:
                continue
            blocks_in_sg[tgt] |= blocks_in_sg[s]
            for b in blocks_in_sg[s]:
                sg_of_block[b] = tgt
            blocks_in_sg[s] = set()
    used = sorted(i for i, s in enumerate(blocks_in_sg) if s)
    remap = {s: i for i, s in enumerate(used)}
    sg_of_block = [remap[s] for s in sg_of_block]

    K = max(sg_of_block) + 1
    members = [set() for _ in range(K)]
    for b, s in enumerate(sg_of_block):
        members[s].add(b)
    cmap = {b: s for b, s in enumerate(sg_of_block)}
    internal = [[0.0, 0.0] for _ in range(K)]
    pool_idx = {'L1': 0, 'UB': 1}
    for pb, cb, _ho, tpos, size in model.tg:
        cs_p = {cmap[b] for b in pb}
        cs_c = {cmap[b] for b in cb}
        if len(cs_p) == 1 and cs_p == cs_c:
            internal[cs_p.pop()][pool_idx.get(tpos, 0)] += size
    cl_preds = defaultdict(set)
    for (bi, bj), _w in model.block_edges:
        ci, cj = cmap[bi], cmap[bj]
        if ci != cj:
            cl_preds[cj].add(ci)
    cl_in_traffic = defaultdict(float)
    for pb, cb, _ho, _pos, size in model.tg:
        cs_p = {cmap[b] for b in pb}
        cs_c = {cmap[b] for b in cb}
        for cs in cs_c - cs_p:
            cl_in_traffic[cs] += size
    work_m, work_v = model.block_work_m, model.block_work_v
    cl_work = [sum(work_m[b] + work_v[b] for b in members[ci])
               for ci in range(K)]
    cl_m = [sum(work_m[b] for b in members[ci]) for ci in range(K)]
    cl_v = [sum(work_v[b] for b in members[ci]) for ci in range(K)]
    bl = [0.0] * K
    for ci in reversed(range(K)):
        best = 0.0
        for cj in range(K):
            if ci in cl_preds[cj] and bl[cj] > best:
                best = bl[cj]
        bl[ci] = cl_work[ci] + best
    indeg = [len(cl_preds.get(ci, ())) for ci in range(K)]
    ready = [(-bl[ci], ci) for ci in range(K) if indeg[ci] == 0]
    heapq.heapify(ready)
    core_of_sg = []
    core_live = [[0.0, 0.0] for _ in range(num_cores)]
    core_free = [0.0] * num_cores
    core_m = [0.0] * num_cores
    core_v = [0.0] * num_cores
    while ready:
        _, ci = heapq.heappop(ready)
        best_c, best_key = None, None
        for c in range(num_cores):
            over = sum(max(0.0, core_live[c][pi] + internal[ci][pi]
                           - mem_budget * MEM[pi]) for pi in (0, 1))
            nm = core_m[c] + cl_m[ci]
            nv = core_v[c] + cl_v[ci]
            imbalance = abs(nm - nv) / max(1.0, nm + nv)
            est = core_free[c] + cl_in_traffic.get(ci, 0.0) / BW_
            key = (over > 0,
                   over / BW_ + imbalance * max(1.0, cl_work[ci]) * mv_weight,
                   est)
            if best_key is None or key < best_key:
                best_key, best_c = key, c
        core_of_sg.append(best_c)
        for pi in (0, 1):
            core_live[best_c][pi] += internal[ci][pi]
        core_free[best_c] += cl_work[ci] + cl_in_traffic.get(ci, 0.0) / BW_
        core_m[best_c] += cl_m[ci]
        core_v[best_c] += cl_v[ci]
        for cj in range(K):
            if ci in cl_preds[cj]:
                indeg[cj] -= 1
                if indeg[cj] == 0:
                    heapq.heappush(ready, (-bl[cj], cj))
    return core_of_sg, sg_of_block
