"""M0 下界攻击模式（LB-Attacker）。

严格下界 LB = max(纯CP, work/N, 原始字节/60)，其中 work 按场景取：
- 场景 A（子图=Task 串行）：work_A = Σ_sg max(wm_sg, wv_sg) / N（下界用块近似）
- 场景 B/C（每核合并 Task，M/V 跨子图乱序交错）：work_B = max(Σwm, Σwv) / N

lb_construct：K 扫描（N / 1.5N / 2N / 容量下限 K_floor）+ 流量优先凸凝聚
+ 场景感知装箱（A：最小化 Σ_sg max(wm,wv)；B/C：最小化 max(Σwm,Σwv)，
M 重簇与 V 重簇同核互补配对）+ 字节门槛回调。
"""
import heapq
import math
from collections import defaultdict

from solution import Sol
from model import BW, L1_CAP, UB_CAP

CAPS = (L1_CAP, UB_CAP)
POOL_IDX = {'L1': 0, 'UB': 1}
MEM_BUDGET = 0.85


def compute_lb(graph_json, orig_bytes, n_cores):
    """严格下界与瓶颈归因（op 级纯 CP + 管道独立均衡 + 原始字节）。"""
    from collections import deque
    ops = {o['id']: o for o in graph_json['ops']}
    op_ids = set(ops)
    succ = defaultdict(list)
    pred = defaultdict(set)
    for e in graph_json['edges']:
        s, t = e['source'], e['target']
        if s in op_ids and t in op_ids and s != t:
            succ[s].append(t)
            pred[t].add(s)
    indeg = {o: len(pred[o]) for o in op_ids}
    q = deque(o for o in op_ids if indeg[o] == 0)
    cpt = {o: 0.0 for o in op_ids}
    order = []
    while q:
        u = q.popleft()
        order.append(u)
        for v in succ[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    for u in order:
        cpt[u] = ops[u]['cycles'] + max((cpt[p] for p in pred[u]), default=0.0)
    cp = max(cpt.values()) if cpt else 0.0
    twm = sum(o['cycles'] for o in graph_json['ops']
              if o['pipe'] == 'PIPE_M')
    twv = sum(o['cycles'] for o in graph_json['ops']
              if o['pipe'] == 'PIPE_V')
    bytes_lb = orig_bytes / BW
    # 场景 A 的块近似 Σmax（用 ops 直接算上界近似：Σ max(wm_op,vv_op)）
    work_a_num = sum(max(o['cycles'], 0.0) if o['pipe'] == 'PIPE_M'
                     else max(0.0, o['cycles'])
                     for o in graph_json['ops']
                     if o['op'] not in ('COPY_IN', 'COPY_OUT'))
    work_a = sum(o['cycles'] for o in graph_json['ops']
                 if o['op'] not in ('COPY_IN', 'COPY_OUT'))
    # op 级 Σmax(wm,vv)：每个 op 只在一个管道上，故 Σmax = 总 cycles
    work_a_lb = work_a / max(1, n_cores)
    work_b_lb = max(twm, twv) / max(1, n_cores)
    terms_a = {'CP': cp, 'work': work_a_lb, 'bytes': bytes_lb}
    lb_a = max(terms_a.values())
    binder_a = max(terms_a, key=terms_a.get)
    # CP 接近（>=85% LB）即标 CP（指引 M0 的条带路由）
    if cp >= 0.85 * lb_a:
        binder_a = 'CP'
    terms_b = {'CP': cp, 'work': work_b_lb, 'bytes': bytes_lb}
    lb_b = max(terms_b.values())
    binder_b = max(terms_b, key=terms_b.get)
    if cp >= 0.85 * lb_b:
        binder_b = 'CP'
    return {'CP': cp, 'twm': twm, 'twv': twv,
            'lb_A': lb_a, 'binder_A': binder_a,
            'lb_B': lb_b, 'binder_B': binder_b}


def _cluster_internal_bytes(model, blocks):
    """簇完全内部张量字节（按池）。"""
    bset = set(blocks)
    internal = [0.0, 0.0]
    for pb, cb, has_out, tpos, size in model.tg:
        if pb and cb and set(pb) <= bset and set(cb) <= bset:
            internal[POOL_IDX.get(tpos, 0)] += size
    return internal


def lb_construct(model, num_cores, scene, k_target):
    """下界攻击构造：按图类路由。

    深图（纯CP > 0.45×总工作量）：拓扑连续条带切 K 段（流水化）；
    宽图：边凝聚 + 容量感知 LPT 装箱到 K（均衡）。"""
    nb = len(model.blocks)
    # ---- 图类判别（块级纯 CP：权重 max(wm,wv)） ----
    from collections import defaultdict as _dd
    _preds = _dd(list)
    for (bi, bj), _w in model.block_edges:
        _preds[bj].append(bi)
    _order = model.block_topo
    _cp = {}
    for b in _order:
        _cp[b] = max(model.block_work_m[b], model.block_work_v[b]) + max(
            (_cp[p] for p in _preds[b]), default=0.0)
    cp_pure = max(_cp.values()) if _cp else 0.0
    work_total = sum(max(model.block_work_m[b], model.block_work_v[b])
                     for b in range(nb))
    if cp_pure > 0.45 * max(1.0, work_total):
        from construct import strip_construct
        return strip_construct(model, num_cores, scene, num_strips=k_target)
    # ---- 宽图细分路由：近独立（分量多）走 LPT 装箱；连通走 HEFT ----
    parent = list(range(nb))
    def _find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for (bi, bj), _w in model.block_edges:
        ra, rb = _find(bi), _find(bj)
        if ra != rb:
            parent[ra] = rb
    n_comp = len({_find(b) for b in range(nb)})
    edge_bytes = defaultdict(float)
    for pb, cb, _ho, _pos, size in model.tg:
        cross = [(b, c) for b in pb for c in cb if b != c]
        if not cross:
            continue
        share = size / len(cross)
        for b, c in cross:
            edge_bytes[(b, c)] += share
    members = [frozenset([b]) for b in range(nb)]
    head = list(range(nb))
    tail = list(range(nb))
    cid_of = list(range(nb))
    alive = set(range(nb))

    def internal_bytes(cid):
        return _cluster_internal_bytes(model, members[cid])

    # 容量预计算（单块超容的照收，后续合并受预算约束）
    cap_ok = {}
    for cid in list(alive):
        cap_ok[cid] = all(x <= MEM_BUDGET * CAPS[p] for p, x in
                          enumerate(internal_bytes(cid)))

    # 初始堆：均衡优先——合并后工作量小者优先（Huffman 式），流量节省次级
    heap = []
    def _pair_key(a, b):
        work = sum(max(model.block_work_m[b], model.block_work_v[b])
                   for b in members[a] | members[b])
        sz = sum(edge_bytes.get((u, v), 0.0)
                 for u in members[a] for v in members[b]) +             sum(edge_bytes.get((u, v), 0.0)
                for u in members[b] for v in members[a])
        return (work, -sz, min(a, b), max(a, b))
    seen_pairs = set()
    for (u, v), sz in edge_bytes.items():
        cu, cv = cid_of[u], cid_of[v]
        if cu != cv:
            key = (min(cu, cv), max(cu, cv))
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            w2, ns, _, _ = _pair_key(cu, cv)
            heapq.heappush(heap, (w2, ns, cu, cv))

    def neighbors(cid):
        res = set()
        for op in model.blocks[tail[cid]]:
            for v2 in model.eligible_succs.get(op, ()):
                cj = cid_of[model.block_of_op[v2]]
                if cj != cid and cj in alive:
                    res.add(cj)
        for op in model.blocks[head[cid]]:
            for p2 in model.eligible_preds.get(op, ()):
                cj = cid_of[model.block_of_op[p2]]
                if cj != cid and cj in alive:
                    res.add(cj)
        return res

    # 阶段1：沿边凝聚（均衡优先）直到无边可合
    while heap:
        _w, _ns, a, b = heapq.heappop(heap)
        if a not in alive or b not in alive:
            continue
        if cid_of[a] != a or cid_of[b] != b:
            continue
        merged_internal = [x + y for x, y in zip(internal_bytes(a),
                                                 internal_bytes(b))]
        both_ok = cap_ok.get(a, True) and cap_ok.get(b, True)
        if both_ok and not all(x <= MEM_BUDGET * CAPS[p]
                               for p, x in enumerate(merged_internal)):
            continue
        if model.block_pos[head[a]] > model.block_pos[head[b]]:
            a, b = b, a
        members[a] = members[a] | members[b]
        if model.block_pos[tail[a]] < model.block_pos[tail[b]]:
            tail[a] = tail[b]
        else:
            head[a] = head[b]
        for blk in members[b]:
            cid_of[blk] = a
        alive.discard(b)
        cap_ok[a] = all(x <= MEM_BUDGET * CAPS[p]
                        for p, x in enumerate(internal_bytes(a)))
        for cj in neighbors(a):
            if cj in alive:
                w2, ns, _, _ = _pair_key(a, cj)
                heapq.heappush(heap, (w2, ns, a, cj))

    # 阶段2：独立簇容量感知 LPT 装箱到 K 目标（处理无边图：独立链等）
    cur_clusters = sorted(alive, key=lambda c: -sum(
        max(model.block_work_m[b], model.block_work_v[b])
        for b in members[c]))
    bins = []           # [ [Σmax, internal_bytes[2], [cluster ids]] ]
    for c in cur_clusters:
        cw = sum(max(model.block_work_m[b], model.block_work_v[b])
                 for b in members[c])
        ci = internal_bytes(c)
        best_bin, best_key = None, None
        for bn in bins:
            if bn[1][0] + ci[0] > MEM_BUDGET * CAPS[0] or                bn[1][1] + ci[1] > MEM_BUDGET * CAPS[1]:
                continue
            t = bn[0] + cw
            key = (t, -len(bn[2]))
            if best_key is None or key < best_key:
                best_key, best_bin = key, bn
        if best_bin is not None and len(bins) >= k_target:
            best_bin[0] += cw
            best_bin[1] = [x + y for x, y in zip(best_bin[1], ci)]
            best_bin[2].append(c)
        else:
            bins.append([cw, list(ci), [c]])
    # 合成最终簇
    final_members = []
    for _w, _ib, clist in bins:
        blk = set()
        for c in clist:
            blk |= members[c]
        final_members.append(frozenset(blk))
    # 兼容后续装箱阶段的数据结构
    alive = set(range(len(final_members)))
    members = final_members
    head = [min(model.block_pos[b] for b in blk) for blk in members]
    tail = [max(model.block_pos[b] for b in blk) for blk in members]
    for cid, blk in enumerate(members):
        for b in blk:
            cid_of[b] = cid

    clusters = sorted(alive, key=lambda c: model.block_pos[head[c]])
    # ---- 场景感知装箱 ----
    cl_work = []
    for c in clusters:
        wm = sum(model.block_work_m[b] for b in members[c])
        wv = sum(model.block_work_v[b] for b in members[c])
        mx = max(wm, wv)
        blks = sorted(members[c])
        cl_work.append((mx, wm, wv, blks, c))
    # A：max(wm,wv) 降序装箱，最小化每核 Σmax；
    # B/C：最小化每核 max(Σwm,Σv)（M/V 互补自然发生）
    order = sorted(cl_work, key=lambda x: (-x[0], x[3][0]))
    core_load = []
    for _ in range(num_cores):
        core_load.append([0.0, 0.0])     # [Σmax, (Σwm, Σv) via max]
    core_mv = [[0.0, 0.0] for _ in range(num_cores)]
    assign = {}
    for mx, wm, wv, blks, c in order:
        best_c, best_key = None, None
        for k in range(num_cores):
            if scene == 'A':
                t = core_load[k][0] + mx
                key = (t, max(core_mv[k][0] + wm, core_mv[k][1] + wv))
            else:
                nm = core_mv[k][0] + wm
                nv = core_mv[k][1] + wv
                key = (max(nm, nv), core_load[k][0] + mx)
            if best_key is None or key < best_key:
                best_key, best_c = key, k
        assign[c] = best_c
        core_load[best_c][0] += mx
        core_mv[best_c][0] += wm
        core_mv[best_c][1] += wv

    sg_of_block = [0] * nb
    core_of_sg = []
    for ci, c in enumerate(clusters):
        core_of_sg.append(assign[c])
        for b in members[c]:
            sg_of_block[b] = ci
    # R0-3 构造后自校验：实测簇间流量（含跨核重复读）占原始字节比，
    # 超 2.5x 判定 LPT 装箱破坏了通信结构，整条回退 HEFT。
    cut_bytes = 0.0
    for pb, cb, _ho, _pos, size in model.tg:
        src_sgs = {sg_of_block[b] for b in pb}
        dst_sgs = {sg_of_block[b] for b in cb}
        if not (len(src_sgs) == 1 and src_sgs == dst_sgs):
            cut_bytes += size * max(1, len(src_sgs | dst_sgs) - 1)
    if cut_bytes > 2.5 * max(1.0, model.original_copy_bytes):
        from construct import heft_construct, SCENE_CONFIG
        n_el = len(model.eligible)
        return heft_construct(
            model, num_cores, scene,
            max_sg_ops=max(1, n_el // k_target),
            balance=SCENE_CONFIG[scene]['traffic_weight'])
    return sg_of_block, core_of_sg


def k_floor(model, num_cores):
    """容量下限 K：凝聚到不能再合（容量闸门挡住）时的簇数。"""
    nb = len(model.blocks)
    # 简化：用 lb_construct 的凝聚过程，k_target=1 时自然停在容量下限
    sg, core = lb_construct(model, num_cores, 'A', 1)
    return max(1, len(set(sg)))
