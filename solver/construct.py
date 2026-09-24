"""构造式初始解：HEFT 风格块调度 + 拓扑条带切分。"""
from collections import defaultdict

from model import WAIT_SAME, WAIT_CROSS, DELAY_B, BW, MEM_CAP


def block_graph(model):
    succs = defaultdict(list)
    preds = defaultdict(list)
    traffic = defaultdict(float)
    for (bi, bj), w in model.block_edges:
        succs[bi].append(bj)
        preds[bj].append(bi)
    # 块间流量（tensor 字节）
    for pb, cb, _ho, _pos, size in model.tg:
        for b in pb:
            for c in cb:
                if b != c:
                    traffic[(b, c)] += size
    return succs, preds, traffic


def block_topo_order(model):
    return list(model.block_topo)


def upward_ranks(model):
    succs, preds, _traffic = block_graph(model)
    nb = len(model.blocks)
    dur = [max(model.block_work_m[b], model.block_work_v[b]) for b in range(nb)]
    order = block_topo_order(model)
    rank = [0.0] * nb
    for b in reversed(order):
        best = 0.0
        for s in succs.get(b, ()):
            if rank[s] > best:
                best = rank[s]
        rank[b] = dur[b] + best
    return rank, dur


def heft_construct(model, num_cores, scene, max_sg_ops=400, balance=1.0):
    """按递减 upward rank 处理块，最早完成时间选核（含通信局部性），
    每核块序列再切成有限规模子图。"""
    nb = len(model.blocks)
    rank, dur = upward_ranks(model)
    succs, preds, traffic = block_graph(model)
    proc_order = sorted(range(nb),
                        key=lambda b: (-rank[b], model.block_pos[b]))
    core_free = [0.0] * num_cores
    end_time = {}
    core_of_block = [0] * nb
    seq_by_core = [[] for _ in range(num_cores)]

    for b in proc_order:
        best_c, best_cost = None, None
        for c in range(num_cores):
            start = core_free[c]
            if scene == 'A' and seq_by_core[c]:
                start += WAIT_SAME
            remote_traffic = 0.0
            for p in preds.get(b, ()):
                if p not in end_time:
                    continue
                if core_of_block[p] != c:
                    start = max(start, end_time[p] +
                                (WAIT_CROSS if scene == 'A' else DELAY_B))
                    remote_traffic += traffic.get((p, b), 0.0)
            fin = start + dur[b] + balance * remote_traffic / BW
            if best_cost is None or fin < best_cost:
                best_cost, best_c = fin, c
        c = best_c
        start = core_free[c]
        if scene == 'A' and seq_by_core[c]:
            start += WAIT_SAME
        for p in preds.get(b, ()):
            if p in end_time and core_of_block[p] != c:
                start = max(start, end_time[p] +
                            (WAIT_CROSS if scene == 'A' else DELAY_B))
        end_time[b] = start + dur[b]
        core_free[c] = end_time[b]
        seq_by_core[c].append(b)
        core_of_block[b] = c

    # 每核序列切子图
    sg_of_block = [0] * nb
    core_of_sg = []
    for c in range(num_cores):
        cur = None
        count = 0
        for b in seq_by_core[c]:
            if cur is None or count >= max_sg_ops:
                cur = len(core_of_sg)
                core_of_sg.append(c)
                count = 0
            sg_of_block[b] = cur
            count += len(model.blocks[b])
    return sg_of_block, core_of_sg


def strip_construct(model, num_cores, scene, num_strips):
    """按拓扑序切 contiguous 条带，条带按贪心列表调度分配到核。"""
    nb = len(model.blocks)
    order = block_topo_order(model)
    num_strips = max(num_cores, min(num_strips, nb))
    strips = []
    step = nb / num_strips
    for i in range(num_strips):
        lo = int(round(i * step))
        hi = int(round((i + 1) * step))
        if lo < hi:
            strips.append(order[lo:hi])
    dur = [max(model.block_work_m[b], model.block_work_v[b]) for b in range(nb)]
    succs, preds, traffic = block_graph(model)
    strip_of_block = {}
    for si, st in enumerate(strips):
        for b in st:
            strip_of_block[b] = si
    sdur = [sum(dur[b] for b in st) for st in strips]
    strip_end = {}
    strip_core_of = {}
    core_free_t = [0.0] * num_cores
    core_of_sg = []
    sg_of_block = [0] * nb
    # 条带按拓扑顺序贪心分配：代价 = max(核空闲, 前驱条带结束+等待) + 时长
    for si, st in enumerate(strips):
        pred_strips = sorted({strip_of_block[p] for b in st
                              for p in preds.get(b, ())
                              if strip_of_block.get(p) != si})
        best_c, best_cost = None, None
        for c in range(num_cores):
            start = core_free_t[c]
            for p in pred_strips:
                if strip_core_of.get(p) != c:
                    wait = WAIT_CROSS if scene == 'A' else DELAY_B
                else:
                    wait = 0
                start = max(start, strip_end[p] + wait)
            fin = start + sdur[si]
            if best_cost is None or fin < best_cost:
                best_cost, best_c = fin, c
        sg = len(core_of_sg)
        core_of_sg.append(best_c)
        for b in st:
            sg_of_block[b] = sg
        strip_end[si] = best_cost
        strip_core_of[si] = best_c
        core_free_t[best_c] = best_cost
    return sg_of_block, core_of_sg


def chain_construct(model, num_cores, scene):
    """面向宽图的构造：每条块链尽量独占，均匀摊到各核。"""
    succs, preds, traffic = block_graph(model)
    nb = len(model.blocks)
    rank, dur = upward_ranks(model)
    order = block_topo_order(model)
    assigned = {}
    chains = []
    for b in order:
        if b in assigned:
            continue
        chain = [b]
        assigned[b] = len(chains)
        cur = b
        while True:
            nxt, w = None, -1.0
            for s in succs.get(cur, ()):
                if s in assigned:
                    continue
                tw = traffic.get((cur, s), 0.0)
                if tw > w:
                    nxt, w = s, tw
            if nxt is None:
                break
            chain.append(nxt)
            assigned[nxt] = len(chains)
            cur = nxt
        chains.append(chain)
    # 链按工作量大到小，LPT 分核
    cdur = []
    for ch in chains:
        cdur.append(sum(max(model.block_work_m[b], model.block_work_v[b])
                        for b in ch))
    idx = sorted(range(len(chains)), key=lambda i: (-cdur[i], min(chains[i])))
    core_free = [0.0] * num_cores
    core_of_sg = []
    sg_of_block = [0] * nb
    for i in idx:
        c = min(range(num_cores), key=lambda c: core_free[c])
        sg = len(core_of_sg)
        core_of_sg.append(c)
        for b in chains[i]:
            sg_of_block[b] = sg
        core_free[c] += cdur[i]
    return sg_of_block, core_of_sg
