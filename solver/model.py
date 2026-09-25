"""NPU 多核调度问题的快速代理模型。

职责：
1. 读取计算图，构造 op 级 DAG（收缩 COPY_IN/COPY_OUT）；
2. 将 op 粗化为 block（降低元启发式搜索维度）；
3. 对给定 (block->subgraph, subgraph->core) 方案：
   - 精确计算切分新增的 DDR 搬运量（与官方评估器的边界 COPY 插入规则一致）；
   - 轻量事件仿真估计 Makespan（Task 等待、跨核延迟、共享带宽、容量压力）；
   - 派生每核合法的子图执行顺序（拓扑一致）。
任何分配方案都可行（DAG 的商图必无环），搜索只需处理分配决策。
"""
import heapq
import json
import math
from collections import defaultdict

COPY_TYPES = {'COPY_IN', 'COPY_OUT'}


def _tarjan_scc(succs):
    """迭代版 Tarjan 强连通分量（避免深图递归爆栈）。"""
    n = len(succs)
    index_counter = [0]
    stack, on_stack = [], [False] * n
    index, lowlink = [-1] * n, [0] * n
    sccs = []
    for root in range(n):
        if index[root] != -1:
            continue
        work = [(root, iter(sorted(succs[root])))]
        index[root] = lowlink[root] = index_counter[0]
        index_counter[0] += 1
        stack.append(root)
        on_stack[root] = True
        while work:
            v, it = work[-1]
            advanced = False
            for w in it:
                if index[w] == -1:
                    index[w] = lowlink[w] = index_counter[0]
                    index_counter[0] += 1
                    stack.append(w)
                    on_stack[w] = True
                    work.append((w, iter(sorted(succs[w]))))
                    advanced = True
                    break
                elif on_stack[w]:
                    lowlink[v] = min(lowlink[v], index[w])
            if advanced:
                continue
            work.pop()
            if work:
                pv = work[-1][0]
                lowlink[pv] = min(lowlink[pv], lowlink[v])
            if lowlink[v] == index[v]:
                scc = []
                while True:
                    w = stack.pop()
                    on_stack[w] = False
                    scc.append(w)
                    if w == v:
                        break
                sccs.append(scc)
    return sccs

# 固定配置（data/config.txt，不得修改）
L1_CAP = 524288
UB_CAP = 131072
BW = 60.0                 # DDR bytes/cycle
WAIT_CROSS = 1000         # 场景A跨核前驱等待
WAIT_SAME = 100           # 场景A同核Task切换等待
DELAY_B = 500             # 场景B跨核COPY同步延迟
L2_CAP = 1048576
L2_BW = 250.0

MEM_CAP = L1_CAP + UB_CAP


class Model:
    def __init__(self, graph_json, block_ops_cap=120):
        self.graph_json = graph_json
        ops = graph_json['ops']
        self.op_by_id = {o['id']: o for o in ops}
        self.tensor_by_id = {t['id']: t for t in graph_json['tensors']}
        eligible = sorted(o['id'] for o in ops if o['op'] not in COPY_TYPES)
        self.eligible = eligible
        self.eligible_set = set(eligible)

        producers = defaultdict(set)   # tensor -> ops producing it
        consumers = defaultdict(set)   # tensor -> ops consuming it
        direct_edges = []
        op_ids = set(self.op_by_id)
        for e in graph_json['edges']:
            s, t = e['source'], e['target']
            if s in op_ids and t in op_ids:
                if s != t:
                    direct_edges.append((s, t))
            elif s in op_ids:
                producers[t].add(s)
            elif t in op_ids:
                consumers[s].add(t)
        self._producers_raw = producers
        self._consumers_raw = consumers

        # ---- 收缩 COPY 节点：eligible 级 DAG ----
        full_succs = defaultdict(set)
        for s, t in direct_edges:
            full_succs[s].add(t)
        for tid, prods in producers.items():
            for s in prods:
                for c in consumers.get(tid, ()):
                    if s != c:
                        full_succs[s].add(c)
        contracted_succs = self._contract(eligible, full_succs)
        contracted_preds = defaultdict(set)
        for u, vs in contracted_succs.items():
            for v in vs:
                contracted_preds[v].add(u)
        self.eligible_succs = contracted_succs
        self.eligible_preds = contracted_preds
        self.topo = self._topo(eligible, contracted_preds, contracted_succs)
        self.pos_in_topo = {op: i for i, op in enumerate(self.topo)}

        self.work_m = {}
        self.work_v = {}
        for oid in eligible:
            op = self.op_by_id[oid]
            if op['pipe'] == 'PIPE_M':
                self.work_m[oid] = op['cycles']
                self.work_v[oid] = 0
            else:
                self.work_m[oid] = 0
                self.work_v[oid] = op['cycles']

        self.block_ops_cap = block_ops_cap
        self.spill_coefs = None      # NNLS 校准系数（默认启发式）
        self.use_lru_spill = False   # op 级 LRU spill 仿真开关
        self.use_overlap = False     # overlap 因子开关
        self.op_lifetimes = None     # 惰性构建
        self._build_blocks()
        self._build_tensor_groups()
        self._build_block_topo()
        self._eval_cache = {}
        # 原图固有搬运量（合成边界 COPY 会替换原图 COPY，added 需扣除）
        self.original_copy_bytes = self._original_copy_traffic()

    def _original_copy_traffic(self):
        """与官方 _copy_traffic_bytes 一致：COPY_IN 输出 + COPY_OUT 输入张量大小。"""
        op_ids = set(self.op_by_id)
        total = 0
        for e in self.graph_json['edges']:
            s, t = e['source'], e['target']
            if s in op_ids and t not in op_ids:
                if self.op_by_id[s]['op'] == 'COPY_IN':
                    total += self.tensor_by_id[t]['size']
            elif s not in op_ids and t in op_ids:
                if self.op_by_id[t]['op'] == 'COPY_OUT':
                    total += self.tensor_by_id[s]['size']
        return total

    # ------------------------------------------------------------- 基础工具
    @staticmethod
    def _contract(eligible, full_succs):
        eligible_set = set(eligible)
        contracted = {u: set() for u in eligible}
        for src in eligible:
            stack = list(full_succs.get(src, ()))
            visited_excluded = set()
            while stack:
                dst = stack.pop()
                if dst in eligible_set:
                    if dst != src:
                        contracted[src].add(dst)
                    continue
                if dst in visited_excluded:
                    continue
                visited_excluded.add(dst)
                stack.extend(full_succs.get(dst, ()))
        return contracted

    @staticmethod
    def _topo(nodes, preds, succs):
        node_set = set(nodes)
        indeg = {n: sum(p in node_set for p in preds.get(n, ())) for n in nodes}
        ready = sorted(n for n in nodes if indeg[n] == 0)
        order = []
        while ready:
            n = ready.pop(0)
            order.append(n)
            for v in sorted(succs.get(n, ())):
                if v not in node_set:
                    continue
                indeg[v] -= 1
                if indeg[v] == 0:
                    ready.append(v)
                    ready.sort()
        if len(order) != len(nodes):
            raise ValueError('graph has a cycle')
        return order

    def _build_blocks(self):
        """每个 op 归入其最高流量前驱所在块（受块容量约束），否则开新块；
        随后对块商图做 SCC 缩聚，保证块 DAG 无环。"""
        ops_cap = self.block_ops_cap
        n = len(self.eligible)
        if n <= ops_cap:
            blocks = [[op] for op in self.topo]
        else:
            edge_traffic = defaultdict(int)
            for tid, t in self.tensor_by_id.items():
                sz = t['size']
                if sz == 0:
                    continue
                for p in self._tensor_eligible_producers(tid):
                    for c in self._tensor_eligible_consumers(tid):
                        if p != c:
                            edge_traffic[(p, c)] += sz
            block_of_op = {}
            blocks = []
            block_load = []
            tail_of_block = []
            for op in self.topo:
                # 优先追加到"前驱恰为该块尾"的块（路径式增长，防交错成环）
                best_b, best_a = None, 0.0
                for p in self.eligible_preds.get(op, ()):
                    b = block_of_op.get(p)
                    if b is None or tail_of_block[b] != p:
                        continue
                    if block_load[b] >= ops_cap:
                        continue
                    a = edge_traffic.get((p, op), 0.0)
                    if a > best_a:
                        best_b, best_a = b, a
                if best_b is None:
                    # 次选：任意有容量且前驱在块内的块
                    aff = defaultdict(float)
                    for p in self.eligible_preds.get(op, ()):
                        b = block_of_op.get(p)
                        if b is not None and block_load[b] < ops_cap:
                            aff[b] += edge_traffic.get((p, op), 0.0)
                    for b, a in aff.items():
                        if a > best_a:
                            best_b, best_a = b, a
                if best_b is None:
                    best_b = len(blocks)
                    blocks.append([])
                    block_load.append(0)
                    tail_of_block.append(None)
                blocks[best_b].append(op)
                block_of_op[op] = best_b
                block_load[best_b] += 1
                tail_of_block[best_b] = op
        # ---- SCC 缩聚：块商图有环时合并同一 SCC 的块 ----
        for _round in range(8):
            nb = len(blocks)
            succs = [set() for _ in range(nb)]
            block_of_op = {op: bi for bi, blk in enumerate(blocks)
                           for op in blk}
            for bi, blk in enumerate(blocks):
                for op in blk:
                    for v in self.eligible_succs.get(op, ()):
                        bj = block_of_op[v]
                        if bj != bi:
                            succs[bi].add(bj)
            sccs = _tarjan_scc(succs)
            if all(len(scc) == 1 for scc in sccs):
                break
            keep = set()
            for scc in sccs:
                if len(scc) == 1:
                    keep.add(scc[0])
            new_blocks = [blocks[i] for i in sorted(keep)]
            for scc in sccs:
                if len(scc) > 1:
                    merged = []
                    for i in sorted(scc):
                        merged.extend(blocks[i])
                    new_blocks.append(merged)
            blocks = [sorted(blk, key=lambda o: self.pos_in_topo[o])
                      for blk in new_blocks]
        self.blocks = blocks
        self.block_of_op = {op: bi for bi, blk in enumerate(blocks) for op in blk}
        nb = len(blocks)
        self.block_work_m = [0.0] * nb
        self.block_work_v = [0.0] * nb
        for bi, blk in enumerate(blocks):
            self.block_work_m[bi] = sum(self.work_m[o] for o in blk)
            self.block_work_v[bi] = sum(self.work_v[o] for o in blk)
        be = defaultdict(int)
        for bi, blk in enumerate(blocks):
            for op in blk:
                for v in self.eligible_succs.get(op, ()):
                    bj = self.block_of_op[v]
                    if bj != bi:
                        be[(bi, bj)] += 1
        self.block_edges = sorted(be.items())

    def _build_block_topo(self):
        """块 DAG 的严格拓扑序 + 块级 cp 权重（串行管道时间 + 搬运时间）。"""
        nb = len(self.blocks)
        preds = defaultdict(set)
        succs = defaultdict(set)
        for (bi, bj), _w in self.block_edges:
            succs[bi].add(bj)
            preds[bj].add(bi)
        self.block_topo = self._topo(list(range(nb)), preds, succs)
        self.block_pos = {b: i for i, b in enumerate(self.block_topo)}
        # 块深度（关键路径层号，供 overlap 估计）
        topo_depth = {}
        for b in self.block_topo:
            ps = [p for p in preds.get(b, ())]
            topo_depth[b] = (max(topo_depth[p] for p in ps) + 1) if ps else 0
        self.block_topo_depth = topo_depth
        # 块的进出流量（DDR 字节，含图输入/输出）
        in_b = [0.0] * nb
        out_b = [0.0] * nb
        for pb, cb, _has_out, _tpos, size in self.tg:
            for b in cb:
                in_b[b] += size
            for b in pb:
                out_b[b] += size
        self.block_cp_weight = [self.block_work_m[b] + self.block_work_v[b]
                                + (in_b[b] + out_b[b]) / BW
                                for b in range(nb)]

    def lower_bounds(self, num_cores):
        """Optimistic work, dependency and mandatory I/O bounds in cycles.

        These omit synchronization, partition copies and spills, so they can
        diagnose headroom without claiming that a schedule is attainable.
        """
        if num_cores < 1:
            raise ValueError('num_cores must be positive')
        path = {}
        for op in self.topo:
            duration = self.work_m[op] + self.work_v[op]
            path[op] = duration + max(
                (path[p] for p in self.eligible_preds.get(op, ())),
                default=0.0)
        parts = {
            'pipe_m_work': sum(self.work_m.values()) / num_cores,
            'pipe_v_work': sum(self.work_v.values()) / num_cores,
            'dependency_path': max(path.values(), default=0.0),
            'mandatory_ddr': self.original_copy_bytes / BW,
        }
        parts['overall'] = max(parts.values(), default=0.0)
        return parts

    def _spill_from_signals(self, sig):
        """由四路信号映射 spill 字节数/时间。
        默认启发式：每池取(峰值超限, 内部总量超限)之大者×2；
        set_spill_coefs 设置 NNLS 校准系数后走线性映射。
        lru=True 时改用 op 级 LRU 仿真结果（sig 键 'L1l'/'UBl'）。"""
        if self.use_lru_spill:
            b = 2.0 * (sig.get('L1l', 0.0) + sig.get('UBl', 0.0))
            return b, b / BW
        if self.spill_coefs is not None:
            a, b2, c, d, e = self.spill_coefs
            bytes_ = max(0.0, a * sig['L1p'] + b2 * sig['L1w']
                         + c * sig['UBp'] + d * sig['UBw'] + e)
        else:
            bytes_ = 2.0 * (max(sig['L1p'], sig['L1w'])
                            + max(sig['UBp'], sig['UBw']))
        return bytes_, bytes_ / BW

    def _build_op_lifetimes(self):
        """构建 op 级张量生命周期（LRU spill 仿真用，惰性一次）。"""
        if self.op_lifetimes is not None:
            return
        block_ops_sorted = [sorted(blk, key=lambda o: self.pos_in_topo[o])
                            for blk in self.blocks]
        step_of = {}
        step = 0
        for bi in self.block_topo:
            for op in block_ops_sorted[bi]:
                step_of[op] = step
                step += 1
        lives = []
        pool_idx = {'L1': 0, 'UB': 1}
        for tid, t in self.tensor_by_id.items():
            prods = self._tensor_eligible_producers(tid)
            cons = self._tensor_eligible_consumers(tid)
            if not prods and not cons:
                continue
            sz = t['size']
            if sz == 0:
                continue
            d = min((step_of[p] for p in prods), default=0)
            u = max((step_of[c] for c in cons), default=d)
            pos = t.get('pos', 'L1')
            lives.append((d, u, sz, pool_idx.get(pos, 0)))
        self.op_lifetimes = lives

    def _lru_spill_bytes(self, span_list):
        """对若干执行单元做 op 级 LRU spill 仿真。

        span_list: 每单元的块位置区间列表 [(first_blk, last_blk), ...]；
        张量生命周期按全局步号（块拓扑展开）落在某区间内即归属该单元。
        返回 (L1 超限字节和, UB 超限字节和)——每次逐出计 1 次写回。
        """
        self._build_op_lifetimes()
        caps = (L1_CAP, UB_CAP)
        overs = [0.0, 0.0]
        span_union = span_list
        for pi in (0, 1):
            cap = caps[pi]
            sub = [(d, u, sz) for d, u, sz, p in self.op_lifetimes
                   if p == pi and any(f <= d and u <= l
                                      for f, l in span_union)]
            if not sub:
                continue
            sub.sort()
            resident = []          # [last_use, size]
            used = 0.0
            n = len(sub)
            ei = 0
            events = sorted({d for d, _, _ in sub})
            for cur_d in events:
                while ei < n and sub[ei][0] == cur_d:
                    _, u, sz = sub[ei]
                    resident.append([u, sz])
                    used += sz
                    ei += 1
                while used > 0.85 * cap and resident:
                    resident.sort(reverse=True)   # 逐出 last_use 最晚者
                    u, sz = resident.pop(0)
                    used -= sz
                    overs[pi] += sz
        return overs

    def set_spill_coefs(self, coefs):
        """coefs = (a, b, c, d, e)：spill_bytes = a·L1p + b·L1w + c·UBp
        + d·UBw + e（NNLS 非负拟合，单位字节）。"""
        self.spill_coefs = coefs

    def _tensor_eligible_producers(self, tid):
        return [p for p in self._producers_raw.get(tid, ())
                if p in self.eligible_set]

    def _tensor_eligible_consumers(self, tid):
        return [c for c in self._consumers_raw.get(tid, ())
                if c in self.eligible_set]

    def _build_tensor_groups(self):
        """按 (生产块, 消费块, has_out, pos) 签名合并 tensor，精确聚合边界流量。"""
        groups = defaultdict(float)
        for tid, t in self.tensor_by_id.items():
            size = t['size']
            if size == 0:
                continue
            prods = self._tensor_eligible_producers(tid)
            cons = self._tensor_eligible_consumers(tid)
            if not prods and not cons:
                continue
            has_out = any(self.op_by_id[c]['op'] == 'COPY_OUT'
                          for c in self._consumers_raw.get(tid, ()))
            pb = tuple(sorted({self.block_of_op[p] for p in prods}))
            cb = tuple(sorted({self.block_of_op[c] for c in cons}))
            pos = t.get('pos', 'L1') if (prods or cons) else 'DDR'
            if pos == 'DDR':
                pos = 'L1'   # 任务图里 DDR 张量会改放核内，按 L1 池估计
            groups[(pb, cb, has_out, pos)] += size
        self.tg = [(pb, cb, ho, pos, sz)
                   for (pb, cb, ho, pos), sz in groups.items()]

    # ---------------------------------------------------------------- 评估
    def evaluate(self, sg_of_block, core_of_sg, scene, num_cores,
                 use_cache=True, orders_override=None, diagnostics=False):
        """返回 (est_makespan, est_added_bytes, info)。scene: 'A'|'B'|'C'.

        orders_override 给定时按该每核顺序仿真（绕过缓存），用于
        Tessel 式顺序精修；顺序必须拓扑可行。"""
        key = None
        if use_cache and orders_override is None and not diagnostics:
            key = (tuple(sg_of_block), tuple(core_of_sg), scene, num_cores)
            hit = self._eval_cache.get(key)
            if hit is not None:
                return hit
        nb = len(self.blocks)
        K = len(core_of_sg)
        sg_wm = [0.0] * K
        sg_wv = [0.0] * K
        for bi in range(nb):
            s = sg_of_block[bi]
            sg_wm[s] += self.block_work_m[bi]
            sg_wv[s] += self.block_work_v[bi]

        sg_in = [0.0] * K     # 场景A COPY_IN 字节
        sg_out = [0.0] * K    # 场景A COPY_OUT 字节
        added_a = 0.0
        added_b = 0.0
        sg_in_b = [0.0] * K   # 场景B 挂靠到子图的输入字节
        sg_out_b = [0.0] * K
        sg_preds = [set() for _ in range(K)]
        sg_succs = [set() for _ in range(K)]
        for (bi, bj), _w in self.block_edges:
            si, sj = sg_of_block[bi], sg_of_block[bj]
            if si != sj:
                sg_succs[si].add(sj)
                sg_preds[sj].add(si)

        sg_rank = [float('inf')] * K
        for bi in range(nb):
            s = sg_of_block[bi]
            r = self.block_pos[bi]
            if r < sg_rank[s]:
                sg_rank[s] = r

        # 子图内部依赖宽度（同层最大块数，估计流水重叠潜力）
        sg_depth_cnt = defaultdict(int)
        for bi in range(nb):
            sg_depth_cnt[(sg_of_block[bi],
                          self.block_topo_depth[bi])] += 1
        self.sg_width = defaultdict(int)
        for (s, _d), cnt in sg_depth_cnt.items():
            if cnt > self.sg_width[s]:
                self.sg_width[s] = cnt

        # 块在子图内部的序（场景A 驻留区间用）
        pos_in_sg = {}
        cnt_sg = [0] * K
        for bi in self.block_topo:
            s = sg_of_block[bi]
            pos_in_sg[bi] = cnt_sg[s]
            cnt_sg[s] += 1

        # 子图内部关键路径（串行管道+搬运的块级最长路）
        blk_preds_in_sg = {}
        for (bi, bj), _w in self.block_edges:
            if sg_of_block[bi] == sg_of_block[bj]:
                blk_preds_in_sg.setdefault(bj, []).append(bi)
        cpv = [0.0] * nb
        sg_cp = [0.0] * K
        for bi in self.block_topo:
            best = 0.0
            for p in blk_preds_in_sg.get(bi, ()):
                if cpv[p] > best:
                    best = cpv[p]
            cpv[bi] = self.block_cp_weight[bi] + best
            s = sg_of_block[bi]
            if cpv[bi] > sg_cp[s]:
                sg_cp[s] = cpv[bi]

        for pb, cb, has_out, tpos, size in self.tg:
            PP = {sg_of_block[b] for b in pb}
            CC = {sg_of_block[b] for b in cb}
            if not PP and not CC:
                continue
            # ---- 场景 A：子图边界规则（与官方评估器一致，精确） ----
            if PP:
                for s in PP:
                    if has_out or (CC - {s}):
                        sg_out[s] += size
                        added_a += size
            if CC:
                for s in CC - PP:
                    sg_in[s] += size
                    added_a += size
            if scene == 'A':
                continue
            # ---- 场景 B：核级规则 ----
            PC = {core_of_sg[s] for s in PP}
            CCores = {core_of_sg[s] for s in CC}
            if CC and not PP:                      # 图输入：每消费核读一次
                for cc in CCores:
                    added_b += size
                for cc in CCores:
                    cands = [x for x in CC if core_of_sg[x] == cc]
                    att = min(cands, key=lambda x: sg_rank[x])
                    sg_in_b[att] += size
            if PP and (has_out or not CC):         # 图输出：每生产核写一次
                for pc in PC:
                    added_b += size
                for pc in PC:
                    cands = [x for x in PP if core_of_sg[x] == pc]
                    att = max(cands, key=lambda x: sg_rank[x])
                    sg_out_b[att] += size
            if PP and CC:                          # 跨核：每对核 COPY_OUT+COPY_IN
                for pc in PC:
                    n_cross = sum(1 for cc in CCores if cc != pc)
                    if n_cross:
                        cands = [x for x in PP if core_of_sg[x] == pc]
                        att = max(cands, key=lambda x: sg_rank[x])
                        sg_out_b[att] += size * n_cross
                for cc in CCores:
                    n_cross = sum(1 for pc in PC if pc != cc)
                    if n_cross:
                        cands = [x for x in CC if core_of_sg[x] == cc]
                        att = min(cands, key=lambda x: sg_rank[x])
                        sg_in_b[att] += size * n_cross
                for pc in PC:
                    for cc in CCores:
                        if pc != cc:
                            added_b += 2 * size

        # ---- 驻留峰值与 spill 估计（按 L1/UB 两个独立容量池） ----
        # 两个信号取大：区间驻留峰值超限（宽图）与内部总字节超限（深链反复换入换出）。
        # 场景A：按子图；场景B：按每核任务。
        spill_time = 0.0
        spill_bytes = 0.0
        n_units = K if scene == 'A' else num_cores
        spans_per_unit = [[] for _ in range(n_units)]
        wint_per_unit = [[0.0, 0.0] for _ in range(n_units)]  # [L1, UB]
        pool_idx = {'L1': 0, 'UB': 1}
        for pb, cb, has_out, tpos, size in self.tg:
            pi = pool_idx.get(tpos, 0)
            if scene == 'A':
                by_unit = defaultdict(list)
                for b in set(pb) | set(cb):
                    by_unit[sg_of_block[b]].append(b)
                for s in by_unit:
                    ps = [pos_in_sg[b] for b in pb if sg_of_block[b] == s]
                    cs = [pos_in_sg[b] for b in cb if sg_of_block[b] == s]
                    if not ps and not cs:
                        continue
                    spans_per_unit[s].append(
                        (min(ps + cs), max(ps + cs), size, pi))
                    if ps and cs:      # 完全内部张量计入 W_int
                        wint_per_unit[s][pi] += size
            else:
                for c in range(num_cores):
                    ps = [self.block_pos[b] for b in pb
                          if core_of_sg[sg_of_block[b]] == c]
                    cs = [self.block_pos[b] for b in cb
                          if core_of_sg[sg_of_block[b]] == c]
                    if not ps and not cs:
                        continue
                    spans_per_unit[c].append(
                        (min(ps + cs), max(ps + cs), size, pi))
                    if ps and cs:
                        wint_per_unit[c][pi] += size
        CAPS = (L1_CAP, UB_CAP)
        sig = {'L1p': 0.0, 'L1w': 0.0, 'UBp': 0.0, 'UBw': 0.0}
        keys = ('L1p', 'L1w', 'UBp', 'UBw')
        for u, spans in enumerate(spans_per_unit):
            if not spans:
                continue
            for pi in (0, 1):
                cap = CAPS[pi]
                sub = [(f, l, sz) for f, l, sz, p in spans if p == pi]
                peak = 0.0
                if sub:
                    points = {p for s in sub for p in s[:2]}
                    for p in points:
                        live = 0.0
                        for f, l, sz in sub:
                            if f <= p <= l:
                                live += sz
                        if live > peak:
                            peak = live
                sig[keys[2 * pi]] += max(0.0, peak - 0.85 * cap)
                sig[keys[2 * pi + 1]] += max(
                    0.0, wint_per_unit[u][pi] - 0.85 * cap)
        spill_bytes, spill_time = self._spill_from_signals(sig)
        if self.use_lru_spill:
            # op 级 LRU 仿真：按执行单元的块位置区间筛选生命周期
            if scene == 'A':
                unit_spans = [[] for _ in range(K)]
                for bi in range(nb):
                    s = sg_of_block[bi]
                    unit_spans[s].append((self.block_pos[bi],
                                          self.block_pos[bi]))
                spans_flat = [(min(f for f, _ in sp), max(l for _, l in sp))
                              for sp in unit_spans if sp]
            else:
                core_blocks = [[] for _ in range(num_cores)]
                for bi in range(nb):
                    core_blocks[core_of_sg[sg_of_block[bi]]].append(
                        self.block_pos[bi])
                spans_flat = [(min(cb), max(cb)) for cb in core_blocks if cb]
            l1o, ubo = self._lru_spill_bytes(spans_flat)
            sig = dict(sig)
            sig['L1l'], sig['UBl'] = l1o, ubo
            spill_bytes, spill_time = self._spill_from_signals(sig)

        if scene == 'A':
            durs = [max(sg_wm[s], sg_wv[s], sg_in[s] / BW, sg_out[s] / BW,
                        sg_cp[s]) for s in range(K)]
        else:
            durs = [max(sg_wm[s], sg_wv[s], sg_in_b[s] / BW, sg_out_b[s] / BW,
                        sg_cp[s]) for s in range(K)]
        # overlap 因子（综述 MAS-Attention/DoubleBuffer 思想）：
        # 完全重叠假设对依赖链式子图过于乐观。按子图内部依赖宽度估计
        # 可重叠比例 φ=宽度/容量，未重叠的搬运时间按 (1-φ) 加回。
        if self.use_overlap:
            for s in range(K):
                copy_t = (sg_in[s] + sg_out[s]) / BW if scene == 'A' else \
                         (sg_in_b[s] + sg_out_b[s]) / BW
                if copy_t <= 0 or durs[s] <= 0:
                    continue
                width = self.sg_width.get(s, 1)
                phi = min(1.0, 0.35 + 0.65 * min(1.0, width / 6.0))
                durs[s] += (1.0 - phi) * copy_t

        if orders_override is not None:
            makespan, orders, end_time = self._simulate_fixed_orders(
                sg_preds, sg_succs, core_of_sg, durs, orders_override,
                scene, num_cores)
        else:
            makespan, orders, end_time = self._simulate(
                sg_preds, sg_succs, core_of_sg, durs, sg_rank, scene,
                num_cores)
        # 全局 DDR 带宽下界：全部搬运量共享 60B/cycle
        total_copy_bytes = self.original_copy_bytes + (
            added_a if scene == 'A' else added_b) + spill_bytes
        makespan = max(makespan + spill_time,
                       total_copy_bytes / BW if total_copy_bytes > 0 else 0.0)
        # 场景 C（B + 只读 L2）：FIFO 复用距离近似 + 双池带宽下界。
        # 两遍仿真：第一遍得各子图结束时序 → 按复用距离估计每子图
        # L2 命中字节 → 从 sg_in_b 扣除后第二遍仿真，使搜索能"看见" L2。
        if scene == 'C':
            def tg_size(gi):
                return self.tg[gi][4]
            # 读者映射：gi -> [(读者子图, 字节)]（每核首个消费子图一次读取）
            readers = {}
            for gi, (pb, cb, has_out, tpos, size) in enumerate(self.tg):
                if not cb or size <= 0 or size > L2_CAP:
                    continue
                cons_sgs = {sg_of_block[b] for b in cb}
                if len({core_of_sg[s] for s in cons_sgs}) < 2:
                    continue
                per_core = {}
                for s in cons_sgs:
                    c = core_of_sg[s]
                    att = min((x for x in cons_sgs if core_of_sg[x] == c),
                              key=lambda x: sg_rank[x])
                    per_core[att] = size
                readers[gi] = list(per_core.items())
            fills = []      # (首读时刻, size, gi)
            extra = []      # (后续读时刻, gi, 读者子图)
            for gi, rlist in readers.items():
                times = sorted((end_time[s], s) for s, _b in rlist)
                if not times:
                    continue
                fills.append((times[0][0], tg_size(gi), gi))
                for t, s in times[1:]:
                    extra.append((t, gi, s))
            fills.sort()
            fill_times = [f[0] for f in fills]
            fill_time_of = {gi: t for t, _s, gi in fills}
            prefix = [0.0]
            for _t, _s, _g in fills:
                prefix.append(prefix[-1] + _s)
            import bisect
            def _window(lo, hi):
                i = bisect.bisect_right(fill_times, lo)
                j = bisect.bisect_right(fill_times, hi)
                return prefix[j] - prefix[i]
            extra.sort()
            prev_t = {}
            l2_served = 0.0
            hit_per_sg = defaultdict(float)
            for t, gi, s in extra:
                if _window(prev_t.get(gi, fill_time_of[gi]), t) < L2_CAP:
                    hb = tg_size(gi)
                    l2_served += hb
                    hit_per_sg[s] += hb
                prev_t[gi] = t
            if hit_per_sg:
                durs = list(durs)
                for s, hb in hit_per_sg.items():
                    durs[s] = max(sg_wm[s], sg_wv[s],
                                  max(0.0, sg_in_b[s] - hb) / BW,
                                  sg_out_b[s] / BW, sg_cp[s])
                if orders_override is not None:
                    makespan, orders, end_time = self._simulate_fixed_orders(
                        sg_preds, sg_succs, core_of_sg, durs, orders,
                        scene, num_cores)
                else:
                    makespan, orders, end_time = self._simulate(
                        sg_preds, sg_succs, core_of_sg, durs,
                        sg_rank, scene, num_cores)
            # L2 服务的字节从 DDR 池转入 250B/cy 独立池（带宽下界）
            ddr_served = max(0.0, total_copy_bytes - l2_served)
            bw_bound = max(ddr_served / BW,
                           l2_served / L2_BW if l2_served else 0.0)
            makespan = max(makespan, bw_bound)

        if scene == 'C':
            added = max(0.0, added_b - self.original_copy_bytes + spill_bytes)
        elif scene == 'A':
            added = max(0.0, added_a - self.original_copy_bytes + spill_bytes)
        else:
            added = max(0.0, added_b - self.original_copy_bytes + spill_bytes)
        info = {'orders': orders, 'K': K, 'durs': durs,
                'sg_wm': sg_wm, 'sg_wv': sg_wv, 'spill_sig': sig,
                'spill_bytes': spill_bytes}
        if diagnostics:
            info.update({'sg_preds': sg_preds, 'end_time': end_time,
                         'schedule_makespan': max(end_time, default=0.0),
                         'copy_bytes': total_copy_bytes,
                         'bandwidth_bound': (bw_bound if scene == 'C'
                                             else total_copy_bytes / BW)})
        result = (makespan, added, info)
        if use_cache and key is not None:
            if len(self._eval_cache) > 8192:
                self._eval_cache.clear()
            self._eval_cache[key] = result
        return result

    def _simulate(self, sg_preds, sg_succs, core_of_sg, durs, sg_rank,
                  scene, num_cores):
        """按就绪事件派生每核顺序并估计 makespan。"""
        K = len(durs)
        indeg = [len(sg_preds[s]) for s in range(K)]
        end_time = [0.0] * K
        orders = [[] for _ in range(num_cores)]
        core_free = [0.0] * num_cores
        heap = []

        def push(s):
            c = core_of_sg[s]
            est = core_free[c]
            if scene == 'A' and orders[c]:
                est += WAIT_SAME
            for p in sg_preds[s]:
                if core_of_sg[p] != c:
                    est = max(est, end_time[p] +
                              (WAIT_CROSS if scene == 'A' else DELAY_B))
            heapq.heappush(heap, (est, sg_rank[s], s))

        for s in range(K):
            if indeg[s] == 0:
                push(s)
        while heap:
            _, _, s = heapq.heappop(heap)
            c = core_of_sg[s]
            start = core_free[c]
            if scene == 'A' and orders[c]:
                start += WAIT_SAME
            for p in sg_preds[s]:
                if core_of_sg[p] != c:
                    start = max(start, end_time[p] +
                                (WAIT_CROSS if scene == 'A' else DELAY_B))
                else:
                    start = max(start, end_time[p])
            end_time[s] = start + durs[s]
            core_free[c] = end_time[s]
            orders[c].append(s)
            for v in sg_succs[s]:
                indeg[v] -= 1
                if indeg[v] == 0:
                    push(v)
        return (max(end_time) if end_time else 0.0), orders, end_time

    def _simulate_fixed_orders(self, sg_preds, sg_succs, core_of_sg, durs,
                               orders, scene, num_cores):
        """按给定每核顺序仿真（顺序须拓扑可行）。轮询各核，队首子图
        前驱全部调度完后发射，直至全部完成。"""
        K = len(durs)
        end_time = [0.0] * K
        done = [False] * K
        core_free = [0.0] * num_cores
        ptr = [0] * num_cores
        scheduled = 0
        while scheduled < K:
            progressed = False
            for c in range(num_cores):
                while ptr[c] < len(orders[c]):
                    s = orders[c][ptr[c]]
                    if not all(done[p] for p in sg_preds[s]):
                        break
                    start = core_free[c]
                    if scene == 'A' and ptr[c] > 0:
                        start += WAIT_SAME
                    for p in sg_preds[s]:
                        w = 0.0 if core_of_sg[p] == c else (
                            WAIT_CROSS if scene == 'A' else DELAY_B)
                        start = max(start, end_time[p] + w)
                    end_time[s] = start + durs[s]
                    core_free[c] = end_time[s]
                    done[s] = True
                    ptr[c] += 1
                    scheduled += 1
                    progressed = True
            if not progressed and scheduled < K:
                return float('inf'), orders, end_time   # 顺序违反拓扑
        return (max(end_time) if end_time else 0.0), orders, end_time

    # ---------------------------------------------------------------- 方案 IO
    def plan_from(self, sg_of_block, orders):
        node_to_subgraph = {}
        for bi, blk in enumerate(self.blocks):
            for op in blk:
                node_to_subgraph[str(op)] = int(sg_of_block[bi])
        return {'node_to_subgraph': node_to_subgraph,
                'core_schedules': [[int(s) for s in order] for order in orders]}


def load_graph(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)
