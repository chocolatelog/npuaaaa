"""解状态与邻域动作：所有元启发式共享的解表示和移动算子。"""
import random
from collections import defaultdict

from model import BW, _tarjan_scc


class Sol:
    """sg_of_block: block -> sg（可稀疏，evaluate 前需 compact）"""

    def __init__(self, sg_of_block, core_of_sg, blocks_in_sg=None):
        self.sg_of_block = list(sg_of_block)
        self.core_of_sg = list(core_of_sg)
        if blocks_in_sg is None:
            blocks_in_sg = [set() for _ in range(len(core_of_sg))]
            for b, s in enumerate(self.sg_of_block):
                blocks_in_sg[s].add(b)
        self.blocks_in_sg = blocks_in_sg

    def clone(self):
        return Sol(self.sg_of_block, self.core_of_sg,
                   [set(x) for x in self.blocks_in_sg])

    def compact(self):
        used = sorted({s for s in self.sg_of_block})
        remap = {s: i for i, s in enumerate(used)}
        self.sg_of_block = [remap[s] for s in self.sg_of_block]
        self.core_of_sg = [self.core_of_sg[s] for s in used]
        self.blocks_in_sg = [self.blocks_in_sg[s] for s in used]

    # ------------------------------------------------------------- 邻域动作
    def move_block(self, b, s):
        old = self.sg_of_block[b]
        if old == s:
            return False
        self.sg_of_block[b] = s
        self.blocks_in_sg[old].discard(b)
        self.blocks_in_sg[s].add(b)
        return True

    def merge_sg(self, s1, s2):
        if s1 == s2 or not self.blocks_in_sg[s2]:
            return False
        # 合到工作量大的子图所在核
        if len(self.blocks_in_sg[s1]) < len(self.blocks_in_sg[s2]):
            s1, s2 = s2, s1
        self.blocks_in_sg[s1] |= self.blocks_in_sg[s2]
        for b in self.blocks_in_sg[s2]:
            self.sg_of_block[b] = s1
        self.blocks_in_sg[s2] = set()
        return True

    def split_sg(self, s, rank):
        """按 rank 中位数把子图一分为二（保持拓扑凸性）。"""
        blk = sorted(self.blocks_in_sg[s], key=lambda b: rank[b])
        if len(blk) < 2:
            return False
        half = len(blk) // 2
        new_s = len(self.core_of_sg)
        self.core_of_sg.append(self.core_of_sg[s])
        self.blocks_in_sg.append(set(blk[half:]))
        self.blocks_in_sg[s] = set(blk[:half])
        for b in blk[half:]:
            self.sg_of_block[b] = new_s
        return True

    def move_sg_core(self, s, c):
        if self.core_of_sg[s] == c:
            return False
        self.core_of_sg[s] = c
        return True

    def swap_block(self, b1, b2):
        s1, s2 = self.sg_of_block[b1], self.sg_of_block[b2]
        if s1 == s2:
            return False
        self.sg_of_block[b1], self.sg_of_block[b2] = s2, s1
        self.blocks_in_sg[s1].discard(b1)
        self.blocks_in_sg[s2].discard(b2)
        self.blocks_in_sg[s1].add(b2)
        self.blocks_in_sg[s2].add(b1)
        return True

    def num_used_sg(self):
        return sum(1 for x in self.blocks_in_sg if x)

    def validate(self, model, num_cores, max_subgraphs=None):
        """检查块覆盖、核编号和子图商图无环，不修改当前解。"""
        nb = len(getattr(model, "blocks", ()))
        if nb == 0 or len(self.sg_of_block) != nb or not self.core_of_sg:
            return False
        if any(s < 0 or s >= len(self.core_of_sg)
               for s in self.sg_of_block):
            return False
        if any(c < 0 or c >= num_cores for c in self.core_of_sg):
            return False
        used = set(self.sg_of_block)
        if max_subgraphs is not None and len(used) > max_subgraphs:
            return False
        succ = [set() for _ in self.core_of_sg]
        for edge in getattr(model, "block_edges", ()):
            if len(edge) == 2 and isinstance(edge[0], (tuple, list)):
                u, v = edge[0]
            else:
                u, v = edge[:2]
            su, sv = self.sg_of_block[int(u)], self.sg_of_block[int(v)]
            if su != sv:
                succ[su].add(sv)
        indeg = [0] * len(succ)
        for out in succ:
            for v in out:
                indeg[v] += 1
        stack = [i for i, d in enumerate(indeg) if d == 0]
        seen = 0
        while stack:
            u = stack.pop()
            seen += 1
            for v in succ[u]:
                indeg[v] -= 1
                if indeg[v] == 0:
                    stack.append(v)
        return seen == len(succ)


class Context:
    """搜索上下文：模型、场景、核数、适应度权重与评估缓存。
    screen_enabled=True 时启用 Mamba 式 SSM 粗筛层：邻域扰动先经
    轻量代理预测，只放行预测改进者做完整仿真。"""

    def __init__(self, model, num_cores, scene, traffic_weight=0.2,
                 screen_enabled=False, seed=0):
        self.model = model
        self.num_cores = num_cores
        self.scene = scene
        self.traffic_weight = traffic_weight
        self.n_evals = 0
        from construct import block_graph, upward_ranks
        self.bsuccs, self.bpreds, self.btraffic = block_graph(model)
        self.brank, self.bdur = upward_ranks(model)
        self.nb = len(model.blocks)
        self.block_core_affinity = self._block_affinity()
        self.mean_block_traffic = (sum(self.btraffic.values()) /
                                   max(1, len(self.btraffic)))
        self.mean_block_dur = (sum(self.bdur) / max(1, self.nb))
        # 逐案真值校正：mk 分量乘子（大用例在线校正后 >1）
        self.fitness_bias = 1.0
        # 完整评估后刷新的轻量快照（特征提取用，非完整重算）
        self._snap = None
        # Mamba 式粗筛层
        self.screen_enabled = screen_enabled
        if screen_enabled:
            from mamba_screen import MambaScreen
            self.screen = MambaScreen(feat_dim=16, state_dim=32, seed=seed)
        else:
            self.screen = None

    # ------------------------------------------------------- 筛选式评估
    FEAT_DIM = 16

    def _move_features(self, sol, mv):
        """邻域扰动的廉价特征（O(块度数)，不做完整流量/CP/仿真重算）。"""
        f = [0.0] * self.FEAT_DIM
        if mv is None:
            return f
        kind = mv[0]
        # 快照缺失时用保守中性特征
        K = max(1, sol.num_used_sg())
        core_load = self._snap['core_load'] if self._snap else [0.0] * self.num_cores
        mean_load = sum(core_load) / max(1, len(core_load)) or 1.0
        imb = (max(core_load) - min(core_load)) / mean_load
        if kind in ('move_block', 'merge_edge'):
            b = mv[1]
            tgt = sol.sg_of_block[mv[2]] if kind == 'merge_edge' else mv[2]
            f[0] = 1.0
            f[5] = max(self.model.block_work_m[b], self.model.block_work_v[b]) / max(1e-9, self.mean_block_dur)
            aff = 0.0
            tot = 0.0
            for v, w in self.block_core_affinity.get(b, {}).items():
                tot += w
                if sol.sg_of_block[v] == tgt:
                    aff += w
            f[6] = aff / max(1.0, self.mean_block_traffic)
            f[7] = tot / max(1.0, self.mean_block_traffic)
            f[8] = aff / max(1.0, tot)
            f[9] = len(sol.blocks_in_sg[tgt]) / max(1.0, self.nb / K)
            f[13] = 1.0 if (self._snap and tgt < len(sol.core_of_sg) and
                            sol.core_of_sg[tgt] == sol.core_of_sg[sol.sg_of_block[b]]) else 0.0
        elif kind == 'move_core':
            f[1] = 1.0
            s = mv[1]
            f[9] = len(sol.blocks_in_sg[s]) / max(1.0, self.nb / K)
            f[13] = 0.0
        elif kind == 'merge':
            f[2] = 1.0
            s1, s2 = mv[1], mv[2]
            f[9] = len(sol.blocks_in_sg[s1]) / max(1.0, self.nb / K)
            f[10] = len(sol.blocks_in_sg[s2]) / max(1.0, self.nb / K)
            f[13] = 1.0 if sol.core_of_sg[s1] == sol.core_of_sg[s2] else 0.0
        elif kind == 'split':
            f[3] = 1.0
            s = mv[1]
            f[9] = len(sol.blocks_in_sg[s]) / max(1.0, self.nb / K)
        elif kind == 'swap':
            f[4] = 1.0
            b1 = mv[1]
            f[5] = max(self.model.block_work_m[b1], self.model.block_work_v[b1]) / max(1e-9, self.mean_block_dur)
        f[11] = K / max(1, self.nb)
        f[12] = imb
        f[14] = self._snap['log_fit'] if self._snap else 0.0
        f[15] = self._snap['phase'] if self._snap else 0.0
        return f

    def predict_move(self, sol, mv):
        """SSM 预测该扰动的改进量（不做完整仿真）。返回 (y_pred, phi)。"""
        x = self._move_features(sol, mv)
        y, phi = self.screen.predict(x)
        return y, phi

    def eval_and_learn(self, sol, phi, f_cur):
        """完整仿真 + 在线学习。返回 (fitness, mk, added)。"""
        mk, added, info = self.evaluate(sol)
        f_new = self.fitness(mk, added)
        improved = f_new < f_cur - 1e-9
        y_true = (f_cur - f_new) / max(1e-9, abs(f_cur))
        self.screen.learn(phi, y_true)
        self.screen.record_admit(improved)
        if improved:
            self.screen.commit(phi)
        return f_new, mk, added

    def screen_skip(self, n=1):
        self.screen.stats['skipped'] += n

    @staticmethod
    def _wrap(cand, mv, f_new, mk_new, ad_new):
        return (cand, f_new, mk_new, ad_new, mv)

    def screen_probe(self, phi, f_cur, sol):
        """ε 探索：对未入选候选做完整仿真以估计漏筛率并补充训练数据。"""
        mk, added, _ = self.evaluate(sol)
        f_new = self.fitness(mk, added)
        y_true = (f_cur - f_new) / max(1e-9, abs(f_cur))
        self.screen.learn(phi, y_true)
        self.screen.record_skip_probe(f_new < f_cur - 1e-9)
        return f_new, mk, added

    def update_snapshot(self, sol, fitness, phase=0.0):
        """完整评估后刷新轻量快照（供特征提取）。"""
        core_load = [0.0] * self.num_cores
        for b in range(self.nb):
            s = sol.sg_of_block[b]
            if s < len(sol.core_of_sg):
                core_load[sol.core_of_sg[s]] += max(
                    self.model.block_work_m[b], self.model.block_work_v[b])
        import math as _m
        self._snap = {'core_load': core_load,
                      'log_fit': _m.log10(max(1.0, abs(fitness))),
                      'phase': phase}

    def _block_affinity(self):
        """块与块间的通信亲和度（供邻域选择偏向高流量边界）。"""
        aff = defaultdict(dict)
        for (u, v), w in self.btraffic.items():
            if w > 0:
                aff[u][v] = w
                aff[v][u] = w
        return aff

    def _block_out_traffic(self, b):
        tot = 0.0
        for v, w in self.block_core_affinity.get(b, {}).items():
            tot += w
        return max(1.0, tot / 2.0)

    def evaluate(self, sol, use_cache=True):
        sol.compact()
        self.repair(sol)                 # 商图有环则合并 SCC，保证方案合法
        sol.compact()
        mk, added, info = self.model.evaluate(
            sol.sg_of_block, sol.core_of_sg, self.scene, self.num_cores,
            use_cache=use_cache)
        self.n_evals += 1
        return mk, added, info

    def repair(self, sol):
        """商图环修复：反复把非平凡 SCC 合并成单子图直至无环。"""
        for _ in range(64):
            K = len(sol.core_of_sg)
            if K <= 1:
                return
            succs = [set() for _ in range(K)]
            for (bi, bj), _w in self.model.block_edges:
                si, sj = sol.sg_of_block[bi], sol.sg_of_block[bj]
                if si != sj:
                    succs[si].add(sj)
            sccs = _tarjan_scc(succs)
            bad = [scc for scc in sccs if len(scc) > 1]
            if not bad:
                return
            for scc in bad:
                tgt = min(scc)
                for s in scc:
                    if s == tgt:
                        continue
                    sol.blocks_in_sg[tgt] |= sol.blocks_in_sg[s]
                    for b in sol.blocks_in_sg[s]:
                        sol.sg_of_block[b] = tgt
                    sol.blocks_in_sg[s] = set()

    def fitness(self, mk, added):
        return mk * self.fitness_bias + self.traffic_weight * added / BW

    def eval_fitness(self, sol):
        mk, added, _ = self.evaluate(sol)
        return self.fitness(mk, added), mk, added

    def random_move(self, sol, rng):
        """随机邻域动作，返回 (动作描述, 是否成功)。"""
        used = [i for i, x in enumerate(sol.blocks_in_sg) if x]
        if not used:
            return None, False
        kind = rng.random()
        if kind < 0.40:                     # 移动块
            b = rng.randrange(self.nb)
            s = rng.choice(used)
            ok = sol.move_block(b, s)
            return ('move_block', b, s), ok
        elif kind < 0.55:                   # 换核
            s = rng.choice(used)
            c = rng.randrange(self.num_cores)
            ok = sol.move_sg_core(s, c)
            return ('move_core', s, c), ok
        elif kind < 0.75:                   # 合并子图
            if len(used) < 2:
                return None, False
            s1, s2 = rng.sample(used, 2)
            ok = sol.merge_sg(s1, s2)
            return ('merge', s1, s2), ok
        elif kind < 0.90:                   # 拆分子图
            s = rng.choice(used)
            ok = sol.split_sg(s, self.brank)
            return ('split', s), ok
        else:                               # 交换块
            b1 = rng.randrange(self.nb)
            b2 = rng.randrange(self.nb)
            ok = sol.swap_block(b1, b2)
            return ('swap', b1, b2), ok

    def targeted_move(self, sol, rng):
        """偏向通信边界的移动：优先搬移高流量边界上的块。"""
        used_set = {i for i, x in enumerate(sol.blocks_in_sg) if x}
        # 随机选一条跨子图高流量边
        edges = []
        for b in range(self.nb):
            s = sol.sg_of_block[b]
            for v, w in self.block_core_affinity.get(b, {}).items():
                sv = sol.sg_of_block[v]
                if sv != s and w > 0:
                    edges.append((w, b, v))
        if edges and rng.random() < 0.8:
            edges.sort(reverse=True)
            w, b, v = edges[rng.randrange(min(20, len(edges)))]
            if rng.random() < 0.5:
                b, v = v, b
            ok = sol.move_block(b, sol.sg_of_block[v])
            return ('merge_edge', b, v), ok
        return self.random_move(sol, rng)
