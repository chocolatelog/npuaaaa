"""多目标 Pareto 归档：支配过滤 + 拥挤距离。"""


def dominates(a, b):
    """a 支配 b：(mk, added) 双目标最小化。"""
    return ((a[0] <= b[0] and a[1] <= b[1]) and (a[0] < b[0] or a[1] < b[1]))


class ParetoArchive:
    """保存 (mk, added, sol) 的非支配集，供 MOPSO 选领袖和最终决策。"""

    def __init__(self, cap=64):
        self.cap = cap
        self.items = []   # [ (mk, added, sol) ]

    def insert(self, mk, added, sol):
        for (m2, a2, _) in self.items:
            if (m2 <= mk and a2 <= added) and (m2 < mk or a2 < added):
                return False               # 被现有解支配
        self.items = [x for x in self.items
                      if not (mk <= x[0] and added <= x[1]
                              and (mk < x[0] or added < x[1]))]
        self.items.append((mk, added, sol.clone()))
        if len(self.items) > self.cap:
            self._prune()
        return True

    def _crowding(self, item, idx):
        vals = sorted(x[idx] for x in self.items)
        lo = min(vals)
        hi = max(vals)
        if hi - lo < 1e-12:
            return 0.0
        i = vals.index(item[idx])
        if i == 0 or i == len(vals) - 1:
            return float('inf')
        return (vals[i + 1] - vals[i - 1]) / (hi - lo)

    def _prune(self):
        scored = []
        for it in self.items:
            c = self._crowding(it, 0) + self._crowding(it, 1)
            scored.append((c, it))
        scored.sort(key=lambda x: -x[0])
        self.items = [it for _, it in scored[:self.cap]]

    def best_by_scalar(self, traffic_weight, bw=60.0):
        """按 makespan + w*added/bw 选最终解。"""
        best, best_f = None, None
        for mk, added, sol in self.items:
            f = mk + traffic_weight * added / bw
            if best_f is None or f < best_f:
                best_f, best = f, (mk, added, sol)
        return best

    def front(self):
        return sorted((mk, added) for mk, added, _ in self.items)
