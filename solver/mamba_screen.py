"""Mamba(S6) 式选择性状态空间轻量代理评估器（候选解粗筛层）。

作用：在 SA/TS 的每次邻域扰动后，先用极廉价的特征 + 递推隐状态预测
"该扰动是否会改进适应度"，只对预测为正的候选执行完整代理仿真，
从而在同等墙钟预算下多做迭代、或同等质量下减少完整仿真次数。

结构（Mamba 风格的选择性 SSM，输入依赖门控 + 在线训练读出）：
    a(x) = σ(A₀ + Uₐ·x)          # 选择性衰减门 (S 维, 输入依赖)
    b(x) = tanh(U_b·x)            # 选择性输入门 (S 维, 输入依赖)
    h_t = a(x) ⊙ h_{t-1} + b(x)   # 离散化递推 (ZOH 简化)
    φ = [h_t ; x]                 # 读出特征
    ŷ = w·φ                       # 预测改进量(归一化)
    w 由 NLMS 在线更新（每次完整仿真得到真值标签后立即学习）

门控投影 Uₐ/U_b 为固定随机投影（selective 但免反向传播），读出层在线自适应，
兼顾稳定性与低开销（每步 ~百次乘加）。
"""
import math
import random


class MambaScreen:
    def __init__(self, feat_dim, state_dim=32, seed=0, lr=0.6,
                 target_precision=0.70, tau=0.0, tau_step=0.02,
                 eps=0.15, eps_min=0.05, eps_decay=0.9995):
        rng = random.Random(seed)
        self.F = feat_dim
        self.S = state_dim
        # 固定随机投影（选择性门控）
        scale = 1.0 / math.sqrt(feat_dim)
        self.Ua = [[rng.gauss(0, scale) for _ in range(feat_dim)]
                   for _ in range(state_dim)]
        self.Ub = [[rng.gauss(0, scale) for _ in range(feat_dim)]
                   for _ in range(state_dim)]
        # 每个状态维的基础衰减（对数均匀 in (0.05, 0.98)）
        self.A0 = [math.log(0.05 + (0.93 * s / max(1, state_dim - 1)))
                   for s in range(state_dim)]
        self.h = [0.0] * state_dim
        # 读出层 w 与 NLMS
        self.w = [0.0] * (state_dim + feat_dim)
        self.lr = lr
        # 特征/标签归一化统计
        self.f_mean = [0.0] * feat_dim
        self.f_m2 = [1.0] * feat_dim
        self.n_seen = 0
        self.y_scale = 1.0
        # 粗筛策略
        self.tau = tau
        self.tau_step = tau_step
        self.target_precision = target_precision
        self.eps = eps
        self.eps_min = eps_min
        self.eps_decay = eps_decay
        # 统计
        self.stats = {'admitted': 0, 'skipped': 0, 'forced': 0,
                      'admit_improved': 0, 'skip_would_improve': 0,
                      'learn': 0}
        self._prec_ema = 0.5      # 滑动窗口精度（tau 自适应用）
        self._n_admit_ema = 0

    # ------------------------------------------------------------ 前向
    def _normalize(self, x):
        self.n_seen += 1
        n = self.n_seen
        out = [0.0] * self.F
        for i, v in enumerate(x):
            if not math.isfinite(v):
                v = 0.0
            d = v - self.f_mean[i]
            self.f_mean[i] += d / n
            self.f_m2[i] += d * (v - self.f_mean[i])
            var = self.f_m2[i] / n
            std = math.sqrt(var) if var > 1e-9 else 1.0
            out[i] = max(-3.0, min(3.0, d / std))
        return out

    def _gates(self, xn):
        a = [0.0] * self.S
        b = [0.0] * self.S
        for s in range(self.S):
            ua = self.Ua[s]
            ub = self.Ub[s]
            za = self.A0[s]
            zb = 0.0
            for i, v in enumerate(xn):
                za += ua[i] * v
                zb += ub[i] * v
            a[s] = 1.0 / (1.0 + math.exp(-za))       # σ
            b[s] = math.tanh(zb)
        return a, b

    def _phi(self, x):
        xn = self._normalize(x)
        a, b = self._gates(xn)
        h_new = [a[s] * self.h[s] + b[s] for s in range(self.S)]
        # 预测不推进正式隐状态（候选可能被拒绝），由 commit() 决定
        return xn, h_new

    def predict(self, x):
        """返回 (预测改进值, phi)。不修改隐状态。"""
        xn, h_new = self._phi(x)
        phi = h_new + xn
        y = sum(w * v for w, v in zip(self.w, phi))
        return y, (xn, h_new)

    def commit(self, phi):
        """接受该候选（进入完整评估或成为当前解）时推进隐状态。"""
        _, h_new = phi
        self.h = h_new

    def decide(self, y, rng):
        """粗筛决策：放行 / 跳过（ε 探索放行记 forced）。"""
        if y > self.tau:
            return True, False
        if rng.random() < self.eps:
            self.eps = max(self.eps_min, self.eps * self.eps_decay)
            return True, True
        return False, False

    # ------------------------------------------------------------ 在线学习
    def learn(self, phi, y_true):
        """NLMS 更新读出层。y_true: 归一化前的改进比例 (f_cur-f_new)/|f_cur|。"""
        xn, h_new = phi
        feats = h_new + xn
        # 标签尺度跟踪
        ay = abs(y_true)
        self.y_scale = 0.98 * self.y_scale + 0.02 * max(ay, 1e-6)
        t = max(-3.0, min(3.0, y_true / max(1e-6, self.y_scale)))
        yhat = sum(w * v for w, v in zip(self.w, feats))
        err = t - yhat
        norm = sum(v * v for v in feats) + 1e-3
        step = self.lr * err / norm
        for i, v in enumerate(feats):
            self.w[i] += step * v
        self.stats['learn'] += 1

    def record_admit(self, improved):
        self.stats['admitted'] += 1
        if improved:
            self.stats['admit_improved'] += 1
        # 滑动窗口精度（半衰期 ~32 次），供 tau 快速自适应
        beta = math.exp(-1.0 / 32.0)
        self._prec_ema = beta * self._prec_ema + (1 - beta) * (1.0 if improved else 0.0)
        self._n_admit_ema = beta * self._n_admit_ema + (1 - beta)
        if self._n_admit_ema >= 5:
            if self._prec_ema < self.target_precision - 0.05:
                self.tau = min(1.0, self.tau + self.tau_step)
            elif self._prec_ema > self.target_precision + 0.10:
                self.tau = max(-0.5, self.tau - self.tau_step)

    def record_skip_probe(self, would_improve):
        """ε 强制评估的被拒候选的真实结果（估计漏筛率）。"""
        self.stats['forced'] += 1
        if would_improve:
            self.stats['skip_would_improve'] += 1

    @property
    def precision(self):
        n = self.stats['admitted']
        return self.stats['admit_improved'] / n if n else float('nan')

    @property
    def miss_rate(self):
        n = self.stats['forced']
        return self.stats['skip_would_improve'] / n if n else float('nan')

    @property
    def skip_ratio(self):
        n = self.stats['admitted'] + self.stats['skipped']
        return self.stats['skipped'] / n if n else 0.0
