# 第二篇综述（开源可复现实现）× 本系统对照

来源：《近两年通用神经网络处理器多核调度：开源论文、可复现实现与技术路线调研》
（2024-01~2026-09，10 个开源项目：LATTICE/DAN-Scheduler、MAGIS、DOPS、Tessel、
ONNXim、NeuPIMs、Mirage、GrapheonRL、XSched、ACT/TAIDL）
前次分析见 `survey_gap_analysis.md`（第一篇综述，方法趋势版），本篇侧重可复现工程。

## 一、综述推荐的四条路线 vs 本系统

| 综述优先路线 | 核心思想 | 本系统现状 | 结论 |
|---|---|---|---|
| LATTICE（场景B核心） | memory-pressure-aware 排序 + 生命周期 + spill/reload + Compute-DMA overlap | L1/UB 双池驻留 + NNLS 校准 spill；op 级 LRU 仿真**已试并回退**（排序 0.295 < 启发式 0.75）；overlap 因子**已试无排序收益** | ✅ 已吸收可用部分；综述原话"不用真实模拟地址，只算近似 live set"反向验证了我们的回退决策 |
| MAGIS（图变换-调度联合） | Mutator→Scheduler→Simulator→Optimizer 闭环，mutator={merge,split,move,swap}，支持 time budget | **结构完全一致**：四动作邻域 + 代理仿真 + TS/SA/ACO/MOPSO + 时间预算流水线 | ✅ 已等价实现（独立重写，未用其 Torch backend） |
| DOPS（统一搜索框架） | task_graph + hardware cost model + HEFT baseline + 候选分层精评 | 构造池含 HEFT；代理→真值分层择优（≤1.2万op 进程内官方评估） | ✅ 已等价且闭环更强 |
| Tessel（placement 后 schedule search） | **固定 placement 专搜 execution order**，再交替 Partition(k)→Schedule(k)→Partition(k+1) | ❌ **未做**：我们的每核顺序由代理事件仿真一次性派生，从未把 core_schedules 作为独立决策变量搜索 | ⚠️ **新差距（本篇最大增量）** |

综述与第一篇一致的两条原则，我们均已满足且更深：
- "官方 Evaluator 是 Judge 不是每步代价函数"（我们：keep-best 单调协议 + 进程内真值校验）；
- "GNN/RL 只做候选动作排序器"（我们：Mamba-SSM 在线 ΔMakespan 排序器，A/B 验证省 40% 仿真）。

## 二、新识别的改进空间（本篇增量，按价值排序）

### 1. Tessel 式交替优化：core_schedules 作为独立搜索变量 ⭐ 新增
现状：顺序 = 代理仿真派生（最早就绪优先），合法但非搜索对象。
方案：固定 node_to_subgraph，对每核顺序做轻量 beam search——只允许交换
**互相独立**的相邻子图（保拓扑合法），场景 A 收益最大（每对跨核依赖
1000 cycles 等待，顺序不同等待暴露不同）；然后与切图交替。
成本：代理已能评估任意合法顺序（plan_from 接受外部 orders，需小改）。
预期：场景 A 中间带 +2~5%，场景 B 较小（同核 Task 内顺序影响驻留与流水）。
风险：低——keep-best 兜底。

### 2. FIFO Reuse-Distance 修正 P_hit（问题 3 代理精化）
现状：场景 C 判据是"张量 ≤1MB 且跨核读者 >1 即命中"（过粗）。
综述公式：ReuseDistance(t) = 两次访问间进入 L2 的字节数；<1MB 才大概率命中；
Benefit(t) ≈ size×(Nread−1)×P_hit×(1/BW_DDR − 1/BW_L2)。
方案：按各核对共享输入的**访问时间序**（代理仿真时间线已有时序）累积
插入字节，估计 FIFO 存活率得 P_hit，替换布尔判据。
预期：P3 代理排序精度提升 → L2 感知聚散决策（C2）才可靠；L2 加速比
现状 1.009~1.035 → 向综述"明显效果"线 1.08 靠近。

### 3. 词典序接受准则（Makespan 严格优先）
现状：标量 `mk + 0.2·added/60`。
综述建议接受解用 `F(S)=(Makespan, AddedDDR)` 词典序，搜索阶段才用代理加权和。
方案：真值校验选优时先比 Makespan、并列（±0.3%）才比 added。
成本：极低；避免 added 折算噪声翻盘 Makespan 差异。

### 4. 离线动作排序器（XGBoost + TopKRecall）
综述与本系统都已有在线版（Mamba-SSM）；本篇新增可落地细节：
用已积累的 (图, 动作, ΔMakespan) 数据（800 任务 × 每任务数千次评估）
离线训练，验收用 **TopKRecall**（好候选是否进 Top-K）而非 MSE。
预期：粗筛跳过率从 75% 提升且漏筛率下降 → 等预算下更多有效迭代。

### 5. 消融与分组报告（论文向，纯分析）
综述列的四层基线消融（B1 纯计算/B2 通信感知/B3 +内存/Full）、
λ 扫描 Pareto 图、L2 三态消融、按图结构分组报告——我们数据已齐，
其中 L2 三态消融是证明"提升来自算法而非免费 L2"的关键。

## 三、本篇综述明确背书的两项既有决策

1. **"没有一个现成系统同时实现本题完整硬件模型"** → 我们的场景 C
   代理（L2 双池）+ FIFO 复用距离是**赛题定制创新点**，不是重复造轮子；
2. **"所有 paper cost model 只是 surrogate，官方 evaluator 是 oracle"**
   → 我们的 keep-best 单调协议（两轮 450/800 替换、零退化）是该原则的
   严格实现，比综述建议的"Top-K 精评"更彻底。

## 四、已证伪方向（本篇综述无法预判，我们有数据）

| 尝试 | 综述预期 | 实测 | 教训 |
|---|---|---|---|
| op 级 LRU spill 仿真 | "精确模拟成本高，需增量更新" | 排序 0.295，级联重读被低估 | 启发式+真值校准更稳 |
| overlap 因子 | "重叠上界过于乐观应折减" | 排序相关无变化 | 折减方向对但量级与官方步进3流水不匹配 |
| NetBenefit 凝聚切分 | MAGIS 式合并收益公式 | 块号/拓扑交错图上非凸，深图劣化 2.4× | 路径式凸增长（heft）是深图正确基元 |

## 五、行动清单（合并两篇综述后的最终待办）

1. Tessel 交替顺序搜索（新，场景 A 主攻）—— 中等工作量；
2. FIFO 复用距离 P_hit（新，问题 3 主攻）—— 中等；
3. 词典序真值选优 —— 极小；
4. 离线 XGBoost 动作排序器 + TopKRecall 验收 —— 中等（数据已备）；
5. 消融/分组/oracle 上界报告 —— 小（纯分析）。
不采纳：GNN+RL 端到端（两篇综述一致评高风险）、硬件改动类（NVR 等）、
直接嵌入外部模拟器（ONNXim 等，题目已有官方模拟器）。
