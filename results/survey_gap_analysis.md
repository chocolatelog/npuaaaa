# 近两年方法综述 × 本系统对照（差距分析）

来源：《近两年多核 NPU / AI 加速器调度与硬件感知编译研究综述》（用户提供）
对照对象：本系统实测数据（1300 项官方评估 + 1200 项基线评估 + 648 真值样本）

## 一、综述的方法脉络（2024→2026）

| 年份 | 主题 | 代表工作 | 核心方法 |
|---|---|---|---|
| 2024 | 搜哪个并行计划 | nnScaler (OSDI'24) | op-trans/op-assign/op-order 三原语 + 约束裁剪 |
| 2025 | Memory + 异构流水 | MAS-Attention (MLSys'25), PipeThreader (OSDI'25), CaMDN | matrix/vector 双流、软件流水、cache-aware mapping |
| 2026 | 显式 Cost Model + Memory Plan + Schedule 联合约束 | LATTICE, NPUMeter (TACO'26), HyperParallel-MoE | 白盒解析模型 + 内存生命周期规划 + 关键路径精修 |

**综述的核心判断**：不要押纯 GNN+RL；主线是
"硬件感知图划分 + 关键路径列表调度 + 局部搜索 + 内存生命周期 + 廉价代理择优"，
学习模块只做可插拔的搜索加速器。官方评估器是 Judge，不是每步搜索的代价函数。

## 二、本系统已对齐（甚至超出）的部分

| 综述推荐 | 本系统现状 | 对齐度 |
|---|---|---|
| 廉价代理筛 10^4~10^6 候选 → Top-K 真值裁决 | 毫秒级代理（搬运量与官方逐字节一致，mk 排序相关 0.953）+ ≤12kop 进程内真值校验 top-5 | ✅ 超出（真值校验直接进搜索末端） |
| 通信感知划分 + List Scheduling + 局部搜索 | HEFT/条带/链式构造 + TS/SA/ACO/MOPSO | ✅ 主线一致 |
| Merge/Split/Move/Swap 四类动作 | 完全相同的四类邻域 + 通信定向变体 | ✅ 完全一致 |
| 多起点（CP 偏置/通信偏置/均衡偏置） | 9 个构造解池（三种偏置×多种规模） | ✅ |
| Memory-pressure 模型（LATTICE 思想） | L1/UB 双池驻留峰值 + 内部总量 + NNLS 校准 spill | ✅ 粗粒度版 |
| 学习型 ΔCost Model（推荐 XGBoost/线性） | Mamba-SSM 在线 NLMS 读出，预测单动作 ΔMakespan，A/B 验证省 40% 完整仿真 | ✅ 在线版 |
| 结果口径（官方评估器为唯一准绳） | 全部结果来自官方 CLI/进程内调用 | ✅ |
| B0 随机基线 | stub 1200 项对比已完成 | ✅ |

**结果对照综述建议目标**：
- 场景 A 5 核：本系统 **2.95**，综述"做得不错"档 2.8-3.2 ✅（冲击档 3.4-3.6 = 我们规划的 3.2-3.4，接近）
- 场景 B 5 核：本系统 **3.24**，综述第一阶段目标 3.1-3.6 ✅ 落在带内
- B 相对 A 改善：本系统 2-12%（随核数增），综述 8-20% ⚠️ 低核数未达标
- spill 相比无 memory-aware 版降低 ≥20%：未做该消融 ⚠️

## 三、差距与改进空间（按优先级）

### 差距 1：L2 利用是最大缺口（问题 3）
- 综述目标：L2/no-L2 加速比 **≥1.08**（明显效果）、较好 1.12-1.20；命中率初态 40-60%
- 本系统：L2 加速比仅 **1.009-1.035**，命中率 10-18% —— **低于综述"明显效果"门槛**
- 根因：代理完全没有 L2 双池模型（场景'C'未实现），P3 直接复用 P2 方案，等于盲跑
- 综述给出的方法（CaMDN 路线，与我们 C1/C2 方案吻合）：共享输入超图 +
  ReuseScore(t)=size×(#consumers−1)×P_hit（1MB FIFO 模拟器）+
  Benefit=HitBytes×(1/B_L2−1/B_DDR,eff)−DelayCost；注意"命中率不是主目标，
  为命中率破坏并行度反而变差"
- **结论：验证了我们 C1/C2 方向的正确性，且这是三问题中唯一未达综述基准线的一项**

### 差距 2：Tensor lifetime 驱动的顺序优化（LATTICE 核心）
- LATTICE 的关键观察：**同样合法的拓扑序，张量 lifetime overlap 不同 →
  L1/UB 峰值与 spill 差异巨大**
- 本系统：核内顺序完全交给官方 step1-3，只通过子图组成间接影响；
  阶段化流水切图（B3）+ 容量装箱（B1）能间接改善，但没有显式
  lifetime-aware 排序/NetBenefit 判据
- 改进：合并判据升级为综述公式
  NetBenefit = SavedCrossCoreDDR + SavedBoundaryCopy − PredictedSpill − ParallelismLoss
  （即我们 B1 的装箱收益函数，综述给出了同构表达）

### 差距 3：Pipeline overlap 建模不足
- Ascend DoubleBuffer 思想：CopyIn/Compute/CopyOut 可重叠，串行时 Vector
  利用率仅 1/3；"重叠上界往往过于乐观"
- 本系统：时长模型取 max(wm, wv, in/60, out/60, cp)=**完全重叠假设**，
  未区分"依赖链式子图（无法重叠）"与"宽子图（可重叠）"
- 改进：加 overlap_potential 因子（按子图依赖宽度估计），
  dur = max(...) + (1−overlap)·copy_time；同时补 Pipe 利用率/OverlapRatio 指标

### 差距 4：ΔCost Model 可升级为离线训练
- 综述建议特征 [ΔcomputeBalance, ΔcutBytes, ΔcriticalPath, ΔL1Peak, ΔUBPeak,
  Δreuse, ΔDDRConcurrency]，XGBoost/LightGBM，评价用 MAPE/Spearman/**TopKRecall**
- 本系统：16 特征 + 在线线性读出；我们已坐拥 1300 官方评估 + 800 任务日志的
  训练数据，可离线训练并以 TopKRecall 验收（这才是"把好候选排进 Top-K"的指标）

### 差距 5：实验体系（论文向）
- 综述要求的四层基线 B1(纯计算 List)/B2(通信感知)/B3(通信+内存) 消融未做；
- λ_comm∈{0,0.25,0.5,1,2,4} 扫描的 Pareto 图（Makespan vs Added DDR）未做；
- L2 三态消融（No-L2 / L2+原调度 / L2+cache-aware 重调度）未做——
  这是证明"提升来自算法而非免费 L2"的关键实验；
- 分结构报告（median/P25/P75/worst-case、按 Compute-bound/DDR-bound/
  Reuse-heavy 分组）未做——我们已有全部数据，纯分析工作；
- L2 命中率的 oracle 上界（每图理论可复用 COPY_IN 字节）未算。

### 不采纳项（综述自身也给出风险评级）
- GNN+RL 直接调度器（综述评"实现风险很高、比赛推荐度★★"）——不做；
- NVR 类硬件改动——题目不允许，只进 Related Work。

## 四、结论

1. **主线架构与 2026 前沿完全同向**（白盒约束感知 + 代理择优 + 局部搜索），
   综述的"HAMPS 四层"与我们已有系统结构等价，且我们的真值校验与
   单调改进协议比综述建议走得更远；
2. **唯一未达综述基准的是问题 3 的 L2 利用**（1.03 vs ≥1.08），
   根因是代理盲区，修法明确（C1/C2，与 CaMDN 思想吻合）；
3. 其余差距（lifetime 判据、overlap 建模、离线 Δmodel、消融体系）
   与我们先前的 P1/P2/P3 最优方案清单高度重合——综述为该清单提供了
   文献佐证，并新增了 NetBenefit 公式、TopKRecall 验收指标、
   L2 三态消融与 oracle 上界四个可落地细节。
