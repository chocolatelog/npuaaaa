# H 分支最新技术路线

版本：`H-v2.0`

更新时间：2026-09-25

> 2026-09-25 后续专项更新：B/5 的同一 81 个官方可比案例均值已达到
> **4.284528864097812x**。新增可证计算下界、局部 CP-SAT 与原始操作级
> 流水线感知列表调度，作为独立可选实验入口，未替换本文的默认流程。
> 方法、完整配置与独立重放结果见 [数学强化实验记录](实验记录_H分支_数学强化_v1.0.md)。
> 本文原有 800 项结果保留为历史记录，不代表对全部场景重新实验。

## 1. 本轮定位

H 分支在主流程的构造池、多种子搜索和官方评估桥接基础上，扩大了官方候选核验范围，并保留了结构差异更大的初始种子。全量 800 项实验采用非回退选择规则：候选只有在官方评估中不劣于对应基线时才替换基线方案。

本轮主结论来自 `verify-k=24 + 8 seeds` 流程；AlphaGo 风格 MCTS 是新增的可选研究模块，默认关闭，尚未作为 800 项主结果的必要条件。

## 2. 求解数据流

```text
图模型与块化
  -> HEFT / STRIP / CHAIN / NETBENEFIT 构造池
  -> coarse / balanced / fine 粒度与 R0-R3 精化
  -> 6 个默认或 8 个高核验结构种子
  -> 禁忌搜索 -> 模拟退火 -> 蚁群 -> 多目标粒子群
  -> Pareto 归档与候选去重
  -> 官方评估器核验
  -> 基线保底、非回退选择
```

`solver/pipeline.py` 负责单任务流程，`solver/run_all.py` 负责任务展开、断点恢复和参数传递，`solver/evaluate_official.py` 是最终真实性来源。超大图无法在预算内完成官方评估时单独标记为代理兜底，不混入官方平均指标。

## 3. 本轮新增和调整

### 3.1 扩大候选核验与结构种子

- `verify_k` 控制送入官方评估器的候选数，默认保持 16。
- `verify_k >= 24` 时保留最多 8 个结构种子；默认流程仍保留 6 个。
- 小图、中图的候选复核预算和 spill 复排预算随 `verify_k` 联动。
- 通过 `warm_start_dir` 提供已评估方案作为保底，避免搜索结果回退。

### 3.2 AlphaGo 风格有界 PUCT 搜索

`solver/mcts_schedule.py` 实现轻量 MCTS：

- 状态：子图到核的合法映射；动作：关键子图换核和关键子图核交换。
- 先验：关键路径等级、子图工作量和目标核负载。
- 树策略：带先验的 PUCT 选择，叶节点由现有代理适应度评估。
- 去重：按 `sg_of_block` 与 `core_of_sg` 签名去重。
- 约束：每一步执行方案合法性检查和压缩，限定搜索深度、分支数和剩余时间。
- 接口：`solve_case(..., use_mcts=True)` 或命令行 `--mcts`。

该模块只在剩余时间内使用约 30% 的预算，若异常则记录错误并回退到原流程；默认关闭，以保证旧流程可复现。

## 4. 评价口径

- `makespan` 是第一目标，越小越好。
- 新增搬运字节是第二目标，仅在 makespan 接近时比较。
- 加速比为 `baseline_makespan / candidate_makespan`。
- 改善率为 `(baseline_makespan - candidate_makespan) / baseline_makespan`。
- 官方可评估结果与超大图代理结果分开统计。

## 5. 已知边界

MCTS 当前是基于代理评估的局部树搜索，不是训练式 AlphaZero，也不包含策略网络、价值网络或自我对弈。它适合探索关键子图的换核和交换，不应在没有官方复核的情况下直接宣称真实加速比提升。下一步应先用固定代表集比较 `--mcts` 与关闭 MCTS 的配对结果，再决定是否进入全量默认流程。

## 6. 复现实验入口

```cmd
.venv\Scripts\python.exe -m pytest tests -q
.venv\Scripts\python.exe -m solver.run_all --cases 1-100 --cores 2,3,4,5 --scenes A,B --workers 8 --seed 0 --verify-k 24 --warm-start-dir results\plans_baseline_stable --output-dir results\plans_verify24_seed8 --log-file results\solve_log_verify24_full.jsonl
```

研究 MCTS 时，在同样的算例、核数、场景、seed 和保底目录下追加 `--mcts`，并使用独立输出目录和日志文件。
