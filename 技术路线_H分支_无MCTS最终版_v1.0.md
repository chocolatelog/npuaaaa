# H 分支正式技术路线（无 MCTS）

版本：`H-v2.1-no-mcts`

更新时间：2026-09-25

## 1. 采用范围

本文件是 H 分支当前正式结果对应的技术路线。800 项全量结果和 96 项代表集结果均来自默认流程，未启用 `--mcts`，因此不包含 AlphaGo 风格树搜索的贡献。

`solver/mcts_schedule.py` 仍保留为独立研究模块，`--mcts` 默认关闭，不属于本报告的结果来源。

## 2. 正式求解流程

```text
计算图读取与块化
  -> HEFT / STRIP / CHAIN / NETBENEFIT 构造池
  -> coarse / balanced / fine 粒度
  -> R0-R3 结构精化与合法性检查
  -> 6 个默认种子；verify-k>=24 时最多 8 个结构种子
  -> 禁忌搜索 -> 模拟退火 -> 蚁群 -> 多目标粒子群
  -> Pareto 归档、候选去重
  -> 官方评估器核验
  -> warm-start 保底与非回退选择
```

核心实现位于 `solver/model.py`、`solver/construct_refine.py`、`solver/order_refine.py`、`solver/pipeline.py` 和 `solver/run_all.py`。官方评估器是最终真实性来源；超过官方评估规模上限的超大图只记为代理兜底，不计入官方平均指标。

## 3. 关键技术点

### 3.1 候选构造与搜索

- 以关键路径等级、通信代价、工作量和内存约束生成多范式、多粒度候选。
- R1/R2/R3 分别覆盖换核、真实块边界移动和合法结构交换。
- 多种子共享时间截止点和 Pareto 归档，避免只搜索单一代理最优候选。
- `verify-k=24` 时扩大官方核验候选并保留 8 个结构种子；默认 `verify-k=16` 保持兼容。

### 3.2 官方选择与保底

候选必须通过覆盖、商图无环、核编号和官方格式检查。经过官方评估后，只有 makespan 不劣于基线的候选才会替换保底方案；新增搬运字节仅作为接近 makespan 时的次目标。

### 3.3 评价口径

- 加速比：`baseline_makespan / candidate_makespan`。
- 改善率：`(baseline_makespan - candidate_makespan) / baseline_makespan`。
- 官方结果与代理结果分开统计。
- 全量汇总只报告固定基线、固定 seed 和同一候选口径下的配对结果。

## 4. 复现命令（不启用 MCTS）

```cmd
.venv\Scripts\python.exe -m pytest tests -q
.venv\Scripts\python.exe -m solver.run_all --cases 1-100 --cores 2,3,4,5 --scenes A,B --workers 8 --seed 0 --verify-k 24 --warm-start-dir results\plans_baseline_stable --output-dir results\plans_verify24_seed8 --log-file results\solve_log_verify24_full.jsonl
```

命令中没有 `--mcts`；这正是本报告全量结果的配置。若研究 MCTS，应使用独立输出目录和日志，不能覆盖本报告的结果。

## 5. 后续路线

先修正 spill / 生命周期代理和后继链评分，再在固定代表集上单独比较 MCTS 开关。只有在官方评估结果稳定优于无 MCTS 基线后，才考虑改变正式默认流程。

