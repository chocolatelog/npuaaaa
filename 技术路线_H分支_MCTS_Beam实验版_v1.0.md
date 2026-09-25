# H 分支 MCTS + Beam 混合搜索实验路线

版本：`H-v2.2-guided-search`

日期：2026-09-25

## 1. 定位

本路线在无 MCTS 正式流程之后追加两个受限候选生成器：核心分配 PUCT
搜索和结构 Beam Search。原禁忌搜索、模拟退火、蚁群、MOPSO、LRU 复排
及官方候选不减少，新增搜索只追加候选，最终仍由官方评估器和 warm-start
非回退规则选择方案。

## 2. 搜索流程

```text
无 MCTS 正式流程完整运行
  -> 固定当前最优分区作为两路搜索根节点
  -> PUCT：子图换核 / 子图核交换
  -> Beam：换核 / 核交换 / 关键边块移动 / 子图合并 / 子图拆分
  -> 两路独立归档和 top-K 候选池
  -> 原候选优先官方核验
  -> MCTS / Beam 候选使用独立追加配额
  -> warm-start 官方复核与非回退选择
```

## 3. 关键实现

- PUCT 价值为相对根节点的归一化代理收益，限制在 `[-1, 1]`。
- transposition table 共享重复状态统计，并用最大评估数和停滞阈值终止。
- Beam 在相同代理评估预算下保留负载/子图数量不同的多样化前沿。
- `A/3`、`A/4`、`B/4` 使用 192 次代理评估，其他组合使用 64 次。
- 高收益且不超过 6000 op 的任务追加 8 个官方候选，其他任务追加 4 个；
  同时启用两路搜索时平均分配给 MCTS 和 Beam。
- 每个官方候选记录来源、代理 makespan、代理搬运量、官方 makespan、
  官方搬运量以及是否最终被选择。

核心文件：

- `solver/mcts_schedule.py`
- `solver/beam_schedule.py`
- `solver/pipeline.py`
- `solver/run_all.py`

## 4. 使用方式

```cmd
.venv\Scripts\python.exe -m solver.run_all --cases 1-100 --cores 2,3,4,5 --scenes A,B --workers 8 --seed 0 --verify-k 24 --mcts --beam --warm-start-dir results\plans_verify24_full --output-dir results\plans_guided_full --log-file results\solve_log_guided_full.jsonl
```

`--mcts` 和 `--beam` 默认关闭，因此不会改变既有正式路线。实验输出必须使用
独立目录，不能覆盖无 MCTS 全量方案。

## 5. 适用结论

96 项代表集达到 +0.75995%，但 648 项官方全量配对增量为 +0.27538%。
该路线确认结构搜索有效且在 warm-start 保护下 0 退化，但代表集明显高估
全量收益。因此当前应保留为可选增强模块，不直接替换无 MCTS 正式默认流程。
