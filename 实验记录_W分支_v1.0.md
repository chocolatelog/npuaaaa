# W 分支实验记录

版本：`W-fix-v1.0.0`

所属分支：`W`

更新时间：2026-09-24 17:21

## 1. 本轮目标

验证 W 分支完成的构造池改造和多种子主流程，确认上一版细粒度候选爆炸问题已得到控制，并为后续生命周期和前瞻改进保留可复现实验基线。

## 2. 本轮代码变化

1. 新增 `solver/construct_refine.py`，建立范式 × 粒度 × 场景 × 精化入口。
2. 修改 `solver/pipeline.py`，正式传入 `Context`，保留最多 6 个结构多样化种子。
3. 修改 `solver/construct.py` 和 `solver/construct_v2.py`，增加子图规模和 NETBENEFIT 粒度控制。
4. 修改 `solver/model.py`，补齐 `spill_bytes` 结果字段。
5. 修改 `solver/solution.py`，增加方案覆盖、商图无环和核编号验证。
6. 新增 `solver/validate_construct_pool.py` 和 `tests/test_construct_refine.py`。

## 3. 验证命令

```cmd
cd /d J:\数学建模\改进 && python -m pytest tests -q
```

```cmd
cd /d J:\数学建模\改进 && python solver\run_all.py --cases 1-100 --cores 2,3,4,5 --scenes A --workers 4 --log-file results\solve_log_fixed_v1.jsonl --output-dir results\plans_fixed_v1
```

## 4. 实验结果

全量任务：400/400 完成。

官方可比结果：每个核数 81 组；19 个超大图结果单独作为代理结果。

| 核数 | 平均加速比 |
|---:|---:|
| 2 | **1.760** |
| 3 | **2.370** |
| 4 | **2.890** |
| 5 | **3.320** |

与 `refined_v2` 相比，平均真实 makespan 下降：2 核 5.17%、3 核 7.43%、4 核 8.49%、5 核 10.00%。

## 5. 失败/退化分析

少数样本仍会出现通信下降而 makespan 上升，重点包括 `case_049/N5` 和 `case_063/N4`。当前代理 added copy 在大图上明显高估，说明下一步应先统一生命周期和目标函数，再引入后继前瞻。

## 6. 下一轮计划

先在 12 个代表算例上做三组消融：

1. 生命周期 `start/end/free` 修正；
2. 生命周期修正 + DOPS 式 H=2 后继前瞻；
3. 上述两项 + 固定切分后的核内顺序搜索。

未完成上述消融前，不提交全量新版本，不覆盖 `main` 的当前最佳流程。
