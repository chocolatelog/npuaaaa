# H 分支实验数据归档

更新时间：2026-09-25。本次按明确要求归档 H 分支现有实验记录、JSON/JSONL、方案目录及汇总，作为通用上传规范的一次例外。不含环境、缓存、密钥、数据库及官方完整 trace。

## 当前正式成果

- [技术路线 H-v3.0](../技术路线_H分支_v3.0.md)
- [800 项实验记录](../实验记录_H分支_操作级800全量_v1.0.md)
- [数据统计与核验报告](../数据统计_H分支_操作级800全量_v1.0.md)、[机器可读统计](oplist_full800.statistics.json)
- [800 项精简记录](oplist_full800.tasks.json)
- [完整运行日志](solve_log_oplist_full800.jsonl)
- [原始汇总](solve_log_oplist_full800.summary.json)、[启动配置与文件指纹](solve_log_oplist_full800.manifest.json)
- [800 份最终方案](plans_oplist_full800/)

100 案例 × A/B × 2/3/4/5 核 = 800 项。每组全 100 案例和历史可比 81 案例分别统计；历史日志中的代理/超时记录按其原始实验状态解读。

## 本轮输入组合

| 输入目录 | 作用 | 对应日志 |
|---|---|---|
| plans_guided_full | A/B、2–5 核历史方案及固定 81 案例范围 | solve_log_guided_full.jsonl |
| plans_n5_expanded_full | A/B 五核扩展历史方案 | solve_log_n5_expanded_full.jsonl |
| plans_n5_oplist_full81 | B/5 的 81 案例历史最优方案 | solve_log_n5_oplist_full81.jsonl |

各输入方案在本轮重新官方评估后择优；单核参照由配置中的 singlecore 文件提供。方案池保底与单核参照不是同一概念。

## 历史记录

- legacy_20260923：项目父目录中 209 份早期 JSON 和 5 个 Python 脚本的原样归档，独立于本轮 800 任务。

- plans_baseline_stable、plans_optimized*、plans_verify24*：原构造、优化与复核。
- plans_mcts*、plans_guided*：MCTS/Beam 和引导搜索。
- plans_n5_portfolio*、plans_n5_expanded*、plans_n5_activefill*、plans_n5_wavefront*：五核方案池和活跃核探索。
- plans_n5_math*、plans_n5_oplist*：数学强化和操作级调度。
- plans_oplist_matrix_smoke16、plans_oplist_full800_gate12：本轮门槛/兼容性试跑；plans_oplist_full800 是正式 800 项。
- solve_log*.jsonl 保存逐项原始记录；配套 manifest 记录配置和指纹、summary 记录汇总，部分早期实验没有两种配套文件。
- 旧 report.md、summary.md、ab_test_900.json 等保留历史范围；文件名含 900 不表示当前运行规模。

## 复算与复现边界

统计复算入口：python solver/export_oplist_report.py。它核验已归档证据并生成两个统计 JSON 文件，不重新运行调度优化器。重跑实验的命令、预算和限制见实验记录，输出必须使用新目录。

原始日志保留采集时路径和配置，在其他机器复现时使用仓库相对目录。manifest 是原始字节哈希，Git 检出换行会影响文本源码字节哈希，需区分字节和逻辑变化。plan_sha256 使用解析 JSON 后的规范化哈希，方法见运行器 signature。
