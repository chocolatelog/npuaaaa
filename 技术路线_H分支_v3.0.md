# H 分支当前技术路线：操作级调度与全量官方核验

版本：H-v3.0；更新时间：2026-09-25；目标分支：H。
代码父版本：1d94da699fe174967ce0f57c5de50ea3e1d1d591；本轮还包含 run_oplist_full.py、新增测试与 run_math_refine.py 的泛化改动。运行时精确代码、输入及保底方案以 results/solve_log_oplist_full800.manifest.json 的指纹为准。
负责模块：操作级候选生成、官方核验编排、历史方案保底、统计与复现。

## 1. 当前完成范围

本轮运行覆盖 100 个案例 × A/B 两个场景 × 2/3/4/5 核，共 800 项。800 个任务全部得到有效官方评估结果，输出 800 份调度方案，无重复、缺失和最终未验证任务。

A 场景对应 problem_1 的 evaluate_scene_a；B 场景对应 problem_2 的 evaluate_scene_b。带 L2 参数的 problem_3/C 分支不在本轮 800 项中。不能将其他历史文件中的 900 项或第三种场景并入本轮结论。

| 核数 | A：全 100 案例 | B：全 100 案例 | A：历史可比 81 案例 | B：历史可比 81 案例 |
|---|---:|---:|---:|---:|
| 2 | 1.720799× | 2.267487× | 1.795990× | 2.308702× |
| 3 | 2.253361× | 3.065740× | 2.428981× | 3.104269× |
| 4 | 2.710850× | 3.677874× | 2.983310× | 3.710305× |
| 5 | 3.078658× | 4.225545× | 3.457662× | 4.284529× |

数据统计详见 [统计报告](数据统计_H分支_操作级800全量_v1.0.md)，运行命令、过程和异常详见 [实验记录](实验记录_H分支_操作级800全量_v1.0.md)。

## 2. 求解流程

1. 读取原始图、单核参照以及同案例/场景/核数的历史方案。
2. 对历史方案按规范化 JSON 签名去重，重新调用原官方评估器。
3. 在有效官方结果中按 (makespan, added_copy_bytes) 字典序选出输入保底。
4. 将图化为保留原始依赖的计算 DAG，生成最多 12 种操作级资源列表候选。
5. 在预算内逐一官方评估候选，只有字典序严格改善才替换当前方案。
6. 计算仅考虑计算约束的下界证书，写出最终计划、签名、逐项日志和全量汇总。

本轮是独立可选入口 solver/run_oplist_full.py，不代表替换 solver/run_all.py 的默认构造与搜索流程。历史方案可能由构造池、启发式搜索、MCTS/Beam、CP-SAT 等早期实验产生；本轮新增候选来自 op_list_candidates，不能把全部历史组合收益归因于本轮某一个算法。

## 3. 图模型与候选生成

bound_certificate.build_compute_dag 检查节点 ID、边端点和 DAG 合法性。COPY_IN/COPY_OUT 及张量节点在计算松弛中不计时，但沿这些节点传递的依赖关系保留。计算时长取 max(1, cycles)，资源类型为 PIPE_M、PIPE_V、PIPE_MTE2、PIPE_MTE3。

op_list_candidates 使用每核、每流水线的占用日历。每次从就绪堆选择操作，为各候选核计算依赖释放时刻，再寻找该流水线可容纳操作的最早空隙；比较完成时刻、跨核代价、开始时刻和核编号。

候选配置是三种优先策略与四种跨核延迟的组合：

- height：后继最长计算路径优先；
- depth：稳定拓扑顺序优先；
- fanout：后继数量优先；
- delay 取 0、100、500、1000，默认最多 12 种，重复方案去重。

启发式跨核传输代价按图边或张量数据量估计为 2 × bytes / 60，再叠加 delay。它只用于候选排序，不是替代官方搬运、缓存或同步模型的真实耗时。每个原始计算操作形成一个子图，降低粗化导致商图成环的风险，但细粒度也可能放大跨核搬运。

最终方案包含 node_to_subgraph 与 core_schedules，官方核验仍是判断有效性和真实收益的依据。

## 4. 官方评估、保底与预算

pipeline.real_evaluate 调用附件中的原官方函数。当前桥接参数固定为 DDR 带宽 60.0、L1 524288、UB 131072；A 调用的附加参数为 1000、100，B 为 500。以上是模拟参数，不是本机 CPU/GPU 性能测量。

本轮组合输入目录为 plans_n5_oplist_full81、plans_n5_expanded_full、plans_guided_full。相同任务的不同输入全部重新核验；输入池中的最优有效结果成为 baseline。baseline_speedup 仍使用共同的单核参照作为分子。

A 场景和大于 12000 操作的大图在隔离进程中核验，并提升历史 verify_cap 以允许大图获得官方结果。输入单次核验上限 360 秒，大图生成隔离上限 120 秒，每项候选生成与核验预算 300 秒。

候选预算在输入保底核验之后开始，不包括所有前处理、输入复核和证书开销。常规 B 场景在进程内核验，预算在调用之间检查；因此 300 秒不是任务端到端严格墙钟上限。常规小图候选生成同样不使用独立进程超时。

如果输入均无法有效核验，运行器保留输入计划但标记 unverified，不纳入官方均值；本轮最终没有此类任务。候选失败或超时时保留有效 incumbent。

输出目录、日志和配套 manifest/summary 必须使用全新路径。该新入口拒绝覆盖已有实验，不提供直接接续写入同一日志的断点续跑参数；不能沿用旧 run_all 的续跑描述。

## 5. 下界与指标定义

计算下界由各类流水线总计算工作量除以核数向上取整，以及计算依赖最长路径取最大值得到。忽略 COPY 时长、跨核同步、DDR 争用、缓存溢出和分区约束，因此是松弛下界，不是可执行最优调度，也不证明当前方案全局最优。

speedup_i = singlecore_makespan_i / official_makespan_i；分组均值为逐案例 speedup 的算术平均，不是总 makespan 的比值。

完整 800 项含每组 100 案例；历史可比 648 项含旧 guided 日志中每组 81 个有效官方案例，集合由明确任务键确定，不是编号前 81 个。剩余 152 项为每组 19 个大案例。

相对本轮输入组合为 266 胜、534 平、0 负；相对 guided 单一历史版本的 648 项为 325 胜、323 平、0 负。不能混用两个基线。

A/5 全 100 案例 3.078658×，输入组合为 3.058258×；历史 81 案例 3.457662×，另外 19 案例 1.462900×。必须用相同范围比较，不能把 81 案例成绩承诺为全 100 案例保底。

## 6. 复现入口

环境：启动 manifest 记录 Python 3.12.7。依赖文件为 requirements.txt 与 requirements-optimization.txt；运行前使用同一 Python 环境安装需要的依赖。具体 CPU/GPU 型号及显存峰值本轮未记录，不以仓库推荐硬件代替实测。

在仓库根目录运行以下命令，使用新的复现输出路径：

```powershell
.venv/Scripts/python.exe solver/run_oplist_full.py --cases 1-100 --scenes A,B --cores 2,3,4,5 --baseline-dir results/plans_n5_oplist_full81 --baseline-dir results/plans_n5_expanded_full --baseline-dir results/plans_guided_full --reference-log results/solve_log_guided_full.jsonl --output-dir results/plans_oplist_reproduce --log-file results/solve_log_oplist_reproduce.jsonl --workers 6 --max-candidates 12 --evaluation-seconds 360 --generation-seconds 120 --candidate-seconds 300
.venv/Scripts/python.exe -m pytest tests -q
.venv/Scripts/python.exe solver/export_oplist_report.py
```

统计脚本复算归档的 solve_log_oplist_full800，并不自动转向新复现实验；若要统计新运行，应显式调整统计输入或使用新运行器自身 summary。本轮入口无随机 seed 参数，候选规则确定，历史输入种子见各自记录；墙钟截止仍可能影响跨机器复现的探索数量。

## 7. 验证与局限

归档前 60 项测试通过，800 份计划规范化 SHA-256 与日志一致，任务唯一性、覆盖、加速比公式及汇总一致；启动 manifest 的 1231 个文件字节指纹全部一致。以上归档检查不等于再跑一遍独立 800 项官方重放。

全量墙钟 11881.969 秒。搜索过程含 98 次候选错误、33 项生成错误、97 项预算耗尽，不能将最终全部有效称为搜索过程零错误。

当前 A 场景大案例收益有限；优先研究通信量、依赖瓶颈、粒度和空核，以及预算耗尽原因。B 场景 2–4 核收益明显，B/5 的历史可比 81 案例与输入历史最优持平。后续应按同一输入预算开展候选消融和独立复核。

原始日志、方案、历史实验、机器可读明细均随本次 H 归档提供，入口见 [results/README.md](results/README.md)。本次结果不等于 main 合并审批，不支持直接推断比赛奖项。
