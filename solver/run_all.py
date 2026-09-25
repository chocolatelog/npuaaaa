"""全量求解入口：并行处理 (用例 × 场景 × 核数)，输出方案 JSON 与求解日志。

用法：
  python run_all.py [--cases 1-100] [--cores 2,3,4,5] [--scenes A,B]
                    [--workers 8] [--time-budget auto]
"""
import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
ATTACH = os.path.normpath(os.path.join(
    HERE, '..', '通用神经网络处理器下的多核调度问题附件'))
DATA = os.path.join(ATTACH, 'data')
OUT_PLANS = os.path.normpath(os.path.join(HERE, '..', 'results', 'plans'))
OUT_LOG = os.path.normpath(os.path.join(HERE, '..', 'results', 'solve_log.jsonl'))


def ensure_run_manifest(log_path, files, settings):
    """实验入口保存输入指纹；拒绝混用不同代码/参数/数据续跑。"""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = log_path.with_suffix('.manifest.json')
    packages = {}
    for name in ('numpy', 'scipy', 'torch'):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    payload = {'protocol_version': 2, 'settings': settings,
               'environment': {'python': sys.version, 'platform': platform.platform(),
                               'packages': packages}, 'files': {
        str(Path(p).resolve()): hashlib.sha256(Path(p).read_bytes()).hexdigest()
        for p in sorted(files, key=str)}}
    fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    if manifest.exists():
        previous = json.loads(manifest.read_text(encoding='utf-8'))
        if previous['fingerprint'] != fingerprint:
            raise ValueError('代码、数据或参数指纹变化，请使用新的日志和方案目录')
    elif log_path.exists() and log_path.stat().st_size:
        raise ValueError('已有日志缺少实验指纹，请使用新的日志和方案目录')
    else:
        manifest.write_text(json.dumps({'fingerprint': fingerprint, **payload},
                                      ensure_ascii=False, indent=2), encoding='utf-8')
    return fingerprint


def stable_seed(case, scene, n, seed_base=0):
    """跨进程、跨机器稳定的任务种子。"""
    raw = f'{case}|{scene}|{int(n)}'
    if seed_base:
        raw += f'|seed_base={int(seed_base)}'
    raw = raw.encode('utf-8')
    return int.from_bytes(hashlib.sha256(raw).digest()[:4], 'big')


def plan_digest(plan):
    return hashlib.sha256(json.dumps({'n': plan['node_to_subgraph'],
                                     'c': plan['core_schedules']}, sort_keys=True).encode()).hexdigest()


def valid_official(real):
    return (isinstance(real, dict) and not real.get('error')
            and isinstance(real.get('makespan'), (int, float))
            and math.isfinite(real['makespan']) and real['makespan'] > 0
            and all(isinstance(real.get(k), (int, float)) and math.isfinite(real[k])
                    and real[k] >= 0 for k in ('added_copy_bytes', 'scheduled_copy_bytes')))


def result_binding(entry):
    payload = {k: entry.get(k) for k in ('case', 'scene', 'N', 'seed', 'fingerprint',
                                        'plan_sha256', 'plan_id', 'real')}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def bind_entry(entry, plan_path):
    path = Path(plan_path)
    plan = json.loads(path.read_text(encoding='utf-8'))
    entry['protocol_version'] = 2
    entry['plan_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    entry['plan_id'] = plan_digest(plan)
    selected = (entry.get('log') or {}).get('selected_plan_id')
    if selected is not None and selected != entry['plan_id']:
        raise ValueError('最终方案与求解器选中方案摘要不一致')
    entry['status'] = ('official_success' if valid_official(entry.get('real')) else
                       'official_pending' if entry.get('real') is None and entry.get('n_ops', 0) > 12000
                       else 'official_error')
    entry['result_binding'] = result_binding(entry)


def resume_status(entry, plans_dir):
    """核验保存方案和结果绑定；代理已求解与官方成功分别计数。"""
    try:
        path = Path(plans_dir) / f"{entry['case']}_{entry['scene']}_N{entry['N']}.json"
        if entry.get('protocol_version') != 2 or not path.is_file():
            return 'retry'
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry.get('plan_sha256'):
            return 'retry'
        plan = json.loads(path.read_text(encoding='utf-8'))
        if len(plan['core_schedules']) != entry['N'] or plan_digest(plan) != entry.get('plan_id'):
            return 'retry'
        if result_binding(entry) != entry.get('result_binding'):
            return 'retry'
        if valid_official(entry.get('real')) and entry.get('status') == 'official_success':
            return 'official_success'
        if (entry.get('real') is None and entry.get('status') == 'official_pending'
                and entry.get('n_ops', 0) > 12000):
            return 'official_pending'
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return 'retry'


def read_run_rows(path):
    """保留最后一次完整尝试；损坏行显式报告，不删除历史记录。"""
    rows, warnings = {}, []
    path = Path(path)
    if path.exists():
        for line_no, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                key = (row['case'], row['scene'], int(row['N']))
                rows[key] = row
            except (ValueError, TypeError, KeyError) as exc:
                warnings.append({'line': line_no, 'error': str(exc)})
    return rows, warnings


def append_run_row(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # 保留中断时的原始尾行，补换行后追加新尝试；读取时报告损坏行。
    needs_newline = False
    if path.exists() and path.stat().st_size:
        with path.open('rb') as stream:
            stream.seek(-1, os.SEEK_END)
            needs_newline = stream.read(1) != b'\n'
    with path.open('a', encoding='utf-8') as stream:
        if needs_newline:
            stream.write('\n')
        stream.write(json.dumps(row, ensure_ascii=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def run_input_files(cases):
    files = list(Path(HERE).glob('*.py'))
    files += list((Path(ATTACH) / 'code').glob('*.py'))
    files += [Path(DATA) / f'{case}.json' for case in cases]
    files += [Path(HERE) / 'spill_coefs.json', Path(DATA) / 'config.txt']
    return files


def budget_for(n_ops):
    if n_ops <= 1500:
        return 12.0
    if n_ops <= 5000:
        return 15.0
    if n_ops <= 12000:
        return 20.0
    return 28.0


def block_cap_for(n_eligible):
    """块粒度：约 400 op 一块的上限并夹在 [24, 240]，保证子图可贴合缓存。"""
    return int(max(24, min(240, n_eligible // 400)))


def solve_task(task):
    from model import load_graph
    from pipeline import solve_case
    case, scene, n = task['case'], task['scene'], task['n']
    graph_path = os.path.join(DATA, f'{case}.json')
    graph = load_graph(graph_path)
    n_ops = len(graph['ops'])
    n_eligible = sum(1 for o in graph['ops']
                     if o['op'] not in ('COPY_IN', 'COPY_OUT'))
    res = solve_case(graph, N=n, scene=scene,
                     time_budget=task.get('budget') or budget_for(n_ops),
                     seed=task.get('seed', stable_seed(case, scene, n)),
                     block_ops_cap=block_cap_for(n_eligible),
                     spill_calibrated=bool(task.get('spill_calibrated', False)),
                     use_lru_spill=bool(task.get('use_lru_spill', False)),
                     event_rerank=bool(task.get('event_rerank', False)),
                     task_order_refine=bool(task.get('task_order_refine', False)),
                     partition_polish=bool(task.get('partition_polish', False)),
                     construct_reservoir=bool(task.get('construct_reservoir', False)),
                     fast_candidate_eval=bool(task.get('fast_candidate_eval', False)),
                     search_rounds=task.get('search_rounds'),
                     terminal_swaps=bool(task.get('terminal_swaps', False)),
                     terminal_cache=bool(task.get('terminal_cache', False)))
    out_plan = os.path.join(task.get('out_plans', OUT_PLANS),
                            f'{case}_{scene}_N{n}.json')
    with open(out_plan, 'w', encoding='utf-8') as f:
        json.dump(res['plan'], f)
    entry = {'case': case, 'scene': scene, 'N': n,
             'seed': task.get('seed', stable_seed(case, scene, n)),
             'est_mk': res['est'][0], 'est_added': res['est'][1],
             'real': res['real'], 'log': res['log'],
             'spill_calibrated': res['log'].get('spill_calibrated', False),
             'spill_mode': res['log'].get('spill_mode', 'heuristic'),
             'proxy_partition_raw_bytes': res['log'].get(
                 'proxy_partition_raw_bytes'),
             'proxy_partition_added_bytes': res['log'].get(
                 'proxy_partition_added_bytes'),
             'proxy_spill_added_bytes': res['log'].get(
                 'proxy_spill_added_bytes'),
             'proxy_total_added_bytes': res['log'].get(
                 'proxy_total_added_bytes'),
             'proxy_memory_overage': res['log'].get('proxy_memory_overage'),
             'proxy_peak_memory_bytes': res['log'].get(
                 'proxy_peak_memory_bytes'),
             'elapsed': round(res['elapsed'], 1), 'n_ops': n_ops,
             'seed_base': task.get('seed_base', 0), 'fingerprint': task.get('fingerprint'),
             'attempt': task.get('attempt', 1)}
    bind_entry(entry, out_plan)
    return entry


def parse_cases(spec):
    out = []
    for part in spec.split(','):
        if '-' in part:
            lo, hi = part.split('-')
            out.extend(f'case_{i:03d}' for i in range(int(lo), int(hi) + 1))
        else:
            out.append(f'case_{int(part):03d}')
    if not out or any(not 1 <= int(c.split('_')[1]) <= 100 for c in out):
        raise ValueError('案例编号必须在 1 到 100 之间')
    return list(dict.fromkeys(out))


def main():
    global OUT_PLANS, OUT_LOG
    parser = argparse.ArgumentParser()
    parser.add_argument('--cases', default='1-100')
    parser.add_argument('--cores', default='2,3,4,5')
    parser.add_argument('--scenes', default='A,B')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--budget', type=float, default=None)
    parser.add_argument('--search-rounds', type=int, default=None, help='固定元启发式轮数；用于复现和计数对照，替代限时停止')
    parser.add_argument('--seed-base', type=int, default=0, help='重复实验种子组；0 保持历史种子')
    parser.add_argument('--max-tasks', type=int, default=0,
                        help='本次最多启动多少个未完成任务，0 表示全部；用于分批与续跑验证')
    parser.add_argument('--output-dir', default=OUT_PLANS)
    parser.add_argument('--log-file', default=OUT_LOG)
    parser.add_argument('--spill-calibrated', action='store_true',
                        help='启用已有 spill 校准系数')
    parser.add_argument('--lru-spill', action='store_true',
                        help='实验性启用 op 生命周期 LRU spill 估计')
    parser.add_argument('--event-rerank', action='store_true',
                        help='场景 A 末端精评并增加最多两个官方复核候选')
    parser.add_argument('--task-order-refine', action='store_true',
                        help='场景 A 固定切分迁核与核内顺序束搜索，最多两次官方复核')
    parser.add_argument('--partition-polish', action='store_true',
                        help='场景 A 末端多粒度拆分与边界字节引导合并')
    parser.add_argument('--construct-reservoir', action='store_true',
                        help='配合末端精评保留构造阶段被裁掉的粒度候选')
    parser.add_argument('--fast-candidate-eval', action='store_true',
                        help='场景 A 新增精化候选使用计数式副本，最终强制原官方复核')
    parser.add_argument('--terminal-swaps', action='store_true',
                        help='完整流程原官方确认后执行迁移/交换，最多八份副本筛选和两份原官方确认')
    parser.add_argument('--terminal-cache', action='store_true',
                        help='为末端交换的副本精评启用每任务局部模板缓存；必须启用 --terminal-swaps')
    args = parser.parse_args()
    if args.terminal_cache and not args.terminal_swaps:
        parser.error('--terminal-cache 必须与 --terminal-swaps 一起使用')
    if args.search_rounds is not None and (args.search_rounds < 1 or any((args.event_rerank, args.task_order_refine, args.partition_polish, args.construct_reservoir, args.fast_candidate_eval))):
        parser.error('固定轮数必须为正，且暂不与额外后处理开关混用')
    if args.construct_reservoir and not args.event_rerank:
        parser.error('--construct-reservoir 必须与 --event-rerank 一起使用')
    if args.workers < 1 or args.max_tasks < 0 or (args.budget is not None and
                                                (not math.isfinite(args.budget) or args.budget <= 0)):
        parser.error('并发数、预算必须为正，分批任务数不能为负')

    OUT_PLANS = os.path.abspath(args.output_dir)
    OUT_LOG = os.path.abspath(args.log_file)
    os.makedirs(OUT_PLANS, exist_ok=True)
    cases = parse_cases(args.cases)
    cores = list(dict.fromkeys(int(x) for x in args.cores.split(',')))
    scenes = list(dict.fromkeys(args.scenes.split(',')))
    if any(n not in (2, 3, 4, 5) for n in cores) or any(s not in ('A', 'B', 'C') for s in scenes):
        parser.error('核心数为 2/3/4/5，场景为 A/B/C')
    fingerprint = ensure_run_manifest(OUT_LOG, run_input_files(cases), {
            'cases': cases, 'cores': cores, 'scenes': scenes, 'budget': args.budget,
            'seed_base': args.seed_base, 'search_rounds': args.search_rounds,
            'workers': args.workers, 'event_rerank': args.event_rerank,
            'task_order_refine': args.task_order_refine,
            'partition_polish': args.partition_polish,
            'construct_reservoir': args.construct_reservoir,
            'fast_candidate_eval': args.fast_candidate_eval,
            'terminal_swaps': args.terminal_swaps, 'terminal_cache': args.terminal_cache,
            'spill_calibrated': args.spill_calibrated, 'lru_spill': args.lru_spill,
            'output_dir': OUT_PLANS})

    previous, warnings = read_run_rows(OUT_LOG)
    if warnings:
        print(f'发现 {len(warnings)} 条损坏日志行，保留原记录并重新核验任务', flush=True)
        Path(OUT_LOG).with_suffix('.recovery.json').write_text(
            json.dumps(warnings, ensure_ascii=False, indent=2), encoding='utf-8')
    expected = {(c, s, n) for c in cases for s in scenes for n in cores}
    states = {key: resume_status(row, OUT_PLANS) for key, row in previous.items()
              if key in expected and row.get('fingerprint') == fingerprint}
    done = {key for key, state in states.items() if state in ('official_success', 'official_pending')}

    tasks = []
    for case in cases:
        for scene in scenes:
            for n in cores:
                if (case, scene, n) not in done:
                    tasks.append({'case': case, 'scene': scene, 'n': n,
                                  'seed': stable_seed(case, scene, n, args.seed_base),
                                  'seed_base': args.seed_base, 'fingerprint': fingerprint,
                                  'search_rounds': args.search_rounds,
                                  'attempt': previous.get((case, scene, n), {}).get('attempt', 0) + 1,
                                  'budget': args.budget,
                                  'out_plans': OUT_PLANS,
                                  'spill_calibrated': args.spill_calibrated,
                                  'use_lru_spill': args.lru_spill,
                                  'event_rerank': args.event_rerank,
                                  'task_order_refine': args.task_order_refine,
                                  'partition_polish': args.partition_polish,
                                  'construct_reservoir': args.construct_reservoir,
                                  'fast_candidate_eval': args.fast_candidate_eval,
                                  'terminal_swaps': args.terminal_swaps,
                                  'terminal_cache': args.terminal_cache})
    remaining = len(tasks)
    if args.max_tasks:
        tasks = tasks[:args.max_tasks]
    official_count = sum(state == 'official_success' for state in states.values())
    pending_count = sum(state == 'official_pending' for state in states.values())
    print(f'[{len(done)}/{len(expected)}] 本批 {len(tasks)} 个任务，待求解 {remaining}；'
          f'官方成功 {official_count}，代理已求解/官方待评 {pending_count}，并发 {args.workers}', flush=True)
    t0 = time.time()
    if tasks:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(solve_task, t): t for t in tasks}
            n_done, failed = 0, 0
            for fut in as_completed(futs):
                t = futs[fut]
                try:
                    entry = fut.result()
                    append_run_row(OUT_LOG, entry)
                    n_done += 1
                    real = entry['real']
                    official_count += entry['status'] == 'official_success'
                    pending_count += entry['status'] == 'official_pending'
                    failed += entry['status'] == 'official_error'
                    real_mk = real.get('makespan', '-') if real else '-'
                    print(f'[{n_done}/{len(tasks)}] {entry["case"]} '
                          f'{entry["scene"]} N={entry["N"]} '
                          f'官方={real_mk}，成功/待评/失败={official_count}/{pending_count}/{failed} '
                          f'({entry["elapsed"]:.0f}s)', flush=True)
                except Exception as e:
                    failed += 1
                    n_done += 1
                    append_run_row(OUT_LOG, {'case': t['case'], 'scene': t['scene'],
                        'N': t['n'], 'status': 'failed', 'error': repr(e),
                        'attempt': t['attempt'], 'protocol_version': 2,
                        'fingerprint': fingerprint})
                    print(f'[{n_done}/{len(tasks)}] 失败 {t["case"]}/{t["scene"]}/N{t["n"]}: {e!r}', flush=True)
    print(f'本批结束，用时 {time.time() - t0:.1f} 秒；官方成功 {official_count}，官方待评 {pending_count}')


if __name__ == '__main__':
    sys.path.insert(0, HERE)
    main()
