"""固定保存方案的官方保底精化；候选凭据、任务检查点及全量清单可恢复。"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import shutil
import time

from official_protocol import (ATTACHMENT, digest, object_digest, input_context,
                               evaluate_job, verified_record, read_ledger, job_key)
from run_all import ensure_run_manifest, parse_cases, append_run_row
from runtime_resources import initialize_worker_threads, resolve_workers, collect_runtime_environment


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.pending')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def completed(checkpoint, source, output, fingerprint):
    """只有输入、配置、输出及官方凭据全部匹配才跳过。"""
    try:
        row = json.loads(Path(checkpoint).read_text(encoding='utf-8'))
        return (row if row['run_fingerprint'] == fingerprint
                and row['input_plan_sha256'] == digest(source)
                and row['output_plan_sha256'] == digest(output)
                and verified_record(row['official_record']) else None)
    except (OSError, KeyError, ValueError, TypeError):
        return None


def load_reuse_history(ledgers):
    """兼容最终清单与完整精化记录；每份候选仍在使用时核验官方凭据。"""
    history = {}
    for ledger in ledgers:
        with Path(ledger).open(encoding='utf-8') as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                records = [row, row.get('official_record', {})]
                records.extend(row.get('candidate_receipts', []))
                for record in records:
                    if record.get('status') == 'official_success' and record.get('fingerprint'):
                        history[record['fingerprint']] = record
    return history


def run_task(job, settings, fingerprint, history):
    started = time.perf_counter()
    out = Path(settings['output_dir'])
    name = f"{job['case']}_{job['plan_scene']}_N{job['N']}.json"
    source = Path(settings['input_dir']) / name
    final = out/'plans'/name
    checkpoint = out/'checkpoints'/name
    old = completed(checkpoint, source, final, fingerprint)
    if old:
        return old, True
    graph = json.loads((ATTACHMENT/'data'/f"{job['case']}.json").read_text(encoding='utf-8'))
    parent = json.loads(source.read_text(encoding='utf-8'))
    from scenario_contract import validate_plan
    validate_plan(graph,parent,job['plan_scene'])
    if len(parent['core_schedules']) != job['N']:
        raise ValueError('输入方案核数错误')
    receipts = {}
    evaluation_costs = []

    def evaluate(graph, plan, scene):
        key = object_digest(plan)
        if key in receipts:
            return receipts[key]['real']
        folder = out/'candidates'/job_key(job)/key
        folder.mkdir(parents=True, exist_ok=True)
        path = folder/name
        if plan == parent:
            content = source.read_bytes()
        else:
            content = json.dumps(plan, ensure_ascii=False).encode('utf-8')
        if path.exists() and path.read_bytes() != content:
            raise ValueError('已冻结候选内容不一致')
        if not path.exists():
            path.write_bytes(content)
        context = input_context(job, folder, ATTACHMENT)[0]
        old = history.get(object_digest(context))
        evaluated_at = time.perf_counter()
        if old is not None and verified_record(old):
            row = old
            reused_history = True
        else:
            row = evaluate_job(job, out/'official', folder, timeout=settings['timeout'])
            reused_history = False
        if row.get('status') != 'official_success':
            raise RuntimeError(row.get('error', '官方评估失败'))
        receipts[key] = row
        evaluation_costs.append({'plan_id': key, 'seconds': time.perf_counter()-evaluated_at,
                                 'reused_history': reused_history, 'record_id': row['record_id']})
        return row['real']

    base = evaluate(graph, parent, job['plan_scene'])
    if settings.get('boundary_tail'):
        if job['plan_scene']!='A':raise ValueError('输出尾部迁核只适用于A')
        from boundary_tail_refine import generate_boundary_tail_candidates
        from candidate_portfolio import select_official_portfolio
        raw=json.loads(Path(receipts[object_digest(parent)]['result_path']).read_text(encoding='utf8'))
        candidates,generation=generate_boundary_tail_candidates(graph,parent,raw,
            max_candidates=settings.get('official_limit',6),max_expanded=settings.get('screen_candidates',24),
            cache_mib=settings.get('stage_cache_mib',128))
        chosen,best,audit=select_official_portfolio(graph,parent,base,'A',candidates,evaluate)
        audit['generation']=generation
        if audit['errors']:raise RuntimeError('输出尾部候选未全部成功：'+str(audit['errors']))
        mismatches=[]
        for item in audit['candidates']:
            observed=json.loads(Path(receipts[item['plan_id']]['result_path']).read_text(encoding='utf8'))
            info=item['source']
            if info['closed_loop_makespan']!=observed['makespan'] or info['closed_loop_movement']!=observed['data_movement_bytes']:
                mismatches.append(dict(source=info,official_makespan=observed['makespan'],official_movement=observed['data_movement_bytes']))
        if mismatches:
            write_json(out/'diagnostics'/name,dict(error='A输出尾部预测与原官方不一致',mismatches=mismatches,generation=generation))
            raise RuntimeError('A输出尾部预测与原官方不一致，保留诊断')
    elif settings.get('boundary_release'):
        if job['plan_scene'] != 'A': raise ValueError('输出屏障细化只适用于A')
        from boundary_release import generate_boundary_release_candidates
        from candidate_portfolio import select_official_portfolio
        raw=json.loads(Path(receipts[object_digest(parent)]['result_path']).read_text(encoding='utf8'))
        candidates,generation=generate_boundary_release_candidates(graph,parent,raw,max_candidates=settings.get('official_limit',6))
        chosen,best,audit=select_official_portfolio(graph,parent,base,'A',candidates,evaluate)
        audit['generation']=generation
    elif settings.get('critical_copy_window'):
        if job['plan_scene'] not in ('B','C'): raise ValueError('关键搬运窗口只适用于B/C')
        from bc_copy_window import generate_critical_copy_candidates
        from candidate_portfolio import select_official_portfolio
        raw=json.loads(Path(receipts[object_digest(parent)]['result_path']).read_text(encoding='utf8'))
        if settings.get('closed_loop_screen'):
            from bc_event_screen import generate_event_screened_candidates
            candidates,generation=generate_event_screened_candidates(graph,parent,raw,job['plan_scene'],
                max_candidates=settings.get('official_limit',6),max_expanded=settings.get('screen_candidates',24),
                cache_mib=settings.get('stage_cache_mib',128))
        else:
            candidates,generation=generate_critical_copy_candidates(graph,parent,raw,max_candidates=settings.get('official_limit',6),dependency_closure=settings.get('critical_copy_closure',False))
        chosen,best,audit=select_official_portfolio(graph,parent,base,job['plan_scene'],candidates,evaluate)
        audit['generation']=generation
        if audit['errors']:raise RuntimeError('关键搬运窗口候选未全部成功，等待恢复：'+str(audit['errors']))
        if settings.get('closed_loop_screen'):
            mismatches=[r for r in audit['candidates'] if r.get('official',{}).get('makespan')!=r['source']['closed_loop_makespan']]
            if mismatches:
                write_json(out/'diagnostics'/name,dict(error='闭环预测与新官方不一致',mismatches=mismatches,generation=generation))
                raise RuntimeError('闭环精筛候选原官方核验不一致，保留诊断，停止当前任务')
    elif settings.get('core_locality'):
        if job['plan_scene'] not in ('B','C'): raise ValueError('亲和迁核只适用于B/C')
        from core_locality_refine import generate_core_locality_candidates
        from candidate_portfolio import select_official_portfolio
        raw=json.loads(Path(receipts[object_digest(parent)]['result_path']).read_text(encoding='utf8'))
        candidates,generation=generate_core_locality_candidates(graph,parent,raw,bandwidth=raw['bandwidth_bytes_per_cycle'],max_candidates=settings.get('official_limit',3),exchange_only=settings.get('core_locality_exchanges',False))
        chosen,best,audit=select_official_portfolio(graph,parent,base,job['plan_scene'],candidates,evaluate)
        audit['generation']=generation
        if audit['errors']:raise RuntimeError('亲和迁核候选未全部成功，等待恢复：'+str(audit['errors']))
    elif settings.get('observed_merge'):
        if job['plan_scene'] != 'A': raise ValueError('观测时间回并只适用于A')
        from observed_task_merge import generate_observed_merge_candidates
        from candidate_portfolio import select_official_portfolio
        raw=json.loads(Path(receipts[object_digest(parent)]['result_path']).read_text(encoding='utf8'))
        candidates,generation=generate_observed_merge_candidates(graph,parent,raw,max_candidates=settings.get('official_limit',3))
        chosen,best,audit=select_official_portfolio(graph,parent,base,'A',candidates,evaluate)
        audit['generation']=generation
        if audit['errors']:raise RuntimeError('观测时间回并候选未全部成功，等待恢复：'+str(audit['errors']))
    elif settings.get('cache_event_window'):
        if job['plan_scene']!='C':raise ValueError('缓存事件窗口只适用于C')
        from cache_event_window import generate_cache_event_candidates
        from candidate_portfolio import select_official_portfolio
        raw=json.loads(Path(receipts[object_digest(parent)]['result_path']).read_text(encoding='utf8'))
        candidates,generation=generate_cache_event_candidates(graph,parent,raw,
            max_candidates=settings.get('official_limit',6),max_expanded=settings.get('screen_candidates',24),
            cache_mib=settings.get('stage_cache_mib',128))
        chosen,best,audit=select_official_portfolio(graph,parent,base,'C',candidates,evaluate)
        audit['generation']=generation
        if audit['errors']:raise RuntimeError('缓存事件候选未全部成功：'+str(audit['errors']))
        mismatches=[]
        for item in audit['candidates']:
            observed=json.loads(Path(receipts[item['plan_id']]['result_path']).read_text(encoding='utf8'))
            candidate_source=item['source']
            if candidate_source['closed_loop_makespan']!=observed['makespan'] or candidate_source['closed_loop_cache_stats']!=observed['cache_stats']:
                mismatches.append(dict(source=candidate_source,official_makespan=observed['makespan'],official_cache_stats=observed['cache_stats']))
        if mismatches:
            write_json(out/'diagnostics'/name,dict(error='缓存事件预测与原官方不一致',mismatches=mismatches,generation=generation))
            raise RuntimeError('缓存事件预测与原官方不一致，保留诊断并停止当前任务')
    elif settings.get('cache_window'):
        if job['plan_scene']!='C':raise ValueError('缓存窗口只适用于C')
        from cache_window import generate_cache_window_candidates
        from candidate_portfolio import select_official_portfolio
        raw=json.loads(Path(receipts[object_digest(parent)]['result_path']).read_text(encoding='utf8'))
        candidates,generation=generate_cache_window_candidates(graph,parent,raw,max_candidates=settings.get('official_limit',6),read_boundary=settings.get('cache_window_read_boundary',False))
        chosen,best,audit=select_official_portfolio(graph,parent,base,'C',candidates,evaluate)
        audit['generation']=generation
        if audit['errors']:raise RuntimeError('缓存窗口候选未全部成功，保留尝试等待恢复：'+str(audit['errors']))
    elif settings.get('op_list'):
        if job['plan_scene'] not in ('B','C'):raise ValueError('操作级生成器只适用于B/C')
        from op_list_candidates import generate_op_candidates
        from candidate_portfolio import select_official_portfolio
        from evaluation_validation import read_evaluation_config
        config=read_evaluation_config(str(ATTACHMENT/'data/config.txt'))
        from multicore_cut_evaluate_problem_2 import read_scene_b_config
        cross_delay=read_scene_b_config(str(ATTACHMENT/'data/config.txt'))['cross_core_copy_delay_cycles']
        candidates,generation=generate_op_candidates(graph,parent,num_cores=job['N'],
            max_candidates=settings.get('op_candidates',12),bandwidth=config['bandwidth'],
            communication_rank=settings.get('op_communication_rank',False),cross_delay=cross_delay)
        chosen,best,audit=select_official_portfolio(graph,parent,base,job['plan_scene'],candidates,evaluate)
        audit['generation']=generation
        # 超时/失败候选不视为完成，成功尝试已在官方目录留档，下次恢复复用。
        if audit['errors']:raise RuntimeError('操作级候选未全部验收，保留官方尝试等待续跑：'+str(audit['errors']))
    elif settings.get('portfolio_only'):
        from candidate_portfolio import load_candidates, select_official_portfolio
        directories = [settings['input_dir']] + settings.get('portfolio_dir', [])
        candidates = load_candidates(job['case'], job['plan_scene'], job['N'], directories,
                                     settings.get('include_lower_cores', False))
        chosen, best, audit = select_official_portfolio(
            graph, parent, base, job['plan_scene'], candidates, evaluate)
    elif job['plan_scene'] == 'A':
        from task_order_search import refine_plan_orders
        # 固定评估次数而非墙钟，避免机器负载影响候选集合。
        chosen, best, audit = refine_plan_orders(
            graph, parent, base, evaluate, enable_swaps=True, enable_window=True,
            search_seconds=None, max_evals=settings['max_evals'],
            window_candidates=settings['window_candidates'])
    elif settings['refine_op_limit'] and len(graph['ops']) > settings['refine_op_limit']:
        # 全部案例仍做官方评定；这里只限制未校准的B/C搜索，不以代理代替结果。
        chosen, best = parent, base
        audit = {'status': 'search_skipped_large_graph', 'baseline_official': base,
                 'final_official': base, 'errors': [], 'n_ops': len(graph['ops']),
                 'refine_op_limit': settings['refine_op_limit'],
                 'reason': '大图多候选校准未完成，本轮保留父方案并作完整官方评估'}
        if settings.get('seed_dir'):
            seed = json.loads((Path(settings['seed_dir'])/name.replace('_C_', '_B_')).read_text(encoding='utf-8'))
            seed_real = evaluate(graph, seed, job['plan_scene'])
            audit['seed_best_official'] = seed_real
            if (seed_real['makespan'], seed_real['added_copy_bytes']) < (best['makespan'], best['added_copy_bytes']):
                chosen, best = seed, seed_real
            audit['final_official'] = best
    else:
        from bc_search import refine_official
        seeds = []
        if settings.get('seed_dir'):
            seed = Path(settings['seed_dir'])/name.replace('_C_', '_B_')
            seeds.append(json.loads(seed.read_text(encoding='utf-8')))
        chosen, best, audit = refine_official(
            graph, parent, job['plan_scene'], evaluate, seeds=seeds,
            max_states=settings['max_states'], official_limit=settings['official_limit'])
    validate_plan(graph,chosen,job['plan_scene'])
    if (best['makespan'], best['added_copy_bytes']) > (base['makespan'], base['added_copy_bytes']):
        raise ValueError('官方保底失效')
    selected = receipts[object_digest(chosen)]
    if job['plan_scene'] == 'C':
        from cache_replay import audit_official_cache
        raw = json.loads(Path(selected['result_path']).read_text(encoding='utf-8'))
        audit['observed_cache_audit'] = audit_official_cache(raw)
        if not audit['observed_cache_audit']['consistent']:
            raise ValueError('观察到的官方缓存时间线审计不一致；保留凭据待诊断')
    candidate_path = out/'candidates'/job_key(job)/object_digest(chosen)/name
    final.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(candidate_path, final)
    if digest(final) != selected['plan_sha256']:
        raise ValueError('输出方案与官方凭据不一致')
    row = dict(job, status='official_success', run_fingerprint=fingerprint,
               input_plan_sha256=digest(source), output_plan_sha256=digest(final),
               baseline=base, real=best, audit=audit, official_record=selected,
               candidate_receipts=list(receipts.values()),
               evaluation_costs=evaluation_costs,
               elapsed=time.perf_counter()-started)
    write_json(checkpoint, row)
    return row, False


def run(args):
    initialize_worker_threads()
    workers = resolve_workers(args.workers).workers
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    settings = {k: v for k, v in vars(args).items() if k not in ('workers', 'max_tasks')}
    settings['input_dir'] = str(Path(args.input_dir).resolve())
    settings['output_dir'] = str(out)
    cases = parse_cases(args.cases)
    cores = list(dict.fromkeys(map(int, args.cores.split(','))))
    if any(n not in (2, 3, 4, 5) for n in cores):
        raise ValueError('核数必须在2～5之间')
    if (min(args.max_evals, args.window_candidates, args.max_states, args.official_limit, args.timeout) < 1
            or args.max_tasks < 0 or args.refine_op_limit < 0):
        raise ValueError('预算必须为正')
    jobs = [dict(case=case, N=n, plan_scene=args.scene,
                 kind='problem_'+str('ABC'.index(args.scene)+1)) for case in cases for n in cores]
    files = list(Path(__file__).parent.glob('*.py')) + list((ATTACHMENT/'code').glob('*.py'))
    files += [ATTACHMENT/'data/config.txt']
    files += [ATTACHMENT/'data'/f'{case}.json' for case in cases]
    files += [Path(args.input_dir)/f"{j['case']}_{args.scene}_N{j['N']}.json" for j in jobs]
    if args.seed_dir:
        files += [Path(args.seed_dir)/f"{j['case']}_B_N{j['N']}.json" for j in jobs]
    if getattr(args, 'portfolio_only', False):
        from candidate_portfolio import candidate_paths
        directories = [args.input_dir] + args.portfolio_dir
        files += [path for j in jobs for path, _ in candidate_paths(
            j['case'], args.scene, j['N'], directories, args.include_lower_cores)]
        files = list(dict.fromkeys(Path(p).resolve() for p in files))
    fingerprint = ensure_run_manifest(out/'refine.jsonl', files, settings)
    write_json(out/'runtime.json', collect_runtime_environment())
    history = load_reuse_history(args.reuse_ledger)
    rows, unfinished = [], []
    for j in jobs:
        name = f"{j['case']}_{args.scene}_N{j['N']}.json"
        old = completed(out/'checkpoints'/name, Path(args.input_dir)/name,
                        out/'plans'/name, fingerprint)
        if old:
            rows.append(old)
        else:
            unfinished.append(j)
    # 首批先验收已冻结代表例的五核；后续恢复不重复这些任务。
    representative = {'case_001', 'case_007', 'case_042', 'case_063', 'case_067', 'case_097'}
    unfinished.sort(key=lambda j: (not (j['case'] in representative and j['N'] == 5), j['case'], j['N']))
    if args.max_tasks:
        unfinished = unfinished[:args.max_tasks]
    print(f'[{len(rows)}/{len(jobs)}] 固定输入精化；场景{args.scene}；并发{workers}；本批{len(unfinished)}项', flush=True)
    with ProcessPoolExecutor(max_workers=workers, initializer=initialize_worker_threads) as pool:
        pending = {pool.submit(run_task, j, settings, fingerprint, history): j for j in unfinished}
        for future in as_completed(pending):
            row, reused = future.result()
            rows.append(row)
            if not reused:
                append_run_row(out/'refine.jsonl', row)
            wins = sum(r['real']['makespan'] < r['baseline']['makespan'] for r in rows)
            print(f"[{len(rows)}/{len(jobs)}] {row['case']}/{row['N']}；改善{wins}；复用{reused}；{row['elapsed']:.1f}秒", flush=True)
    if len(rows) != len(jobs):
        write_json(out/'batch_summary.json', {'expected': len(jobs), 'completed': len(rows),
                   'wins': sum(r['real']['makespan'] < r['baseline']['makespan'] for r in rows),
                   'losses': sum(r['real']['makespan'] > r['baseline']['makespan'] for r in rows),
                   'errors': sum(len(r['audit'].get('errors', [])) for r in rows)})
        print('本批结束，剩余任务使用相同命令续跑。', flush=True)
        return
    # 单核分母也只能复用完整官方凭据，否则现场评估。
    singles = [dict(case=case, N=1, kind='singlecore', plan_scene=None) for case in cases]
    expected = jobs + singles
    ensure_run_manifest(out/'official/results.jsonl', [Path(__file__)], {'jobs': expected, 'run_fingerprint': fingerprint})
    ledger = out/'official/results.jsonl'
    existing, _ = read_ledger(ledger)
    for row in rows:
        record = row['official_record']
        if existing.get(record['job_id'], {}).get('record_id') != record['record_id']:
            append_run_row(ledger, record)
    for i, job in enumerate(singles, 1):
        context = input_context(job, out/'plans', ATTACHMENT)[0]
        record = history.get(object_digest(context))
        if record is None or not verified_record(record):
            record = evaluate_job(job, out/'official', out/'plans', timeout=args.timeout)
        if existing.get(record['job_id'], {}).get('record_id') != record['record_id']:
            append_run_row(ledger, record)
        print(f'单核分母[{i}/{len(singles)}] {record["status"]}', flush=True)
    from aggregate import write_reports
    summary = write_reports(ledger, out/'official')
    write_json(out/'paired_summary.json', {
        'tasks': len(rows), 'wins': sum(r['real']['makespan'] < r['baseline']['makespan'] for r in rows),
        'losses': sum(r['real']['makespan'] > r['baseline']['makespan'] for r in rows),
        'errors': sum(len(r['audit'].get('errors', [])) for r in rows),
        'groups': summary['groups']})
    if summary['official_success'] != len(expected):
        raise RuntimeError('官方全量不完整')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', default='1-100')
    parser.add_argument('--cores', default='2,3,4,5')
    parser.add_argument('--scene', choices=list('ABC'), required=True)
    parser.add_argument('--input-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--seed-dir')
    parser.add_argument('--boundary-release',action='store_true',help='A同核输出祖先闭包拆分，最多6候选，原官方保底')
    parser.add_argument('--boundary-tail',action='store_true',help='A固定输出边界拆分后迁尾/改插入位置，动态带宽评分与结构控制')
    parser.add_argument('--observed-merge',action='store_true',help='A父轨迹时间余量筛选，同核相邻小块不重叠回并；预算最多3')
    parser.add_argument('--core-locality',action='store_true',help='B/C单算子切分亲和迁核，边界服务下降且每流水线最大工作量不增加')
    parser.add_argument('--core-locality-exchanges',action='store_true',help='亲和迁核改为同流水线双算子交换对照，最多16次交换')
    parser.add_argument('--critical-copy-window',action='store_true',help='B/C关键搬运前移，真实展开合法性，最多12展开与6官方候选')
    parser.add_argument('--critical-copy-closure',action='store_true',help='前移窗口中必要本核计算前驱闭包，最多16子图，内存补边重新验证')
    parser.add_argument('--closed-loop-screen',action='store_true',help='关键搬运依赖闭包采用完整动态带宽/缓存事件精筛，仍由原官方裁决')
    parser.add_argument('--screen-candidates',type=int,default=24,help='闭环展开候选预算6～48，默认24')
    parser.add_argument('--stage-cache-mib',type=int,default=128,help='每任务阶段缓存序列化负载上限0～128MiB')
    parser.add_argument('--op-list',action='store_true',help='B/C操作级就绪日历候选，原官方父方案保底')
    parser.add_argument('--op-communication-rank',action='store_true',help='操作级通信感知向上高度，仅两个通信偏好配置')
    parser.add_argument('--cache-window',action='store_true',help='C固定切分分核，按官方未命中事件生成最多六个局部核序候选')
    parser.add_argument('--cache-event-window',action='store_true',help='C按首次读完成/淘汰事件定位读入窗口，双向依赖闭包与动态事件精筛')
    parser.add_argument('--cache-window-read-boundary',action='store_true',help='缓存窗口只选择跨过实际读入子图的位置，窗口仍最多8个子图')
    parser.add_argument('--op-candidates',type=int,default=12,help='固定操作级配置前缀预算，1～12')
    parser.add_argument('--portfolio-only', action='store_true',
                        help='只在父方案与外部/少核候选间官方取优，不运行搜索')
    parser.add_argument('--portfolio-dir', action='append', default=[],
                        help='外部完整方案目录，可重复；保持当前场景')
    parser.add_argument('--include-lower-cores', action='store_true',
                        help='允许同场景少核方案补空核，目标核数重新官方核验')
    parser.add_argument('--reuse-ledger', action='append', default=[])
    parser.add_argument('--workers', default='auto')
    parser.add_argument('--max-evals', type=int, default=4000)
    parser.add_argument('--window-candidates', type=int, default=500)
    parser.add_argument('--max-states', type=int, default=64)
    parser.add_argument('--official-limit', type=int, default=6)
    parser.add_argument('--timeout', type=int, default=3600)
    parser.add_argument('--max-tasks', type=int, default=0, help='本次最多处理的未完成任务，0表示全部')
    parser.add_argument('--refine-op-limit', type=int, default=12000,
                        help='B/C大图保留父方案和显式种子，仍做全部官方评测；0表示不限')
    args = parser.parse_args()
    if args.boundary_tail and (args.scene!='A' or args.boundary_release or args.observed_merge or args.core_locality or args.critical_copy_window or args.op_list or args.cache_window or args.cache_event_window or args.portfolio_only or args.seed_dir or not 3<=args.official_limit<=6):
        parser.error('输出尾部迁核只用于A，含控制组预算3～6，不能与其他候选模式混用')
    if args.cache_event_window and (args.scene!='C' or args.boundary_release or args.observed_merge or args.core_locality or args.critical_copy_window or args.op_list or args.cache_window or args.portfolio_only or args.seed_dir or not 1<=args.official_limit<=6):
        parser.error('缓存事件窗口只用于C，预算1～6，不能与其他候选模式混用')
    if args.boundary_release and (args.scene!='A' or args.critical_copy_window or args.core_locality or args.observed_merge or args.op_list or args.cache_window or args.portfolio_only or args.seed_dir or not 1<=args.official_limit<=6):
        parser.error('输出屏障细化只用于A，预算1～6，不能与其他候选模式混用')
    if args.critical_copy_closure and not args.critical_copy_window:
        parser.error('搬运依赖闭包必须同时启用关键搬运窗口')
    if args.closed_loop_screen and not (args.critical_copy_window and args.critical_copy_closure):
        parser.error('闭环精筛必须同时启用关键搬运窗口和依赖闭包')
    if not 6<=args.screen_candidates<=48 or not 0<=args.stage_cache_mib<=128:
        parser.error('闭环候选预算必须6～48，阶段缓存预算必须0～128MiB')
    if args.critical_copy_window and (args.scene=='A' or args.core_locality or args.observed_merge or args.op_list or args.cache_window or args.portfolio_only or args.seed_dir or not 1<=args.official_limit<=6):
        parser.error('关键搬运窗口只适用于B/C，预算1～6，不能与其他候选模式混用')
    if args.core_locality_exchanges and not args.core_locality:
        parser.error('双算子交换必须同时启用亲和迁核')
    if args.core_locality and (args.scene=='A' or args.observed_merge or args.op_list or args.cache_window or args.portfolio_only or args.seed_dir or not 1<=args.official_limit<=3):
        parser.error('亲和迁核只适用于B/C，预算1～3，不能与其他候选模式混用')
    if args.observed_merge and (args.scene!='A' or args.op_list or args.cache_window or args.portfolio_only or args.seed_dir or not 1<=args.official_limit<=3):
        parser.error('观测时间回并只适用于A，预算1～3，不能与其他候选模式混用')
    if args.op_communication_rank and not args.op_list:
        parser.error('通信高度必须与操作级生成器同时启用')
    if args.cache_window_read_boundary and not args.cache_window:
        parser.error('读入边界开关必须与缓存窗口一起使用')
    if args.cache_window and (args.scene!='C' or args.op_list or args.portfolio_only or args.seed_dir or not 1<=args.official_limit<=6):
        parser.error('缓存窗口只适用于C，预算1～6，不能与其他候选模式混用')
    if args.op_list and (args.scene=='A' or args.portfolio_only or args.seed_dir):
        parser.error('操作级候选只适用于B/C，不能与外部组合或种子模式混用')
    if not 1<=args.op_candidates<=12:parser.error('操作级候选预算必须在1～12')
    if (args.portfolio_dir or args.include_lower_cores) and not args.portfolio_only:
        parser.error('外部/少核组合参数必须显式启用 --portfolio-only')
    # 保留锁文件，仅释放锁；禁止同目录并发启动，进程异常退出由操作系统释放。
    folder = Path(args.output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    with (folder/'run.lock').open('a+b') as lock:
        import os
        if lock.tell() == 0:
            lock.write(b'0'); lock.flush()
        lock.seek(0)
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args)


if __name__ == '__main__':
    main()
