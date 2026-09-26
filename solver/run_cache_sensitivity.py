"""100例四核固定方案扫描：原官方独立配置、内存门槛、进度和内容指纹恢复。"""
import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import json
import os
from pathlib import Path
import sys
import time

from cache_sensitivity import sweep_points, sensitivity_config, summarize_rows
from official_protocol import (ATTACHMENT, digest, object_digest, evaluate_job, input_context,
                               read_ledger, verified_record)
from online_shared import process_lock
from run_saved_refine import write_json
from runtime_resources import initialize_worker_threads, memory_snapshot, GIB, collect_runtime_environment
from validated_bundle import freeze_files, verify_files


def point_key(capacity, bandwidth):
    return f'capacity_{capacity}_bandwidth_{bandwidth}'


def prepare(out, baseline):
    sources = [(p, p.relative_to(ATTACHMENT)) for p in sorted((ATTACHMENT/'code').glob('*.py'))]
    sources += [(ATTACHMENT/'data'/f'case_{i:03}.json', Path('data')/f'case_{i:03}.json') for i in range(1, 101)]
    sources += [(ATTACHMENT/'data/config.txt', Path('data/config.txt'))]
    frozen = out/'frozen_attachment'; manifest = out/'source_manifest.json'
    if manifest.exists():
        records = json.loads(manifest.read_text(encoding='utf8')); verify_files(frozen, records)
        if any(digest(source) != next(r['sha256'] for r in records if r['path'] == rel.as_posix())
               for source, rel in sources):
            raise ValueError('官方源内容变化，需使用新实验目录')
    else:
        records = freeze_files(sources, frozen, lambda i, n: print(f'冻结官方输入[{i}/{n}]', flush=True))
        write_json(manifest, records)
    plan_sources = [(baseline/'plans'/f'case_{i:03}_C_N4.json', f'case_{i:03}_C_N4.json') for i in range(1, 101)]
    plans = freeze_files(plan_sources, out/'plans')
    config = (frozen/'data/config.txt').read_bytes().decode('utf8')
    for capacity, bandwidth in sweep_points():
        attachment = out/'points'/point_key(capacity, bandwidth)/'attachment'
        for record in records:
            if record['path'] == 'data/config.txt':
                continue
            target = attachment/record['path']; target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                os.link(frozen/record['path'], target)
            if digest(target) != record['sha256']:
                raise ValueError('扫描输入摘要变化')
        target = attachment/'data/config.txt'
        content = sensitivity_config(config, capacity, bandwidth).encode('utf8')
        if target.exists() and target.read_bytes() != content:
            raise ValueError('冻结参数变化')
        if not target.exists():
            target.write_bytes(content)
    return dict(source_manifest_sha256=digest(manifest), plans=plans, points=sweep_points(),
                wrapper_sha256=digest(__file__), scoring_sha256=digest(Path(__file__).with_name('cache_sensitivity.py')),
                protocol_sha256=digest(Path(__file__).with_name('official_protocol.py')), cores=4,
                cases=list(range(1, 101)), baseline=str(baseline.resolve()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', default='results/p3_best_combined_r45')
    parser.add_argument('--output-dir', default='results/p3_cache_sensitivity_r51')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--max-new', type=int, default=0, help='本轮新增评测上限，0为全部；不改变恢复身份')
    args = parser.parse_args()
    if not 1 <= args.workers <= 4 or args.max_new < 0:
        parser.error('并发范围1～4；新增上限不能为负')
    out=Path(args.output_dir).resolve(); baseline=Path(args.baseline).resolve()
    out.mkdir(parents=True, exist_ok=True); initialize_worker_threads()
    with process_lock(out/'run.lock'):
        specification=prepare(out, baseline); manifest=out/'manifest.json'
        specification=json.loads(json.dumps(specification))
        if manifest.exists() and json.loads(manifest.read_text(encoding='utf8')) != specification:
            raise ValueError('扫描配置或代码变化，必须使用新实验目录')
        if not manifest.exists(): write_json(manifest, specification)
        write_json(out/'runtime.json', collect_runtime_environment())
        history, warnings=read_ledger(baseline/'official/results.jsonl')
        if warnings: raise ValueError('基线清单损坏')
        by_fp={r['fingerprint']:r for r in history.values()}
        baseline_times={r['case']:r['real']['makespan'] for r in history.values() if r['N']==4}
        if len(baseline_times)!=100: raise ValueError('基线必须覆盖100份四核方案')
        done=[];pending=[]
        for i in range(1,101):
            case=f'case_{i:03}'; job=dict(case=case,kind='problem_3',N=4,plan_scene='C')
            for capacity, bandwidth in sweep_points():
                folder=out/'points'/point_key(capacity,bandwidth)
                context=input_context(job,out/'plans',folder/'attachment')[0]; fp=object_digest(context)
                checkpoint=out/'checkpoints'/f'{case}_{point_key(capacity,bandwidth)}.json'
                row=json.loads(checkpoint.read_text(encoding='utf8')) if checkpoint.exists() else None
                if row and row['official_record']['fingerprint']==fp and verified_record(row['official_record']):
                    done.append(row);continue
                task=dict(job=job,capacity_bytes=capacity,cache_bandwidth=bandwidth,folder=folder,
                          checkpoint=checkpoint,baseline_makespan=baseline_times[case])
                record=by_fp.get(fp)
                if record and verified_record(record):
                    row=save_result(task,record);done.append(row)
                else: pending.append(task)
        print(f'[{len(done)}/1200] 内容核验后复用；待评{len(pending)}；并发上限{args.workers}',flush=True)
        publish(out,done)
        started=0;active={};last_wait=0
        with ProcessPoolExecutor(max_workers=args.workers,initializer=initialize_worker_threads) as pool:
            while pending or active:
                while pending and len(active)<args.workers and (not args.max_new or started<args.max_new):
                    free=memory_snapshot()['available_bytes']
                    if free is None or free < int(3.5*GIB):
                        if time.monotonic()-last_wait>30:
                            print(f'[{len(done)}/1200] 内存保护等待，可用{free}；需保留2GiB并预留新任务1.5GiB',flush=True)
                            last_wait=time.monotonic()
                        break
                    task=pending.pop(0)
                    future=pool.submit(evaluate_job,task['job'],task['folder']/'official',out/'plans',task['folder']/'attachment')
                    active[future]=task;started+=1
                if not active:
                    if args.max_new and started>=args.max_new: break
                    time.sleep(5);continue
                finished,_=wait(active,timeout=5,return_when=FIRST_COMPLETED)
                for future in finished:
                    task=active.pop(future);record=future.result()
                    if record.get('status')!='official_success':
                        write_json(out/'failure.json',record);raise RuntimeError(record.get('error','官方失败'))
                    done.append(save_result(task,record));publish(out,done)
                    print(f'[{len(done)}/1200] {task["job"]["case"]} 容量{task["capacity_bytes"]} 带宽{task["cache_bandwidth"]}',flush=True)
        publish(out,done)


def save_result(task, record):
    row=dict(case=task['job']['case'],N=4,capacity_bytes=task['capacity_bytes'],cache_bandwidth=task['cache_bandwidth'],
             real=record['real'],baseline_makespan=task['baseline_makespan'],official_record=record,
             experiment_class='cache_hardware_sensitivity')
    write_json(task['checkpoint'],row)
    return row


def publish(out, rows):
    summary=summarize_rows(rows,[f'case_{i:03}' for i in range(1,101)])
    write_json(out/'summary.json',summary)
    write_json(out/'metrics.json', [{k:v for k,v in r.items() if k!='official_record'} for r in rows])


if __name__=='__main__':main()
