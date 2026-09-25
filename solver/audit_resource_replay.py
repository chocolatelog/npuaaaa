"""E06固定方案三方完整字段验证；显卡批特征、逐条清单与中断恢复。"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import time

from batch_features import BatchFeatures, scalar_features
from experiment_branch_partition import seal_result
from experiment_order_pair import binding
from model import Model
from run_all import parse_cases, block_cap_for, ensure_run_manifest, run_input_files, append_run_row
from scene_a_replay import build_resource_evaluator
from scene_a_fast import official, evaluate_scene_a_fast

ROOT=Path(__file__).resolve().parents[1]


def resource_summary(result):
    timelines=result['per_core_timeline'];mk=result['makespan']
    tasks={t['task_id']:(c['core_id'],t) for c in timelines for t in c['tasks']}
    preds={t:[] for t in tasks}
    for e in result['task_dependencies']:preds[e['target']].append(e['source'])
    rows=[]
    for core in timelines:
        previous=0;data_wait=sync_wait=residual=0
        for task in core['tasks']:
            tid=task['task_id'];p=preds[tid]
            ready=max(previous,max((tasks[x][1]['end'] for x in p),default=0))
            release=max(previous+(100 if previous else 0),
                        max((tasks[x][1]['end']+1000 for x in p if tasks[x][0]!=core['core_id']),default=0))
            data_wait+=max(0,ready-previous)
            sync_wait+=max(0,release-ready)
            residual+=task['start']-max(ready,release)
            previous=task['end']
        rows.append({'core':core['core_id'],'active_cycles':sum(t['duration'] for t in core['tasks']),
                     'idle_cycles':mk-sum(t['duration'] for t in core['tasks']),
                     'dependency_wait_cycles':data_wait,'synchronization_after_data_cycles':sync_wait,
                     'residual_before_task_cycles':residual,'tail_idle_cycles':mk-previous})
    return {'cores':rows,'ddr_issues':len(result['ddr_contention_log']),
            'contended_issues':sum(e['active_count']>1 for e in result['ddr_contention_log']),
            'max_ddr_requests':max((e['active_count'] for e in result['ddr_contention_log']),default=0),
            'tasks':[{'task':tid,'global_duration':t['duration'],
                      'local_duration':result['step3_by_task'][tid]['local_makespan'],
                      'global_minus_local':t['duration']-result['step3_by_task'][tid]['local_makespan']}
                     for tid,(_,t) in tasks.items()],
            'scope':'时间线诊断；全局减局部不是严格带宽因果分解，空闲包含依赖等待与尾部空闲'}


def good(row,fp):
    try:
        content=dict(row);expected=content.pop('binding')
        return (row['fingerprint']==fp and row['full_result_equal'] and expected==binding(content) and
                all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in row['artifacts'].items()))
    except (KeyError,OSError,ValueError):return False


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--cases',default='1,5,7,12,15,23,24,35,42,44,45,49,55,63,65,67,75,78,82,83,95,97')
    p.add_argument('--cores',default='2,3,4,5')
    p.add_argument('--max-tasks',type=int,default=0)
    p.add_argument('--unified',action='store_true',help='同时启用核心/任务状态和完成事件计数')
    p.add_argument('--reference-dir',help='复用此前已绑定的完整原官方结果，避免重复调用官方')
    p.add_argument('--output-dir',required=True)
    args=p.parse_args()
    import torch
    if not torch.cuda.is_available():p.error('需要已有显卡环境用于真实批量特征核验')
    cases=parse_cases(args.cases);cores=list(dict.fromkeys(map(int,args.cores.split(','))))
    if args.max_tasks < 0 or any(n not in (2,3,4,5) for n in cores):
        p.error('分批数或核数非法')
    if args.reference_dir and not args.unified:p.error('当前完整结果复用仅供统一表对照')
    source=ROOT/'results/p1_e03_r02/order_pairs.jsonl'
    mapping={(r['case'],r['N']):r for r in map(json.loads,source.read_text(encoding='utf-8').splitlines())}
    tasks=[mapping[c,n] for c in cases for n in cores]
    out=Path(args.output_dir).resolve();out.mkdir(parents=True,exist_ok=True);ledger=out/'resource_replay.jsonl'
    paths=[Path(r['variants']['swap']['plan_path']) for r in tasks]
    references={}
    if args.reference_dir:
        ref_dir=Path(args.reference_dir).resolve();ref_log=ref_dir/'resource_replay.jsonl'
        ref_manifest=ref_log.with_suffix('.manifest.json')
        ref_fp=json.loads(ref_manifest.read_text(encoding='utf-8'))['fingerprint']
        references={r['id']:r for r in map(json.loads,ref_log.read_text(encoding='utf-8').splitlines())}
        for entry in tasks:
            ref=references[entry['id']]
            if not good(ref,ref_fp) or ref['source_sha256']!=entry['variants']['swap']['plan_sha256']:
                raise ValueError('完整官方参照身份或产物绑定无效')
            paths.extend(Path(p) for p in ref['artifacts'])
        paths.extend([ref_log,ref_manifest])
    fp=ensure_run_manifest(ledger,run_input_files(cases)+[source]+paths,
                           {'cases':cases,'cores':cores,'mode':'原官方/计数副本/资源表三方',
                            'unified':args.unified,'reference_dir':str(Path(args.reference_dir).resolve()) if args.reference_dir else None,
                            'gpu':torch.cuda.get_device_name(0)})
    saved={}
    if ledger.exists():
        for line in ledger.read_text(encoding='utf-8').splitlines():
            try:r=json.loads(line);saved[r['id']]=r
            except (ValueError,KeyError):pass
    done={k:r for k,r in saved.items() if good(r,fp)}
    pending=[r for r in tasks if r['id'] not in done]
    if args.max_tasks:pending=pending[:args.max_tasks]
    print(f'[{len(done)}/{len(tasks)}] E06资源表，待执行本批{len(pending)}',flush=True)
    if pending:
        start=time.perf_counter();replay=build_resource_evaluator(unified=args.unified)
        ddr_control=build_resource_evaluator() if references else None
        initialization=time.perf_counter()-start
    feature_cache={}
    for entry in pending:
        case,n=entry['case'],entry['N'];source_plan=entry['variants']['swap']
        graph=json.loads((ROOT/'通用神经网络处理器下的多核调度问题附件/data'/f'{case}.json').read_text(encoding='utf-8'))
        path=Path(source_plan['plan_path'])
        assert hashlib.sha256(path.read_bytes()).hexdigest()==source_plan['plan_sha256']
        plan=json.loads(path.read_text(encoding='utf-8'))
        if case not in feature_cache:
            start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
            model=Model(graph,block_ops_cap=block_cap_for(sum(o['op'] not in ('COPY_IN','COPY_OUT') for o in graph['ops'])))
            aa,cc=[],[]
            for group in (r for r in pending if r['case']==case):
                pp=json.loads(Path(group['variants']['swap']['plan_path']).read_text(encoding='utf-8'))
                a=[pp['node_to_subgraph'][str(b[0])] for b in model.blocks];c=[0]*(max(a)+1)
                for core,order in enumerate(pp['core_schedules']):
                    for task in order:c[task]=core
                aa.append(a);cc.append(c)
            values=BatchFeatures(model,5).evaluate(aa,cc)
            assert values==[scalar_features(model,a,c,5) for a,c in zip(aa,cc)]
            feature_cache[case]={'batch_candidates':len(aa),'seconds':time.perf_counter()-start,
                                 'peak_allocated':torch.cuda.max_memory_allocated(),'equal':True}
            feature_owner=True
        else:feature_owner=False
        functions=[('official',official.evaluate_scene_a),('counter',evaluate_scene_a_fast),('resource',replay)]
        if references:functions=[('ddr_control',ddr_control),('resource',replay)]
        offset=tasks.index(entry)%len(functions);functions=functions[offset:]+functions[:offset]
        results={};timing={};artifacts={}
        params=(graph,plan,60,{'L1':524288,'UB':131072},1000,100)
        for label,fn in functions:
            start=time.perf_counter();results[label]=fn(*params);timing[label]=time.perf_counter()-start
        equal=all(r==results['resource'] for r in results.values())
        hashes={k:binding(json.loads(json.dumps(v))) for k,v in results.items()}
        reference_note=None
        if references:
            ref=references[entry['id']];ref_path=next(iter(ref['artifacts']))
            full=json.loads(gzip.decompress(Path(ref_path).read_bytes()).decode('utf-8'))
            equal=equal and full==json.loads(json.dumps(results['resource']))
            hashes['official_reference']=binding(full)
            if hashes['official_reference']!=ref['result_sha256']['official']:
                raise ValueError('原官方完整结果内容摘要无效')
            reference_note={'ledger':str(ref_log),'result_path':ref_path,'record_binding':ref['binding'],
                            'source_fingerprint':ref_fp,'official_calls_this_round':0}
        # 完整相等只存一份压缩结果；不相等时各自保存，便于定位。
        for label in (['resource'] if equal else list(results)):
            target=out/f'{entry["id"]}_{label}.json.gz'
            target.write_bytes(gzip.compress(json.dumps(results[label],ensure_ascii=False).encode('utf-8'),mtime=0))
            artifacts[str(target)]=hashlib.sha256(target.read_bytes()).hexdigest()
        truth=results['resource'];dm=truth['data_movement_bytes']
        record=seal_result({'id':entry['id'],'case':case,'N':n,'full_result_equal':equal,
            'source_plan':str(path),'source_sha256':source_plan['plan_sha256'],'result_sha256':hashes,
            'real':{'makespan':truth['makespan'],'added_copy_bytes':dm['added_copy_bytes'],
                    'scheduled_copy_bytes':dm['scheduled_copy_bytes'],'partition_added':dm['partition_added_copy_bytes'],
                    'spill_added':dm['spill_added_copy_bytes']},'timing':timing,'first':functions[0][0],
            'initialization_seconds':initialization if not done else 0,
            'unified':args.unified,'official_reference':reference_note,
            'gpu_features':feature_cache[case] if feature_owner else {'shared_with_case':case},
            'resource_diagnosis':resource_summary(truth),'artifacts':artifacts},fp)
        assert record['real']==source_plan['real']
        append_run_row(ledger,record);done[entry['id']]=record
        costs=' / '.join(f'{label}={seconds:.3f}' for label,seconds in timing.items())
        print(f'[{len(done)}/{len(tasks)}] {entry["id"]} 完整一致={equal}，{costs} 秒',flush=True)
        if not equal:raise ValueError('资源表不等价，保留失败证据并停止')
    summary={'expected':len(tasks),'completed':len(done),'all_equal':all(r['full_result_equal'] for r in done.values()),
             'seconds':{k:sum(r['timing'].get(k,0) for r in done.values()) for k in sorted({k for r in done.values() for k in r['timing']})},
             'scope':('核心/任务/外存状态与从头重放；区域表及完整快照未完成；固定方案质量不变；无后缀恢复'
                      if args.unified else '仅外存表及从头重放；核心/任务/区域统一表未完成；固定方案质量不变；不含后缀恢复'),
             'original_official_calls_this_round':0 if references else len(done)}
    target=out/'summary.json';text=json.dumps(summary,ensure_ascii=False,indent=2)
    if not target.exists() or target.read_text(encoding='utf-8')!=text:target.write_text(text,encoding='utf-8')
    print(text,flush=True)


if __name__=='__main__':main()
