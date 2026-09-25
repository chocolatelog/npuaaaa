"""E08固定切分区域原子放置实验；显卡特征、完整重放与官方保底。"""
import argparse
import ast
import gzip
import hashlib
import json
from pathlib import Path
import time
import zipfile

from audit_resource_replay import good
from batch_features import BatchFeatures,scalar_features
from experiment_branch_partition import seal_result,verified
from local_template_cache import TemplateCache
from lookahead_place import compact_result,order_key
from model import Model
from pipeline import real_evaluate
from region_group_refine import choose_regions,generate_joint_candidates,select_joint_candidates,region_trace
from run_all import parse_cases,block_cap_for,ensure_run_manifest,run_input_files,append_run_row,plan_digest,valid_official
from scene_a_event import derive_multicore_plan
from scene_a_replay import build_resource_evaluator

ROOT=Path(__file__).resolve().parents[1]


def run_variant(entry,reference,out,torch,mode,prior=None):
    started=time.perf_counter();case,n=entry['case'],entry['N'];source=entry['variants']['swap'];eid=entry['id']
    source_path=Path(source['plan_path'])
    if hashlib.sha256(source_path.read_bytes()).hexdigest()!=source['plan_sha256']:raise ValueError('源计划摘要不一致')
    graph=json.loads((ROOT/'通用神经网络处理器下的多核调度问题附件/data'/f'{case}.json').read_text(encoding='utf-8'))
    plan=json.loads(source_path.read_text(encoding='utf-8'))
    raw=json.loads(gzip.decompress(Path(next(iter(reference['artifacts']))).read_bytes()))
    baseline=compact_result(raw,plan['core_schedules'])
    if baseline['real']!=source['real'] or reference['source_sha256']!=source['plan_sha256']:raise ValueError('官方基线不对应输入')
    durations={t:r['duration'] for t,r in baseline['tasks'].items()};preds=derive_multicore_plan(graph,plan)['subgraph_preds']
    regions=choose_regions(durations,preds,plan['core_schedules'])
    baseline_regions=region_trace(raw,regions)
    known={};known_official={}
    if prior:
        if prior['source_sha256']!=source['plan_sha256'] or prior['baseline']!=baseline['real'] or prior['regions']!=[list(x) for x in regions]:
            raise ValueError('复用来源的输入或区域不同')
        known={order_key(r['orders']):r for r in prior['evaluated']}
        known_official={order_key(json.loads(Path(c['plan_path']).read_text(encoding='utf-8'))['core_schedules']):c for c in prior['checks'] if c['equal']}
    before=time.perf_counter();rows,generation=generate_joint_candidates(durations,preds,plan['core_schedules'],regions,mode=mode)
    generation_seconds=time.perf_counter()-before
    model=Model(graph,block_ops_cap=block_cap_for(sum(o['op'] not in ('COPY_IN','COPY_OUT') for o in graph['ops'])))
    assignment=[plan['node_to_subgraph'][str(b[0])] for b in model.blocks]
    if model.plan_from(assignment,plan['core_schedules'])!=plan:raise ValueError('块映射不保持输入')
    torch.cuda.reset_peak_memory_stats();batch=BatchFeatures(model,n);owners=[]
    for row in rows:
        owner=[0]*(max(assignment)+1)
        for c,q in enumerate(row['orders']):
            for t in q:owner[t]=c
        owners.append(owner)
    before=time.perf_counter();values=batch.evaluate([assignment]*len(owners),owners) if owners else []
    if values!=[scalar_features(model,assignment,c,n) for c in owners]:raise ValueError('区域显卡粗特征不一致')
    gpu_seconds=time.perf_counter()-before
    for row,value in zip(rows,values):row['features']=value
    selected,screen=select_joint_candidates(rows);selected_keys={order_key(r['orders']) for r in selected}
    for row in rows:row['admitted']=order_key(row['orders']) in selected_keys
    cache=TemplateCache();before=time.perf_counter();evaluator=build_resource_evaluator(unified=True,template_cache=cache)
    setup_seconds=time.perf_counter()-before;artifacts={};evaluated=[]
    for index,candidate in enumerate(selected):
        proposal={**plan,'core_schedules':candidate['orders']}
        path=out/f'{eid}_candidate_{index:02d}.json';path.write_text(json.dumps(proposal),encoding='utf-8')
        artifacts[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        prior_value=known.get(order_key(candidate['orders']))
        if prior_value:
            result={'real':prior_value['resource']};elapsed=0.;trace=prior_value['regions']
            # 旧审计保存的组外结束差及组内完成事件共同覆盖所有任务。
            old_effect=prior_value['outside_effects'][0]
            ends={int(t):baseline['tasks'][int(t)]['end']+d for t,d in old_effect['outside_end_delta'].items()}
            for region_info in trace:
                for event in region_info['releases']:ends[event['task']]=event['time']
            if set(ends)!=set(durations):raise ValueError('复用来源缺少完整任务结束信息')
        else:
            before=time.perf_counter();raw=evaluator(graph,proposal,60,{'L1':524288,'UB':131072},1000,100)
            elapsed=time.perf_counter()-before;result=compact_result(raw,candidate['orders'])
            trace=region_trace(raw,regions);ends={t:r['end'] for t,r in result['tasks'].items()}
        if any(result['real'][k]!=baseline['real'][k] for k in baseline['real'] if k!='makespan'):raise ValueError('固定区域切分搬运变化')
        affected=[]
        for origin in candidate['origins']:
            outside=set(durations)-set(origin['tasks'])
            changes={str(t):ends[t]-baseline['tasks'][t]['end'] for t in sorted(outside)}
            affected.append({'region':origin['region'],'outside_end_delta':changes,
                'positive_sum':sum(max(0,d) for d in changes.values()),'max_delay':max(changes.values(),default=0)})
        evaluated.append({'plan_path':str(path),'orders':candidate['orders'],'origins':candidate['origins'],'coarse':candidate['coarse'],
                          'resource':result['real'],'seconds':elapsed,'regions':trace,'outside_effects':affected,
                          'reused':prior_value is not None,'source_binding':prior['binding'] if prior_value else None,
                          'source_evaluation_seconds':prior_value['seconds'] if prior_value else None})
        if (index+1)%4==0 or index+1==len(selected):print(f'  {eid} [{index+1}/{len(selected)}] 区域完整重放',flush=True)
    ranked=sorted(evaluated,key=lambda r:(r['resource']['makespan'],order_key(r['orders'])))
    best,best_plan=source['real'],plan;checks=[];errors=[]
    for row in [r for r in ranked if r['resource']['makespan']<best['makespan']][:2]:
        proposal=json.loads(Path(row['plan_path']).read_text(encoding='utf-8'))
        prior_check=known_official.get(order_key(proposal['core_schedules']))
        if prior_check:truth=prior_check['official'];elapsed=0.
        else:
            before=time.perf_counter();truth=real_evaluate(graph,proposal,'A');elapsed=time.perf_counter()-before
        equal=truth==row['resource'];checks.append({'plan_path':row['plan_path'],'official':truth,'resource':row['resource'],'equal':equal,'seconds':elapsed,
                                                  'reused':prior_check is not None,'source_binding':prior['binding'] if prior_check else None})
        if not valid_official(truth) or not equal:errors.append({'plan_path':row['plan_path'],'official':truth});continue
        if truth['makespan']<best['makespan']:best,best_plan=truth,proposal
    target=out/f'{eid}_selected.json';target.write_text(json.dumps(best_plan),encoding='utf-8')
    artifacts[str(target)]=hashlib.sha256(target.read_bytes()).hexdigest()
    return {'id':eid,'case':case,'N':n,'mode':mode,'status':'official_success','baseline':source['real'],'real':best,
            'source_plan':str(source_path),'source_sha256':source['plan_sha256'],'baseline_reference_binding':reference['binding'],
            'plan_path':str(target),'plan_id':plan_digest(best_plan),'artifacts':artifacts,'checks':checks,'errors':errors,
            'regions':regions,'baseline_regions':baseline_regions,'generation':generation,'generation_seconds':generation_seconds,
            'screen':screen,'candidate_features':rows,'evaluated':evaluated,'official_count':sum(not c['reused'] for c in checks),
            'official_reused_count':sum(c['reused'] for c in checks),
            'gpu_features':{'equal':True,'candidates':len(owners),'seconds':gpu_seconds,'peak_allocated':torch.cuda.max_memory_allocated()},
            'cache_stats':cache.stats,'setup_seconds':setup_seconds,'seconds':time.perf_counter()-started,
            'scope':'两个已有任务原子放置；2000廉价上限/12完整资源/2原官方；没有区域内部切分，非完整E08'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--cases',default='5,44,49,67,82');p.add_argument('--cores',default='2,4')
    p.add_argument('--mode',choices=('adaptive','fixed2'),default='adaptive');p.add_argument('--max-tasks',type=int,default=0)
    p.add_argument('--reuse-ledger',help='复用同输入、同评估器依赖的已保存完整核序；成本分开记账')
    p.add_argument('--output-dir',required=True);args=p.parse_args();import torch
    if not torch.cuda.is_available():p.error('需要已有显卡环境')
    cases=parse_cases(args.cases);cores=list(dict.fromkeys(map(int,args.cores.split(','))))
    if args.max_tasks<0 or any(n not in (2,3,4,5) for n in cores):p.error('参数非法')
    source=ROOT/'results/p1_e03_r02/order_pairs.jsonl';ref_log=ROOT/'results/p1_e06_ddr_r01/resource_replay.jsonl'
    ref_manifest=ref_log.with_suffix('.manifest.json');ref_fp=json.loads(ref_manifest.read_text(encoding='utf-8'))['fingerprint']
    mapping={(r['case'],r['N']):r for r in map(json.loads,source.read_text(encoding='utf-8').splitlines())}
    refs={r['id']:r for r in map(json.loads,ref_log.read_text(encoding='utf-8').splitlines())}
    tasks=[mapping[c,n] for c in cases for n in cores];files=run_input_files(cases)+[source,ref_log,ref_manifest]
    for entry in tasks:
        ref=refs[entry['id']]
        if not good(ref,ref_fp):raise ValueError('原官方参照无效')
        files+=[Path(entry['variants']['swap']['plan_path'])]+[Path(s) for s in ref['artifacts']]
    priors={}
    if args.reuse_ledger:
        prior_log=Path(args.reuse_ledger).resolve();prior_meta_path=prior_log.with_suffix('.manifest.json')
        prior_meta=json.loads(prior_meta_path.read_text(encoding='utf-8'));runner=Path(__file__).resolve()
        frozen=prior_log.parent/'frozen_source/source.zip'
        with zipfile.ZipFile(frozen) as z:old_runner=z.read('solver/experiment_region_group.py')
        if hashlib.sha256(old_runner).hexdigest()!=prior_meta['files'][str(runner)]:raise ValueError('旧入口快照无效')
        def eval_calls(src):
            return sorted(ast.dump(x) for x in ast.walk(ast.parse(src)) if isinstance(x,ast.Call) and isinstance(x.func,ast.Name) and x.func.id in ('evaluator','real_evaluate'))
        if eval_calls(old_runner.decode('utf-8'))!=eval_calls(runner.read_text(encoding='utf-8')):raise ValueError('评估调用或硬件参数改变，拒绝复用')
        for path,digest in prior_meta['files'].items():
            if Path(path).resolve()!=runner and hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest:raise ValueError('评估器依赖改变，拒绝复用')
        for row in map(json.loads,prior_log.read_text(encoding='utf-8').splitlines()):
            if not verified(row,prior_meta['fingerprint']):raise ValueError('旧结果绑定无效')
            priors[row['id']]=row;files.extend(map(Path,row['artifacts']))
        files.extend([prior_log,prior_meta_path,frozen])
    out=Path(args.output_dir).resolve();out.mkdir(parents=True,exist_ok=True);ledger=out/'region_pairs.jsonl'
    fp=ensure_run_manifest(ledger,files,{'cases':cases,'cores':cores,'mode':args.mode,'regions':2,'region_size':2,
        'coarse_limit':2000,'pool_limit':20,'resource_limit':12,'official_limit':2,'gpu':torch.cuda.get_device_name(0),'reuse_ledger':args.reuse_ledger})
    previous={}
    if ledger.exists():
        for line in ledger.read_text(encoding='utf-8').splitlines():
            try:r=json.loads(line);previous[r['id']]=r
            except (ValueError,KeyError):pass
    done={k:r for k,r in previous.items() if verified(r,fp)}
    pending=[r for r in tasks if r['id'] not in done]
    if args.max_tasks:pending=pending[:args.max_tasks]
    print(f'[{len(done)}/{len(tasks)}] E08区域原子放置，本批{len(pending)}',flush=True)
    for entry in pending:
        try:result=seal_result(run_variant(entry,refs[entry['id']],out,torch,args.mode,priors.get(entry['id'])),fp)
        except Exception as exc:
            append_run_row(ledger,{'id':entry['id'],'status':'failed','fingerprint':fp,'error':repr(exc)});raise
        append_run_row(ledger,result);done[entry['id']]=result
        print(f'[{len(done)}/{len(tasks)}] {entry["id"]} 官方{result["baseline"]["makespan"]}→{result["real"]["makespan"]}，候选官方{result["official_count"]}',flush=True)
    summary={'expected':len(tasks),'completed':len(done),'wins':sum(r['real']['makespan']<r['baseline']['makespan'] for r in done.values()),
             'losses':sum(r['real']['makespan']>r['baseline']['makespan'] for r in done.values())}
    target=out/'summary.json';text=json.dumps(summary,ensure_ascii=False,indent=2)
    if not target.exists() or target.read_text(encoding='utf-8')!=text:target.write_text(text,encoding='utf-8')
    print(text,flush=True)


if __name__=='__main__':main()
