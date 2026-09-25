"""E08结构接口首轮：固定已选E05分支切分，只改变区域核心与完整队列。"""
import argparse
import gzip
import hashlib
from itertools import combinations
import json
from pathlib import Path
import time

from batch_features import BatchFeatures,scalar_features
from experiment_branch_partition import seal_result,verified
from local_template_cache import TemplateCache
from lookahead_place import compact_result,order_key
from model import Model
from pipeline import real_evaluate
from region_group_refine import region_trace,select_joint_candidates
from region_structure import split_region_identity,complete_region_orders
from run_all import parse_cases,block_cap_for,ensure_run_manifest,run_input_files,append_run_row,plan_digest,valid_official
from scene_a_event import derive_multicore_plan
from scene_a_replay import build_resource_evaluator

ROOT=Path(__file__).resolve().parents[1]


def run_variant(source,out,torch):
    started=time.perf_counter();eid=source['id'];case,n=source['case'],source['N']
    matched=[r for r in source['evaluated'] if r['plan_id']==source['plan_id']]
    if not matched:raise ValueError('此接口试验要求已有经官方确认的分支切分')
    root=matched[0]['source']['task']
    plan=json.loads(Path(source['plan_path']).read_text(encoding='utf-8'))
    original=json.loads(Path(source['source']).read_text(encoding='utf-8'))
    graph=json.loads((ROOT/'通用神经网络处理器下的多核调度问题附件/data'/f'{case}.json').read_text(encoding='utf-8'))
    model=Model(graph,block_ops_cap=block_cap_for(sum(o['op'] not in ('COPY_IN','COPY_OUT') for o in graph['ops'])))
    old=[original['node_to_subgraph'][str(b[0])] for b in model.blocks]
    assignment=[plan['node_to_subgraph'][str(b[0])] for b in model.blocks]
    if model.plan_from(assignment,plan['core_schedules'])!=plan:raise ValueError('输入分支切分块映射无损性失败')
    region,outside_map=split_region_identity(old,assignment,root)
    cache=TemplateCache();evaluator=build_resource_evaluator(unified=True,template_cache=cache)
    before=time.perf_counter();raw=evaluator(graph,plan,60,{'L1':524288,'UB':131072},1000,100);baseline_seconds=time.perf_counter()-before
    baseline=compact_result(raw,plan['core_schedules'])
    if baseline['real']!=source['real']:raise ValueError('完整资源时间线与E05已知官方指标不同')
    # 这是首次保存该输入的完整资源时间线；不是新增原官方结果。
    trace_path=out/f'{eid}_baseline_resource.json.gz';trace_path.write_bytes(gzip.compress(json.dumps(raw).encode(),mtime=0))
    artifacts={str(trace_path):hashlib.sha256(trace_path.read_bytes()).hexdigest()}
    baseline_trace=region_trace(raw,[region]);durations={t:r['duration'] for t,r in baseline['tasks'].items()}
    preds=derive_multicore_plan(graph,plan)['subgraph_preds'];rows={};attempts=invalid=0
    before=time.perf_counter()
    for width in range(1,n+1):
        for subset in combinations(range(n),width):
            for priority in ('critical','earliest'):
                attempts+=1;candidate=complete_region_orders(durations,preds,plan['core_schedules'],region,subset,priority)
                if candidate is None:invalid+=1;continue
                key=order_key(candidate['orders'])
                if key==order_key(plan['core_schedules']):continue
                origin={'region':0,'tasks':sorted(region),'old_task':root,'width':len(candidate['actual_cores']),
                    'cores':candidate['actual_cores'],'requested_cores':list(subset),'priority':priority}
                row=rows.setdefault(key,{'orders':candidate['orders'],'coarse':candidate['coarse'],'origins':[]})
                row['origins'].append(origin)
    generation_seconds=time.perf_counter()-before;rows=list(rows.values())
    torch.cuda.reset_peak_memory_stats();batch=BatchFeatures(model,n);owners=[]
    for r in rows:
        owner=[0]*(max(assignment)+1)
        for c,q in enumerate(r['orders']):
            for t in q:owner[t]=c
        owners.append(owner)
    before=time.perf_counter();values=batch.evaluate([assignment]*len(owners),owners) if owners else []
    if values!=[scalar_features(model,assignment,c,n) for c in owners]:raise ValueError('显卡与中央处理器粗特征不同')
    gpu_seconds=time.perf_counter()-before
    for row,value in zip(rows,values):row['features']=value
    selected,screen=select_joint_candidates(rows);selected_keys={order_key(r['orders']) for r in selected}
    for row in rows:row['admitted']=order_key(row['orders']) in selected_keys
    evaluated=[]
    for i,candidate in enumerate(selected):
        proposal={**plan,'core_schedules':candidate['orders']};path=out/f'{eid}_candidate_{i:02d}.json'
        path.write_text(json.dumps(proposal),encoding='utf-8');artifacts[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        before=time.perf_counter();raw=evaluator(graph,proposal,60,{'L1':524288,'UB':131072},1000,100);elapsed=time.perf_counter()-before
        result=compact_result(raw,candidate['orders'])
        if any(result['real'][k]!=source['real'][k] for k in source['real'] if k!='makespan'):raise ValueError('同一新切分重排改变搬运')
        outside={str(t):r['end']-baseline['tasks'][t]['end'] for t,r in result['tasks'].items() if t not in region}
        evaluated.append({'plan_path':str(path),'resource':result['real'],'coarse':candidate['coarse'],'orders':candidate['orders'],
            'origins':candidate['origins'],'seconds':elapsed,'region_trace':region_trace(raw,[region]),'outside_end_delta':outside})
        if (i+1)%4==0 or i+1==len(selected):print(f'  {eid} [{i+1}/{len(selected)}] 固定分支结构区域重放',flush=True)
    best,best_plan=source['real'],plan;checks=[]
    ranked=sorted(evaluated,key=lambda r:(r['resource']['makespan'],order_key(r['orders'])))
    for candidate in [r for r in ranked if r['resource']['makespan']<source['real']['makespan']][:2]:
        proposal=json.loads(Path(candidate['plan_path']).read_text(encoding='utf-8'))
        before=time.perf_counter();truth=real_evaluate(graph,proposal,'A');elapsed=time.perf_counter()-before
        if not valid_official(truth) or truth!=candidate['resource']:raise ValueError('原官方确认不一致')
        checks.append({'plan_path':candidate['plan_path'],'official':truth,'resource':candidate['resource'],'seconds':elapsed})
        if truth['makespan']<best['makespan']:best,best_plan=truth,proposal
    target=out/f'{eid}_selected.json';target.write_text(json.dumps(best_plan),encoding='utf-8');artifacts[str(target)]=hashlib.sha256(target.read_bytes()).hexdigest()
    return {'id':eid,'case':case,'N':n,'status':'official_success','baseline':source['real'],'real':best,
            'source_plan':source['plan_path'],'source_sha256':source['artifacts'][source['plan_path']],
            'source_binding':source['binding'],'original_E03':source['baseline'],'region':sorted(region),'old_task':root,'outside_identity':outside_map,
            'plan_path':str(target),'plan_id':plan_digest(best_plan),'artifacts':artifacts,'checks':checks,'official_count':len(checks),
            'baseline_resource_seconds':baseline_seconds,'baseline_trace':baseline_trace,'generation_seconds':generation_seconds,
            'generation':{'attempts':attempts,'invalid':invalid,'distinct':len(rows)},'candidate_features':rows,'screen':screen,'evaluated':evaluated,
            'gpu_features':{'seconds':gpu_seconds,'candidates':len(rows),'equal':True,'peak_allocated':torch.cuda.max_memory_allocated()},
            'cache_stats':cache.stats,'seconds':time.perf_counter()-started,
            'scope':'固定E05已选官方切分/最多N核区域完整构造接口试验；不是20结构菜单联合选择，不与E03混基线'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--cases',default='5,49,82');p.add_argument('--cores',default='2,4')
    p.add_argument('--max-tasks',type=int,default=0);p.add_argument('--output-dir',required=True);args=p.parse_args();import torch
    if not torch.cuda.is_available():p.error('需要已有显卡环境')
    cases=parse_cases(args.cases);cores=list(dict.fromkeys(map(int,args.cores.split(','))))
    if args.max_tasks<0 or any(n not in (2,3,4,5) for n in cores):p.error('参数非法')
    source=ROOT/'results/p1_e05_branch_r03/branch_experiments.jsonl';meta=source.with_suffix('.manifest.json')
    source_fp=json.loads(meta.read_text(encoding='utf-8'))['fingerprint']
    mapping={(r['case'],r['N']):r for r in map(json.loads,source.read_text(encoding='utf-8').splitlines())}
    tasks=[mapping[c,n] for c in cases for n in cores];files=run_input_files(cases)+[source,meta]
    for r in tasks:
        if not verified(r,source_fp):raise ValueError('E05保存证据无效')
        files.extend(map(Path,r['artifacts']));files.append(Path(r['source']))
    out=Path(args.output_dir).resolve();out.mkdir(parents=True,exist_ok=True);ledger=out/'structure_pairs.jsonl'
    fp=ensure_run_manifest(ledger,files,{'cases':cases,'cores':cores,'structure':'fixed_selected_E05','resource_limit':12,'official_limit':2,
        'priorities':['critical','earliest'],'gpu':torch.cuda.get_device_name(0)})
    previous={}
    if ledger.exists():
        for line in ledger.read_text(encoding='utf-8').splitlines():
            try:r=json.loads(line);previous[r['id']]=r
            except (ValueError,KeyError):pass
    done={k:r for k,r in previous.items() if verified(r,fp)};pending=[r for r in tasks if r['id'] not in done]
    if args.max_tasks:pending=pending[:args.max_tasks]
    print(f'[{len(done)}/{len(tasks)}] E08固定分支结构接口，本批{len(pending)}',flush=True)
    for r in pending:
        try:result=seal_result(run_variant(r,out,torch),fp)
        except Exception as exc:append_run_row(ledger,{'id':r['id'],'status':'failed','fingerprint':fp,'error':repr(exc)});raise
        append_run_row(ledger,result);done[r['id']]=result
        print(f'[{len(done)}/{len(tasks)}] {r["id"]} 官方{result["baseline"]["makespan"]}→{result["real"]["makespan"]}',flush=True)
    summary={'expected':len(tasks),'completed':len(done),'wins':sum(r['real']['makespan']<r['baseline']['makespan'] for r in done.values()),
             'losses':sum(r['real']['makespan']>r['baseline']['makespan'] for r in done.values())}
    text=json.dumps(summary,ensure_ascii=False,indent=2);path=out/'summary.json'
    if not path.exists() or path.read_text(encoding='utf-8')!=text:path.write_text(text,encoding='utf-8')
    print(text,flush=True)


if __name__=='__main__':main()
