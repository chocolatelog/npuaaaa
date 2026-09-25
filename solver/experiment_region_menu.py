"""E08内部结构菜单：两范式、实际核心集合、完整核序与独立官方保底。"""
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
from region_group_refine import select_joint_candidates,region_trace
from region_structure import build_structure_candidates
from solution import Sol
from run_all import parse_cases,block_cap_for,ensure_run_manifest,run_input_files,append_run_row,plan_digest,valid_official
from scene_a_event import derive_multicore_plan
from scene_a_replay import build_resource_evaluator
from evaluation_validation import validate_task_order

ROOT=Path(__file__).resolve().parents[1]


def merge_prior_rows(first,second):
    """只合并已验证且同输入的完整方案证据；冲突不以较新时间掩盖。"""
    rows=[r for r in (first,second) if r is not None]
    if not rows:raise ValueError('缺少复用来源')
    base=rows[0]
    for row in rows:
        if any(row[k]!=base[k] for k in ('id','source_sha256','baseline')):raise ValueError('复用清单输入冲突')
    merged={**base}
    for field,metric in (('evaluated','resource'),('checks','official')):
        known={}
        for row in rows:
            for value in row[field]:
                key=value.get('plan_id') or plan_digest(json.loads(Path(value['plan_path']).read_text(encoding='utf-8')))
                if key in known and known[key][metric]!=value[metric]:raise ValueError('同计划复用指标冲突')
                if key not in known:
                    origin=value.get('evidence_binding') or (value.get('source_binding') if value.get('reused') else row['binding'])
                    known[key]={**value,'evidence_binding':origin}
        merged[field]=list(known.values())
    return merged


def run_variant(entry,reference,out,torch,core_policy='tied',prior=None,menu_policy='single'):
    started=time.perf_counter();source=entry['variants']['swap'];case,n=entry['case'],entry['N'];eid=entry['id']
    path=Path(source['plan_path'])
    if hashlib.sha256(path.read_bytes()).hexdigest()!=source['plan_sha256']:raise ValueError('源计划失效')
    plan=json.loads(path.read_text(encoding='utf-8'))
    graph=json.loads((ROOT/'通用神经网络处理器下的多核调度问题附件/data'/f'{case}.json').read_text(encoding='utf-8'))
    raw=json.loads(gzip.decompress(Path(next(iter(reference['artifacts']))).read_bytes()))
    baseline=compact_result(raw,plan['core_schedules'])
    if baseline['real']!=source['real'] or reference['source_sha256']!=source['plan_sha256']:raise ValueError('官方参照不对应输入')
    model=Model(graph,block_ops_cap=block_cap_for(sum(o['op'] not in ('COPY_IN','COPY_OUT') for o in graph['ops'])))
    assignment=[plan['node_to_subgraph'][str(b[0])] for b in model.blocks];owners=[0]*(max(assignment)+1)
    for c,q in enumerate(plan['core_schedules']):
        for t in q:owners[t]=c
    if model.plan_from(assignment,plan['core_schedules'])!=plan:raise ValueError('输入块映射不一致')
    known={};known_official={}
    if prior:
        if prior['source_sha256']!=source['plan_sha256'] or prior['baseline']!=source['real']:raise ValueError('复用结果输入不同')
        known={r['plan_id']:r for r in prior['evaluated']}
        known_official={plan_digest(json.loads(Path(c['plan_path']).read_text(encoding='utf-8'))):c for c in prior['checks']}
    before=time.perf_counter();rows,generation=build_structure_candidates(graph,model,Sol(assignment,owners),plan,baseline,core_policy=core_policy,menu_policy=menu_policy)
    generation_seconds=time.perf_counter()-before
    torch.cuda.reset_peak_memory_stats();batch=BatchFeatures(model,n)
    before=time.perf_counter();features=batch.evaluate([r['assignment'] for r in rows],[r['owners'] for r in rows]) if rows else []
    if features!=[scalar_features(model,r['assignment'],r['owners'],n) for r in rows]:raise ValueError('显卡批特征不一致')
    gpu_seconds=time.perf_counter()-before
    for row,feature in zip(rows,features):row['features']=feature
    selected,screen=select_joint_candidates(rows)
    selected_keys={(tuple(r['assignment']),order_key(r['orders'])) for r in selected}
    for row in rows:row['admitted']=(tuple(row['assignment']),order_key(row['orders'])) in selected_keys
    cache=TemplateCache();before=time.perf_counter();evaluator=build_resource_evaluator(unified=True,template_cache=cache);setup_seconds=time.perf_counter()-before
    evaluated=[];artifacts={};movement_by_partition={}
    for i,candidate in enumerate(selected):
        proposal=model.plan_from(candidate['assignment'],candidate['orders'])
        validate_task_order(derive_multicore_plan(graph,proposal))
        target=out/f'{eid}_candidate_{i:02d}.json';target.write_text(json.dumps(proposal),encoding='utf-8')
        artifacts[str(target)]=hashlib.sha256(target.read_bytes()).hexdigest()
        prior_value=known.get(plan_digest(proposal))
        if prior_value:
            result={'real':prior_value['resource']};elapsed=0.;trace=prior_value['trace']
        else:
            before=time.perf_counter();raw=evaluator(graph,proposal,60,{'L1':524288,'UB':131072},1000,100);elapsed=time.perf_counter()-before
            result=compact_result(raw,candidate['orders']);trace=[]
            for origin in candidate['origins']:
                changes={str(old):result['tasks'][int(new)]['end']-baseline['tasks'][int(old)]['end'] for old,new in origin['outside_identity'].items()}
                trace.append({'origin':origin,'region_timeline':region_trace(raw,[origin['tasks']]),'outside_end_delta':changes})
        if result['real']['partition_added']!=max(0,candidate['features']['boundary_bytes']-model.original_copy_bytes):raise ValueError('真实边界与批特征不一致')
        movement={k:v for k,v in result['real'].items() if k!='makespan'};key=tuple(candidate['assignment'])
        if key in movement_by_partition and movement_by_partition[key]!=movement:raise ValueError('同一切分不同分核的搬运变化')
        movement_by_partition[key]=movement
        evaluated.append({'plan_path':str(target),'plan_id':plan_digest(proposal),'assignment':candidate['assignment'],'orders':candidate['orders'],
            'origins':candidate['origins'],'coarse':candidate['coarse'],'features':candidate['features'],'resource':result['real'],
            'seconds':elapsed,'trace':trace,'reused':prior_value is not None,
            'source_binding':prior_value.get('evidence_binding',prior['binding']) if prior_value else None,
            'source_seconds':(prior_value.get('source_seconds') if prior_value.get('reused') else prior_value['seconds']) if prior_value else None})
        if (i+1)%4==0 or i+1==len(selected):print(f'  {eid} [{i+1}/{len(selected)}] 结构菜单完整重放',flush=True)
    best,best_plan=source['real'],plan;checks=[]
    ranked=sorted(evaluated,key=lambda r:(r['resource']['makespan'],r['resource']['added_copy_bytes'],r['plan_id']))
    for row in [r for r in ranked if (r['resource']['makespan'],r['resource']['added_copy_bytes'])<(source['real']['makespan'],source['real']['added_copy_bytes'])][:2]:
        proposal=json.loads(Path(row['plan_path']).read_text(encoding='utf-8'));prior_check=known_official.get(plan_digest(proposal))
        if prior_check:truth=prior_check['official'];elapsed=0.
        else:
            before=time.perf_counter();truth=real_evaluate(graph,proposal,'A');elapsed=time.perf_counter()-before
        if not valid_official(truth) or truth!=row['resource']:raise ValueError('结构菜单原官方不同')
        checks.append({'plan_path':row['plan_path'],'resource':row['resource'],'official':truth,'seconds':elapsed,
                       'reused':prior_check is not None,'source_binding':prior_check.get('evidence_binding',prior['binding']) if prior_check else None})
        if (truth['makespan'],truth['added_copy_bytes'])<(best['makespan'],best['added_copy_bytes']):best,best_plan=truth,proposal
    target=out/f'{eid}_selected.json';target.write_text(json.dumps(best_plan),encoding='utf-8');artifacts[str(target)]=hashlib.sha256(target.read_bytes()).hexdigest()
    return {'id':eid,'case':case,'N':n,'status':'official_success','baseline':source['real'],'real':best,
        'source_plan':str(path),'source_sha256':source['plan_sha256'],'baseline_reference_binding':reference['binding'],
        'plan_path':str(target),'plan_id':plan_digest(best_plan),'artifacts':artifacts,'checks':checks,'official_count':sum(not c['reused'] for c in checks),
        'official_reused_count':sum(c['reused'] for c in checks),'core_policy':core_policy,'menu_policy':menu_policy,
        'generation':generation,'generation_seconds':generation_seconds,'candidate_features':rows,'screen':screen,'evaluated':evaluated,
        'gpu_features':{'equal':True,'candidates':len(rows),'seconds':gpu_seconds,'peak_allocated':torch.cuda.max_memory_allocated()},
        'cache_stats':cache.stats,'setup_seconds':setup_seconds,'seconds':time.perf_counter()-started,
        'scope':'固定E03输入；两区域结构菜单、2000廉价/20池/12资源/2官方；与E05仅同输入参照，非等墙钟'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--cases',default='5,49,82');p.add_argument('--cores',default='2,4')
    p.add_argument('--core-policy',choices=('tied','all'),default='tied')
    p.add_argument('--menu-policy',choices=('single','diverse','seed_combinations'),default='single')
    p.add_argument('--reuse-ledger')
    p.add_argument('--extra-reuse-ledger',action='append',default=[])
    p.add_argument('--max-tasks',type=int,default=0);p.add_argument('--output-dir',required=True);args=p.parse_args();import torch
    if not torch.cuda.is_available():p.error('需已有显卡环境')
    cases=parse_cases(args.cases);cores=list(dict.fromkeys(map(int,args.cores.split(','))))
    if args.max_tasks<0 or any(n not in (2,3,4,5) for n in cores):p.error('参数非法')
    source=ROOT/'results/p1_e03_r02/order_pairs.jsonl';ref_log=ROOT/'results/p1_e06_ddr_r01/resource_replay.jsonl'
    ref_meta=ref_log.with_suffix('.manifest.json');ref_fp=json.loads(ref_meta.read_text(encoding='utf-8'))['fingerprint']
    mapping={(r['case'],r['N']):r for r in map(json.loads,source.read_text(encoding='utf-8').splitlines())}
    refs={r['id']:r for r in map(json.loads,ref_log.read_text(encoding='utf-8').splitlines())};tasks=[mapping[c,n] for c in cases for n in cores]
    files=run_input_files(cases)+[source,ref_log,ref_meta]
    for r in tasks:
        ref=refs[r['id']]
        if not good(ref,ref_fp):raise ValueError('原官方参照无效')
        files.extend(map(Path,ref['artifacts']));files.append(Path(r['variants']['swap']['plan_path']))
    priors={}
    for reuse_path in ([args.reuse_ledger] if args.reuse_ledger else [])+args.extra_reuse_ledger:
        log=Path(reuse_path).resolve();meta_path=log.with_suffix('.manifest.json');meta=json.loads(meta_path.read_text(encoding='utf-8'))
        runner=Path(__file__).resolve();generator=runner.with_name('region_structure.py');frozen=log.parent/'frozen_source/source.zip'
        with zipfile.ZipFile(frozen) as z:old_runner=z.read('solver/experiment_region_menu.py')
        if hashlib.sha256(old_runner).hexdigest()!=meta['files'][str(runner)]:raise ValueError('源入口快照不匹配')
        def calls(src):return sorted(ast.dump(x) for x in ast.walk(ast.parse(src)) if isinstance(x,ast.Call) and isinstance(x.func,ast.Name) and x.func.id in ('evaluator','real_evaluate'))
        if calls(old_runner.decode('utf-8'))!=calls(runner.read_text(encoding='utf-8')):raise ValueError('评估调用或硬件参数变化')
        for path,digest in meta['files'].items():
            if Path(path).resolve() not in (runner,generator) and hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest:raise ValueError('评估依赖变化，不能复用')
        for row in map(json.loads,log.read_text(encoding='utf-8').splitlines()):
            if not verified(row,meta['fingerprint']):raise ValueError('原结果绑定失效')
            priors[row['id']]=merge_prior_rows(priors.get(row['id']),row);files.extend(map(Path,row['artifacts']))
        files.extend([log,meta_path,frozen])
    out=Path(args.output_dir).resolve();out.mkdir(parents=True,exist_ok=True);ledger=out/'menu_pairs.jsonl'
    fp=ensure_run_manifest(ledger,files,{'cases':cases,'cores':cores,'regions':2,'structures_limit':20,'coarse_limit':2000,'pool_limit':20,
        'resource_limit':12,'official_limit':2,'families':['original','branch','balanced'],'gpu':torch.cuda.get_device_name(0),
        'core_policy':args.core_policy,'menu_policy':args.menu_policy,'reuse_ledger':args.reuse_ledger,'extra_reuse_ledgers':args.extra_reuse_ledger})
    previous={}
    if ledger.exists():
        for line in ledger.read_text(encoding='utf-8').splitlines():
            try:r=json.loads(line);previous[r['id']]=r
            except (ValueError,KeyError):pass
    done={k:r for k,r in previous.items() if verified(r,fp)};pending=[r for r in tasks if r['id'] not in done]
    if args.max_tasks:pending=pending[:args.max_tasks]
    print(f'[{len(done)}/{len(tasks)}] E08内部结构菜单，本批{len(pending)}',flush=True)
    for r in pending:
        try:result=seal_result(run_variant(r,refs[r['id']],out,torch,args.core_policy,priors.get(r['id']),args.menu_policy),fp)
        except Exception as exc:append_run_row(ledger,{'id':r['id'],'status':'failed','fingerprint':fp,'error':repr(exc)});raise
        append_run_row(ledger,result);done[r['id']]=result
        print(f'[{len(done)}/{len(tasks)}] {r["id"]} 官方{result["baseline"]["makespan"]}→{result["real"]["makespan"]}',flush=True)
    summary={'expected':len(tasks),'completed':len(done),'wins':sum(r['real']['makespan']<r['baseline']['makespan'] for r in done.values()),
        'losses':sum(r['real']['makespan']>r['baseline']['makespan'] for r in done.values())}
    text=json.dumps(summary,ensure_ascii=False,indent=2);path=out/'summary.json'
    if not path.exists() or path.read_text(encoding='utf-8')!=text:path.write_text(text,encoding='utf-8')
    print(text,flush=True)


if __name__=='__main__':main()
