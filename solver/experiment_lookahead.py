"""E07固定切分深度一/三对照；全图资源重放、显卡粗特征、两次原官方。"""
import argparse
import ast
import gzip
import hashlib
import json
from pathlib import Path
import time
import zipfile

from audit_resource_replay import good as valid_reference
from batch_features import BatchFeatures,scalar_features
from experiment_branch_partition import seal_result,verified
from local_template_cache import TemplateCache
from lookahead_place import CandidateOracle,search_lookahead,order_key
from model import Model
from pipeline import real_evaluate
from run_all import parse_cases,block_cap_for,ensure_run_manifest,run_input_files,append_run_row,plan_digest,valid_official
from scene_a_event import derive_multicore_plan
from scene_a_replay import build_resource_evaluator

ROOT=Path(__file__).resolve().parents[1]


def run_variant(entry,reference,depth,out,torch,future_policy='chain',cut_policy='start',keep_current=False):
    started=time.perf_counter();case,n=entry['case'],entry['N'];eid=f'{entry["id"]}_D{depth}'
    source=entry['variants']['swap'];path=Path(source['plan_path'])
    assert hashlib.sha256(path.read_bytes()).hexdigest()==source['plan_sha256']
    plan=json.loads(path.read_text(encoding='utf-8'))
    graph=json.loads((ROOT/'通用神经网络处理器下的多核调度问题附件/data'/f'{case}.json').read_text(encoding='utf-8'))
    ref_path=Path(next(iter(reference['artifacts'])))
    baseline=json.loads(gzip.decompress(ref_path.read_bytes()).decode('utf-8'))
    assert reference['source_sha256']==source['plan_sha256'] and reference['real']==source['real']
    view=derive_multicore_plan(graph,plan)
    model=Model(graph,block_ops_cap=block_cap_for(sum(o['op'] not in ('COPY_IN','COPY_OUT') for o in graph['ops'])))
    assignment=[plan['node_to_subgraph'][str(b[0])] for b in model.blocks]
    assert model.plan_from(assignment,plan['core_schedules'])==plan
    torch.cuda.reset_peak_memory_stats();batch=BatchFeatures(model,n)
    feature_stats={'seconds':0.0,'candidates':0,'calls':0,'equal':True}
    def features(orders_batch):
        before=time.perf_counter();owners=[]
        for orders in orders_batch:
            owner=[0]*(max(assignment)+1)
            for core,queue in enumerate(orders):
                for task in queue:owner[task]=core
            owners.append(owner)
        values=batch.evaluate([assignment]*len(owners),owners)
        if values!=[scalar_features(model,assignment,c,n) for c in owners]:
            raise ValueError('前瞻显卡粗特征与中央处理器不同')
        feature_stats['seconds']+=time.perf_counter()-before
        feature_stats['candidates']+=len(owners);feature_stats['calls']+=1
        return values
    cache=TemplateCache();before=time.perf_counter()
    evaluator=build_resource_evaluator(unified=True,template_cache=cache)
    setup=time.perf_counter()-before
    oracle=CandidateOracle(graph,plan,baseline,evaluator,max_evals=32,
        progress=lambda stats:print(f'  {eid} [{stats["evaluations"]}/32] 全图影子重放，缓存复用{stats["cache_hits"]}',flush=True))
    ranked,audit=search_lookahead(oracle,view['subgraph_preds'],depth=depth,feature_batch=features,future_policy=future_policy,cut_policy=cut_policy,keep_current=keep_current)
    baseline_key=order_key(plan['core_schedules'])
    shortlist=[r for r in ranked if order_key(r['orders'])!=baseline_key and
               (r['real']['makespan'],r['real']['added_copy_bytes']) <
               (source['real']['makespan'],source['real']['added_copy_bytes'])][:2]
    best,best_plan=source['real'],plan;artifacts={};checks=[];errors=[]
    for i,candidate in enumerate(shortlist):
        proposal={**plan,'core_schedules':candidate['orders']}
        candidate_path=out/f'{eid}_candidate_{i}.json'
        candidate_path.write_text(json.dumps(proposal),encoding='utf-8')
        artifacts[str(candidate_path)]=hashlib.sha256(candidate_path.read_bytes()).hexdigest()
        truth=real_evaluate(graph,proposal,'A')
        equal=truth==candidate['real']
        checks.append({'plan_path':str(candidate_path),'plan_id':plan_digest(proposal),
                       'resource':candidate['real'],'official':truth,'equal':equal})
        if not valid_official(truth) or not equal:
            errors.append({'stage':'official_guard','candidate':i,'result':truth});continue
        if (truth['makespan'],truth['added_copy_bytes']) < (best['makespan'],best['added_copy_bytes']):
            best,best_plan=truth,proposal
    target=out/f'{eid}_selected.json';target.write_text(json.dumps(best_plan),encoding='utf-8')
    artifacts[str(target)]=hashlib.sha256(target.read_bytes()).hexdigest()
    return {'id':eid,'case':case,'N':n,'depth':depth,'status':'official_success','baseline':source['real'],
        'source_plan':str(path),'source_sha256':source['plan_sha256'],
        'baseline_reference_binding':reference['binding'],'real':best,'plan_path':str(target),'plan_id':plan_digest(best_plan),
        'artifacts':artifacts,'official_count':len(checks),'checks':checks,'errors':errors,'search':audit,
        'archive':[{'orders':r['orders'],'resource':r['real']} for r in ranked],
        'cache_stats':cache.stats,'setup_seconds':setup,'gpu_features':feature_stats,
        'gpu_peak_allocated':torch.cuda.max_memory_allocated(),'seconds':time.perf_counter()-started,
        'scope':'固定E03切分，32全图资源重放上限、3当前决策；实际次数不相同，非等墙钟；最后2原官方保底'}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--cases',default='44,49,65,67');p.add_argument('--cores',default='2,4')
    p.add_argument('--depths',default='1,3')
    p.add_argument('--future-policy',choices=('chain','ready'),default='chain')
    p.add_argument('--cut-policy',choices=('start','data_ready'),default='start')
    p.add_argument('--keep-current',action='store_true',help='补入当前完整核序作为可选择动作，不新增当前方案重放')
    p.add_argument('--max-tasks',type=int,default=0);p.add_argument('--output-dir',required=True)
    p.add_argument('--reuse-depth1-ledger',help='仅复用已证明不受首次前缀修复影响的深度1记录；保留原执行指纹')
    args=p.parse_args();import torch
    if not torch.cuda.is_available():p.error('需已有显卡环境批量评分')
    cases=parse_cases(args.cases);cores=list(dict.fromkeys(map(int,args.cores.split(','))))
    depths=list(dict.fromkeys(map(int,args.depths.split(','))))
    if args.max_tasks<0 or any(n not in (2,3,4,5) for n in cores):p.error('参数非法')
    if not depths or any(d not in (1,3) for d in depths):p.error('仅支持深度1/3')
    source=ROOT/'results/p1_e03_r02/order_pairs.jsonl'
    ref_log=ROOT/'results/p1_e06_ddr_r01/resource_replay.jsonl'
    ref_manifest=ref_log.with_suffix('.manifest.json')
    ref_fp=json.loads(ref_manifest.read_text(encoding='utf-8'))['fingerprint']
    mapping={(r['case'],r['N']):r for r in map(json.loads,source.read_text(encoding='utf-8').splitlines())}
    refs={r['id']:r for r in map(json.loads,ref_log.read_text(encoding='utf-8').splitlines())}
    tasks=[(mapping[c,n],depth) for c in cases for n in cores for depth in depths]
    files=run_input_files(cases)+[source,ref_log,ref_manifest]
    reusable=[]
    if args.reuse_depth1_ledger:
        prior_log=Path(args.reuse_depth1_ledger).resolve()
        prior_manifest=prior_log.with_suffix('.manifest.json')
        prior_meta=json.loads(prior_manifest.read_text(encoding='utf-8'))
        frozen=prior_log.parent/'frozen_interrupted/source.zip'
        module_path=Path(__file__).with_name('lookahead_place.py').resolve()
        with zipfile.ZipFile(frozen) as z:
            old_module=z.read('solver/lookahead_place.py')
            old_runner=z.read('solver/experiment_lookahead.py')
        for path,raw in ((module_path,old_module),(Path(__file__).resolve(),old_runner)):
            if hashlib.sha256(raw).hexdigest()!=prior_meta['files'][str(path)]:
                raise ValueError('旧源码快照与原实验清单不一致')
        expected=old_module.decode('utf-8').replace('\r\n','\n').replace(
            'locked|{root},feature_batch)[:2]','locked,feature_batch)[:2]')
        if module_path.read_text(encoding='utf-8')!=expected:
            raise ValueError('当前修改超出已证明与深度1无关的单行修复，不能复用')
        def variant_ast(source_text):
            return ast.dump(next(n for n in ast.parse(source_text).body if isinstance(n,ast.FunctionDef) and n.name=='run_variant'))
        if variant_ast(old_runner.decode('utf-8'))!=variant_ast(Path(__file__).read_text(encoding='utf-8')):
            raise ValueError('模式执行逻辑已改变，不能复用深度1')
        for path,digest in prior_meta['files'].items():
            if Path(path).resolve() in (module_path,Path(__file__).resolve()):continue
            if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest:
                raise ValueError('其他依赖已变化，不能复用深度1')
        for row in map(json.loads,prior_log.read_text(encoding='utf-8').splitlines()):
            if row.get('depth')==1 and verified(row,prior_meta['fingerprint']):reusable.append(row)
        files.extend([prior_log,prior_manifest,frozen])
        files.extend(Path(p) for r in reusable for p in r['artifacts'])
    for entry,_ in tasks:
        ref=refs[entry['id']]
        if not valid_reference(ref,ref_fp):raise ValueError('完整官方参照无效')
        files+=[Path(entry['variants']['swap']['plan_path'])]+[Path(s) for s in ref['artifacts']]
    out=Path(args.output_dir).resolve();out.mkdir(parents=True,exist_ok=True);ledger=out/'lookahead_pairs.jsonl'
    fp=ensure_run_manifest(ledger,files,{'cases':cases,'cores':cores,'depths':depths,'full_evals':32,'future_policy':args.future_policy,'cut_policy':args.cut_policy,'keep_current':args.keep_current,
        'decisions':3,'closure_limit':8,'future_cores':2,'official_limit':2,'cache_entries':128,'payload_mb':64,
        'gpu':torch.cuda.get_device_name(0),'reuse_depth1_ledger':args.reuse_depth1_ledger})
    previous={}
    if ledger.exists():
        for line in ledger.read_text(encoding='utf-8').splitlines():
            try:r=json.loads(line);previous[r['id']]=r
            except (ValueError,KeyError):pass
    done={k:r for k,r in previous.items() if verified(r,fp)}
    for prior in reusable:
        if prior['id'] in done or (prior['case'],prior['N']) not in {(r['case'],r['N']) for r,_ in tasks}:continue
        reused=dict(prior);old_binding=reused.pop('binding')
        reused['reuse_provenance']={'original_ledger':str(prior_log),'original_fingerprint':prior['fingerprint'],
            'original_binding':old_binding,'executed_this_round':False,
            'proof':'唯一算法差异在未来层，深度1没有未来层；执行函数语法树与其他依赖摘要完全相同'}
        reused=seal_result(reused,fp);append_run_row(ledger,reused);done[prior['id']]=reused
    pending=[(e,d) for e,d in tasks if f'{e["id"]}_D{d}' not in done]
    if args.max_tasks:pending=pending[:args.max_tasks]
    print(f'[{len(done)}/{len(tasks)}] E07配置×前瞻深度，本批{len(pending)}',flush=True)
    for entry,depth in pending:
        eid=f'{entry["id"]}_D{depth}'
        try:
            result=seal_result(run_variant(entry,refs[entry['id']],depth,out,torch,future_policy=args.future_policy,cut_policy=args.cut_policy,keep_current=args.keep_current),fp)
            done[eid]=result
        except Exception as exc:
            append_run_row(ledger,{'id':eid,'status':'failed','fingerprint':fp,'error':repr(exc)})
            raise
        append_run_row(ledger,result)
        print(f'[{len(done)}/{len(tasks)}] {eid} 官方{result["baseline"]["makespan"]}→{result["real"]["makespan"]}，'
              f'影子{result["search"]["oracle"]["evaluations"]}次，候选官方{result["official_count"]}次',flush=True)
    summary={str(d):{'expected':len(cases)*len(cores),'completed':sum(r['depth']==d for r in done.values()),
                    'wins':sum(r['depth']==d and r['real']['makespan']<r['baseline']['makespan'] for r in done.values()),
                    'losses':sum(r['depth']==d and r['real']['makespan']>r['baseline']['makespan'] for r in done.values())}
             for d in depths}
    target=out/'summary.json';text=json.dumps(summary,ensure_ascii=False,indent=2)
    if not target.exists() or target.read_text(encoding='utf-8')!=text:target.write_text(text,encoding='utf-8')
    print(text,flush=True)


if __name__=='__main__':main()
