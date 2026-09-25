"""固定初始方案完整插入邻域审计；诊断成本独立、逐候选恢复。"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import time

from audit_hotspot_candidates import load_rows,reusable
from audit_resource_replay import good
from batch_features import BatchFeatures,scalar_features
from experiment_branch_partition import seal_result,verified
from local_template_cache import TemplateCache
from lookahead_place import compact_result,decision_window,order_key,propose_placements
from model import Model
from pipeline import real_evaluate
from run_all import append_run_row,block_cap_for,ensure_run_manifest,plan_digest,run_input_files,valid_official
from scene_a_event import derive_multicore_plan
from scene_a_replay import build_resource_evaluator
from task_order_search import replay_tasks

ROOT=Path(__file__).resolve().parents[1]


def all_positions(orders,task,preds,durations,locked):
    if task in locked:return []
    stripped=[[t for t in q if t!=task] for q in orders];rows=[]
    for core,queue in enumerate(stripped):
        prefix=0
        while prefix<len(queue) and queue[prefix] in locked:prefix+=1
        if any(t in locked for t in queue[prefix:]):raise ValueError('锁定集合不是前缀')
        for slot in range(prefix,len(queue)+1):
            trial=[list(q) for q in stripped];trial[core].insert(slot,task)
            value=replay_tasks(durations,preds,trial)
            if value is not None:rows.append({'orders':trial,'core':core,'slot':slot,'coarse':value['makespan']})
    return rows


def main():
    p=argparse.ArgumentParser();p.add_argument('--input-dir',required=True);p.add_argument('--output-dir',required=True)
    p.add_argument('--max-candidates',type=int,default=0)
    args=p.parse_args();import torch
    if not torch.cuda.is_available():p.error('需要已有显卡环境')
    source=Path(args.input_dir).resolve()/'lookahead_pairs.jsonl'
    meta_path=source.with_suffix('.manifest.json');meta=json.loads(meta_path.read_text(encoding='utf-8'))
    # 当前审计只能复用所有既有依赖完全相同的源结果；新增加的审计模块不在旧依赖中。
    for path,digest in meta['files'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest:raise ValueError('源实验依赖变化，拒绝复用')
    groups=load_rows(source)
    if not all(verified(r,meta['fingerprint']) for r in groups.values()):raise ValueError('源清单绑定无效')
    ref_log=ROOT/'results/p1_e06_ddr_r01/resource_replay.jsonl';refs=load_rows(ref_log)
    ref_manifest=ref_log.with_suffix('.manifest.json');ref_meta=json.loads(ref_manifest.read_text(encoding='utf-8'))
    files=run_input_files(sorted({r['case'] for r in groups.values()}))+[source,meta_path,ref_log,ref_manifest]
    for r in groups.values():
        ref=refs[f'{r["case"]}_N{r["N"]}']
        if not good(ref,ref_meta['fingerprint']) or ref['source_sha256']!=r['source_sha256']:raise ValueError('原官方参照无效')
        files.extend(map(Path,ref['artifacts']));files.append(Path(r['source_plan']));files.extend(map(Path,r['artifacts']))
    out=Path(args.output_dir).resolve();out.mkdir(parents=True,exist_ok=True)
    ledger=out/'coverage.jsonl';fp=ensure_run_manifest(ledger,files,{'source':str(source),'initial_windows_only':True,'official_limit':2,'gpu':torch.cuda.get_device_name(0)})
    saved={k:r for k,r in load_rows(ledger).items() if reusable(r,fp)}
    prepared_path=out/'prepared.json'
    if prepared_path.exists():
        prepared=json.loads(prepared_path.read_text(encoding='utf-8'))
        if not reusable(prepared,fp):raise ValueError('固定候选清单无效')
    else:
        prepared={'id':'prepared','groups':{},'artifacts':{}}
        for gid,g in groups.items():
            started=time.perf_counter();graph=json.loads((ROOT/'通用神经网络处理器下的多核调度问题附件/data'/f'{g["case"]}.json').read_text(encoding='utf-8'))
            plan=json.loads(Path(g['source_plan']).read_text(encoding='utf-8'))
            if hashlib.sha256(Path(g['source_plan']).read_bytes()).hexdigest()!=g['source_sha256']:raise ValueError('原计划改变')
            ref=refs[f'{g["case"]}_N{g["N"]}'];raw=json.loads(gzip.decompress(Path(next(iter(ref['artifacts']))).read_bytes()))
            baseline=compact_result(raw,plan['core_schedules']);preds=derive_multicore_plan(graph,plan)['subgraph_preds']
            durations={t:r['duration'] for t,r in baseline['tasks'].items()}
            model=Model(graph,block_ops_cap=block_cap_for(sum(o['op'] not in ('COPY_IN','COPY_OUT') for o in graph['ops'])))
            assignment=[plan['node_to_subgraph'][str(b[0])] for b in model.blocks];batch=BatchFeatures(model,g['N'])
            torch.cuda.reset_peak_memory_stats();feature_seconds=0.;feature_count=0
            def features(orders_batch):
                nonlocal feature_seconds,feature_count
                started=time.perf_counter();owners=[]
                for orders in orders_batch:
                    owner=[0]*(max(assignment)+1)
                    for c,q in enumerate(orders):
                        for t in q:owner[t]=c
                    owners.append(owner)
                values=batch.evaluate([assignment]*len(owners),owners)
                if values!=[scalar_features(model,assignment,c,g['N']) for c in owners]:raise ValueError('显卡特征不一致')
                feature_seconds+=time.perf_counter()-started;feature_count+=len(owners);return values
            candidates={};roots=list(dict.fromkeys(d['root'] for d in g['search']['decisions']))
            for root in roots:
                cut,locked,_=decision_window(root,preds,baseline['tasks'],g['search'].get('cut_policy','start'))
                selected={order_key(r['orders']) for r in propose_placements(plan['core_schedules'],root,preds,durations,locked,features)}
                for row in all_positions(plan['core_schedules'],root,preds,durations,locked):
                    key=order_key(row['orders']);cid=plan_digest({**plan,'core_schedules':row['orders']})
                    candidate=candidates.setdefault(cid,{'id':gid+'_'+cid,'group':gid,'orders':row['orders'],'coarse':row['coarse'],'origins':[]})
                    candidate['origins'].append({'root':root,'cut':cut,'core':row['core'],'slot':row['slot'],'admitted':key in selected})
            prepared['groups'][gid]={'candidates':list(candidates.values()),'preparation_seconds':time.perf_counter()-started,
                'gpu_seconds':feature_seconds,'gpu_candidates':feature_count,'gpu_peak_bytes':torch.cuda.max_memory_allocated()}
        prepared=seal_result(prepared,fp);prepared_path.write_text(json.dumps(prepared,ensure_ascii=False,indent=2),encoding='utf-8')
    total=sum(len(x['candidates']) for x in prepared['groups'].values());count=0
    print(f'[{len(saved)}/{total}] 固定初始合法插入候选；逐候选恢复',flush=True)
    for gid,g in groups.items():
        todo=[r for r in prepared['groups'][gid]['candidates'] if r['id'] not in saved]
        if not todo:continue
        graph=json.loads((ROOT/'通用神经网络处理器下的多核调度问题附件/data'/f'{g["case"]}.json').read_text(encoding='utf-8'))
        plan=json.loads(Path(g['source_plan']).read_text(encoding='utf-8'))
        known={order_key(plan['core_schedules']):g['baseline']}
        known.update((order_key(r['orders']),r['real']) for r in g['search']['history'])
        known.update((order_key(r['orders']),r['resource']) for r in g['archive'])
        evaluator=build_resource_evaluator(unified=True,template_cache=TemplateCache())
        for candidate in todo:
            if args.max_candidates and count>=args.max_candidates:return
            started=time.perf_counter();key=order_key(candidate['orders']);reused=key in known
            value=known[key] if reused else compact_result(evaluator(graph,{**plan,'core_schedules':candidate['orders']},60,{'L1':524288,'UB':131072},1000,100),candidate['orders'])['real']
            if any(value[k]!=g['baseline'][k] for k in g['baseline'] if k!='makespan'):raise ValueError('搬运不变量失败')
            row=seal_result({**candidate,'resource':value,'reused':reused,'source_binding':g['binding'],
                'seconds':time.perf_counter()-started,'artifacts':{},'scope':'额外诊断，不并入原32上限'},fp)
            append_run_row(ledger,row);saved[row['id']]=row;count+=1
            if len(saved)%10==0 or len(saved)==total:print(f'[{len(saved)}/{total}] 新评估{sum(not r["reused"] for r in saved.values())}',flush=True)
    checks_path=out/'official_checks.jsonl';checks={k:r for k,r in load_rows(checks_path).items() if reusable(r,fp)}
    for gid,g in groups.items():
        ranked=sorted((r for r in saved.values() if r['group']==gid and r['resource']['makespan']<g['baseline']['makespan']),key=lambda r:(r['resource']['makespan'],r['id']))[:2]
        for r in ranked:
            if r['id'] in checks:continue
            graph=json.loads((ROOT/'通用神经网络处理器下的多核调度问题附件/data'/f'{g["case"]}.json').read_text(encoding='utf-8'))
            plan=json.loads(Path(g['source_plan']).read_text(encoding='utf-8'));plan['core_schedules']=r['orders']
            path=out/(r['id']+'.json');path.write_text(json.dumps(plan),encoding='utf-8')
            started=time.perf_counter();truth=real_evaluate(graph,plan,'A')
            if not valid_official(truth) or truth!=r['resource']:raise ValueError('诊断候选原官方不一致')
            checked=seal_result({'id':r['id'],'group':gid,'official':truth,'resource':r['resource'],'seconds':time.perf_counter()-started,
                'artifacts':{str(path):hashlib.sha256(path.read_bytes()).hexdigest()},'scope':'额外诊断官方，不作为原预算成绩'},fp)
            append_run_row(checks_path,checked);checks[r['id']]=checked
    report=[]
    for gid,g in groups.items():
        rows=[r for r in saved.values() if r['group']==gid]
        selected=[r for r in rows if any(o['admitted'] for o in r['origins'])]
        best=min(rows,key=lambda r:(r['resource']['makespan'],r['id']))
        report.append({'id':gid,'baseline':g['baseline'],'candidates':len(rows),'new_evaluations':sum(not r['reused'] for r in rows),
            'reused':sum(r['reused'] for r in rows),'new_seconds':sum(r['seconds'] for r in rows if not r['reused']),
            'best_resource':best['resource'],'best_origins':best['origins'],'best_selected_resource':min(r['resource']['makespan'] for r in selected),
            'improvements':sum(r['resource']['makespan']<g['baseline']['makespan'] for r in rows),
            'missed_improvements':sum(r['resource']['makespan']<g['baseline']['makespan'] and not any(o['admitted'] for o in r['origins']) for r in rows),
            'coarse_mape':sum(abs(r['coarse']-r['resource']['makespan'])/r['resource']['makespan'] for r in rows)/len(rows),
            'official_count':sum(r['group']==gid for r in checks.values()),'preparation':{k:v for k,v in prepared['groups'][gid].items() if k!='candidates'}})
    text=json.dumps({'groups':report,'scope':'固定初始方案，访问根的全部单插入；非所有根、非多步联合穷举；额外诊断'},ensure_ascii=False,indent=2)
    target=out/'mechanism_analysis.json'
    if not target.exists() or target.read_text(encoding='utf-8')!=text:target.write_text(text,encoding='utf-8')
    print(text,flush=True)


if __name__=='__main__':main()
