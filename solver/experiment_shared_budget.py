"""E09六次共同复核/一次原官方；先冻结全部名单，再查询候选历史证据。"""
import argparse
import ast
import json
import time
import uuid
import zipfile
from pathlib import Path
from audit_evidence_index import evidence_paths
from evidence_index import load_evidence,verified_rows,file_sha,plan_sha,digest
from baseline_evidence import add_baselines,read_full_official
from experiment_branch_partition import seal_result,verified
from experiment_menu_order import read_rows,valid_record,read_journal,journal_write
from run_all import run_input_files,ensure_run_manifest,append_run_row,block_cap_for,valid_official
from shared_candidate_pool import generate_pools
from shared_budget import select_shared,unify_scores
from batch_features import BatchFeatures,scalar_features
from model import Model
from scene_a_event import derive_multicore_plan
from evaluation_validation import validate_task_order
from scene_a_replay import build_resource_evaluator
from local_template_cache import TemplateCache
from lookahead_place import compact_result
from pipeline import real_evaluate

ROOT=Path(__file__).resolve().parents[1]
IDS=[f'case_{c:03d}_N{n}' for c in (5,44,49,67,82) for n in (2,4)]

def freeze(path,value):
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8'))!=value:raise ValueError('冻结文件内容变化')
        return
    temporary=path.with_name(path.name+'.'+uuid.uuid4().hex+'.partial')
    temporary.write_text(json.dumps(value,ensure_ascii=False),encoding='utf-8');temporary.rename(path)

def sources():
    menus={};files=[]
    for directory in ('p1_e08_diverse_r01','p1_e08_diverse_memory_r01','p1_e08_seed_combinations_r01','p1_e08_seed_combinations_r02'):
        log=ROOT/'results'/directory/'menu_pairs.jsonl';meta=json.loads(log.with_suffix('.manifest.json').read_text(encoding='utf-8'))
        if digest({k:v for k,v in meta.items() if k!='fingerprint'})!=meta['fingerprint']:raise ValueError('菜单源指纹无效')
        for row in verified_rows(log,meta['fingerprint']):
            if row['N']==4 or directory.startswith('p1_e08_seed_combinations'):menus[row['id']]=row
        files.extend([log,log.with_suffix('.manifest.json'),log.parent/'frozen_source/source.zip'])
    log=ROOT/'results/p1_e08_menu_order_r01/menu_order_pairs.jsonl'
    meta=json.loads(log.with_suffix('.manifest.json').read_text(encoding='utf-8'));orders={}
    if digest({k:v for k,v in meta.items() if k!='fingerprint'})!=meta['fingerprint']:raise ValueError('顺序源指纹无效')
    for row in verified_rows(log,meta['fingerprint']):
        gp=log.parent/f'{row["id"]}_generation.json';g=json.loads(gp.read_text(encoding='utf-8'))
        if str(gp) not in row['artifacts'] or not valid_record(g,meta['fingerprint']):raise ValueError('历史完整提案未绑定')
        if g['source_binding']!=menus[row['id']]['binding']:raise ValueError('顺序提案的冻结父菜单不同')
        orders[row['id']]=g;files.append(gp)
    files.extend([log,log.with_suffix('.manifest.json'),log.parent/'frozen_source/source.zip'])
    ref_log=ROOT/'results/p1_e06_ddr_r01/resource_replay.jsonl';rm=json.loads(ref_log.with_suffix('.manifest.json').read_text(encoding='utf-8'))
    if digest({k:v for k,v in rm.items() if k!='fingerprint'})!=rm['fingerprint']:raise ValueError('基线源指纹无效')
    refs={r['id']:r for r in verified_rows(ref_log,rm['fingerprint']) if r['id'] in IDS}
    for r in refs.values():
        read_full_official(r);files.extend(map(Path,r['artifacts']));files.append(Path(r['source_plan']))
        if menus[r['id']]['source_sha256']!=r['source_sha256']:raise ValueError('区域菜单非同一E03输入')
    files.extend([ref_log,ref_log.with_suffix('.manifest.json'),ref_log.parent/'frozen_source/source.zip'])
    original=ROOT/'results/p1_e03_r02/order_pairs.jsonl';files.extend([original,original.with_suffix('.manifest.json')])
    if any(cid not in menus or cid not in orders or cid not in refs for cid in IDS):raise ValueError('十配置冻结源不完整')
    return menus,orders,refs,files

def score_config(cid,generation,ref,out,fp,index,journal,done,limit,counter):
    start=time.perf_counter();case=ref['case'];n=ref['N'];graph_path=ROOT/'通用神经网络处理器下的多核调度问题附件/data'/f'{case}.json'
    graph=json.loads(graph_path.read_text(encoding='utf-8'));graph_sha=file_sha(graph_path)
    plan=json.loads(Path(ref['source_plan']).read_text(encoding='utf-8'));baseline=ref['real'];best=baseline;best_plan=plan
    model=Model(graph,block_ops_cap=block_cap_for(sum(o['op'] not in ('COPY_IN','COPY_OUT') for o in graph['ops'])))
    known=index.lookup_for_evaluation(case,'A',n,graph_sha,plan_sha(plan),'official')
    if known is None or known['metrics']!=baseline:raise ValueError('E03完整输入缺少同环境原官方保底')
    evaluated=[];evaluator=None;artifacts={str(out/f'{cid}_frozen.json'):file_sha(out/f'{cid}_frozen.json')}
    for i,candidate in enumerate(generation['selected']):
        pid=candidate['plan_id'];key=cid+'_score_'+pid
        if key in done:row=done[key]
        else:
            known=index.lookup_for_evaluation(case,'A',n,graph_sha,pid,'resource');kind='resource'
            if known is None:
                known=index.lookup_for_evaluation(case,'A',n,graph_sha,pid,'official');kind='official' if known else 'resource'
            if known is None and limit and counter[0]>=limit:return None
            proposal=candidate['plan'];validate_task_order(derive_multicore_plan(graph,proposal))
            path=out/f'{cid}_candidate_{i:02d}.json';freeze(path,proposal)
            if known:metrics=known['metrics'];elapsed=0.;provenance=known['sources']
            else:
                if evaluator is None:evaluator=build_resource_evaluator(unified=True,template_cache=TemplateCache())
                before=time.perf_counter();raw=evaluator(graph,proposal,60,{'L1':524288,'UB':131072},1000,100)
                elapsed=time.perf_counter()-before;metrics=compact_result(raw,proposal['core_schedules'])['real'];provenance=[];counter[0]+=1
            if metrics['partition_added']!=max(0,candidate['features']['boundary_bytes']-model.original_copy_bytes):raise ValueError('新切分真实边界与显卡特征不符')
            row=journal_write(journal,{'id':key,'config_id':cid,'status':'scored_candidate','plan_id':pid,'plan_path':str(path),
                'allocated_family':candidate['allocated_family'],'origin':candidate['origin'],'coarse':candidate['coarse'],
                'resource':metrics,'evaluation_kind':kind,'reused':known is not None,'seconds':elapsed,
                'evidence_sources':provenance,'timeline_level':'metrics_only','artifacts':{str(path):file_sha(path)}},fp,done)
        evaluated.append(row);artifacts.update(row['artifacts'])
        print(f'  {cid} 共同评分[{i+1}/{len(generation["selected"])}] 复用={row["reused"]}',flush=True)
    ranked=sorted(evaluated,key=lambda r:(r['resource']['makespan'],r['resource']['added_copy_bytes'],r['plan_id']))
    shortlist=[r for r in ranked if (r['resource']['makespan'],r['resource']['added_copy_bytes'])<(baseline['makespan'],baseline['added_copy_bytes'])][:1]
    checks=[]
    for row in shortlist:
        key=cid+'_official_'+row['plan_id']
        if key in done:check=done[key]
        else:
            known=index.lookup_for_evaluation(case,'A',n,graph_sha,row['plan_id'],'official')
            if known:truth=known['metrics'];elapsed=0.
            else:
                proposal=json.loads(Path(row['plan_path']).read_text(encoding='utf-8'));before=time.perf_counter()
                truth=real_evaluate(graph,proposal,'A');elapsed=time.perf_counter()-before
            check=journal_write(journal,{'id':key,'config_id':cid,'status':'official_candidate','plan_id':row['plan_id'],
                'plan_path':row['plan_path'],'official':truth,'resource':row['resource'],
                'matches':bool(valid_official(truth) and truth==row['resource']),'reused':known is not None,
                'seconds':elapsed,'evidence_sources':known['sources'] if known else [],'artifacts':row['artifacts']},fp,done)
        checks.append(check)
        if check['matches'] and (check['official']['makespan'],check['official']['added_copy_bytes'])<(best['makespan'],best['added_copy_bytes']):
            best=check['official'];best_plan=json.loads(Path(check['plan_path']).read_text(encoding='utf-8'))
    path=out/f'{cid}_selected.json';freeze(path,best_plan);artifacts[str(path)]=file_sha(path)
    return {'id':cid,'case':case,'N':n,'status':'official_success','baseline':baseline,'real':best,
        'source_plan':ref['source_plan'],'source_sha256':ref['source_sha256'],'baseline_binding':ref['binding'],
        'plan_path':str(path),'plan_id':plan_sha(best_plan),'artifacts':artifacts,'evaluated':evaluated,'checks':checks,
        'generation':generation['audit'],'selection':generation['selection'],'generation_binding':generation['binding'],
        'active_batch_seconds':time.perf_counter()-start,'scope':'固定E03输入的六逻辑评分/一官方机制组；历史区域生成费用不等于在线零成本'}

def main():
    p=argparse.ArgumentParser();p.add_argument('--output-dir',required=True);p.add_argument('--max-tasks',type=int,default=0)
    p.add_argument('--max-generations',type=int,default=0);p.add_argument('--max-new-candidates',type=int,default=0)
    p.add_argument('--replay-pools',help='已完成共同预算目录；复用候选池，不生成新计划')
    p.add_argument('--reuse-unified-scores',action='store_true',help='复用已绑定的统一近似分数，只重排名额')
    p.add_argument('--tie-break',choices=('boundary','total_copy'),default='boundary');a=p.parse_args()
    if min(a.max_tasks,a.max_generations,a.max_new_candidates)<0:p.error('停止计数必须非负')
    if (a.reuse_unified_scores or a.tie_break=='total_copy') and not a.replay_pools:p.error('该消融必须提供冻结来源池')
    import torch
    if not torch.cuda.is_available():p.error('需要现有显卡环境')
    out=Path(a.output_dir).resolve();out.mkdir(parents=True,exist_ok=True)
    # 持久锁文件只加锁/释放；不删除文件。系统在进程退出时自动释放字节锁。
    import msvcrt
    with (out/'process.lock').open('a+b') as lock:
        if lock.tell()==0:lock.write(b'0');lock.flush()
        lock.seek(0)
        try:msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        except OSError:raise SystemExit('同目录实验正在运行，拒绝重复启动')
        run(a,out,torch)

def run(a,out,torch):
    menus,orders,refs,files=sources();paths=evidence_paths()+[ROOT/'results/p1_e08_menu_order_r01/menu_order_pairs.jsonl']
    frozen_sources={};replay_source=None
    if a.replay_pools:
        replay_source=Path(a.replay_pools).resolve()
        if replay_source==out:raise ValueError('禁止把自身输出作为重排输入')
        previous_log=replay_source/'shared_pairs.jsonl';previous_meta=previous_log.with_suffix('.manifest.json')
        old_meta=json.loads(previous_meta.read_text(encoding='utf-8'))
        if digest({k:v for k,v in old_meta.items() if k!='fingerprint'})!=old_meta['fingerprint']:raise ValueError('历史共同预算指纹失效')
        if a.reuse_unified_scores:
            for name in ('task_order_search.py','scene_a_event.py','spill_events.py','model.py','batch_features.py'):
                dependency=ROOT/'solver'/name
                if old_meta['files'].get(str(dependency))!=file_sha(dependency):raise ValueError('统一近似模型依赖已改变，不能复用分数')
            with zipfile.ZipFile(replay_source/'frozen_source/source.zip') as z:old_source=z.read('solver/shared_budget.py')
            import hashlib
            if hashlib.sha256(old_source).hexdigest()!=old_meta['files'][str(ROOT/'solver/shared_budget.py')]:raise ValueError('统一近似源快照失效')
            def unified_ast(src):
                return ast.dump(next(n for n in ast.parse(src).body if isinstance(n,ast.FunctionDef) and n.name=='unify_scores'))
            if unified_ast(old_source.decode('utf-8'))!=unified_ast((ROOT/'solver/shared_budget.py').read_text(encoding='utf-8')):
                raise ValueError('统一近似公式改变，不能复用历史分数')
        old_rows={r['id']:r for r in verified_rows(previous_log,old_meta['fingerprint'])}
        if set(old_rows)!=set(IDS):raise ValueError('重排需要完整已完成十配置')
        for cid in IDS:
            gp=replay_source/f'{cid}_frozen.json';g=json.loads(gp.read_text(encoding='utf-8'))
            if str(gp) not in old_rows[cid]['artifacts'] or not valid_record(g,old_meta['fingerprint']):raise ValueError('历史冻结池失效')
            if g['baseline_binding']!=refs[cid]['binding']:raise ValueError('重排输入基线不同')
            frozen_sources[cid]=g;files.append(gp)
        paths.append(previous_log);files.append(previous_meta)
        # 来源链上的已完成实验也须导入；最近一轮未选中的旧计划仍可能已测。
        ancestry={out,replay_source};ancestor=old_meta['settings'].get('replay_pools')
        while ancestor:
            ancestor=Path(ancestor).resolve()
            if ancestor in ancestry:raise ValueError('历史候选来源链成环')
            ancestry.add(ancestor);log=ancestor/'shared_pairs.jsonl'
            if log not in paths:paths.append(log)
            m=json.loads(log.with_suffix('.manifest.json').read_text(encoding='utf-8'))
            if digest({k:v for k,v in m.items() if k!='fingerprint'})!=m['fingerprint']:raise ValueError('祖先清单指纹无效')
            ancestor=m['settings'].get('replay_pools')
    for path in paths:
        files.extend([path,path.with_suffix('.manifest.json')])
        for row in read_rows(path):files.extend(map(Path,row.get('artifacts',{})))
        archive=path.parent/'frozen_source/source.zip'
        if archive.exists():files.append(archive)
    ledger=out/'shared_pairs.jsonl';journal=out/'candidate_journal.jsonl'
    fp=ensure_run_manifest(ledger,run_input_files(sorted({r['case'] for r in refs.values()}))+files,
        {'ids':IDS,'family_slots':[2,2,2],'score_limit':6,'official_limit':1,'schedule_evals':4000,
         'partition_local_evals':400,'gpu':torch.cuda.get_device_name(0),'baseline':'冻结E03，E06同环境原官方确认',
         'replay_pools':str(replay_source) if replay_source else None,'ranking_model':'unified_local' if replay_source else 'historical_mixed',
         'reuse_unified_scores':a.reuse_unified_scores,'tie_break':a.tie_break,
         'selection_policy':'家族内近似耗时/'+('边界加溢出总复制' if a.tie_break=='total_copy' else '真实边界')+
                            '/完整摘要，先不同成员切分；缺额按已得名额最少转移'})
    archive=out/'frozen_source/source.zip'
    if not archive.exists():
        archive.parent.mkdir(exist_ok=True)
        with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
            for path in (ROOT/'solver').glob('*.py'):z.write(path,path.relative_to(ROOT))
    generated={};new_count=0
    for cid in IDS:
        path=out/f'{cid}_frozen.json'
        if path.exists():
            g=json.loads(path.read_text(encoding='utf-8'))
            if not valid_record(g,fp):raise ValueError('已冻结候选改变')
        else:
            if a.max_generations and new_count>=a.max_generations:print('生成停止额度已到，未查询候选真值',flush=True);return
            ref=refs[cid];graph=json.loads((ROOT/'通用神经网络处理器下的多核调度问题附件/data'/f'{ref["case"]}.json').read_text(encoding='utf-8'))
            plan=json.loads(Path(ref['source_plan']).read_text(encoding='utf-8'))
            model=Model(graph,block_ops_cap=block_cap_for(sum(o['op'] not in ('COPY_IN','COPY_OUT') for o in graph['ops'])))
            if replay_source:
                original=frozen_sources[cid]
                if a.reuse_unified_scores:
                    if any('unified_local' not in r for rows in original['pools'].values() for r in rows):raise ValueError('来源不是统一近似池')
                    pools=original['pools'];details={'seconds':0.,'local_seconds':0.,'local_models':0,'replay_seconds':0.,'queue_replays':0,
                        'scope':'同一输入/硬件/源清单下复用已绑定统一近似分数；未重新局部建模或搜索'}
                else:
                    pools,details=unify_scores(graph,original['pools'],
                        progress=lambda i,n,k:print(f'  {cid} 统一近似重排[{i}/{n}]，局部模型{k}',flush=True))
                if any([r['plan_id'] for r in pools[f]]!=[r['plan_id'] for r in original['pools'][f]] for f in pools):raise ValueError('重排改变候选池')
                batch=BatchFeatures(model,len(plan['core_schedules']));all_rows=[r for rows in pools.values() for r in rows]
                torch.cuda.reset_peak_memory_stats();before=time.perf_counter()
                features=batch.evaluate([r['assignment'] for r in all_rows],[r['owners'] for r in all_rows])
                if features!=[r['features'] for r in all_rows] or features!=[scalar_features(model,r['assignment'],r['owners'],len(plan['core_schedules'])) for r in all_rows]:raise ValueError('重排显卡特征不同')
                gpu={'equal':True,'seconds':time.perf_counter()-before,'peak_allocated':torch.cuda.max_memory_allocated(),'candidates':len(all_rows)}
                audit={'seconds':details['seconds']+gpu['seconds'],'local_seconds':details['local_seconds'],'search_seconds':0.,
                    'search_evaluations':0,'unified_scoring':details,'gpu':gpu,'source_generation_binding':original['binding'],
                    'source_generation_seconds':original['audit']['seconds'],
                    'scope':('复用统一分数，只改变第二排序键；历史生成/建模成本不计为零' if a.reuse_unified_scores else
                             '同冻结池仅统一近似分数；历史生成成本不计为零')}
            else:
                pools,audit=generate_pools(graph,model,plan,menus[cid],orders[cid],torch,
                    progress=lambda i,n:print(f'  {cid} 重建局部提案[{i}/{n}]',flush=True) if i%6==0 or i==n else None)
            before=time.perf_counter();selected,selection=select_shared(pools,tie_break=a.tie_break)
            audit['selection_seconds']=time.perf_counter()-before;audit['seconds']+=audit['selection_seconds']
            g=seal_result({'id':cid+'_frozen','status':'frozen_candidates','pools':pools,'selected':selected,
                'selection':selection,'audit':audit,'baseline_binding':ref['binding'],'menu_binding':menus[cid]['binding'],
                'order_binding':orders[cid]['binding'],'artifacts':{},'candidate_truth_queried':False},fp)
            freeze(path,g);new_count+=1
        generated[cid]=g;print(f'[{len(generated)}/{len(IDS)}] {cid} 已冻结{len(g["selected"])}份入围名单',flush=True)
    # 所有候选及名额已经持久化，此后才导入真值索引。
    before=time.perf_counter();index=load_evidence(ROOT,paths,IDS);add_baselines(index,ROOT,IDS);index.require_runtime()
    index_seconds=time.perf_counter()-before
    index_path=out/'evidence_inventory.json'
    if not index_path.exists():freeze(index_path,{'summary':index.summary(),'records':list(index.records.values()),'seconds':index_seconds,
        'frozen_sha256':{cid:file_sha(out/f'{cid}_frozen.json') for cid in IDS}})
    completed={r['id']:r for r in read_rows(ledger) if verified(r,fp)} if ledger.exists() else {}
    pending=[cid for cid in IDS if cid not in completed]
    if a.max_tasks:pending=pending[:a.max_tasks]
    done=read_journal(journal,fp);counter=[0];print(f'[{len(completed)}/{len(IDS)}] 共同预算，本批{len(pending)}',flush=True)
    for cid in pending:
        result=score_config(cid,generated[cid],refs[cid],out,fp,index,journal,done,a.max_new_candidates,counter)
        if result is None:print('新增资源停止额度已到，逐候选断点已保存',flush=True);return
        row=seal_result(result,fp);append_run_row(ledger,row);completed[cid]=row
        print(f'[{len(completed)}/{len(IDS)}] {cid} 官方{row["baseline"]["makespan"]}→{row["real"]["makespan"]}',flush=True)
    summary={'expected':len(IDS),'completed':len(completed),'wins':sum(r['real']['makespan']<r['baseline']['makespan'] for r in completed.values()),
        'losses':sum(r['real']['makespan']>r['baseline']['makespan'] for r in completed.values()),
        'logical_scores':sum(len(r['evaluated']) for r in completed.values()),
        'new_resource':sum(not v['reused'] for r in completed.values() for v in r['evaluated']),
        'new_official':sum(not v['reused'] for r in completed.values() for v in r['checks'])}
    text=json.dumps(summary,ensure_ascii=False,indent=2);target=out/'summary.json'
    if not target.exists() or target.read_text(encoding='utf-8')!=text:target.write_text(text,encoding='utf-8')
    print(text,flush=True)

if __name__=='__main__':main()
