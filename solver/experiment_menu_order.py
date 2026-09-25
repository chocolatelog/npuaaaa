"""冻结菜单的全图顺序精化；逐候选恢复，强保底，历史证据分级复用。"""
import argparse,json,time,hashlib,zipfile
from pathlib import Path
from audit_evidence_index import evidence_paths
from evidence_index import load_evidence,plan_sha,file_sha,digest
from menu_order_polish import generate_order_variants,select_order_variants
from model import Model
from batch_features import BatchFeatures,scalar_features
from run_all import block_cap_for,run_input_files,ensure_run_manifest,append_run_row,valid_official
from experiment_branch_partition import seal_result,verified
from scene_a_event import derive_multicore_plan
from evaluation_validation import validate_task_order
from scene_a_replay import build_resource_evaluator
from local_template_cache import TemplateCache
from lookahead_place import compact_result
from pipeline import real_evaluate

ROOT=Path(__file__).resolve().parents[1]

def read_rows(path):return list(map(json.loads,Path(path).read_text(encoding='utf-8').splitlines()))
def valid_record(row,fp):
    content=dict(row);expected=content.pop('binding',None)
    return expected==digest(content) and row.get('fingerprint')==fp and all(file_sha(p)==sha for p,sha in row.get('artifacts',{}).items())
def read_journal(path,fp):
    result={}
    if not path.exists():return result
    for line in path.read_text(encoding='utf-8').splitlines():
        try:row=json.loads(line)
        except json.JSONDecodeError:continue  # 保留中断产生的尾片段，不覆盖或删除。
        if not valid_record(row,fp):raise ValueError('候选断点绑定无效')
        result[row['id']]=row
    return result
def journal_write(path,row,fp,done):
    sealed=seal_result(row,fp)
    if path.exists() and path.stat().st_size and not path.read_bytes().endswith(b'\n'):
        with path.open('a',encoding='utf-8') as f:f.write('\n')
    append_run_row(path,sealed);done[row['id']]=sealed
    return sealed

def experiment(source,out,fp,index,torch,journal,done,new_counter,max_new):
    cid=source['id'];case,n=source['case'],source['N'];started=time.perf_counter()
    graph_path=ROOT/'通用神经网络处理器下的多核调度问题附件/data'/f'{case}.json'
    graph=json.loads(graph_path.read_text(encoding='utf-8'));graph_sha=file_sha(graph_path)
    baseline_plan=json.loads(Path(source['plan_path']).read_text(encoding='utf-8'));baseline=source['real']
    evidence=index.lookup_for_evaluation(case,'A',n,graph_sha,plan_sha(baseline_plan),'official')
    if evidence is None or evidence['metrics']!=baseline:raise ValueError('当前菜单强保底缺少相符官方证据')
    model=Model(graph,block_ops_cap=block_cap_for(sum(o['op'] not in ('COPY_IN','COPY_OUT') for o in graph['ops'])))
    generation_path=out/f'{cid}_generation.json'
    if generation_path.exists():
        generated=json.loads(generation_path.read_text(encoding='utf-8'))
        if not valid_record(generated,fp):raise ValueError('冻结顺序提案失效')
    else:
        before=time.perf_counter();rows,audit=generate_order_variants(graph,model,source['evaluated'],
            progress=lambda i,total:print(f'  {cid} 局部建模/搜索[{i}/{total}]',flush=True))
        selected=select_order_variants(rows)
        batch=BatchFeatures(model,n);torch.cuda.reset_peak_memory_stats();gpu_start=time.perf_counter()
        features=batch.evaluate([r['assignment'] for r in rows],[r['owners'] for r in rows]) if rows else []
        if features!=[scalar_features(model,r['assignment'],r['owners'],n) for r in rows]:raise ValueError('显卡候选特征与标量不一致')
        for row,feature in zip(rows,features):row['features']=feature
        generated=seal_result({'id':cid+'_generation','status':'proposal_generation','artifacts':{},'source_binding':source['binding'],
            'rows':rows,'selected_ids':[r['plan_id'] for r in selected],'audit':audit,'seconds':time.perf_counter()-before,
            'gpu':{'seconds':time.perf_counter()-gpu_start,'equal':True,'peak_allocated':torch.cuda.max_memory_allocated()}},fp)
        temporary=out/f'{cid}_generation.partial.json';temporary.write_text(json.dumps(generated,ensure_ascii=False),encoding='utf-8')
        temporary.rename(generation_path)
    candidates={r['plan_id']:r for r in generated['rows']};evaluated=[];artifacts={str(generation_path):file_sha(generation_path)};evaluator=None
    for i,pid in enumerate(generated['selected_ids']):
        key=cid+'_score_'+pid;candidate=candidates[pid]
        if key in done:record=done[key]
        else:
            known=index.lookup_for_evaluation(case,'A',n,graph_sha,pid,'resource');kind='resource'
            if known is None:
                known=index.lookup_for_evaluation(case,'A',n,graph_sha,pid,'official')
                if known is not None:kind='official'
            if known is None and max_new and new_counter[0]>=max_new:return None
            plan=candidate['plan'];validate_task_order(derive_multicore_plan(graph,plan))
            path=out/f'{cid}_candidate_{i:02d}.json';path.write_text(json.dumps(plan),encoding='utf-8')
            if known:metrics=known['metrics'];seconds=0.;provenance=known['sources']
            else:
                if evaluator is None:evaluator=build_resource_evaluator(unified=True,template_cache=TemplateCache())
                before=time.perf_counter();raw=evaluator(graph,plan,60,{'L1':524288,'UB':131072},1000,100)
                seconds=time.perf_counter()-before;metrics=compact_result(raw,plan['core_schedules'])['real'];provenance=[];new_counter[0]+=1
            expected=candidate['origins'][0]['baseline_resource']
            if any(metrics[k]!=expected[k] for k in metrics if k!='makespan'):raise ValueError('固定成员切分顺序精化改变搬运或溢出')
            if metrics['partition_added']!=max(0,candidate['features']['boundary_bytes']-model.original_copy_bytes):raise ValueError('真实边界与显卡特征不一致')
            record=journal_write(journal,{'id':key,'status':'scored_candidate','config_id':cid,'plan_id':pid,'plan_path':str(path),
                'assignment':candidate['assignment'],'origins':candidate['origins'],'coarse':candidate['coarse'],'resource':metrics,
                'evaluation_kind':kind,'reused':known is not None,'evidence_sources':provenance,'trace':[],
                'timeline_level':'metrics_only','seconds':seconds,'artifacts':{str(path):file_sha(path)}},fp,done)
        evaluated.append(record);artifacts.update(record['artifacts'])
        print(f'  {cid} 候选评分[{i+1}/{len(generated["selected_ids"])}]，复用={record["reused"]}',flush=True)
    best=baseline;best_plan=baseline_plan;checks=[]
    ranked=sorted(evaluated,key=lambda r:(r['resource']['makespan'],r['resource']['added_copy_bytes'],r['plan_id']))
    shortlist=[r for r in ranked if (r['resource']['makespan'],r['resource']['added_copy_bytes'])<(baseline['makespan'],baseline['added_copy_bytes'])][:2]
    for row in shortlist:
        key=cid+'_official_'+row['plan_id'];plan=json.loads(Path(row['plan_path']).read_text(encoding='utf-8'))
        if key in done:check=done[key]
        else:
            known=index.lookup_for_evaluation(case,'A',n,graph_sha,row['plan_id'],'official')
            if known:truth=known['metrics'];seconds=0.
            else:
                before=time.perf_counter();truth=real_evaluate(graph,plan,'A');seconds=time.perf_counter()-before
            if not valid_official(truth) or truth!=row['resource']:raise ValueError('顺序精化原官方与资源评分不一致')
            check=journal_write(journal,{'id':key,'status':'official_candidate','config_id':cid,'plan_path':row['plan_path'],
                'plan_id':row['plan_id'],'official':truth,'resource':row['resource'],'reused':known is not None,
                'evidence_sources':known['sources'] if known else [],'seconds':seconds,'artifacts':row['artifacts']},fp,done)
        checks.append(check)
        if (check['official']['makespan'],check['official']['added_copy_bytes'])<(best['makespan'],best['added_copy_bytes']):
            best=check['official'];best_plan=plan
    path=out/f'{cid}_selected.json';path.write_text(json.dumps(best_plan),encoding='utf-8');artifacts[str(path)]=file_sha(path)
    return {'id':cid,'case':case,'N':n,'status':'official_success','baseline':baseline,'real':best,
        'source_sha256':file_sha(source['plan_path']),'source_plan':source['plan_path'],'source_binding':source['binding'],
        'plan_id':plan_sha(best_plan),'plan_path':str(path),'artifacts':artifacts,'generation':generated['audit'],
        'generation_seconds':generated['seconds'],'gpu':generated['gpu'],'evaluated':evaluated,'checks':checks,
        'resource_new_count':sum(not r['reused'] for r in evaluated),'score_reused_count':sum(r['reused'] for r in evaluated),
        'official_count':sum(not r['reused'] for r in checks),'official_reused_count':sum(r['reused'] for r in checks),
        'active_batch_seconds':time.perf_counter()-started,'scope':'强菜单保底后的额外12评分/2官方阶段；局部时间近似；非原菜单等预算'}

def main():
    p=argparse.ArgumentParser();p.add_argument('--output-dir',required=True);p.add_argument('--max-tasks',type=int,default=0)
    p.add_argument('--max-new-candidates',type=int,default=0);a=p.parse_args();import torch
    if not torch.cuda.is_available() or min(a.max_tasks,a.max_new_candidates)<0:p.error('需要显卡环境和非负停止参数')
    sources={}
    for directory in ('p1_e08_diverse_r01','p1_e08_diverse_memory_r01','p1_e08_seed_combinations_r01','p1_e08_seed_combinations_r02'):
        log=ROOT/'results'/directory/'menu_pairs.jsonl';meta=json.loads(log.with_suffix('.manifest.json').read_text(encoding='utf-8'))
        for row in read_rows(log):
            if not verified(row,meta['fingerprint']):raise ValueError('输入菜单绑定失效')
            if row['N']==4 or directory.startswith('p1_e08_seed_combinations'):sources[row['id']]=row
    tasks=sorted(sources.values(),key=lambda r:(r['id'] not in ('case_005_N2','case_067_N2'),r['case'],r['N']))
    index=load_evidence(ROOT,evidence_paths(),sources,progress=lambda i,n,c:print(f'证据[{i}/{n}] {c}条',flush=True));index.require_runtime()
    files=run_input_files(sorted({r['case'] for r in tasks}))
    for path in evidence_paths():
        files.extend([path,path.with_suffix('.manifest.json')])
        for row in read_rows(path):files.extend(map(Path,row.get('artifacts',{})))
        archive=path.parent/'frozen_source/source.zip'
        if archive.exists():files.append(archive)
    out=Path(a.output_dir).resolve();out.mkdir(exist_ok=True,parents=True);ledger=out/'menu_order_pairs.jsonl';journal=out/'candidate_journal.jsonl'
    fp=ensure_run_manifest(ledger,files,{'ids':[r['id'] for r in tasks],'max_parents':12,'local_search_evals':400,
        'rounds':2,'beam_width':4,'keep':2,'swaps':True,'score_limit':12,'official_limit':2,'strong_menu_baseline':True,
        'contracts':index.contracts,'gpu':torch.cuda.get_device_name(0)})
    archive=out/'frozen_source/source.zip'
    if not archive.exists():
        archive.parent.mkdir(exist_ok=True)
        with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
            for path in (ROOT/'solver').glob('*.py'):z.write(path,path.relative_to(ROOT))
    completed={r['id']:r for r in read_rows(ledger) if verified(r,fp)} if ledger.exists() else {}
    pending=[r for r in tasks if r['id'] not in completed]
    if a.max_tasks:pending=pending[:a.max_tasks]
    done=read_journal(journal,fp);new_counter=[0]
    print(f'[{len(completed)}/{len(tasks)}] 菜单固定切分顺序精化，本批{len(pending)}',flush=True)
    for source in pending:
        result=experiment(source,out,fp,index,torch,journal,done,new_counter,a.max_new_candidates)
        if result is None:print('新增候选停止额度已到，断点已保存',flush=True);return
        result=seal_result(result,fp);append_run_row(ledger,result);completed[result['id']]=result
        print(f'[{len(completed)}/{len(tasks)}] {result["id"]} 官方强保底{result["baseline"]["makespan"]}→{result["real"]["makespan"]}',flush=True)
    summary={'expected':len(tasks),'completed':len(completed),'wins':sum(r['real']['makespan']<r['baseline']['makespan'] for r in completed.values()),
             'losses':sum(r['real']['makespan']>r['baseline']['makespan'] for r in completed.values())}
    target=out/'summary.json';text=json.dumps(summary,ensure_ascii=False,indent=2)
    if not target.exists() or target.read_text(encoding='utf-8')!=text:target.write_text(text,encoding='utf-8')
    print(text,flush=True)

if __name__=='__main__':main()
