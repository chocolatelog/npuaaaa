"""固定有界切分，隔离验证三个就绪优先级；显卡批特征、事件评分、官方保底。"""
import argparse
import json
from pathlib import Path
import shutil
import time

from runtime_resources import initialize_worker_threads
from official_protocol import ATTACHMENT,object_digest,digest,verified_record,evaluate_job,input_context
from run_all import ensure_run_manifest,parse_cases,append_run_row
from run_saved_refine import write_json
from dag_priority import compute_certificate,construct_block_plan
from model import Model
from batch_features import BatchFeatures,scalar_features
from scene_a_candidate_score import SceneACandidateScorer
from scene_a_fast import official
from evaluation_validation import read_evaluation_config


def make_priority_model(graph,num_cores,block_cap,balanced_frontier=False,preserve_entries=False,layer_window=None,entry_ancestors=False):
    if type(block_cap) is not int or block_cap<1:raise ValueError('块上限必须为正整数')
    if type(num_cores) is not int or num_cores not in (2,3,4,5):raise ValueError('核数必须为2～5')
    work=sum(max(1,o.get('cycles',1)) for o in graph['ops'] if o['op'] not in ('COPY_IN','COPY_OUT'))
    return Model(graph,block_ops_cap=block_cap,block_policy='bounded',
                 block_work_cap=work/(num_cores*4),pack_frontiers=True,
                 frontier_width=num_cores if balanced_frontier or layer_window is not None else None,preserve_entries=preserve_entries,layer_window=layer_window,entry_ancestors=entry_ancestors)


def evaluate_with_history(job, output, folder, history):
    fingerprint=object_digest(input_context(job,folder,ATTACHMENT)[0])
    previous=history.get(fingerprint)
    if previous is not None and verified_record(previous):
        return previous,True
    return evaluate_job(job,output,folder),False


def load_candidate_checkpoint(path, fingerprint, plan, job):
    if not path.exists():
        return None
    record = json.loads(path.read_text(encoding='utf-8'))
    if record['fingerprint'] != fingerprint:
        raise ValueError('候选检查点配置失效')
    truth = record['official']
    if truth.get('status') != 'official_success':
        return None  # 原官方失败可重试；原尝试凭据仍保留。
    if not verified_record(truth):
        raise ValueError('候选官方凭据失效')
    if truth['plan_sha256'] != digest(plan):
        raise ValueError('候选方案与凭据不一致')
    if truth['fingerprint'] != object_digest(input_context(job, plan.parent, ATTACHMENT)[0]):
        raise ValueError('候选官方上下文不一致')
    return record


def completed_priority(checkpoint, fingerprint, plan):
    if not checkpoint.exists() or not plan.exists():
        return None
    row = json.loads(checkpoint.read_text(encoding='utf-8'))
    if (row['fingerprint'] == fingerprint
            and verified_record(row['selected_official'])
            and digest(plan) == row['selected_official']['plan_sha256']
            and row.get('candidates')
            and all(verified_record(c['official']) for c in row['candidates'])):
        return row
    return None


def run(args):
    initialize_worker_threads()
    import torch
    if not torch.cuda.is_available():raise RuntimeError('本实验要求已配置显卡批特征')
    torch.set_num_threads(1)
    out=Path(args.output_dir).resolve();out.mkdir(parents=True,exist_ok=True)
    baseline=Path(args.baseline_dir).resolve();cases=parse_cases(args.cases);config=ATTACHMENT/'data/config.txt'
    history={}
    for ledger in args.reuse_ledger:
        for line in Path(ledger).read_text(encoding='utf-8').splitlines():
            if line.strip():
                record=json.loads(line)
                if record.get('status')=='official_success':history[record['fingerprint']]=record
    settings=read_evaluation_config(str(config));scene=official.read_scene_a_config(str(config))
    files=list(Path(__file__).parent.glob('*.py'))+list((ATTACHMENT/'code').glob('*.py'))+[config]
    for case in cases:files += [ATTACHMENT/'data'/f'{case}.json',baseline/'plans'/f'{case}_A_N{args.core}.json',baseline/'checkpoints'/f'{case}_A_N{args.core}.json']
    files += [Path(p).resolve() for p in args.reuse_ledger]
    fp=ensure_run_manifest(out/'results.jsonl',files,dict(cases=cases,N=args.core,cap=args.block_cap,divisor=4,
        alphas=[0,.25,.5],traffic_weight=.2,pack_frontiers=True,gpu=torch.cuda.get_device_name(),dtype='float64',
        duration_mode=args.duration_mode,placement=args.placement,balanced_frontier=args.balanced_frontier,preserve_entries=args.preserve_entries,layer_window=args.layer_window,entry_ancestors=args.entry_ancestors,skip_replay=args.skip_replay,reuse_ledger=[str(Path(p).resolve()) for p in args.reuse_ledger]))
    for index,case in enumerate(cases,1):
        name=f'{case}_A_N{args.core}.json';checkpoint=out/'checkpoints'/name
        if completed_priority(checkpoint,fp,out/'plans'/name):
            print(f'[{index}/{len(cases)}] 复用{case}',flush=True);continue
        start=time.perf_counter();print(f'[{index}/{len(cases)}] {case} 固定切分优先级对照',flush=True)
        graph=json.loads((ATTACHMENT/'data'/f'{case}.json').read_text(encoding='utf-8'))
        parent=json.loads((baseline/'checkpoints'/name).read_text(encoding='utf-8'))['official_record']
        if not verified_record(parent):raise ValueError('父方案凭据失效')
        job=dict(case=case,N=args.core,kind='problem_1',plan_scene='A')
        if object_digest(input_context(job,baseline/'plans',ATTACHMENT)[0])!=parent['fingerprint']:raise ValueError('父评测上下文不一致')
        certificate=compute_certificate(graph,args.core)
        if certificate['lower_bound']>parent['real']['makespan']:raise ValueError('下界与官方结果冲突')
        model=make_priority_model(graph,args.core,args.block_cap,args.balanced_frontier,args.preserve_entries,args.layer_window,args.entry_ancestors)
        profiles=None
        if args.duration_mode!='legacy':
            from task_resource_profiles import task_resource_profiles
            profiles=task_resource_profiles(graph,model.block_of_op,settings['bandwidth'])
        proposals=[];seen={};identities=[];owners=[]
        for alpha in (0,.25,.5):
            plan,audit=construct_block_plan(model,args.core,alpha=alpha,
                duration_mode=args.duration_mode,profiles=profiles,placement=args.placement,
                same_wait=scene['task_same_core_wait_cycles'],cross_wait=scene['task_cross_core_wait_cycles'],
                bandwidth=settings['bandwidth'],traffic_weight=.2)
            key=object_digest(plan)
            if key in seen:seen[key]['alphas'].append(alpha);continue
            row=dict(plan_id=key,alphas=[alpha],plan=plan,dispatch_estimate=audit['estimated_makespan'])
            seen[key]=row;proposals.append(row);identities.append(list(range(len(model.blocks))))
            owners.append([audit['core_of'][i] for i in range(len(model.blocks))])
        torch.cuda.reset_peak_memory_stats();gpu_start=time.perf_counter()
        batch=BatchFeatures(model,args.core,device='cuda',batch_size=4,memory_mb=256)
        features=batch.evaluate(identities,owners);torch.cuda.synchronize();gpu_seconds=time.perf_counter()-gpu_start
        if features!=[scalar_features(model,a,c,args.core) for a,c in zip(identities,owners)]:raise ValueError('显卡特征不一致')
        gpu=dict(seconds=gpu_seconds,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved())
        scorer=None if args.skip_replay else SceneACandidateScorer(graph,config)
        best=parent;best_path=baseline/'plans'/name;results=[]
        for k,(row,feature) in enumerate(zip(proposals,features),1):
            folder=out/'candidates'/row['plan_id'];path=folder/name;write_json(path,row['plan'])
            saved=out/'candidate_checkpoints'/f'{case}_{row["plan_id"]}.json'
            record=load_candidate_checkpoint(saved,fp,path,job)
            if record is None:
                score=scorer.evaluate(row['plan']) if scorer is not None else None
                truth,reused=evaluate_with_history(job,out/'official',folder,history)
                record=dict(fingerprint=fp,plan_id=row['plan_id'],alphas=row['alphas'],feature=feature,
                    dispatch_estimate=row['dispatch_estimate'],score=score,official=truth,reused_official=reused)
                write_json(saved,record)
            truth=record['official'];results.append(record)
            if truth.get('status')=='official_success':
                if record['score'] is not None and truth['real']!=record['score']['real']:raise ValueError('新候选事件重放与官方不一致')
                if certificate['lower_bound']>truth['real']['makespan']:raise ValueError('新候选违反计算下界')
                if (truth['real']['makespan'],truth['real']['added_copy_bytes'])<(best['real']['makespan'],best['real']['added_copy_bytes']):best,best_path=truth,path
            print(f'  [{k}/{len(proposals)}] α={row["alphas"]}；官方{truth.get("status")}；当前{best["real"]["makespan"]}',flush=True)
        (out/'plans').mkdir(exist_ok=True);shutil.copyfile(best_path,out/'plans'/name)
        row=dict(case=case,N=args.core,fingerprint=fp,baseline_official=parent,selected_official=best,
            candidates=results,certificate=certificate,gpu=gpu,coarsening=model.coarsening_audit,
            duration_mode=args.duration_mode,placement=args.placement,resource_profiles=profiles,elapsed=time.perf_counter()-start)
        write_json(checkpoint,row);append_run_row(out/'results.jsonl',row)
    rows=[json.loads((out/'checkpoints'/f'{c}_A_N{args.core}.json').read_text(encoding='utf-8')) for c in cases]
    completed=sum(completed_priority(out/'checkpoints'/f'{c}_A_N{args.core}.json',fp,out/'plans'/f'{c}_A_N{args.core}.json') is not None for c in cases)
    write_json(out/'summary.json',dict(expected=len(cases),completed=completed,
        official_candidates=sum(verified_record(c['official']) for r in rows for c in r['candidates']),
        improved=sum(r['selected_official']['real']['makespan']<r['baseline_official']['real']['makespan'] for r in rows),
        regressed=sum(r['selected_official']['real']['makespan']>r['baseline_official']['real']['makespan'] for r in rows),
        scope='固定切分的开发集优先级对照，不是全量成绩'))
    if completed != len(cases):
        raise RuntimeError('部分候选官方评估未完成；保留父方案，使用原命令续跑')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cases',default='3,14,48,72,76,87');p.add_argument('--core',type=int,default=5,choices=[2,3,4,5])
    p.add_argument('--baseline-dir',required=True);p.add_argument('--output-dir',required=True)
    p.add_argument('--block-cap',type=int,default=240,help='有界块算子上限；工作量上限仍固定总工作/(4N)')
    p.add_argument('--duration-mode',choices=['legacy','resource','additive'],default='legacy',help='原计算估计、边界资源估计、计算搬运相加估计')
    p.add_argument('--placement',choices=['append','insert'],default='append',help='任务尾部追加或双侧等待安全插空')
    p.add_argument('--balanced-frontier',action='store_true',help='按每层工作量和核数限制同层打包，不改变依赖合并上限')
    p.add_argument('--preserve-entries',action='store_true',help='拒绝让后续算子为旧块新增外部入口等待')
    p.add_argument('--layer-window',type=int,help='限制块不跨固定依赖层窗口，同时按核数限制窗口工作量和同层打包')
    p.add_argument('--entry-ancestors',action='store_true',help='允许已是初始入口祖先的前驱，不扩大有效入口等待集合')
    p.add_argument('--skip-replay',action='store_true',help='关闭额外事件重放，仍逐候选原官方确认')
    p.add_argument('--reuse-ledger',action='append',default=[],help='复用内容上下文一致且凭据有效的官方记录')
    args=p.parse_args()
    if args.block_cap<1:p.error('块上限必须为正整数')
    if args.layer_window is not None and args.layer_window<1:p.error('层窗口必须为正整数')
    if args.entry_ancestors and not args.preserve_entries:p.error('祖先入口放宽必须与入口保持一起使用')
    from online_shared import process_lock
    with process_lock(Path(args.output_dir)/'run.lock'):
        run(args)
