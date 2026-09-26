"""固定大图上验证防成环块化，显卡批特征筛选，原官方结果保底。"""
import argparse
import json
from pathlib import Path
import shutil
import time

from model import Model
from construct import heft_construct
from batch_features import BatchFeatures, scalar_features
from official_protocol import (ATTACHMENT, object_digest, digest, verified_record,
                               input_context, evaluate_job)
from run_all import ensure_run_manifest, parse_cases, append_run_row
from run_saved_refine import write_json
from scene_a_event import derive_multicore_plan, validate_task_order
from runtime_resources import initialize_worker_threads


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases',default='3')
    parser.add_argument('--core',type=int,default=5)
    parser.add_argument('--baseline-dir',required=True)
    parser.add_argument('--output-dir',required=True)
    parser.add_argument('--pack-frontiers',action='store_true')
    args=parser.parse_args()
    if args.core not in (2,3,4,5):raise ValueError('核数必须为2～5')
    initialize_worker_threads()
    import torch
    if not torch.cuda.is_available():raise RuntimeError('本试验显式要求显卡批特征，不悄悄改为CPU')
    torch.set_num_threads(1)
    out=Path(args.output_dir).resolve();out.mkdir(parents=True,exist_ok=True)
    baseline_dir=Path(args.baseline_dir).resolve()
    cases=parse_cases(args.cases)
    files=list(Path(__file__).parent.glob('*.py'))+list((ATTACHMENT/'code').glob('*.py'))+[ATTACHMENT/'data/config.txt']
    for case in cases:
        files += [ATTACHMENT/'data'/f'{case}.json',baseline_dir/'plans'/f'{case}_A_N{args.core}.json',
                  baseline_dir/'checkpoints'/f'{case}_A_N{args.core}.json']
    fingerprint=ensure_run_manifest(out/'results.jsonl',files,dict(cases=cases,N=args.core,
        block_caps=[120,240,480],work_cap_divisor=4,balances=[0,.2,1,2],official_per_granularity=2,
        gpu=torch.cuda.get_device_name(),dtype='float64',batch_size=4,memory_mb=256,
        pack_frontiers=args.pack_frontiers))
    for index,case in enumerate(cases,1):
        filename=f'{case}_A_N{args.core}.json'
        checkpoint=out/'checkpoints'/filename
        if checkpoint.exists():
            old=json.loads(checkpoint.read_text(encoding='utf-8'))
            if old['fingerprint']==fingerprint and verified_record(old['selected_official']):
                print(f'[{index}/{len(cases)}] {case} 复用已完成检查点',flush=True);continue
        print(f'[{index}/{len(cases)}] {case} 有界块化候选',flush=True)
        start=time.perf_counter()
        graph=json.loads((ATTACHMENT/'data'/f'{case}.json').read_text(encoding='utf-8'))
        parent=json.loads((baseline_dir/'checkpoints'/filename).read_text(encoding='utf-8'))
        base=parent['official_record'];assert verified_record(base)
        job=dict(case=case,N=args.core,kind='problem_1',plan_scene='A')
        if object_digest(input_context(job,baseline_dir/'plans',ATTACHMENT)[0])!=base['fingerprint']:
            raise ValueError('父方案官方上下文与当前实验不一致')
        eligible=[o for o in graph['ops'] if o['op'] not in ('COPY_IN','COPY_OUT')]
        work_cap=sum(max(1,o.get('cycles',1)) for o in eligible)/(args.core*4)
        proposals=[];model_audits=[];seen=set()
        for cap in (120,240,480):
            built=time.perf_counter()
            model=Model(graph,block_ops_cap=cap,block_policy='bounded',block_work_cap=work_cap,
                        pack_frontiers=args.pack_frontiers)
            assignments=[];owners=[];candidates=[]
            for balance in (0,.2,1,2):
                groups,cores=heft_construct(model,args.core,'A',max_sg_ops=1,balance=balance)
                # 一个已验证有界块对应一个任务，避免构造末端再次把跨核依赖并成环。
                orders=[[] for _ in range(args.core)]
                for block in model.block_topo:
                    sg=groups[block];orders[cores[sg]].append(sg)
                plan=model.plan_from(groups,orders)
                validate_task_order(derive_multicore_plan(graph,plan))
                key=object_digest(plan)
                if key in seen:continue
                seen.add(key);assignments.append(groups);owners.append(cores)
                candidates.append(dict(plan=plan,plan_id=key,block_cap=cap,balance=balance))
            built_seconds=time.perf_counter()-built
            if not candidates:continue
            torch.cuda.reset_peak_memory_stats()
            gpu_started=time.perf_counter()
            batch=BatchFeatures(model,args.core,device='cuda',batch_size=4,memory_mb=256)
            features=batch.evaluate(assignments,owners)
            torch.cuda.synchronize();gpu_seconds=time.perf_counter()-gpu_started
            cpu_started=time.perf_counter()
            reference=[scalar_features(model,a,c,args.core) for a,c in zip(assignments,owners)]
            cpu_seconds=time.perf_counter()-cpu_started
            if reference!=features:raise ValueError('显卡特征与逐候选CPU参考不一致')
            for candidate,feature in zip(candidates,features):
                candidate['feature']=feature
                path=out/'candidates'/candidate['plan_id']/filename
                write_json(path,candidate['plan'])
                candidate['plan_path']=str(path)
            proposals.extend(sorted(candidates,key=lambda r:(r['feature']['coarse_score'],r['plan_id']))[:2])
            model_audits.append(dict(block_cap=cap,coarsening=model.coarsening_audit,build_seconds=built_seconds,
                gpu_seconds_including_setup=gpu_seconds,cpu_reference_seconds=cpu_seconds,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                exact_feature_match=True,candidates=[{k:v for k,v in r.items() if k!='plan'} for r in candidates]))
            write_json(out/'generation'/filename,dict(models=model_audits,official_proposals=[p['plan_id'] for p in proposals]))
            print(f'  块上限{cap}：{len(model.blocks)}块，最大{max(map(len,model.blocks))}算子，合法候选{len(candidates)}',flush=True)
            del batch
        best=base;best_path=baseline_dir/'plans'/filename;records=[]
        for candidate_index,candidate in enumerate(proposals,1):
            row=evaluate_job(job,out/'official',Path(candidate['plan_path']).parent)
            records.append(dict(plan_id=candidate['plan_id'],block_cap=candidate['block_cap'],official=row,
                                feature=candidate['feature']))
            append_run_row(out/'candidate_official.jsonl',row)
            if row.get('status')=='official_success' and (
                row['real']['makespan'],row['real']['added_copy_bytes'])<(
                    best['real']['makespan'],best['real']['added_copy_bytes']):
                best,best_path=row,Path(candidate['plan_path'])
            print(f'  官方[{candidate_index}/{len(proposals)}] {row["status"]}；当前{best["real"]["makespan"]}，父{base["real"]["makespan"]}',flush=True)
        (out/'plans').mkdir(exist_ok=True)
        shutil.copyfile(best_path,out/'plans'/filename)
        row=dict(case=case,N=args.core,fingerprint=fingerprint,baseline_official=base,
                 selected_official=best,models=model_audits,candidates=records,elapsed=time.perf_counter()-start,
                 gpu_scope='只加速计算负载与分区边界批特征；官方评估始终CPU')
        write_json(checkpoint,row);append_run_row(out/'results.jsonl',row)


if __name__=='__main__':main()
