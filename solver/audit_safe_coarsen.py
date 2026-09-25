"""真实收缩候选逐轮比较显卡与中央处理器，并作官方候选质量对照。"""
import argparse
import json
import time
from pathlib import Path

from batch_features import BatchFeatures
from coarsen_safe import safe_coarsen
from construct_v2 import netbenefit_construct
from construct_refine import resolve_granularity, _coarsen_solution
from model import Model
from solution import Sol, Context
from pipeline import real_evaluate
from run_all import (block_cap_for, append_run_row, ensure_run_manifest,
                     run_input_files, plan_digest, valid_official)

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output-dir',required=True)
    parser.add_argument('--cases',default='49,65,82')
    parser.add_argument('--cores',default='2,4')
    args=parser.parse_args()
    import torch
    if not torch.cuda.is_available():
        raise SystemExit('本实验要求现有显卡环境')
    cases=[f'case_{int(c):03d}' for c in args.cases.split(',')]
    cores=list(map(int,args.cores.split(',')))
    out=Path(args.output_dir).resolve();out.mkdir(parents=True,exist_ok=True)
    ledger=out/'safe_coarsen.jsonl'
    fingerprint=ensure_run_manifest(ledger,run_input_files(cases),
        {'cases':cases,'cores':cores,'grain':'balanced','max_candidates':512,'merge_limit':256,
         'ops_cap_multiplier':2,'gpu':torch.cuda.get_device_name(0),'batch_size':32,'tensor_chunk':512})
    previous={}
    if ledger.exists():
        for line in ledger.read_text(encoding='utf-8').splitlines():
            r=json.loads(line);previous[r['id']]=r
    total=len(cases)*len(cores)
    for index,(case,n) in enumerate(((c,n) for c in cases for n in cores),1):
        key=f'{case}_N{n}'
        if previous.get(key,{}).get('status')=='official_success':
            print(f'[{index}/{total}] 已完成 {key}',flush=True)
            continue
        print(f'[{index}/{total}] 真实候选显卡/中央处理器配对 {key}',flush=True)
        graph=json.loads((ROOT/'通用神经网络处理器下的多核调度问题附件/data'/f'{case}.json').read_text(encoding='utf-8'))
        eligible=sum(o['op'] not in ('COPY_IN','COPY_OUT') for o in graph['ops'])
        model=Model(graph,block_ops_cap=block_cap_for(eligible));profile=resolve_granularity(model,n,'balanced')
        raw=Sol(*netbenefit_construct(model,n,'A',max_sg_ops=profile.target_ops,corrected=True))
        before=time.perf_counter()
        cpu,ca=safe_coarsen(model,raw,n,profile.target_strips,2*profile.target_ops)
        cpu_seconds=time.perf_counter()-before
        torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();before=time.perf_counter()
        evaluator=BatchFeatures(model,n,tensor_chunk=512)
        gpu,ga=safe_coarsen(model,raw,n,profile.target_strips,2*profile.target_ops,evaluator=evaluator)
        torch.cuda.synchronize();gpu_seconds=time.perf_counter()-before
        if (cpu.sg_of_block!=gpu.sg_of_block or cpu.core_of_sg!=gpu.core_of_sg
                or ca['trace']!=ga['trace']):
            raise AssertionError('显卡与中央处理器的合并轨迹不同')
        variants={};ctx=Context(model,n,'A')
        for label,sol in [('raw',raw),('old_coarsen',_coarsen_solution(model,raw,profile.target_strips)),('safe_coarsen',gpu)]:
            sol=sol.clone();valid_before=sol.validate(model,n)
            pm,pa,info=ctx.evaluate(sol,use_cache=False)
            plan=model.plan_from(sol.sg_of_block,info['orders'])
            truth=real_evaluate(graph,plan,'A')
            if not valid_official(truth):
                raise ValueError(truth)
            path=out/f'{key}_{label}.json'
            path.write_text(json.dumps(plan),encoding='utf-8')
            variants[label]={'real':truth,'proxy_makespan':pm,'proxy_added':pa,'tasks':sol.num_used_sg(),
                             'valid_before_repair':valid_before,'plan_id':plan_digest(plan),'plan_path':str(path)}
        row={'id':key,'status':'official_success','fingerprint':fingerprint,'case':case,'N':n,
             'cpu_seconds':cpu_seconds,'gpu_seconds_including_setup':gpu_seconds,
             'cpu_gpu_same_trace':True,'gpu_peak_allocated':torch.cuda.max_memory_allocated(),
             'gpu_audit':ga,'variants':variants,
             'scope':'保守收缩独立候选及真实特征批次试验；默认构造池未替换，未评端到端搜索收益'}
        append_run_row(ledger,row);previous[key]=row
        print(f"[{index}/{total}] {key} 候选 {ga['candidate_count']}，CPU（中央处理器）{cpu_seconds:.3f} 秒，GPU（显卡）{gpu_seconds:.3f} 秒；官方 "+str({k:v['real']['makespan'] for k,v in variants.items()}),flush=True)
    (out/'summary.json').write_text(json.dumps({'completed':len(previous),'expected':total,'rows':list(previous.values())},ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
