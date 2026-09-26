"""操作级映射后A安全分组：显卡批特征、生成检查点、独立官方保底入口。"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from official_protocol import ATTACHMENT,digest
from run_all import ensure_run_manifest,parse_cases
from run_saved_refine import write_json
from online_shared import process_lock
from runtime_resources import initialize_worker_threads


def main(args):
    initialize_worker_threads()
    import torch
    from mapped_blocks import generate_mapped_block_candidates
    from model import Model
    from batch_features import BatchFeatures,scalar_features
    from evaluation_validation import read_evaluation_config
    if not torch.cuda.is_available():raise RuntimeError('需要可用的显卡批特征环境')
    torch.set_num_threads(1)
    out=Path(args.output_dir).resolve();base=Path(args.baseline_dir).resolve()
    cases=parse_cases(args.cases);config=ATTACHMENT/'data/config.txt'
    settings=read_evaluation_config(str(config))
    out.mkdir(parents=True,exist_ok=True)
    with process_lock(out/'generation.lock'):
        files=list(Path(__file__).parent.glob('*.py'))+list((ATTACHMENT/'code').glob('*.py'))+[config]
        for case in cases:files.extend([ATTACHMENT/'data'/f'{case}.json',base/'plans'/f'{case}_A_N{args.core}.json'])
        fp=ensure_run_manifest(out/'generation.jsonl',files,dict(cases=cases,N=args.core,caps=[120,240],delays=[0,100],policy='height',divisor=4,gpu=torch.cuda.get_device_name(),batch=4,memory_mb=256))
        for i,case in enumerate(cases,1):
            saved=out/'generation_checkpoints'/f'{case}.json'
            if saved.exists():
                row=json.loads(saved.read_text(encoding='utf8'))
                if row['fingerprint']!=fp or any(digest(p)!=sha for p,sha in row['artifacts'].items()):raise ValueError('生成检查点或候选被修改')
                print(f'[{i}/{len(cases)}] 复用已冻结生成 {case}',flush=True);continue
            start=time.perf_counter();name=f'{case}_A_N{args.core}.json'
            graph=json.loads((ATTACHMENT/'data'/f'{case}.json').read_text(encoding='utf8'))
            parent=json.loads((base/'plans'/name).read_text(encoding='utf8'))
            candidates=generate_mapped_block_candidates(graph,parent,args.core,bandwidth=settings['bandwidth'])
            features=[];gpu={}
            if candidates:
                model=Model(graph,block_ops_cap=1,block_policy='bounded')
                aa=[];cc=[]
                for candidate in candidates:
                    plan=candidate['plan'];owners={sg:c for c,row in enumerate(plan['core_schedules']) for sg in row}
                    aa.append([plan['node_to_subgraph'][str(block[0])] for block in model.blocks]);cc.append([owners[j] for j in range(len(owners))])
                torch.cuda.reset_peak_memory_stats();gpu_start=time.perf_counter()
                batch=BatchFeatures(model,args.core,device='cuda',batch_size=4,memory_mb=256)
                features=batch.evaluate(aa,cc);torch.cuda.synchronize()
                if features!=[scalar_features(model,a,c,args.core) for a,c in zip(aa,cc)]:raise ValueError('显卡和标量特征不一致')
                gpu=dict(seconds=time.perf_counter()-gpu_start,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved())
            artifacts={};details=[]
            for k,candidate in enumerate(candidates):
                path=out/f'candidate_{k}'/name;write_json(path,candidate['plan']);artifacts[str(path)]=digest(path)
                details.append(dict(source=candidate['source'],audit=candidate['audit'],features=features[k],path=str(path)))
            write_json(saved,dict(case=case,N=args.core,fingerprint=fp,candidates=details,artifacts=artifacts,gpu=gpu,seconds=time.perf_counter()-start))
            print(f'[{i}/{len(cases)}] {case}：生成{len(candidates)}份合法候选，{time.perf_counter()-start:.1f}秒',flush=True)
    if args.generate_only:return
    cmd=[sys.executable,'-X','utf8',str(Path(__file__).with_name('run_saved_refine.py')),'--cases',args.cases,'--cores',str(args.core),'--scene','A','--portfolio-only',
         '--input-dir',str(base/'plans'),'--output-dir',str(out/'evaluation'),'--workers',str(args.workers),'--reuse-ledger',str(base/'official/results.jsonl')]
    for k in range(4):cmd.extend(['--portfolio-dir',str(out/f'candidate_{k}')])
    for path in args.reuse_ledger:cmd.extend(['--reuse-ledger',str(Path(path).resolve())])
    subprocess.run(cmd,check=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cases',required=True);p.add_argument('--core',type=int,choices=[2,3,4,5],required=True)
    p.add_argument('--baseline-dir',required=True);p.add_argument('--output-dir',required=True);p.add_argument('--workers',type=int,default=2)
    p.add_argument('--reuse-ledger',action='append',default=[]);p.add_argument('--generate-only',action='store_true')
    main(p.parse_args())
