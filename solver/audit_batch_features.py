"""固定真实候选上核验显卡特征一致性、传输后总成本及显存。"""
import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

from batch_features import BatchFeatures, scalar_features
from model import Model
from run_all import ensure_run_manifest, run_input_files, block_cap_for, append_run_row

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pairs', default=str(ROOT/'results/p1_e02_r01/pairs.jsonl'))
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--cases', default='1,49,65,82,97')
    parser.add_argument('--core', type=int, default=4)
    args = parser.parse_args()
    import torch
    if not torch.cuda.is_available():
        raise SystemExit('本实验要求现有显卡环境，未自动退回中央处理器')
    cases = [f'case_{int(x):03d}' for x in args.cases.split(',')]
    entries = [json.loads(line) for line in Path(args.pairs).read_text(encoding='utf-8').splitlines()]
    selected = [r for r in entries if r.get('status') == 'official_success' and r['case'] in cases and r['N'] == args.core]
    paths = [Path(v['plan_path']) for r in selected for v in r['variants'].values()]
    out = Path(args.output_dir).resolve(); out.mkdir(parents=True, exist_ok=True)
    ledger = out/'gpu_features.jsonl'
    fingerprint = ensure_run_manifest(ledger, run_input_files(cases)+paths+[Path(args.pairs)],
        {'cases':cases,'core':args.core,'device':torch.cuda.get_device_name(0),'batch_size':32,
         'tensor_chunk':64,'memory_mb':256,'torch':torch.__version__,'cuda':torch.version.cuda})
    previous = {}
    if ledger.exists():
        for line in ledger.read_text(encoding='utf-8').splitlines():
            r=json.loads(line);previous[r['id']]=r
    for index,case in enumerate(cases,1):
        if previous.get(case,{}).get('status')=='success':
            continue
        print(f'[{index}/{len(cases)}] 显卡特征核验 {case}',flush=True)
        graph=json.loads((ROOT/'通用神经网络处理器下的多核调度问题附件/data'/f'{case}.json').read_text(encoding='utf-8'))
        eligible=sum(o['op'] not in ('COPY_IN','COPY_OUT') for o in graph['ops'])
        model=Model(graph,block_ops_cap=block_cap_for(eligible))
        assignments,owners=[],[]
        for path in [Path(v['plan_path']) for r in selected if r['case']==case for v in r['variants'].values()]:
            plan=json.loads(path.read_text(encoding='utf-8'))
            maps=plan['node_to_subgraph']
            if any(len({maps[str(op)] for op in block})!=1 for block in model.blocks):
                raise ValueError('保存方案不兼容当前块化')
            a=[maps[str(block[0])] for block in model.blocks]
            core_map={s:c for c,seq in enumerate(plan['core_schedules']) for s in seq}
            used=sorted(set(a));remap={s:i for i,s in enumerate(used)}
            assignments.append([remap[s] for s in a]);owners.append([core_map[s] for s in used])
        if not assignments:
            raise ValueError('没有匹配候选')
        experiments=[]
        for target in (len(assignments),128):
            aa=[assignments[i%len(assignments)] for i in range(target)]
            cc=[owners[i%len(owners)] for i in range(target)]
            torch.cuda.reset_peak_memory_stats()
            init=time.perf_counter();batch=BatchFeatures(model,args.core)
            torch.cuda.synchronize();setup=time.perf_counter()-init
            cpu_times,gpu_times=[],[]
            first_gpu=None
            for repeat in range(3):
                start=time.perf_counter()
                reference=[scalar_features(model,a,c,args.core) for a,c in zip(aa,cc)]
                cpu_times.append(time.perf_counter()-start)
                torch.cuda.synchronize();start=time.perf_counter()
                values=batch.evaluate(aa,cc)
                torch.cuda.synchronize();gpu_times.append(time.perf_counter()-start)
                if values != reference:
                    raise AssertionError('显卡与逐候选参考特征不一致')
            experiments.append({'count':target,'unique_plans':len({json.dumps(a)+json.dumps(c) for a,c in zip(aa,cc)}),
                'replicated_for_throughput':target>len(assignments),'exact_match':True,
                'cpu_seconds_median':statistics.median(cpu_times),'gpu_seconds_median':statistics.median(gpu_times),
                'gpu_first_seconds':gpu_times[0],'gpu_setup_seconds':setup,
                'speedup_including_transfers':statistics.median(cpu_times)/statistics.median(gpu_times),
                'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'batch_stats':batch.last_stats,
                'features_sha256':hashlib.sha256(json.dumps(reference,sort_keys=True).encode()).hexdigest()})
        row={'id':case,'status':'success','fingerprint':fingerprint,'gpu':torch.cuda.get_device_name(0),
             'experiments':experiments,'scope':'特征和粗分数数值核验；不是完整代理模拟或官方评估显卡化，重复候选仅测吞吐'}
        append_run_row(ledger,row);previous[case]=row
        print(json.dumps({'case':case,'experiments':experiments},ensure_ascii=False),flush=True)
    summary={'completed':len(previous),'expected':len(cases),'rows':[previous[c] for c in cases if c in previous]}
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
