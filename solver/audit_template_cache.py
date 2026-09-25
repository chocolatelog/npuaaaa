"""E04：重放 E03 已生成候选，缓存前后完整结果逐字段相等。"""
import argparse
import hashlib
import json
from pathlib import Path
import time

from local_template_cache import TemplateCache
from scene_a_fast import build_counter_evaluator, evaluate_scene_a_fast, official
from run_all import ensure_run_manifest,run_input_files,append_run_row
from experiment_branch_partition import seal_result
from experiment_order_pair import binding

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--pairs',default=str(ROOT/'results/p1_e03_r02/order_pairs.jsonl'))
    parser.add_argument('--output-dir',required=True)
    parser.add_argument('--backend',choices=('counter','unified'),default='counter')
    parser.add_argument('--reference-ledger',help='旧完整等价实验摘要；对应候选已有真值时不重复调用官方')
    parser.add_argument('--max-tasks',type=int,default=0)
    args=parser.parse_args()
    rows=[json.loads(line) for line in Path(args.pairs).read_text(encoding='utf-8').splitlines()]
    selected=[r for r in rows if r.get('status')=='official_success' and r['variants']['swap']['audit']['screened_candidates']]
    references={}
    if args.reference_ledger:
        references={r['id']:r for r in map(json.loads,Path(args.reference_ledger).read_text(encoding='utf-8').splitlines())}
        if any(not references.get(r['id'],{}).get('complete_results_equal') for r in selected):
            raise ValueError('旧完整结果核验记录不齐全')
    cases=sorted({r['case'] for r in selected})
    out=Path(args.output_dir).resolve();out.mkdir(parents=True,exist_ok=True)
    ledger=out/'cache_audit.jsonl'
    fingerprint=ensure_run_manifest(ledger,run_input_files(cases)+[Path(args.pairs)]+[Path(r['source']) for r in selected]+
                                   ([Path(args.reference_ledger)] if args.reference_ledger else []),
                                   {'cache_entries':128,'payload_mb':64,'candidate_source':'E03 R02 fixed screened candidates',
                                    'backend':args.backend,'reference_ledger':args.reference_ledger})
    previous={}
    if ledger.exists():
        for line in ledger.read_text(encoding='utf-8').splitlines():
            try:r=json.loads(line);previous[r['id']]=r
            except (ValueError,KeyError):pass
    def valid(row):
        content=dict(row);expected=content.pop('binding',None)
        return (row.get('status')=='success' and row.get('fingerprint')==fingerprint and
                expected==binding(content) and row.get('complete_results_equal'))
    pending=[r for r in selected if not valid(previous.get(r['id'],{}))]
    if args.max_tasks<0:parser.error('分批数不能为负')
    if args.max_tasks:pending=pending[:args.max_tasks]
    print(f'[{sum(valid(r) for r in previous.values())}/{len(selected)}] 模板缓存核验，本批{len(pending)}',flush=True)
    if args.backend=='unified' and pending:
        import torch
        if not torch.cuda.is_available():parser.error('需要显卡环境核验批量特征')
        from scene_a_replay import build_resource_evaluator
        plain=build_resource_evaluator(unified=True)
        factory=lambda template_cache:build_resource_evaluator(unified=True,template_cache=template_cache)
    else:plain=evaluate_scene_a_fast;factory=build_counter_evaluator
    for index,row in enumerate(selected,1):
        key=row['id']
        if row not in pending:continue
        graph=json.loads((ROOT/'通用神经网络处理器下的多核调度问题附件/data'/f"{row['case']}.json").read_text(encoding='utf-8'))
        source=json.loads(Path(row['source']).read_text(encoding='utf-8'))
        candidates=row['variants']['swap']['audit']['screened_candidates']
        cache=TemplateCache()
        before=time.perf_counter();cached=factory(template_cache=cache)
        setup=time.perf_counter()-before
        plain_seconds=cached_seconds=0.0;digests=[];first_plain=first_cached=None
        gpu_stats=None
        if args.backend=='unified':
            from batch_features import BatchFeatures,scalar_features
            from model import Model
            from run_all import block_cap_for
            before=time.perf_counter();torch.cuda.reset_peak_memory_stats()
            model=Model(graph,block_ops_cap=block_cap_for(sum(o['op'] not in ('COPY_IN','COPY_OUT') for o in graph['ops'])))
            assignment=[source['node_to_subgraph'][str(b[0])] for b in model.blocks]
            owners=[]
            for candidate in candidates:
                cc=[0]*(max(assignment)+1)
                for core,order in enumerate(candidate['orders']):
                    for task in order:cc[task]=core
                owners.append(cc)
            features=BatchFeatures(model,row['N']).evaluate([assignment]*len(owners),owners)
            assert features==[scalar_features(model,assignment,c,row['N']) for c in owners]
            gpu_stats={'seconds':time.perf_counter()-before,'peak_allocated':torch.cuda.max_memory_allocated(),
                       'equal':True,'candidates':len(candidates)}
        for j,candidate in enumerate(candidates):
            plan={**source,'core_schedules':candidate['orders']}
            call=(graph,plan,60,{'L1':524288,'UB':131072},1000,100)
            # 交替先后减少固定次序偏差；每组第一次为冷缓存。
            results={}
            modes=[('plain',plain),('cached',cached)]
            if (index+j)%2:modes.reverse()
            for name,evaluator in modes:
                start=time.perf_counter();results[name]=evaluator(*call);elapsed=time.perf_counter()-start
                if name=='plain':
                    plain_seconds+=elapsed
                    if j==0:first_plain=elapsed
                else:
                    cached_seconds+=elapsed
                    if j==0:first_cached=elapsed
            if results['plain']!=results['cached']:
                raise AssertionError(f'{key} 第 {j} 份候选缓存结果不一致')
            digest=hashlib.sha256(json.dumps(results['cached'],sort_keys=True).encode()).hexdigest()
            if references:
                if digest!=references[key]['result_sha256'][j]:
                    raise AssertionError('当前完整结果与既有完整核验摘要不同')
            elif row['case'] in ('case_044','case_049','case_065','case_082') and j==0:
                if official.evaluate_scene_a(*call)!=results['cached']:
                    raise AssertionError('缓存结果与原官方完整字段不一致')
            digests.append(digest)
        result={'id':key,'case':row['case'],'N':row['N'],'status':'success','fingerprint':fingerprint,
                'count':len(candidates),'complete_results_equal':True,'result_sha256':digests,
                'plain_seconds':plain_seconds,'cached_seconds':cached_seconds,'setup_seconds':setup,
                'first_plain_seconds':first_plain,'first_cached_seconds':first_cached,
                'cache_stats':cache.stats,
                'backend':args.backend,'gpu_features':gpu_stats,
                'reference_fingerprint':references[key]['fingerprint'] if references else None,
                'reference_digests_equal':True if references else None,
                'scope':'固定候选冷启动及后续复用完整结果；字节上限为缓存有效载荷，不是进程内存硬上限'}
        result=seal_result(result,fingerprint)
        append_run_row(ledger,result);previous[key]=result
        print(f'[{index}/{len(selected)}] {key} 完整结果相等，耗时 {plain_seconds:.3f}->{cached_seconds:.3f} 秒，命中 {cache.stats["hits"]} / {cache.stats["hits"]+cache.stats["misses"]}',flush=True)
    completed=[previous[r['id']] for r in selected if valid(previous.get(r['id'],{}))]
    summary={'expected':len(selected),'completed':len(completed),'candidates':sum(r['count'] for r in completed),
             'plain_seconds':sum(r['plain_seconds'] for r in completed),'cached_seconds':sum(r['cached_seconds'] for r in completed),
             'setup_seconds':sum(r['setup_seconds'] for r in completed),'max_cached_payload':max((r['cache_stats']['peak_payload_bytes'] for r in completed),default=0)}
    target=out/'summary.json';text=json.dumps(summary,ensure_ascii=False,indent=2)
    if not target.exists() or target.read_text(encoding='utf-8')!=text:target.write_text(text,encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
