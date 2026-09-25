"""E03：同一完整保存方案，迁移与迁移加交换的固定计数配对。"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import statistics

from experiment_construct_pair import DEVELOPMENT
from run_all import (ensure_run_manifest,run_input_files,parse_cases,append_run_row,
                     plan_digest,valid_official)
from pipeline import real_evaluate, fast_evaluate
from task_order_search import refine_plan_orders

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'通用神经网络处理器下的多核调度问题附件/data'


def binding(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def verified(row,fingerprint):
    try:
        if row.get('status')!='official_success' or row.get('fingerprint')!=fingerprint:
            return False
        if binding(row['variants'])!=row['binding']:
            return False
        return all(valid_official(v['real']) and hashlib.sha256(Path(v['plan_path']).read_bytes()).hexdigest()==v['plan_sha256']
                   for v in row['variants'].values())
    except (KeyError,OSError,ValueError):
        return False


def run_pair(task):
    graph=json.loads((DATA/f"{task['case']}.json").read_text(encoding='utf-8'))
    plan=json.loads(Path(task['source']).read_text(encoding='utf-8'))
    base=real_evaluate(graph,plan,'A')
    if not valid_official(base):
        raise ValueError(f'官方基线无效: {base}')
    variants={}
    modes=(False,True) if task['index']%2==0 else (True,False)
    for swaps in modes:
        name='swap' if swaps else 'migration'
        selected,best,audit=refine_plan_orders(graph,plan,base,real_evaluate,
            enable_swaps=swaps,search_seconds=None,max_evals=task['max_evals'],
            candidate_evaluator=fast_evaluate if task['fast_rerank'] and swaps else None,
            rerank_keep=8)
        if audit['errors'] or not valid_official(best):
            raise ValueError(f'候选/搬运不变量核验失败: {audit}')
        path=Path(task['output'])/f"{task['id']}_{name}.json"
        path.write_text(json.dumps(selected),encoding='utf-8')
        variants[name]={'real':best,'audit':audit,'plan_path':str(path),
                        'plan_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'plan_id':plan_digest(selected)}
    return {'id':task['id'],'case':task['case'],'N':task['N'],'source':task['source'],
            'source_sha256':hashlib.sha256(Path(task['source']).read_bytes()).hexdigest(),
            'status':'official_success','fingerprint':task['fingerprint'],'attempt':task['attempt'],
            'baseline':base,'variants':variants,'binding':binding(variants),
            'swap_vs_migration':variants['swap']['real']['makespan']/variants['migration']['real']['makespan']-1,
            'swap_vs_saved':variants['swap']['real']['makespan']/base['makespan']-1}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--cases',default=DEVELOPMENT)
    parser.add_argument('--cores',default='2,3,4,5')
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--max-evals',type=int,default=4000)
    parser.add_argument('--fast-rerank',action='store_true',help='交换候选最多八份快速精评，再原官方确认两份；成本单列')
    parser.add_argument('--max-pairs',type=int,default=0)
    parser.add_argument('--output-dir',required=True)
    args=parser.parse_args()
    if args.workers<1 or args.max_evals<1 or args.max_pairs<0:
        parser.error('预算或并发非法')
    cases=parse_cases(args.cases);cores=list(dict.fromkeys(map(int,args.cores.split(','))))
    if any(n not in (2,3,4,5) for n in cores):parser.error('核数必须为 2/3/4/5')
    out=Path(args.output_dir).resolve();out.mkdir(parents=True,exist_ok=True)
    source_dirs=[ROOT/'results/plans_fast_three_stage_v1',ROOT/'results/plans_fast_three_stage_validation_v1']
    tasks=[]
    for case in cases:
        for n in cores:
            matches=[p/f'{case}_A_N{n}.json' for p in source_dirs if (p/f'{case}_A_N{n}.json').is_file()]
            if len(matches)!=1:raise ValueError(f'{case}/{n} 需唯一已冻结输入，实际 {len(matches)}')
            tasks.append({'id':f'{case}_N{n}','case':case,'N':n,'source':str(matches[0]),
                          'output':str(out),'index':len(tasks),'max_evals':args.max_evals,
                          'fast_rerank':args.fast_rerank})
    ledger=out/'order_pairs.jsonl'
    fingerprint=ensure_run_manifest(ledger,run_input_files(cases)+[Path(t['source']) for t in tasks],
        {'cases':cases,'cores':cores,'max_evals':args.max_evals,'beam':4,'depth':3,
         'seconds':None,'official_limit':2,'workers':args.workers,
         'fast_rerank':args.fast_rerank,'rerank_keep':8})
    previous={};warnings=[]
    if ledger.exists():
        for i,line in enumerate(ledger.read_text(encoding='utf-8').splitlines(),1):
            try:r=json.loads(line);previous[r['id']]=r
            except (ValueError,KeyError):warnings.append(i)
    done={k for k,r in previous.items() if verified(r,fingerprint)}
    pending=[]
    for t in tasks:
        if t['id'] not in done:
            pending.append({**t,'fingerprint':fingerprint,'attempt':previous.get(t['id'],{}).get('attempt',0)+1})
    if args.max_pairs:pending=pending[:args.max_pairs]
    print(f'[{len(done)}/{len(tasks)}] E03 固定计数配对，本批 {len(pending)}，损坏行 {warnings}',flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(run_pair,t):t for t in pending}
        for future in as_completed(futures):
            t=futures[future]
            try:row=future.result();done.add(t['id'])
            except Exception as exc:row={**t,'status':'failed','error':repr(exc)}
            append_run_row(ledger,row);previous[t['id']]=row
            print(f"[{len(done)}/{len(tasks)}] {t['id']} {row['status']}：交换相对迁移 {row.get('swap_vs_migration')}；相对保存方案 {row.get('swap_vs_saved')}",flush=True)
    rows=[previous[t['id']] for t in tasks if t['id'] in done]
    summary={'expected':len(tasks),'completed':len(rows),'failed':len(tasks)-len(done),
        'wins_vs_migration':sum(r['swap_vs_migration']<0 for r in rows),
        'losses_vs_migration':sum(r['swap_vs_migration']>0 for r in rows),
        'wins_vs_saved':sum(r['swap_vs_saved']<0 for r in rows),
        'losses_vs_saved':sum(r['swap_vs_saved']>0 for r in rows),
        'mean_vs_saved':statistics.mean(r['swap_vs_saved'] for r in rows) if rows else None,
        'mean_vs_migration':statistics.mean(r['swap_vs_migration'] for r in rows) if rows else None,
        'scope':'固定完整保存输入、相同重放上限与官方候选数；实际评估次数可能不同，独立报告成本，非完整求解器等墙钟比较'}
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False),flush=True)
    if any(r.get('status')=='failed' for r in previous.values()):raise SystemExit(1)


if __name__=='__main__':main()
