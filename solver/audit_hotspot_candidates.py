"""固定热点候选潜力审计；复用已有精评，每候选落盘，不计入两次官方实验成绩。"""
import argparse
import hashlib
import json
from pathlib import Path
import time

from experiment_branch_partition import seal_result
from experiment_order_pair import binding
from model import Model
from pipeline import fast_evaluate, real_evaluate
from run_all import block_cap_for, ensure_run_manifest, run_input_files, append_run_row, valid_official, plan_digest
from scene_a_event import SceneAEventModel, derive_multicore_plan
from solution import Sol
from task_order_search import search_orders
from partition_identity import partition_key

ROOT = Path(__file__).resolve().parents[1]


def reusable(row, fingerprint):
    copy = dict(row); expected = copy.pop('binding',None)
    return (expected == binding(copy) and row.get('fingerprint') == fingerprint and
            all(Path(p).is_file() and hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
                for p,h in row.get('artifacts',{}).items()))


def load_rows(path):
    rows = {}
    if path.exists():
        for line in path.read_text(encoding='utf-8').splitlines():
            try:
                r = json.loads(line); rows[r['id']] = r
            except (ValueError,KeyError):
                pass
    return rows


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input-dir',required=True)
    parser.add_argument('--output-dir',required=True)
    parser.add_argument('--max-candidates',type=int,default=0)
    args=parser.parse_args()
    source=Path(args.input_dir).resolve()/'branch_experiments.jsonl'
    groups=load_rows(source)
    out=Path(args.output_dir).resolve();out.mkdir(parents=True,exist_ok=True)
    ledger=out/'potential_candidates.jsonl'
    files=run_input_files(sorted({r['case'] for r in groups.values()}))+[source]
    files += sorted({Path(p) for r in groups.values() for p in r['artifacts']})
    fp=ensure_run_manifest(ledger,files,{'input':str(source),'replays':400,'replica_limit':8,
                                     'official_limit_per_configuration':1,'diagnostic_only':True})
    saved=load_rows(ledger); results={k:v for k,v in saved.items() if reusable(v,fp)}
    total=sum(len(r['candidate_features']) for r in groups.values())
    print(f'[{len(results)}/{total}] 固定候选局部精评，已有项复用',flush=True)
    count=0
    for gid,group in groups.items():
        candidates=group['candidate_features']
        if not candidates:
            continue
        graph=json.loads((ROOT/'通用神经网络处理器下的多核调度问题附件/data'/f'{group["case"]}.json').read_text(encoding='utf-8'))
        model=Model(graph,block_ops_cap=block_cap_for(sum(o['op'] not in ('COPY_IN','COPY_OUT') for o in graph['ops'])))
        evaluator=SceneAEventModel(graph)
        admitted={binding(r['source']):r for r in group['evaluated']}
        for index,candidate in enumerate(candidates):
            cid=f'{gid}_{index:03d}'
            if cid in results:
                continue
            if args.max_candidates and count >= args.max_candidates:
                return
            start=time.perf_counter()
            prior=admitted.get(binding(candidate['source']))
            if prior:
                record=dict(prior)
                path=Path(prior['plan_path'])
                assert hashlib.sha256(path.read_bytes()).hexdigest()==group['artifacts'][str(path)]
                record['local_reused']=True
            else:
                sol=Sol(candidate['assignment'],candidate['owners'])
                if not sol.validate(model,group['N']):
                    raise ValueError('保存候选非法')
                _,_,info=model.evaluate(sol.sg_of_block,sol.core_of_sg,'A',group['N'])
                plan=model.plan_from(sol.sg_of_block,info['orders'])
                estimate=evaluator.evaluate(plan)
                view=derive_multicore_plan(graph,plan)
                orders,stats=search_orders({s:t['duration'] for s,t in estimate['tasks'].items()},
                    view['subgraph_preds'],plan['core_schedules'],bandwidth_floor=estimate['total_copy_bytes']/60,
                    max_evals=400,rounds=2,seconds=None,keep=1)
                predicted=estimate['makespan']
                if orders and orders[0]['makespan'] < predicted:
                    plan['core_schedules']=orders[0]['orders'];predicted=orders[0]['makespan']
                from evaluation_validation import validate_task_order
                validate_task_order(derive_multicore_plan(graph,plan))
                path=out/f'{cid}.json';path.write_text(json.dumps(plan),encoding='utf-8')
                record={'source':candidate['source'],'estimate':predicted,
                        'partition_added':estimate['partition_added_bytes'],'spill_added':estimate['spill_bytes'],
                        'total_added':estimate['total_added_bytes'],'local_search':stats,
                        'plan_id':plan_digest(plan),'plan_path':str(path),'local_reused':False}
            record.update(id=cid,group=gid,admitted=candidate['admitted'],
                          artifacts={str(path):hashlib.sha256(path.read_bytes()).hexdigest()},
                          diagnostic_seconds=time.perf_counter()-start)
            record=seal_result(record,fp);append_run_row(ledger,record);results[cid]=record;count+=1
            if len(results)%10==0 or len(results)==total:
                print(f'[{len(results)}/{total}] {cid}，本轮新增局部精评{sum(not r["local_reused"] for r in results.values())}',flush=True)
    screening_path=out/'potential_screening.jsonl'
    screened=load_rows(screening_path)
    for gid,group in groups.items():
        if gid in screened and reusable(screened[gid],fp):
            continue
        rows=[r for r in results.values() if r['group']==gid]
        if not rows:
            selected=[]
        else:
            # 四份时间代理、四份总搬运优先；预算耗尽后不再追加。
            selected=[]; seen=set()
            for field in ('estimate','total_added'):
                added=0
                for r in sorted(rows,key=lambda r:(r[field],r['estimate'],r['id'])):
                    plan=json.loads(Path(r['plan_path']).read_text(encoding='utf-8'))
                    key=partition_key(plan)
                    if key in seen:continue
                    seen.add(key);selected.append(r);added+=1
                    if added==4:break
        graph=json.loads((ROOT/'通用神经网络处理器下的多核调度问题附件/data'/f'{group["case"]}.json').read_text(encoding='utf-8'))
        checked=[]; start=time.perf_counter()
        for r in selected:
            if 'official' in r:
                truth=r['official']; reused=True
            else:
                truth=fast_evaluate(graph,json.loads(Path(r['plan_path']).read_text(encoding='utf-8')),'A');reused=False
            if not valid_official(truth):raise ValueError(truth)
            if truth['partition_added']!=r['partition_added'] or truth['spill_added']!=r['spill_added']:
                raise ValueError('局部搬运与事件评估不同')
            checked.append({'candidate_id':r['id'],'estimate':r['estimate'],'result':truth,
                            'official_reused':reused,'admitted':r['admitted'],'plan_path':r['plan_path']})
        best=min(checked,key=lambda r:(r['result']['makespan'],r['result']['added_copy_bytes']),default=None)
        confirmation=None
        if best and not best['official_reused']:
            confirmation=real_evaluate(graph,json.loads(Path(best['plan_path']).read_text(encoding='utf-8')),'A')
            if confirmation!=best['result']:raise ValueError('副本与原官方不一致')
        row=seal_result({'id':gid,'checked':checked,'best':best,'extra_official':int(confirmation is not None),
            'official_confirmation':confirmation,'baseline':group['baseline'],
            'screen_seconds':time.perf_counter()-start,'artifacts':{},'scope':'诊断预算，不能并入原两次官方成绩'},fp)
        append_run_row(screening_path,row)
        print(f'{gid}：筛选{len(checked)}份，最佳{best["result"]["makespan"] if best else None}，原方案{group["baseline"]["makespan"]}',flush=True)


if __name__=='__main__':
    main()
