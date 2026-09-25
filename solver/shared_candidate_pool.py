"""E09候选生成：不接收候选真值；局部近似建模与历史冻结提案明确分开。"""
import itertools
import json
import time
from pathlib import Path
from branch_partition import generate_branch_candidates
from partition_polish import split_solution,merge_solution,boundary_savings
from scene_a_event import SceneAEventModel,derive_multicore_plan
from evaluation_validation import validate_task_order
from task_order_search import search_orders
from solution import Sol
from batch_features import BatchFeatures,scalar_features
from evidence_index import plan_sha

def make_candidate(model,plan,coarse,origin):
    assignment=[plan['node_to_subgraph'][str(b[0])] for b in model.blocks]
    if model.plan_from(assignment,plan['core_schedules'])!=plan:raise ValueError('候选破坏完整块')
    owners=[0]*(max(assignment)+1)
    for core,queue in enumerate(plan['core_schedules']):
        for task in queue:owners[task]=core
    return {'plan':plan,'plan_id':plan_sha(plan),'assignment':assignment,'owners':owners,
            'coarse':coarse,'origin':origin}

def generate_pools(graph,model,baseline_plan,menu,order_generation,torch,progress=None,local=None):
    started=time.perf_counter();n=len(baseline_plan['core_schedules']);base_id=plan_sha(baseline_plan)
    base=make_candidate(model,baseline_plan,1,'base');parent=Sol(base['assignment'],base['owners'])
    local=local or SceneAEventModel(graph);before=time.perf_counter();estimate=local.evaluate(baseline_plan)
    local_seconds=time.perf_counter()-before;search_seconds=0.;search_evals=0
    view=derive_multicore_plan(graph,baseline_plan)
    schedule,stats=search_orders({t:r['duration'] for t,r in estimate['tasks'].items()},view['subgraph_preds'],
        baseline_plan['core_schedules'],bandwidth_floor=estimate['total_copy_bytes']/60,
        max_evals=4000,rounds=3,beam_width=4,max_sources=12,keep=8,enable_swaps=True,seconds=None)
    search_seconds+=stats['seconds'];search_evals+=stats['evaluations']
    pools={'schedule':[make_candidate(model,{**baseline_plan,'core_schedules':r['orders']},r['makespan'],
                                     '冻结E03切分迁移或交换') for r in schedule], 'partition':[], 'region':[]}
    targets=sorted(estimate['tasks'],key=lambda t:(-estimate['tasks'][t]['duration'],t))[:3]
    generated,branch_audit=generate_branch_candidates(model,parent,targets,n)
    # 保留旧分支生成的任务代表和粗筛12份；此处尚无新方案全图真值。
    for row in generated:row['features']=scalar_features(model,row['sol'].sg_of_block,row['sol'].core_of_sg,n)
    ranked=sorted(generated,key=lambda r:(r['features']['coarse_score'],tuple(r['sol'].sg_of_block),tuple(r['sol'].core_of_sg)))
    branch=[]
    for task in targets:
        representative=next((r for r in ranked if r['source']['task']==task),None)
        if representative is not None:branch.append(representative)
    chosen={id(r) for r in branch}
    branch += [r for r in ranked if id(r) not in chosen][:max(0,12-len(branch))]
    variants=[({'kind':'independent_branch','source':r['source']},r['sol']) for r in branch]
    merge_rows=[]
    for a,b in view['dependency_pairs']:
        saved=boundary_savings(local,view['mapping'],a,b)
        if saved>0:merge_rows.append((saved,a,b))
    merges=[({'kind':'legacy_merge','tasks':[a,b]},merge_solution(model,parent,a,b,n))
            for saved,a,b in sorted(merge_rows,key=lambda r:(-r[0],r[1],r[2]))[:6]]
    loads=[sum(estimate['tasks'][s]['duration'] for s in q) for q in baseline_plan['core_schedules']]
    heavy=sorted((s for s in estimate['tasks'] if len(parent.blocks_in_sg[s])>=2),
                 key=lambda s:(-estimate['tasks'][s]['duration'],s))[:3]
    splits=[]
    for fraction in (1/3,1/2,2/3):
        for task in heavy:
            removed=estimate['tasks'][task]['duration']*(1-fraction)
            target=min(range(n),key=lambda c:(loads[c]-(removed if c==base['owners'][task] else 0),c))
            splits.append(({'kind':'legacy_split','task':task,'fraction':fraction},split_solution(model,parent,task,fraction,target,n)))
    legacy=[];seen=set();invalid=0
    for pair in itertools.zip_longest(merges,splits):
        for value in pair:
            if value is None:continue
            origin,sol=value
            if sol is None:invalid+=1;continue
            sig=tuple(sol.sg_of_block),tuple(sol.core_of_sg)
            if sig in seen or len(legacy)>=12:continue
            seen.add(sig);legacy.append(value)
    variants+=legacy
    for i,(origin,sol) in enumerate(variants):
        _,_,info=model.evaluate(sol.sg_of_block,sol.core_of_sg,'A',n)
        plan=model.plan_from(sol.sg_of_block,info['orders']);before=time.perf_counter();prediction=local.evaluate(plan)
        local_seconds+=time.perf_counter()-before
        candidate_view=derive_multicore_plan(graph,plan)
        proposals,stat=search_orders({t:r['duration'] for t,r in prediction['tasks'].items()},candidate_view['subgraph_preds'],
            plan['core_schedules'],bandwidth_floor=prediction['total_copy_bytes']/60,max_evals=400,
            rounds=2,seconds=None,keep=1)
        search_seconds+=stat['seconds'];search_evals+=stat['evaluations'];coarse=prediction['makespan']
        if proposals and proposals[0]['makespan']<coarse:plan['core_schedules']=proposals[0]['orders'];coarse=proposals[0]['makespan']
        pools['partition'].append(make_candidate(model,plan,coarse,origin))
        if progress:progress(i+1,len(variants))
    # 显式白名单：忽略admitted/real/resource/checks及真值保底字段。
    for row in menu['candidate_features']:
        pools['region'].append(make_candidate(model,model.plan_from(row['assignment'],row['orders']),row['coarse'],
                                             {'kind':'frozen_structure_menu','origins':row['origins']}))
    for row in order_generation['rows']:
        pools['region'].append(make_candidate(model,row['plan'],row['coarse'],
            {'kind':'frozen_menu_order','parents':[o['parent_plan_id'] for o in row['origins']]}))
    for family,rows in pools.items():
        unique={}
        for row in rows:
            if row['plan_id']==base_id:continue
            validate_task_order(derive_multicore_plan(graph,row['plan']))
            old=unique.get(row['plan_id'])
            if old is None or row['coarse']<old['coarse']:unique[row['plan_id']]=row
        pools[family]=list(unique.values())
    all_rows=[r for rows in pools.values() for r in rows];torch.cuda.reset_peak_memory_stats();before=time.perf_counter()
    batch=BatchFeatures(model,n);features=batch.evaluate([r['assignment'] for r in all_rows],[r['owners'] for r in all_rows])
    if features!=[scalar_features(model,r['assignment'],r['owners'],n) for r in all_rows]:raise ValueError('显卡特征与标量不一致')
    for row,feature in zip(all_rows,features):row['features']=feature
    gpu={'equal':True,'seconds':time.perf_counter()-before,'peak_allocated':torch.cuda.max_memory_allocated(),
         'peak_reserved':torch.cuda.max_memory_reserved(),'candidates':len(all_rows)}
    return pools,{'seconds':time.perf_counter()-started,'local_seconds':local_seconds,'search_seconds':search_seconds,
        'search_evaluations':search_evals,'branch':branch_audit,'branch_selected':len(branch),'legacy_selected':len(legacy),
        'legacy_invalid':invalid,'gpu':gpu,'scope':'调度/分支/工作量本轮重建；区域读取已冻结的真值前提案，历史生成成本另列'}
