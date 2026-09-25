"""冻结成员切分的全图迁移/交换提案；局部近似时长不能当官方结果。"""
import json,time
from pathlib import Path
from evidence_index import plan_sha
from scene_a_event import SceneAEventModel,derive_multicore_plan
from evaluation_validation import validate_task_order
from task_order_search import search_orders


def partition_key(assignment):
    groups={}
    for b,t in enumerate(assignment):groups.setdefault(t,[]).append(b)
    return tuple(sorted(map(tuple,groups.values())))


def generate_order_variants(graph,model,parents,progress=None):
    if not 1<=len(parents)<=12:raise ValueError('冻结输入候选数必须为1至12')
    local=SceneAEventModel(graph);cache={};rows={};audits=[];local_seconds=0.
    parent_plans=[json.loads(Path(p['plan_path']).read_text(encoding='utf-8')) for p in parents]
    parent_ids={plan_sha(plan) for plan in parent_plans}
    for i,(parent,plan) in enumerate(zip(parents,parent_plans)):
        view=derive_multicore_plan(graph,plan);validate_task_order(view)
        assignment=[plan['node_to_subgraph'][str(block[0])] for block in model.blocks]
        if model.plan_from(assignment,plan['core_schedules'])!=plan:raise ValueError('冻结成员切分无法按完整块表示')
        key=tuple(assignment);cached=key in cache
        if not cached:
            before=time.perf_counter();cache[key]=local.evaluate(plan);local_seconds+=time.perf_counter()-before
        estimate=cache[key]
        # 只缓存相同成员编号的局部任务统计；不复用依赖原核序的整图makespan。
        movement={'partition_added':max(0,estimate['partition_added_bytes']),
                  'spill_added':estimate['spill_bytes'],'scheduled_copy_bytes':estimate['total_copy_bytes']}
        if any(value!=parent['resource'][name] for name,value in movement.items()):raise ValueError('固定切分局部搬运统计与已有资源证据不一致')
        proposals,stats=search_orders({t:r['duration'] for t,r in estimate['tasks'].items()},view['subgraph_preds'],
            plan['core_schedules'],same_wait=100,cross_wait=1000,bandwidth_floor=estimate['total_copy_bytes']/60,
            beam_width=4,rounds=2,max_evals=400,seconds=None,max_sources=12,keep=2,enable_swaps=True)
        audits.append({'parent_plan_id':plan_sha(plan),'local_cache_hit':cached,'search':stats,
                       'baseline_resource':parent['resource'],'time_model':'局部管道/带宽近似，非背景竞争真实时长'})
        for proposal in proposals:
            candidate={**plan,'core_schedules':proposal['orders']};pid=plan_sha(candidate)
            if pid in parent_ids:continue
            validate_task_order(derive_multicore_plan(graph,candidate))
            origin={'parent_plan_id':plan_sha(plan),'parent_plan_path':parent['plan_path'],
                    'neighborhood':proposal['origin'],'baseline_resource':parent['resource']}
            if pid in rows:rows[pid]['origins'].append(origin);continue
            owners=[0]*(max(assignment)+1)
            for c,q in enumerate(candidate['core_schedules']):
                for task in q:owners[task]=c
            rows[pid]={'plan':candidate,'plan_id':pid,'assignment':assignment,'owners':owners,
                      'coarse':proposal['makespan'],'origins':[origin]}
        if progress:progress(i+1,len(parents))
    return list(rows.values()),{'parents':len(parents),'local_models':len(cache),'local_seconds':local_seconds,
        'search_evaluations':sum(a['search']['evaluations'] for a in audits),
        'search_seconds':sum(a['search']['seconds'] for a in audits),'proposals':len(rows),'by_parent':audits}


def select_order_variants(rows,limit=12):
    if limit<1:raise ValueError('资源候选预算必须为正')
    ranked=sorted(rows,key=lambda r:(r['coarse'],r['plan_id']));selected=[];used=set();partitions=set()
    for row in ranked:
        part=partition_key(row['assignment'])
        if part not in partitions and len(selected)<limit:
            selected.append(row);used.add(row['plan_id']);partitions.add(part)
    for kind in ('migration','swap'):
        if any(any(o['neighborhood']==kind for o in r['origins']) for r in selected):continue
        options=[r for r in ranked if r['plan_id'] not in used and any(o['neighborhood']==kind for o in r['origins'])]
        if options and len(selected)<limit:selected.append(options[0]);used.add(options[0]['plan_id'])
    for row in ranked:
        if row['plan_id'] not in used and len(selected)<limit:selected.append(row);used.add(row['plan_id'])
    return selected
