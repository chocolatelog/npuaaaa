"""四来源分区候选在固定十二精评预算内融合；默认生产流程尚未启用。"""
from collections import Counter

from branch_partition import generate_branch_candidates
from hotspot_partition import generate_hotspot_candidates
from partition_polish import split_solution, merge_solution, boundary_savings
from scene_a_event import derive_multicore_plan

ORIGINS=('split','branch','merge','hotspot')


def member_key(sol):
    return tuple(sorted(tuple(sorted(group)) for group in sol.blocks_in_sg if group))


def generate_fusion_candidates(model,parent,plan,num_cores,evaluator,estimate):
    heavy=sorted(estimate['tasks'],key=lambda s:(-estimate['tasks'][s]['duration'],s))[:3]
    branches,branch_audit=generate_branch_candidates(model,parent,heavy,num_cores)
    hot,hot_audit=generate_hotspot_candidates(model,parent,plan,num_cores,evaluator,estimate)
    rows=[]
    for origin,generated in (('branch',branches),('hotspot',hot)):
        for r in generated:
            r['source']['origin']=origin;rows.append(r)
    view=derive_multicore_plan(evaluator.graph,plan)
    merge_rank=[]
    for a,b in view['dependency_pairs']:
        saved=boundary_savings(evaluator,view['mapping'],a,b)
        if saved>0:merge_rank.append((saved,a,b))
    invalid=0
    for saved,a,b in sorted(merge_rank,key=lambda r:(-r[0],r[1],r[2]))[:6]:
        child=merge_solution(model,parent,a,b,num_cores)
        if child is None:invalid+=1;continue
        rows.append({'sol':child,'source':{'origin':'merge','kind':'merge','task':a,
                       'tasks':[a,b],'boundary_saved_bytes':saved}})
    loads=[sum(estimate['tasks'][s]['duration'] for s in order) for order in plan['core_schedules']]
    split_tasks=sorted((s for s in estimate['tasks'] if len(parent.blocks_in_sg[s])>=2),
                        key=lambda s:(-estimate['tasks'][s]['duration'],s))[:3]
    for fraction in (1/3,1/2,2/3):
        for task in split_tasks:
            removed=estimate['tasks'][task]['duration']*(1-fraction)
            target=min(range(num_cores),key=lambda c:(loads[c]-(removed if c==parent.core_of_sg[task] else 0),c))
            child=split_solution(model,parent,task,fraction,target,num_cores)
            if child is None:invalid+=1;continue
            rows.append({'sol':child,'source':{'origin':'split','kind':'split','task':task,'fraction':fraction}})
    return rows,{'legal':len(rows),'generated':len(rows)+invalid,'legacy_invalid':invalid,
                 'branch':branch_audit,'hotspot':hot_audit,
                 'target_tasks':heavy,'counts':dict(Counter(r['source']['origin'] for r in rows))}


def select_fusion_candidates(rows,limit=12):
    if limit!=12:
        raise ValueError('当前冻结融合配置固定十二候选')
    def stable(r):
        return (r['features']['coarse_score'],tuple(r['sol'].sg_of_block),tuple(r['sol'].core_of_sg))
    representative={}
    for r in sorted(rows,key=stable):
        representative.setdefault(member_key(r['sol']),r)
    candidates=list(representative.values())
    # 使用负载与真实边界字节的非支配层；不把理想省掉的溢出混入代价。
    dominates=[[] for _ in candidates]; degree=[0]*len(candidates)
    for i,a in enumerate(candidates):
        av=tuple(a['features'][k] for k in ('compute_load','boundary_bytes'))
        for j,b in enumerate(candidates):
            if i==j:continue
            bv=tuple(b['features'][k] for k in ('compute_load','boundary_bytes'))
            if all(x<=y for x,y in zip(av,bv)) and any(x<y for x,y in zip(av,bv)):
                dominates[i].append(j);degree[j]+=1
    frontier=[i for i,d in enumerate(degree) if d==0];rank=0
    while frontier:
        next_front=[]
        for i in frontier:
            candidates[i]['fusion_front']=rank
            for j in dominates[i]:
                degree[j]-=1
                if degree[j]==0:next_front.append(j)
        frontier=next_front;rank+=1
    selected=[]; chosen=set();counts=Counter();tasks=Counter();kinds=Counter()
    def pick(allowed,reason):
        remaining=[r for r in candidates if id(r) not in chosen and r['source']['origin'] in allowed]
        if not remaining:return False
        def priority(r):
            source=r['source'];origin=source['origin']
            if origin=='hotspot':
                subtype='aggregate' if 'aggregate' in source['kind'] else 'single'
                # 两种动作各留机会，第一次优先聚合，避免全被细动作挤占。
                novelty=(kinds[origin,subtype],int(subtype!='aggregate'))
            elif origin=='split':
                novelty=(kinds[origin,source.get('fraction')],0)
            else:novelty=(0,0)
            return (counts[origin],*novelty,tasks[origin,source.get('task')],r['fusion_front'],stable(r))
        r=min(remaining,key=priority);source=r['source'];origin=source['origin']
        selected.append(r);chosen.add(id(r));counts[origin]+=1;tasks[origin,source.get('task')]+=1
        subtype=('aggregate' if 'aggregate' in source['kind'] else 'single') if origin=='hotspot' else source.get('fraction')
        kinds[origin,subtype]+=1;r['fusion_selection_reason']=reason
        return True
    for origin in ORIGINS:
        for _ in range(2):pick({origin},'来源保留')
    for family in ({'split','branch'},{'merge','hotspot'}):
        while sum(counts[o] for o in family)<4 and len(selected)<limit:
            if not pick(family,'同类空额补足'):break
    while len(selected)<limit:
        remaining=[r for r in candidates if id(r) not in chosen]
        if not remaining:break
        r=min(remaining,key=lambda r:(r['fusion_front'],counts[r['source']['origin']],
                                     tasks[r['source']['origin'],r['source'].get('task')],stable(r)))
        selected.append(r);chosen.add(id(r));counts[r['source']['origin']]+=1
        tasks[r['source']['origin'],r['source'].get('task')]+=1
        r['fusion_selection_reason']='非支配层与来源补足'
    return selected,{'selected':len(selected),'member_duplicates':len(rows)-len(candidates),
                     'available_by_origin':{o:sum(r['source']['origin']==o for r in candidates) for o in ORIGINS},
                     'selected_by_origin':dict(counts),'policy':'2×四来源，类别补至4+4，余4非支配/多样性；空额释放'}
