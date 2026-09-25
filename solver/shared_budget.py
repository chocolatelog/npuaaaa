"""三家族共同六次评分；只依近似分数与结构身份分配名额。"""
import math
import time
from copy import deepcopy
from menu_order_polish import partition_key

FAMILIES=('schedule','partition','region')

def unify_scores(graph,pools,progress=None):
    """成员编号相同才共享局部统计；每份全图队列必须独立重放。"""
    from scene_a_event import SceneAEventModel,derive_multicore_plan
    from task_order_search import replay_tasks
    from evaluation_validation import validate_task_order
    from evidence_index import plan_sha
    local=SceneAEventModel(graph);cache={};output={};local_seconds=0.;replay_seconds=0.;count=0
    total=sum(map(len,pools.values()));started=time.perf_counter()
    for family,rows in pools.items():
        output[family]=[]
        for row in rows:
            plan=row['plan']
            if plan_sha(plan)!=row['plan_id']:raise ValueError('重排输入的完整计划摘要错误')
            # 不以不含算子身份的数组独自为键；完整算子映射决定局部图。
            key=tuple(sorted(plan['node_to_subgraph'].items()))
            if key not in cache:
                before=time.perf_counter();estimate=local.evaluate(plan)
                view=derive_multicore_plan(graph,plan);validate_task_order(view)
                cache[key]=({'durations':{t:r['duration'] for t,r in estimate['tasks'].items()},
                    'preds':view['subgraph_preds'],'bandwidth_floor':estimate['total_copy_bytes']/60,
                    'partition_added':estimate['partition_added_bytes'],'spill_bytes':estimate['spill_bytes']})
                local_seconds+=time.perf_counter()-before
            info=cache[key];before=time.perf_counter()
            score=replay_tasks(info['durations'],info['preds'],plan['core_schedules'],same_wait=100,cross_wait=1000,
                               bandwidth_floor=info['bandwidth_floor'])
            replay_seconds+=time.perf_counter()-before
            if score is None:raise ValueError('统一近似重放发现非法完整核序')
            copy=deepcopy(row);copy['previous_coarse']=row['coarse'];copy['coarse']=score['makespan']
            copy['unified_local']={k:v for k,v in info.items() if k not in ('durations','preds')}
            output[family].append(copy);count+=1
            if progress and (count%100==0 or count==total):progress(count,total,len(cache))
    return output,{'seconds':time.perf_counter()-started,'local_seconds':local_seconds,'local_models':len(cache),
        'replay_seconds':replay_seconds,'queue_replays':count,'scope':'统一局部时长近似及带宽下界，非完整真实带宽竞争'}

def ranking_key(row,tie_break='boundary'):
    if tie_break=='boundary':secondary=row['features']['boundary_bytes']
    elif tie_break=='total_copy':
        if 'unified_local' not in row:raise ValueError('总复制排序需要已验证的局部溢出统计')
        local=row['unified_local'];spill=local.get('spill_bytes');floor=local.get('bandwidth_floor')
        if spill is None or floor is None or not math.isfinite(spill) or spill<0:raise ValueError('缺少有效溢出/带宽统计')
        secondary=row['features']['boundary_bytes']+spill
        if not math.isclose(secondary,floor*60,rel_tol=1e-12,abs_tol=1e-6):raise ValueError('总复制量与局部字节计数不一致')
    else:raise ValueError('未知第二排序键')
    return row['coarse'],secondary,row['plan_id']

def select_shared(pools,tie_break='boundary'):
    if set(pools)!=set(FAMILIES):raise ValueError('候选家族必须完整且固定')
    ranked={}
    for family in FAMILIES:
        for row in pools[family]:
            if not math.isfinite(row['coarse']) or row['coarse']<=0:raise ValueError('近似耗时非法')
        ranked[family]=sorted(pools[family],key=lambda r:ranking_key(r,tie_break))
    selected=[];used=set();counts=dict.fromkeys(FAMILIES,0);parts={f:set() for f in FAMILIES}
    def take(family):
        available=[r for r in ranked[family] if r['plan_id'] not in used]
        if not available:return False
        row=next((r for r in available if partition_key(r['assignment']) not in parts[family]),available[0])
        selected.append({**row,'allocated_family':family});used.add(row['plan_id'])
        parts[family].add(partition_key(row['assignment']));counts[family]+=1
        return True
    for family in FAMILIES:
        for _ in range(2):take(family)
    while len(selected)<6:
        if not any(take(f) for f in sorted(FAMILIES,key=lambda f:(counts[f],FAMILIES.index(f)))):break
    return selected,{'counts':counts,'logical_limit':6,'selected_ids':[r['plan_id'] for r in selected],
        'pool_sizes':{f:len(pools[f]) for f in FAMILIES},
        'ranking':('近似耗时、真实边界、完整计划摘要；优先不同成员切分' if tie_break=='boundary' else
                   '近似耗时、真实边界加溢出总复制、完整计划摘要；优先不同成员切分')}
