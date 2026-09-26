"""A输出边界拆分后的尾部迁核/插入候选；固定拆分对照和动态带宽评分。"""
from collections import deque
from copy import deepcopy
import bisect
import time

from a_event_model import simulate_a_plan,CachedAExpansion
from boundary_release import generate_boundary_release_candidates
from official_protocol import object_digest
from scenario_contract import validate_plan


def generate_boundary_tail_candidates(graph,parent,raw,*,max_expanded=24,max_candidates=6,cache_mib=128):
    if type(max_expanded) is not int or not 6<=max_expanded<=48:raise ValueError('动态评分预算必须6～48')
    if type(max_candidates) is not int or not 3<=max_candidates<=6:raise ValueError('需保留控制组，官方预算必须3～6')
    if type(cache_mib) is not int or not 0<=cache_mib<=128:raise ValueError('缓存预算必须0～128MiB')
    started=time.perf_counter();expander=CachedAExpansion(cache_mib*1024*1024)
    hardware={k:raw[k] for k in ('bandwidth_bytes_per_cycle','capacity_bytes','task_cross_core_wait_cycles','task_same_core_wait_cycles')}
    baseline=simulate_a_plan(graph,parent,hardware,expander=expander)
    observed={(o['task_id'],o['op_id']):(o['start'],o['end']) for c in raw['per_core_timeline'] for o in c['ops']}
    if baseline['makespan']!=raw['makespan'] or {k:(o['start'],o['end']) for k,o in baseline['operations'].items()}!=observed:
        raise ValueError('A动态模型父方案全操作核验失败')
    seeds,split_audit=generate_boundary_release_candidates(graph,parent,raw,max_candidates=3)
    seeds=[row for row in seeds if len(row['source']['cuts'])==1][:2]
    prior_groups=set(parent['node_to_subgraph'].values());seen={object_digest(parent)}
    rows=[];states=[];invalid=[];scored=0
    def score(plan,source):
        nonlocal scored
        key=object_digest(plan)
        if key in seen:return None
        seen.add(key)
        try:validate_plan(graph,plan,'A')
        except (ValueError,RuntimeError) as exc:
            invalid.append(dict(source=source,error=str(exc)));return None
        result=simulate_a_plan(graph,plan,hardware,expander=expander);scored+=1
        rows.append(dict(plan=plan,source=dict(source,
            closed_loop_makespan=result['makespan'],closed_loop_movement=result['data_movement_bytes'],
            event_seconds=result['event_seconds'],
            scope='固定拆分的尾部布局/顺序探针，动态模型非原官方；不代表临时核心组完整消融')))
        return result
    for seed in seeds:
        plan=seed['plan'];cut=seed['source']['cuts'][0]
        new_groups=set(plan['node_to_subgraph'].values())-prior_groups
        if len(new_groups)!=1:raise ValueError('单切必须恰好新增一个尾部组')
        tail=next(iter(new_groups));source=dict(module='boundary_tail_refine',mode='split_only',
            split_plan_id=object_digest(plan),prefix_group=cut['group'],tail_group=tail,
            original_core=cut['core'],target_core=cut['core'],cuts=[cut])
        result=score(plan,source)
        if result is not None:states.append((plan,source,result['task_times']))
    streams=deque()
    # 核/切口轮转；固定队外顺序，只移一个真实尾部任务。锚点取拆分后的动态释放时刻。
    for core in range(len(parent['core_schedules'])):
        for plan,source,times in states:
            tail=source['tail_group'];order=[g for g in plan['core_schedules'][core] if g!=tail]
            release=times[source['prefix_group']][1]
            starts=[times[g][0] for g in order]
            anchor=bisect.bisect_left(starts,release)
            positions=sorted({0,len(order),*(min(len(order),max(0,anchor+d)) for d in (-2,-1,0,1,2))},key=lambda i:(abs(i-anchor),i))
            streams.append((plan,source,core,iter(positions)))
    while streams and scored<max_expanded:
        split,source,core,positions=streams.popleft()
        position=next(positions,None)
        if position is None:continue
        streams.append((split,source,core,positions))
        plan=deepcopy(split);tail=source['tail_group']
        plan['core_schedules']=[[g for g in row if g!=tail] for row in plan['core_schedules']]
        plan['core_schedules'][core].insert(position,tail)
        if [[g for g in row if g!=tail] for row in plan['core_schedules']]!=parent['core_schedules']:
            raise ValueError('尾部动作改变组外顺序')
        if plan['node_to_subgraph']!=split['node_to_subgraph']:raise ValueError('固定切分动作修改了分区')
        score(plan,dict(source,mode='same_core_order' if core==source['original_core'] else 'tail_migration',
            target_core=core,insert_position=position))
    ranked=sorted(rows,key=lambda r:(r['source']['closed_loop_makespan'],object_digest(r['plan'])))
    selected=[];selected_ids=set()
    def add(row):
        key=object_digest(row['plan'])
        if key not in selected_ids and len(selected)<max_candidates:selected.append(row);selected_ids.add(key)
    if ranked:add(ranked[0])
    # 留结构控制和两种布局家族的原官方复核机会，不把所有退化动作藏在预筛后面。
    for mode in ('split_only','same_core_order','tail_migration'):
        row=next((r for r in ranked if r['source']['mode']==mode),None)
        if row is not None:add(row)
    for row in ranked:add(row)
    return selected,dict(parent_verified=True,split_generation=split_audit,split_controls=len(states),
        scored=scored,max_expanded=max_expanded,invalid=invalid,candidates=len(selected),
        predicted_improving=sum(r['source']['closed_loop_makespan']<raw['makespan'] for r in rows),
        scored_proposals=[dict(plan_id=object_digest(r['plan']),**r['source']) for r in rows],
        cache_stats=dict(expander.memo.stats),resident_serialized_bytes=expander.memo.resident_bytes,
        elapsed_seconds=time.perf_counter()-started,
        scope='拆分固定后比较仅顺序/迁核；不是组宽1～5与切分的完整联合消融')
