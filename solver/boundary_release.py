"""A输出屏障定向细化：同核祖先闭包前缀，原官方决定最终收益。"""
from collections import defaultdict
import copy

from dag_priority import build_compute_dag
from official_protocol import object_digest
from scenario_contract import validate_plan
from task_order_search import replay_tasks
from task_resource_profiles import task_resource_profiles


def _split(parent, actions):
    plan=copy.deepcopy(parent)
    next_group=max(plan['node_to_subgraph'].values())+1
    replacements={}
    for action in actions:
        group=action['group'];prefix=set(action['prefix'])
        if group in replacements:raise ValueError('同一子图不能同时应用两个切口')
        replacements[group]=next_group
        for op,g in list(plan['node_to_subgraph'].items()):
            if g==group and int(op) not in prefix:plan['node_to_subgraph'][op]=next_group
        next_group+=1
    plan['core_schedules']=[[h for g in row for h in ([g,replacements[g]] if g in replacements else [g])]
                            for row in parent['core_schedules']]
    return plan


def _boundary_bytes(graph,plan,bandwidth):
    return sum(p['input_bytes']+p['output_bytes'] for p in task_resource_profiles(graph,plan['node_to_subgraph'],bandwidth).values())


def generate_boundary_release_candidates(graph,parent,raw,*,max_candidates=6,max_hotspots=16,max_group_ops=480):
    if type(max_candidates) is not int or not 1<=max_candidates<=6:raise ValueError('候选预算必须1～6')
    if type(max_hotspots) is not int or max_hotspots<1:raise ValueError('热点预算必须为正整数')
    validate_plan(graph,parent,'A')
    mapping={int(i):g for i,g in parent['node_to_subgraph'].items()}
    _,preds,_,_=build_compute_dag(graph)
    members=defaultdict(set)
    for i,g in mapping.items():members[g].add(i)
    tasks={};by_group={};observed={}
    for core in raw['per_core_timeline']:
        for task in core['tasks']:
            if task['task_id'] in tasks or task['subgraph_id'] in by_group:raise ValueError('重复父任务')
            tasks[task['task_id']]=dict(task,core=core['core_id']);by_group[task['subgraph_id']]=task['task_id']
        for op in core['ops']:
            if op['op_id'] in mapping:
                if op['op_id'] in observed:raise ValueError('原算子观测重复')
                observed[op['op_id']]=dict(op,core=core['core_id'])
    if set(by_group)!=set(members) or set(observed)!=set(mapping):raise ValueError('父方案与轨迹覆盖不符')
    orders=[[by_group[g] for g in row] for row in parent['core_schedules']]
    for c,order in enumerate(orders):
        if any(tasks[t]['core']!=c for t in order):raise ValueError('父分核与轨迹不符')
    for i,o in observed.items():
        task=tasks[by_group[mapping[i]]]
        if o['core']!=task['core'] or o['task_id']!=task['task_id'] or not task['start']<=o['start']<=o['end']<=task['end']:
            raise ValueError('算子观测与子图不符')
    task_preds={t:set() for t in tasks}
    for edge in raw['task_dependencies']:task_preds[edge['target']].add(edge['source'])
    replay=replay_tasks({t:r['duration'] for t,r in tasks.items()},task_preds,orders,
                        raw['task_same_core_wait_cycles'],raw['task_cross_core_wait_cycles'])
    if replay is None or replay['makespan']!=raw['makespan'] or any(replay['finish'][t]!=r['end'] or r['end']-r['duration']!=r['start'] for t,r in tasks.items()):
        raise ValueError('父轨迹重建失败')
    pair_seeds=defaultdict(set)
    for j,ps in preds.items():
        for p in ps:
            if mapping[p]!=mapping[j]:pair_seeds[mapping[p],mapping[j]].add(p)
    actions={}
    for (g,h),seeds in sorted(pair_seeds.items()):
        a,b=tasks[by_group[g]],tasks[by_group[h]]
        if a['core']==b['core'] or not 2<=len(members[g])<=max_group_ops:continue
        if b['start']!=a['end']+raw['task_cross_core_wait_cycles']:continue
        prefix=set(seeds);pending=sorted(seeds)
        while pending:
            i=pending.pop()
            for p in preds[i]:
                if mapping[p]==g and p not in prefix:prefix.add(p);pending.append(p)
        if len(prefix)==len(members[g]):continue
        finish=max(observed[i]['end'] for i in prefix)
        potential=a['end']-finish-raw['task_same_core_wait_cycles']
        if potential<=0:continue
        ident=(g,tuple(sorted(prefix)))
        if ident not in actions:
            actions[ident]=dict(group=g,prefix=sorted(prefix),consumers=[],core=a['core'],
                original_group_size=len(members[g]),observed_prefix_end=finish,observed_group_end=a['end'],
                release_potential_cycles=potential)
        actions[ident]['consumers'].append(h)
    audit=dict(hotspots=len(actions),costed_hotspots=0,candidates=0,invalid_candidates=[],
               scope='观测释放空间及分区字节筛选，不含新溢出/竞争/核序，非总时间预测')
    if not actions:return [],audit
    ordered=sorted(actions.values(),key=lambda a:(-a['release_potential_cycles'],a['group'],a['prefix']))[:max_hotspots]
    bandwidth=raw['bandwidth_bytes_per_cycle'];old_bytes=_boundary_bytes(graph,parent,bandwidth)
    for action in ordered:
        child=_split(parent,[action]);validate_plan(graph,child,'A')
        action['boundary_bytes_delta']=_boundary_bytes(graph,child,bandwidth)-old_bytes
        action['screen_score']=action['release_potential_cycles']-action['boundary_bytes_delta']/bandwidth
    audit['costed_hotspots']=len(ordered)
    ranked=sorted(ordered,key=lambda a:(-a['screen_score'],-a['release_potential_cycles'],a['group'],a['prefix']))
    distinct=[];groups=set()
    for action in ranked:
        if action['group'] not in groups:distinct.append(action);groups.add(action['group'])
    menus=[('score_single_1',[ranked[0]])]
    if len(ranked)>1:menus.append(('score_single_2',[ranked[1]]))
    menus.append(('potential_single',[ordered[0]]))
    menus.extend((f'joint_{count}',distinct[:count]) for count in (2,4,8) if len(distinct)>=2)
    seen={object_digest(parent)};rows=[]
    for menu,selected in menus:
        child=_split(parent,selected);signature=object_digest(child)
        if signature in seen:continue
        seen.add(signature)
        try:validate_plan(graph,child,'A')
        except (ValueError,KeyError,TypeError) as exc:
            audit['invalid_candidates'].append(dict(menu=menu,error=str(exc)));continue
        rows.append(dict(plan=child,source=dict(module='boundary_release',menu=menu,cuts=selected,
            boundary_bytes_delta=_boundary_bytes(graph,child,bandwidth)-old_bytes,
            added_groups=len(selected),scope=audit['scope'])))
        if len(rows)>=max_candidates:break
    audit['candidates']=len(rows)
    return rows,audit
