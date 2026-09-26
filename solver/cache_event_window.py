"""C事件锚点重排：首次读入完成/实际淘汰驱动，同核依赖闭包与动态精筛。"""
from collections import defaultdict
from copy import deepcopy
import heapq
import time

from bc_event_model import simulate_bc_plan
from bc_event_screen import verify_parent_prediction
from bc_observed_graph import observe_parent
from dag_priority import build_compute_dag
from official_protocol import object_digest
from stage_cache import CachedSceneExpansion


def cache_event_targets(raw):
    """只读父轨迹定位事件目标；目标时间不是修改后保证达到的时间。"""
    timeline={(c['core_id'],o['op_id']):o for c in raw['per_core_timeline'] for o in c['ops']}
    pending=defaultdict(list);evicted={};targets=[]
    for event in raw['cache_events']:
        tid=event['tensor_id'];now=event['time']
        if event['event']=='insert':
            for old in event.get('evicted_tensor_ids',[]):evicted[old]=now
            continue
        if event['event'] not in ('hit','miss'):continue
        queue=pending[tid]
        while queue and queue[0]<=now:heapq.heappop(queue)
        op=timeline[event['core_id'],event['op_id']];size=event['size_bytes']
        if event['event']=='miss' and 0<size<=raw['cache_capacity_bytes']:
            reason='inflight_duplicate' if queue else 'evicted' if tid in evicted else None
            if reason:
                potential=size*(1/raw['bandwidth_bytes_per_cycle']-1/raw['cache_bandwidth_bytes_per_cycle'])
                if potential>0:
                    targets.append(dict(core=event['core_id'],op_id=event['op_id'],subgraph=op['subgraph_id'],
                        tensor_id=tid,size_bytes=size,time=now,target_time=queue[0] if queue else evicted[tid],
                        reason=reason,potential_saved_cycles=potential))
        heapq.heappush(queue,op['end'])
    return targets


def move_with_local_closure(parent,core,at,to,preds,succ,max_groups=16):
    """提前带上窗口内祖先，延后带上窗口内后继，保持闭包内部原顺序。"""
    row=parent['core_schedules'][core];positions={g:i for i,g in enumerate(row)}
    if at==to or not 0<=at<len(row) or not 0<=to<len(row):return None,[]
    moved={row[at]};pending=list(moved);relations=preds if to<at else succ
    while pending:
        g=pending.pop()
        for neighbor in relations.get(g,()):
            pos=positions.get(neighbor)
            if pos is None or neighbor in moved:continue
            if (to<at and pos>at) or (to>at and pos<at):return None,[]
            if min(at,to)<=pos<=max(at,to):
                moved.add(neighbor);pending.append(neighbor)
                if len(moved)>max_groups:return None,[]
    moving=[g for g in row if g in moved]
    split=to if to<at else to+1
    order=[g for g in row[:split] if g not in moved]+moving+[g for g in row[split:] if g not in moved]
    if order==row:return None,moving
    plan=deepcopy(parent);plan['core_schedules'][core]=order
    return plan,moving


def generate_cache_event_candidates(graph,parent,raw,*,max_expanded=24,max_candidates=6,cache_mib=128,max_span=64):
    if type(max_expanded) is not int or not 6<=max_expanded<=48:raise ValueError('展开预算必须6～48')
    if type(max_candidates) is not int or not 1<=max_candidates<=6:raise ValueError('官方候选预算必须1～6')
    if type(cache_mib) is not int or not 0<=cache_mib<=128:raise ValueError('阶段缓存预算必须0～128MiB')
    if type(max_span) is not int or not 1<=max_span<=64:raise ValueError('移动跨度必须1～64')
    started=time.perf_counter();expander=CachedSceneExpansion(cache_mib*1024*1024)
    hardware={k:raw[k] for k in ('bandwidth_bytes_per_cycle','capacity_bytes','cross_core_copy_delay_cycles','cache_capacity_bytes','cache_bandwidth_bytes_per_cycle')}
    baseline=simulate_bc_plan(graph,parent,hardware,'C',expander=expander)
    verify_parent_prediction(baseline,raw,'C')
    state=observe_parent(graph,parent,raw,expander=expander)
    positions={g:(c,i) for c,row in enumerate(parent['core_schedules']) for i,g in enumerate(row)}
    mapping={int(k):v for k,v in parent['node_to_subgraph'].items()}
    _,op_preds,_,_=build_compute_dag(graph)
    preds=defaultdict(set);succ=defaultdict(set)
    for i,ps in op_preds.items():
        for p in ps:
            u,v=mapping[p],mapping[i]
            if u!=v:preds[v].add(u);succ[u].add(v)
    read_groups=defaultdict(dict)
    for core in raw['per_core_timeline']:
        for op in core['ops']:
            if op['op']=='COPY_IN' and op['subgraph_id'] in positions:
                g=op['subgraph_id'];read_groups[core['core_id']][g]=min(op['start'],read_groups[core['core_id']].get(g,op['start']))
    targets=cache_event_targets(raw);actions={}
    for target in targets:
        core=target['core'];g=target['subgraph']
        if g not in positions or positions[g][0]!=core:continue
        at=positions[g][1];delay=target['reason']=='inflight_duplicate'
        available=[(positions[h][1],start) for h,start in read_groups[core].items()
            if 0<(positions[h][1]-at if delay else at-positions[h][1])<=max_span]
        if not available:continue
        # 精确事件邻域与1/4/8个真实读入边界并存；不以总体核序距离冒充读时序。
        by_event=sorted(available,key=lambda a:(abs(a[1]-target['target_time']),abs(a[0]-at)))[:3]
        by_position=sorted(available,key=lambda a:abs(a[0]-at))
        choices=by_event+[by_position[d-1] for d in (1,4,8) if len(by_position)>=d]
        slack=state['replay']['slack'][core,target['op_id']]
        priority=max(0,target['potential_saved_cycles']-slack)
        for to,anchor_start in choices:
            action=dict(target,at=at,to=to,anchor_start=anchor_start,span=abs(to-at),
                observed_slack=slack,critical_potential=priority)
            key=(core,g,to)
            if key not in actions or (priority,target['potential_saved_cycles'])>(actions[key]['critical_potential'],actions[key]['potential_saved_cycles']):actions[key]=action
    ranked=sorted(actions.values(),key=lambda a:(-a['critical_potential'],-a['potential_saved_cycles'],
        abs(a['anchor_start']-a['target_time']),a['span'],a['core'],a['subgraph'],a['to']))
    families={r:[a for a in ranked if a['reason']==r] for r in ('inflight_duplicate','evicted')}
    ordered=[]
    for i in range(max((len(v) for v in families.values()),default=0)):
        for values in families.values():
            if i<len(values):ordered.append(values[i])
    seen={object_digest(parent)};proposals=[];rejected=[];attempted=0;closure_rejected=0
    for action in ordered:
        if attempted>=max_expanded:break
        plan,moved=move_with_local_closure(parent,action['core'],action['at'],action['to'],preds,succ)
        if plan is None:closure_rejected+=1;continue
        plan_id=object_digest(plan)
        if plan_id in seen:continue
        seen.add(plan_id);attempted+=1
        try:tasks,links=expander(graph,plan,hardware)
        except (ValueError,RuntimeError) as exc:rejected.append(dict(action=action,error=str(exc)));continue
        prediction=simulate_bc_plan(graph,plan,hardware,'C',expander=lambda *_:(tasks,links))
        source=dict(module='cache_event_window',**action,moved_subgraphs=moved,
            closed_loop_makespan=prediction['makespan'],closed_loop_cache_stats=prediction['cache_stats'],
            closed_loop_event_seconds=prediction['event_seconds'],closed_loop_events=prediction['events'],
            scope='事件目标仅取父观测；新方案时序/命中自主预测，原官方最终裁决')
        proposals.append(dict(plan=plan,source=source))
    proposals.sort(key=lambda r:(r['source']['closed_loop_makespan'],r['source']['span'],object_digest(r['plan'])))
    chosen=[r for r in proposals if r['source']['closed_loop_makespan']<raw['makespan']][:max_candidates]
    return chosen,dict(status='ok',parent_verified=True,hotspots=len(targets),actions=len(actions),
        expanded=attempted,legal_expanded=len(proposals),closure_rejected=closure_rejected,rejected=rejected,
        candidates=len(chosen),max_expanded=max_expanded,max_span=max_span,
        scored_proposals=[dict(plan_id=object_digest(r['plan']),**r['source']) for r in proposals],
        cache_stats=dict(expander.memo.stats),resident_serialized_bytes=expander.memo.resident_bytes,
        elapsed_seconds=time.perf_counter()-started)
