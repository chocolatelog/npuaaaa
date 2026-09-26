"""C固定分组/分核的观测驱动局部窗口；只生成候选，官方总时间裁决。"""
from collections import Counter,defaultdict
from copy import deepcopy
import heapq

from official_protocol import object_digest
from scenario_contract import validate_plan


def cache_window_hotspots(raw):
    """使用官方实际完成时间分辨可复用未命中；不预测修改后的命中。"""
    timeline={(core['core_id'],op['op_id']):op for core in raw['per_core_timeline'] for op in core['ops']}
    capacity=raw['cache_capacity_bytes'];ddr=raw['bandwidth_bytes_per_cycle'];cache=raw['cache_bandwidth_bytes_per_cycle']
    if ddr<=0 or cache<=0:raise ValueError('官方带宽必须为正')
    pending=Counter();queue=[];inserted=set();rows=[];index=0
    for event in raw['cache_events']:
        now=event['time'];tid=event['tensor_id']
        while queue and queue[0][0]<=now:
            _,_,done=heapq.heappop(queue);pending[done]-=1
        if event['event']=='insert':inserted.add(tid);continue
        if event['event'] not in ('hit','miss'):continue
        op=timeline[event['core_id'],event['op_id']];size=event['size_bytes']
        reason='inflight_duplicate' if pending[tid] else 'evicted' if tid in inserted else None
        if event['event']=='miss' and reason and 0<size<=capacity and cache>ddr:
            rows.append(dict(tensor_id=tid,core=event['core_id'],subgraph=op['subgraph_id'],time=now,reason=reason,
                             size=size,potential_saved_cycles=size*(1/ddr-1/cache)))
        pending[tid]+=1;heapq.heappush(queue,(op['end'],index,tid));index+=1
    return rows


def generate_cache_window_candidates(graph,parent,raw,*,max_candidates=6,window=8,read_boundary=False):
    if type(max_candidates) is not int or not 1<=max_candidates<=6:raise ValueError('候选预算必须为1～6')
    if type(window) is not int or not 1<=window<=8:raise ValueError('局部窗口必须为1～8')
    if type(read_boundary) is not bool:raise ValueError('读入边界开关必须为布尔值')
    validate_plan(graph,parent,'C')
    # 首版保留更强的整任务依赖契约，明确不覆盖B/C特有的跨流水线交错解。
    try:validate_plan(graph,parent,'A')
    except ValueError:return [],dict(status='parent_outside_conservative_task_order',scope='首版仅覆盖整任务核序无环的固定映射')
    positions={sg:(c,i) for c,row in enumerate(parent['core_schedules']) for i,sg in enumerate(row)}
    read_groups=defaultdict(set)
    if read_boundary:
        timeline={(c['core_id'],op['op_id']):op['subgraph_id'] for c in raw['per_core_timeline'] for op in c['ops']}
        for event in raw['cache_events']:
            if event['event'] in ('hit','miss'):
                read_groups[event['core_id']].add(timeline[event['core_id'],event['op_id']])
    work=defaultdict(float)
    for op in graph['ops']:
        if str(op['id']) in parent['node_to_subgraph']:work[parent['node_to_subgraph'][str(op['id'])]]+=max(1,op.get('cycles',1))
    grouped={}
    for hit in cache_window_hotspots(raw):
        sg=hit['subgraph'];core=hit['core']
        if sg not in positions or positions[sg][0]!=core:continue
        key=(core,sg,hit['reason'],hit['tensor_id'])
        if key not in grouped:grouped[key]=dict(hit)
        else:
            grouped[key]['size']+=hit['size'];grouped[key]['potential_saved_cycles']+=hit['potential_saved_cycles']
    actions=[]
    for hit in grouped.values():
        core=hit['core'];sg=hit['subgraph'];_,at=positions[sg];row=parent['core_schedules'][core]
        direction=1 if hit['reason']=='inflight_duplicate' else -1
        distances=sorted({1,min(4,window),window})
        if read_boundary:
            eligible=[d for d in range(1,window+1) if 0<=at+direction*d<len(row) and row[at+direction*d] in read_groups[core]]
            # 仅跨越实际观测到的其他读入子图；不把跨计算管线换序当缓存时序变化。
            distances=sorted(set(eligible[:2]+eligible[-1:]))
        for distance in distances:
            to=at+direction*distance
            if not 0<=to<len(row):continue
            crossed=row[to:at] if direction<0 else row[at+1:to+1]
            cost=sum(work[g] for g in crossed)
            actions.append(dict(hit,at=at,to=to,distance=distance,crossed_compute=cost,
                                read_boundary=read_boundary,
                                heuristic_ratio=hit['potential_saved_cycles']/(1+cost)))
    actions.sort(key=lambda a:(-a['heuristic_ratio'],-a['potential_saved_cycles'],a['distance'],a['core'],a['subgraph'],a['tensor_id']))
    # 两类未命中交替取动作，避免单一大张量将所有预算占满。
    families={reason:[a for a in actions if a['reason']==reason] for reason in ('inflight_duplicate','evicted')}
    ordered=[]
    for i in range(max((len(v) for v in families.values()),default=0)):
        for family in families.values():
            if i<len(family):ordered.append(family[i])
    seen={object_digest(parent)};out=[];rejected=0;attempted=0
    action_seen=set();examined=0;duplicates=0
    for action in ordered:
        if attempted>=24:break
        examined+=1
        # 多个张量可以提出相同换序；检查预算只计不同方案，防止热点挤占。
        action_key=(action['core'],action['subgraph'],action['to'])
        if action_key in action_seen:
            duplicates+=1;continue
        action_seen.add(action_key)
        candidate=deepcopy(parent);row=candidate['core_schedules'][action['core']]
        sg=row.pop(action['at']);row.insert(action['to'],sg)
        key=object_digest(candidate)
        if key in seen:
            duplicates+=1;continue
        seen.add(key);attempted+=1
        try:validate_plan(graph,candidate,'A')
        except ValueError:rejected+=1;continue
        out.append(dict(plan=candidate,source=dict(family='fixed_cache_window',**action)))
        if len(out)>=max_candidates:break
    return out,dict(status='ok',hotspots=len(grouped),actions=len(actions),attempted=attempted,rejected=rejected,
                    examined_actions=examined,duplicate_actions=duplicates,
                    read_boundary=read_boundary,
                    candidates=len(out),window=window,scope='固定切分/分核，只改核序；观测字节潜力非修改后耗时预测',
                    conservative_task_order=True)
