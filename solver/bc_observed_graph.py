"""B/C原官方展开图的只读适配与固定实测时长重建，不模拟反事实竞争。"""
import heapq

from scenario_contract import validate_plan
from multicore_cut_evaluate_problem_2 import _build_scene_b_tasks
from evaluation_validation import validate_execution


def expand(graph,plan,raw):
    validate_plan(graph,plan,'B')
    tasks,links,_,_,_=_build_scene_b_tasks(graph,plan,raw['bandwidth_bytes_per_cycle'],raw['capacity_bytes'])
    validate_execution(tasks,links)
    return tasks,links


def operation_tags(tasks):
    tags={}
    for core,task in tasks.items():
        def tensors(ids):
            return tuple((i,task['tensor_by_id'][i]['size'],task['tensor_by_id'][i].get('pos')) for i in sorted(ids))
        for i,op in task['op_by_id'].items():
            tags[core,i]=(op['op'],op.get('pipe'),op.get('cycles'),
                tensors(task['in_tids'].get(i,())),tensors(task['out_tids'].get(i,())))
    return tags


def replay_expanded(tasks,links,durations,cross_delay):
    nodes={(c,i) for c,t in tasks.items() for i in t['op_by_id']}
    if nodes!=set(durations):return None
    preds={k:{} for k in nodes};succ={k:{} for k in nodes}
    def edge(u,v,lag):
        preds[v][u]=max(preds[v].get(u,0),lag);succ[u][v]=preds[v][u]
    for c,t in tasks.items():
        for i,ps in t['op_preds'].items():
            for p in ps:edge((c,p),(c,i),0)
        for seq in t['pipe_ops'].values():
            for u,v in zip(seq,seq[1:]):edge((c,u),(c,v),0)
    for link in links:
        edge((link['source_core'],link['source_copy_out_id']),
             (link['target_core'],link['target_copy_in_id']),cross_delay)
    count={k:len(p) for k,p in preds.items()};ready=[k for k,d in count.items() if not d];heapq.heapify(ready)
    start={k:0 for k in nodes};end={};parent={};topo=[]
    while ready:
        k=heapq.heappop(ready);topo.append(k);end[k]=start[k]+durations[k]
        for v,lag in sorted(succ[k].items()):
            if end[k]+lag>start[v]:start[v]=end[k]+lag;parent[v]=k
            count[v]-=1
            if not count[v]:heapq.heappush(ready,v)
    if len(topo)!=len(nodes):return None
    makespan=max(end.values(),default=0);tail={}
    for k in reversed(topo):tail[k]=durations[k]+max((lag+tail[v] for v,lag in succ[k].items()),default=0)
    slack={k:makespan-start[k]-tail[k] for k in nodes}
    return dict(makespan=makespan,starts=start,ends=end,slack=slack,
                critical=sorted(k for k,s in slack.items() if s==0),scope='固定实测时长的顺序代理，不含修改后的带宽/缓存反馈')


def observe_parent(graph,plan,raw,*,expander=None):
    tasks,links=(expand if expander is None else expander)(graph,plan,raw)
    observed={(c['core_id'],o['op_id']):o for c in raw['per_core_timeline'] for o in c['ops']}
    durations={k:o['duration'] for k,o in observed.items()}
    replay=replay_expanded(tasks,links,durations,raw['cross_core_copy_delay_cycles'])
    if replay is None or replay['makespan']!=raw['makespan'] or any(
            replay['starts'][k]!=o['start'] or replay['ends'][k]!=o['end'] for k,o in observed.items()):
        raise ValueError('父官方起止时间重建不一致')
    return dict(tasks=tasks,links=links,observed=observed,durations=durations,
                tags=operation_tags(tasks),replay=replay)
