"""计算松弛下界与依赖安全的就绪列表调度；启发式分数不冒充官方预测。"""
from collections import defaultdict
import heapq
import math

PIPES=('PIPE_M','PIPE_V','PIPE_MTE2','PIPE_MTE3')


def _topology(preds):
    succ={i:set() for i in preds}
    count={i:len(ps) for i,ps in preds.items()}
    for i,ps in preds.items():
        for p in ps:
            if p not in preds or p==i:raise ValueError('依赖端点不存在或自环')
            succ[p].add(i)
    ready=[i for i in preds if count[i]==0];heapq.heapify(ready);topo=[]
    while ready:
        i=heapq.heappop(ready);topo.append(i)
        for j in sorted(succ[i]):
            count[j]-=1
            if not count[j]:heapq.heappush(ready,j)
    if len(topo)!=len(preds):raise ValueError('输入依赖有环')
    return topo,succ


def priority_features(preds,durations):
    if set(preds)!=set(durations):raise ValueError('工作量与节点集合不一致')
    if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or v<0 for v in durations.values()):
        raise ValueError('时长必须非负有限')
    topo,succ=_topology(preds);earliest={};height={};pressure={}
    for i in topo:earliest[i]=max((earliest[p]+durations[p] for p in preds[i]),default=0)
    for i in reversed(topo):
        height[i]=durations[i]+max((height[j] for j in succ[i]),default=0)
        pressure[i]=durations[i]+sum(durations[j]/max(1,len(preds[j])) for j in succ[i])
    return dict(topological=topo,successors=succ,earliest=earliest,height=height,successor_pressure=pressure)


def build_compute_dag(graph):
    """收缩复制/张量中介，保留原图拓扑位置供操作级候选复现。"""
    ops={o['id']:o for o in graph['ops']};tensors={t['id'] for t in graph.get('tensors',[])}
    if len(ops)!=len(graph['ops']) or len(tensors)!=len(graph.get('tensors',[])) or set(ops)&tensors:
        raise ValueError('原始节点编号重复')
    nodes=set(ops)|tensors
    if any(type(i) is not int for i in nodes):raise ValueError('节点编号必须为整数')
    preds={i:set() for i in nodes}
    for e in graph.get('edges',[]):
        if any(type(e[k]) is not int or e[k] not in nodes for k in ('source','target')):raise ValueError('边端点不存在或非整数')
        preds[e['target']].add(e['source'])
    topo,succ=_topology(preds)
    compute={i:o for i,o in ops.items() if o['op'] not in ('COPY_IN','COPY_OUT')}
    frontier=defaultdict(set);contracted={i:set() for i in compute};duration={}
    for i in topo:
        if i in compute:
            op=compute[i];p=op.get('cycles',1)
            if op['pipe'] not in PIPES or type(p) is not int:raise ValueError('流水线或整数时长不支持')
            duration[i]=max(1,p);contracted[i]=set(frontier[i]);passing={i}
        else:passing=frontier[i]
        for j in succ[i]:frontier[j].update(passing)
    return compute,contracted,[i for i in topo if i in compute],duration


def compute_certificate(graph,num_cores):
    if type(num_cores) is not int or num_cores<1:raise ValueError('核数必须为正整数')
    compute,contracted,_,duration=build_compute_dag(graph)
    features=priority_features(contracted,duration)
    work={q:sum(duration[i] for i in compute if compute[i]['pipe']==q) for q in PIPES}
    path=max((features['earliest'][i]+duration[i] for i in compute),default=0)
    return dict(scope='global_compute_only_relaxation',dependency_path=path,
        lower_bound=max([path]+[(w+num_cores-1)//num_cores for w in work.values()]),
        pipe_work=work,predecessors={i:sorted(ps) for i,ps in contracted.items()},
        height=features['height'],earliest=features['earliest'],
        omitted=['copy','memory','task_wait','shared_bandwidth'],attainment_proven=False)


def _ranks(ready,values):
    ordered=sorted(ready,key=lambda i:(values[i],i));result={};lo=0
    while lo<len(ordered):
        hi=lo+1
        while hi<len(ordered) and values[ordered[hi]]==values[ordered[lo]]:hi+=1
        rank=(lo+hi-1)/(2*max(1,len(ordered)-1))
        for i in ordered[lo:hi]:result[i]=rank
        lo=hi
    return result


def schedule_ready(preds,durations,traffic,num_cores,*,alpha=0,same_wait=100,cross_wait=1000,
                   bandwidth=60,traffic_weight=.2,placement='append'):
    if placement not in ('append','insert'):raise ValueError('未知任务分配方式')
    if type(num_cores) is not int or num_cores<1:raise ValueError('核数必须为正整数')
    if not math.isfinite(alpha) or not 0<=alpha<=1:raise ValueError('权重范围必须在0至1')
    if not all(math.isfinite(x) and x>=0 for x in (same_wait,cross_wait,traffic_weight)) or not math.isfinite(bandwidth) or bandwidth<=0:
        raise ValueError('等待或带宽参数非法')
    if any(not math.isfinite(v) or v<0 for v in traffic.values()):raise ValueError('流量必须非负有限')
    f=priority_features(preds,durations);succ=f['successors'];pending={i:len(ps) for i,ps in preds.items()}
    ready={i for i in preds if not pending[i]};orders=[[] for _ in range(num_cores)]
    free=[0]*num_cores;starts={};ends={};core={};dispatch=[]
    while ready:
        if alpha==0:i=min(ready,key=lambda x:(-f['height'][x],-durations[x],x))
        else:
            hr=_ranks(ready,f['height']);ur=_ranks(ready,f['successor_pressure'])
            i=min(ready,key=lambda x:(-((1-alpha)*hr[x]+alpha*ur[x]),-f['height'][x],x))
        choices=[]
        for c in range(num_cores):
            release=max((ends[p]+(cross_wait if core[p]!=c else 0) for p in preds[i]),default=0)
            if placement=='insert':
                from task_calendar import earliest_task_slot
                earliest=max((k+1 for k,t in enumerate(orders[c]) if t in preds[i]),default=0)
                start,position=earliest_task_slot([(starts[t],ends[t]) for t in orders[c]],
                    release,durations[i],same_wait,min_position=earliest)
            else:
                start=max(free[c]+(same_wait if orders[c] else 0),release)
                position=len(orders[c])
            remote=sum(traffic.get((p,i),0) for p in preds[i] if core[p]!=c)
            choices.append((start+durations[i]+traffic_weight*remote/bandwidth,start,c,position))
        _,start,c,position=min(choices)
        ready.remove(i);dispatch.append(i);orders[c].insert(position,i)
        core[i]=c;starts[i]=start;ends[i]=start+durations[i];free[c]=max(free[c],ends[i])
        for j in succ[i]:
            pending[j]-=1
            if not pending[j]:ready.add(j)
    return dict(orders=orders,core_of=core,starts=starts,ends=ends,dispatch_order=dispatch,
                estimated_makespan=max(ends.values(),default=0),alpha=alpha,placement=placement,
                scope='heuristic_list_schedule_without_shared_bandwidth')


def construct_block_plan(model,num_cores,alpha=0,duration_mode='legacy',profiles=None,**settings):
    from construct import block_graph
    from scenario_contract import validate_plan
    _,pred,traffic=block_graph(model)
    preds={i:set(pred.get(i,())) for i in range(len(model.blocks))}
    if duration_mode=='legacy':
        durations={i:max(model.block_work_m[i],model.block_work_v[i]) for i in preds}
    elif duration_mode in ('resource','additive'):
        if profiles is None:
            from task_resource_profiles import task_resource_profiles
            profiles=task_resource_profiles(model.graph_json,model.block_of_op,settings.get('bandwidth',60))
        key='resource_estimate' if duration_mode=='resource' else 'additive_estimate'
        durations={i:profiles[i][key] for i in preds}
    else:raise ValueError('未知块时长估计模式')
    result=schedule_ready(preds,durations,traffic,num_cores,alpha=alpha,**settings)
    result['duration_mode']=duration_mode
    plan=model.plan_from(list(range(len(model.blocks))),result['orders'])
    validate_plan(model.graph_json,plan,'A')
    return plan,result
