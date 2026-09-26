"""操作级就绪/流水线插空候选；沿用H提交33feb614的候选规则。

只生成合法候选，不承担官方评分或结果恢复。通信时间是独立流量偏好，
没有模拟共享外存竞争，不能当作官方总时间。带宽由调用方读取官方配置。
"""
import bisect
import heapq
import json
import math
import time
from collections import defaultdict

from dag_priority import build_compute_dag,PIPES
from scenario_contract import validate_plan


def _fit(calendar,release,duration):
    start=release
    for left,right in calendar:
        if start+duration<=left:break
        if start<right:start=right
    return start


def _schedule(ops,preds,topo,durations,n,delay,traffic,policy):
    succ={i:[] for i in ops}
    for i in ops:
        for p in preds[i]:succ[p].append(i)
    height={}
    for i in reversed(topo):
        height[i]=durations[i]+max((height[j]+((1-1/n)*(delay+traffic.get((i,j),0)) if policy=='communication_height' else 0) for j in succ[i]),default=0)
    pos={i:j for j,i in enumerate(topo)}
    def priority(i):
        if policy in ('height','communication_height'):return (-height[i],-durations[i],i)
        if policy=='depth':return (pos[i],-height[i],i)
        return (-len(succ[i]),-height[i],i)
    pending={i:len(preds[i]) for i in ops}
    ready=[(priority(i),i) for i in ops if not pending[i]];heapq.heapify(ready)
    calendars={(c,p):[] for c in range(n) for p in PIPES}
    core,starts,ends={},{},{}
    while ready:
        _,i=heapq.heappop(ready);candidates=[]
        for c in range(n):
            release=max((ends[p]+(delay+traffic.get((p,i),0) if core[p]!=c else 0) for p in preds[i]),default=0)
            start=_fit(calendars[c,ops[i]['pipe']],release,durations[i])
            crossing=sum(traffic.get((p,i),0)+delay for p in preds[i] if core[p]!=c)
            candidates.append((start+durations[i],crossing,start,c))
        finish,_,start,c=min(candidates)
        core[i],starts[i],ends[i]=c,start,finish
        bisect.insort(calendars[c,ops[i]['pipe']],(start,finish))
        for j in succ[i]:
            pending[j]-=1
            if not pending[j]:heapq.heappush(ready,(priority(j),j))
    ordered=sorted(ops,key=lambda i:(starts[i],pos[i],i));rank={i:k for k,i in enumerate(ordered)}
    return dict(node_to_subgraph={str(i):rank[i] for i in ordered},
                core_schedules=[[rank[i] for i in ordered if core[i]==c] for c in range(n)])


def generate_op_candidates(graph_json,plan,num_cores=5,max_candidates=12,*,bandwidth=60,communication_rank=False,cross_delay=None):
    if type(num_cores) is not int or not 1<=num_cores<=5:raise ValueError('核数必须在1～5之间')
    if type(max_candidates) is not int or not 1<=max_candidates<=12:raise ValueError('固定配置预算必须在1～12之间')
    if isinstance(bandwidth,bool) or not isinstance(bandwidth,(int,float)) or not math.isfinite(bandwidth) or bandwidth<=0:raise ValueError('带宽必须为正有限数')
    if type(communication_rank) is not bool:raise ValueError('通信高度开关必须为布尔值')
    if communication_rank and (isinstance(cross_delay,bool) or not isinstance(cross_delay,(int,float)) or not math.isfinite(cross_delay) or cross_delay<0):raise ValueError('通信高度必须提供官方非负跨核延迟')
    if len(plan['core_schedules'])!=num_cores:raise ValueError('父方案核数不匹配')
    validate_plan(graph_json,plan,'B')
    started=time.monotonic();ops,preds,topo,durations=build_compute_dag(graph_json)
    traffic=defaultdict(float);producers=defaultdict(set);consumers=defaultdict(set)
    tensors={t['id']:t for t in graph_json.get('tensors',[])}
    def copy_time(size):
        if isinstance(size,bool) or not isinstance(size,(int,float)) or not math.isfinite(size) or size<0:raise ValueError('流量必须非负有限')
        return 2*size/bandwidth
    for e in graph_json.get('edges',[]):
        u,v=e['source'],e['target']
        if u in ops and v in tensors:producers[v].add(u)
        if u in tensors and v in ops:consumers[u].add(v)
        if u in ops and v in ops:traffic[u,v]+=copy_time(e.get('data_size',0))
    for t in tensors:
        size=copy_time(tensors[t]['size'])
        for u in producers[t]:
            for v in consumers[t]:traffic[u,v]+=size
    configs=[(policy,delay) for delay in (0,100,500,1000) for policy in ('height','depth','fanout')]
    if communication_rank:configs=[('communication_height',delay) for delay in dict.fromkeys((0,cross_delay))]
    seen={json.dumps(plan,sort_keys=True)};out=[]
    for policy,delay in configs[:max_candidates]:
        candidate=_schedule(ops,preds,topo,durations,num_cores,delay,traffic,policy)
        signature=json.dumps(candidate,sort_keys=True)
        if signature in seen:continue
        validate_plan(graph_json,candidate,'B');seen.add(signature)
        out.append(dict(plan=candidate,source=f'op_list_{policy}_delay{delay}',
                        stats=dict(scope='操作级日历启发式候选，非官方评分',delay=delay)))
    return out,dict(status='ok',candidates=len(out),configurations=min(max_candidates,len(configs)),
                    elapsed=time.monotonic()-started,bandwidth=bandwidth,
                    communication_rank=communication_rank,cross_delay=cross_delay,
                    provenance='H/33feb614/op_list_candidates；本地场景契约与输入防护')
