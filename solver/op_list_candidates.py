"""Operation-level resource list proposals; official evaluation is mandatory."""
import bisect
import heapq
import json
import time
from collections import defaultdict

from bound_certificate import build_compute_dag, PIPES


def _fit(calendar, release, duration):
    start=release
    for left,right in calendar:
        if start+duration<=left:
            break
        if start<right:
            start=right
    return start


def _schedule(ops,preds,topo,durations,n,delay,traffic,policy):
    succ={i:[] for i in ops}
    for i in ops:
        for p in preds[i]:
            succ[p].append(i)
    height={}
    for i in reversed(topo):
        height[i]=durations[i]+max((height[j] for j in succ[i]),default=0)
    pos={i:j for j,i in enumerate(topo)}
    def priority(i):
        if policy=='height':
            return (-height[i],-durations[i],i)
        if policy=='depth':
            return (pos[i],-height[i],i)
        return (-len(succ[i]),-height[i],i)
    pending={i:len(preds[i]) for i in ops}
    ready=[(priority(i),i) for i in ops if not pending[i]]
    heapq.heapify(ready)
    calendars={(c,p):[] for c in range(n) for p in PIPES}
    core,starts,ends={},{},{}
    while ready:
        _,i=heapq.heappop(ready)
        candidates=[]
        for c in range(n):
            release=max((ends[p]+(delay+traffic.get((p,i),0) if core[p]!=c else 0)
                         for p in preds[i]),default=0)
            start=_fit(calendars[c,ops[i]['pipe']],release,durations[i])
            crossing=sum(traffic.get((p,i),0)+delay for p in preds[i] if core[p]!=c)
            candidates.append((start+durations[i],crossing,start,c))
        finish,_,start,c=min(candidates)
        core[i],starts[i],ends[i]=c,start,finish
        bisect.insort(calendars[c,ops[i]['pipe']],(start,finish))
        for j in succ[i]:
            pending[j]-=1
            if not pending[j]:
                heapq.heappush(ready,(priority(j),j))
    ordered=sorted(ops,key=lambda i:(starts[i],pos[i],i))
    rank={i:k for k,i in enumerate(ordered)}
    # One original compute op per group: no quotient cycles from coarsening.
    return {'node_to_subgraph':{str(i):rank[i] for i in ordered},
            'core_schedules':[[rank[i] for i in ordered if core[i]==c] for c in range(n)]}


def generate_op_candidates(graph_json,plan,num_cores=5,max_candidates=12,**kwargs):
    started=time.monotonic()
    ops,preds,topo,durations=build_compute_dag(graph_json)
    traffic=defaultdict(float)
    producers=defaultdict(set)
    consumers=defaultdict(set)
    tensors={t['id']:t for t in graph_json.get('tensors',[])}
    for e in graph_json.get('edges',[]):
        u,v=e['source'],e['target']
        if u in ops and v in tensors:
            producers[v].add(u)
        if u in tensors and v in ops:
            consumers[u].add(v)
        if u in ops and v in ops:
            traffic[u,v]+=2*max(0,e.get('data_size',0))/60
    for t in tensors:
        for u in producers[t]:
            for v in consumers[t]:
                traffic[u,v]+=2*max(0,tensors[t]['size'])/60
    out=[]
    seen={json.dumps(plan,sort_keys=True)}
    configs=[(policy,delay) for delay in (0,100,500,1000)
             for policy in ('height','depth','fanout')]
    for policy,delay in configs[:max_candidates]:
        candidate=_schedule(ops,preds,topo,durations,num_cores,delay,traffic,policy)
        sig=json.dumps(candidate,sort_keys=True)
        if sig in seen:
            continue
        seen.add(sig)
        out.append(dict(plan=candidate,source=f'op_list_{policy}_delay{delay}',
                        stats=dict(scope='heuristic resource list proposal',delay=delay)))
    return out,dict(status='ok',candidates=len(out),elapsed=time.monotonic()-started)
