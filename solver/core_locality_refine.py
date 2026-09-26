"""B/C单算子切分的有界亲和迁核；精确边界连接计数不等于总时间预测。"""
from collections import Counter, defaultdict
import bisect
import copy
import heapq
import math

from dag_priority import build_compute_dag, PIPES
from scenario_contract import validate_plan


class CoreTrafficIndex:
    """按张量的生产/消费核心计数维护B/C展开前的搬运边界。"""
    def __init__(self, graph, core_of, bandwidth):
        if isinstance(bandwidth,bool) or not math.isfinite(bandwidth) or bandwidth<=0:
            raise ValueError('带宽必须为正有限数')
        self.core = dict(core_of)
        self.edges, self.incident = {}, defaultdict(set)
        tensors = {t['id']: t for t in graph.get('tensors',[])}
        ops = {o['id']: o for o in graph['ops']}
        prod, cons, output = defaultdict(set), defaultdict(set), set()
        for i,e in enumerate(graph.get('edges',[])):
            u,v=e['source'],e['target']
            if u in core_of and v in tensors: prod[v].add(u)
            if u in tensors and v in core_of: cons[u].add(v)
            if u in tensors and v in ops and ops[v].get('op')=='COPY_OUT': output.add(u)
            if u in core_of and v in core_of:
                self._add(('direct',i),{u},{v},max(0,int(e.get('data_size',0))),False,bandwidth)
        for t,row in tensors.items():
            size=row['size']
            if type(size) is not int or size<0:raise ValueError('张量大小必须为非负整数')
            self._add(('tensor',t),prod[t],cons[t],size,t in output,bandwidth)

    def _add(self, key, p, c, size, output, bandwidth):
        self.edges[key]=dict(producers=p,consumers=c,size=size,unit=max(1,math.ceil(size/bandwidth)),
            output=output,p=Counter(self.core[i] for i in p),c=Counter(self.core[i] for i in c))
        for i in p|c:self.incident[i].add(key)

    @staticmethod
    def _cost(edge,p=None,c=None):
        ps={k for k,v in (edge['p'] if p is None else p).items() if v}
        cs={k for k,v in (edge['c'] if c is None else c).items() if v}
        links=len(ps)*len(cs)-len(ps&cs)
        copies=2*links+(len(cs) if not ps else 0)+(len(ps) if edge['output'] or not cs else 0)
        return dict(bytes=copies*edge['size'],service=copies*edge['unit'],links=links)

    def total(self):
        result=dict(bytes=0,service=0,links=0)
        for e in self.edges.values():
            for k,v in self._cost(e).items():result[k]+=v
        return result

    def move_delta(self,op,target):
        return self.changes_delta({op:target})

    def changes_delta(self,changes):
        result=dict(bytes=0,service=0,links=0)
        affected=set().union(*(self.incident[op] for op in changes))
        for key in affected:
            e=self.edges[key];p=e['p'].copy();c=e['c'].copy()
            for op,target in changes.items():
                source=self.core[op]
                if op in e['producers']:p[source]-=1;p[target]+=1
                if op in e['consumers']:c[source]-=1;c[target]+=1
            old,new=self._cost(e),self._cost(e,p,c)
            for k in result:result[k]+=old[k]-new[k]
        return result

    def move(self,op,target):
        source=self.core[op]
        for key in self.incident[op]:
            e=self.edges[key]
            if op in e['producers']:e['p'][source]-=1;e['p'][target]+=1
            if op in e['consumers']:e['c'][source]-=1;e['c'][target]+=1
        self.core[op]=target


def _ordered_plan(parent, assignment, original_core, group, preds, observed_start):
    """未迁移操作的相对核序固定；迁入操作按实测起点插入后拓扑修复。"""
    reverse={g:i for i,g in group.items()};schedules=[]
    for c,old in enumerate(parent['core_schedules']):
        kept=[reverse[g] for g in old if assignment[reverse[g]]==c]
        incoming=sorted((i for i in assignment if assignment[i]==c and original_core[i]!=c),
                        key=lambda i:(observed_start[i],i))
        proposed=list(kept)
        for i in incoming:
            at=next((j for j,k in enumerate(proposed) if observed_start[k]>observed_start[i]),len(proposed))
            proposed.insert(at,i)
        local=set(proposed);adj={i:set() for i in local};degree={i:0 for i in local}
        def edge(u,v):
            if v not in adj[u]:adj[u].add(v);degree[v]+=1
        for i in local:
            for p in preds[i]&local:edge(p,i)
        for u,v in zip(kept,kept[1:]):edge(u,v)
        rank={i:j for j,i in enumerate(proposed)}
        ready=[(rank[i],i) for i,d in degree.items() if not d];heapq.heapify(ready);order=[]
        while ready:
            _,i=heapq.heappop(ready);order.append(group[i])
            for v in sorted(adj[i]):
                degree[v]-=1
                if not degree[v]:heapq.heappush(ready,(rank[v],v))
        if len(order)!=len(local):return None
        schedules.append(order)
    plan=copy.deepcopy(parent);plan['core_schedules']=schedules
    return plan


def generate_core_locality_candidates(graph,parent,raw,*,bandwidth=60,max_candidates=3,max_moves=16,hotspot_limit=256,exchange_only=False):
    """最多1/4/16次迁移的三个快照；每步正搬运收益且各流水线最大负载不增加。"""
    if type(max_candidates) is not int or not 1<=max_candidates<=3:raise ValueError('候选预算必须1～3')
    if type(max_moves) is not int or not 1<=max_moves<=16:raise ValueError('迁移预算必须1～16')
    if type(hotspot_limit) is not int or hotspot_limit<1:raise ValueError('热点预算必须为正')
    if type(exchange_only) is not bool:raise ValueError('交换开关必须为布尔值')
    validate_plan(graph,parent,'B')
    ops,preds,_,durations=build_compute_dag(graph)
    group={int(k):v for k,v in parent['node_to_subgraph'].items()}
    audit=dict(moves=0,rejections=Counter(),scope='静态B/C边界精确连接计数；不含溢出、带宽竞争与缓存，非总时间代理')
    if len(set(group.values()))!=len(group):
        audit['status']='skipped_non_singleton_partition';audit['rejections']={};return [],audit
    original_group_core={g:c for c,row in enumerate(parent['core_schedules']) for g in row}
    original={i:original_group_core[group[i]] for i in ops};assignment=dict(original)
    observed={o['op_id']:o['start'] for c in raw['per_core_timeline'] for o in c['ops'] if o['op_id'] in ops}
    if set(observed)!=set(ops):raise ValueError('父官方时间线缺少计算操作')
    for c in raw['per_core_timeline']:
        if any(original[o['op_id']]!=c['core_id'] for o in c['ops'] if o['op_id'] in ops):raise ValueError('父时间线核心不一致')
    n=len(parent['core_schedules']);index=CoreTrafficIndex(graph,assignment,bandwidth);before=index.total()
    loads={(c,p):0 for c in range(n) for p in PIPES}
    for i in ops:loads[assignment[i],ops[i]['pipe']]+=durations[i]
    caps={p:max(loads[c,p] for c in range(n)) for p in PIPES}
    # 按真实边界边际收益筛热点，避免本地大张量挤掉能消除跨核连接的小热点。
    exposure={i:max((index.move_delta(i,c)['service'] for c in range(n) if c!=assignment[i]),default=0) for i in ops}
    focus=sorted(ops,key=lambda i:(-exposure[i],i))[:hotspot_limit]
    locked=set();moves=[];rows=[];last=None
    def save(plan):
        if any(plan==r['plan'] for r in rows):return
        after=index.total()
        rows.append(dict(plan=plan,source=dict(module='core_locality',moves=copy.deepcopy(moves),
            boundary_bytes_saved=before['bytes']-after['bytes'],boundary_service_saved=before['service']-after['service'],
            cross_links_removed=before['links']-after['links'],max_pipe_load_caps=caps,exchange_only=exchange_only)))
    for step in range(1,max_moves+1):
        proposals=[];seen_pairs=set()
        buckets=defaultdict(lambda:defaultdict(list))
        if exchange_only:
            for j in ops:
                if j not in locked:buckets[assignment[j],ops[j]['pipe']][durations[j]].append(j)
            for bucket in buckets.values():
                for duration,values in bucket.items():bucket[duration]=sorted(values,key=lambda j:(-exposure[j],j))[:32]
            bucket_times={key:sorted(bucket) for key,bucket in buckets.items()}
        for i in focus:
            if i in locked:continue
            source=assignment[i];pipe=ops[i]['pipe']
            for target in range(n):
                if target==source:continue
                if exchange_only:
                    times=bucket_times.get((target,pipe),[]);bucket=buckets[target,pipe]
                    hi=bisect.bisect_left(times,durations[i]);lo=hi-1;nearby=[]
                    while len(nearby)<32 and (lo>=0 or hi<len(times)):
                        left=durations[i]-times[lo] if lo>=0 else math.inf
                        right=times[hi]-durations[i] if hi<len(times) else math.inf
                        if left<=right:nearby.extend(bucket[times[lo]]);lo-=1
                        if right<=left:nearby.extend(bucket[times[hi]]);hi+=1
                    partners=sorted(nearby,key=lambda j:(abs(durations[j]-durations[i]),-exposure[j],j))[:32]
                    for j in partners:
                        pair=tuple(sorted((i,j)))
                        if pair in seen_pairs:continue
                        seen_pairs.add(pair)
                        if (loads[target,pipe]-durations[j]+durations[i]>caps[pipe] or
                                loads[source,pipe]-durations[i]+durations[j]>caps[pipe]):
                            audit['rejections']['pipe_load']+=1;continue
                        delta=index.changes_delta({i:target,j:source})
                        if delta['service']<=0:audit['rejections']['nonpositive_service_gain']+=1;continue
                        proposals.append((-delta['service'],-delta['links'],-delta['bytes'],i,target,j,delta))
                    continue
                if loads[target,pipe]+durations[i]>caps[pipe]:audit['rejections']['pipe_load']+=1;continue
                delta=index.move_delta(i,target)
                if delta['service']<=0:audit['rejections']['nonpositive_service_gain']+=1;continue
                proposals.append((-delta['service'],-delta['links'],-delta['bytes'],i,target,-1,delta))
        selected=None
        for _,_,_,i,target,j,delta in sorted(proposals):
            changes={i:target}
            if j>=0:changes[j]=assignment[i]
            trial=dict(assignment);trial.update(changes)
            plan=_ordered_plan(parent,trial,original,group,preds,observed)
            if plan is None:audit['rejections']['local_order_cycle']+=1;continue
            try:validate_plan(graph,plan,'B')
            except ValueError:audit['rejections']['invalid_plan']+=1;continue
            selected=(changes,delta,plan);break
        if selected is None:break
        changes,delta,last=selected
        for i,target in changes.items():
            source=assignment[i];pipe=ops[i]['pipe']
            loads[source,pipe]-=durations[i];loads[target,pipe]+=durations[i]
            assignment[i]=target;index.move(i,target);locked.add(i)
            moves.append(dict(op=i,source=source,target=target,action_step=step,joint_delta=delta))
        if step in (1,4,16):save(last)
        if len(rows)>=max_candidates:break
    if last is not None and len(rows)<max_candidates:save(last)
    audit.update(status='ok',moves=len(moves),hotspots=len(focus),exchange_only=exchange_only,initial=index.total() if not moves else before,
                 final=index.total(),rejections=dict(audit['rejections']))
    return rows,audit
