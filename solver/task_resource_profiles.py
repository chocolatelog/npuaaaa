"""A固定切分的计算、依赖和边界搬运估计；不做官方内存/全局事件模拟。"""
from collections import defaultdict
import math

from dag_priority import build_compute_dag,PIPES


def task_resource_profiles(graph,mapping,bandwidth):
    if isinstance(bandwidth,bool) or not isinstance(bandwidth,(int,float)) or not math.isfinite(bandwidth) or bandwidth<=0:
        raise ValueError('带宽必须为正有限数')
    ops,preds,topo,duration=build_compute_dag(graph)
    mapping={int(k):v for k,v in mapping.items()}
    if set(mapping)!=set(ops) or any(type(v) is not int or v<0 for v in mapping.values()):
        raise ValueError('切分必须覆盖全部且仅计算算子，子图编号为非负整数')
    groups=set(mapping.values());all_ops={o['id']:o for o in graph['ops']}
    tensors={t['id']:t for t in graph.get('tensors',[])}
    producers=defaultdict(set);consumers=defaultdict(set)
    for e in graph.get('edges',[]):
        u,v=e['source'],e['target']
        if u in all_ops and v in tensors:producers[v].add(u)
        if u in tensors and v in all_ops:consumers[u].add(v)
    profiles={g:dict(input_bytes=0,output_bytes=0,input_service=0,output_service=0,
                     compute_pipe_work={p:0 for p in PIPES},scope='固定切分忽略溢出与跨任务竞争的资源估计') for g in groups}
    input_release=defaultdict(int);output_reads=defaultdict(list)
    for tid,tensor in tensors.items():
        size=tensor.get('size')
        if type(size) is not int or size<0:raise ValueError('张量大小必须为非负整数')
        cost=max(1,math.ceil(size/bandwidth))
        prod=defaultdict(set);cons=defaultdict(set)
        for op in producers[tid]:
            if op in mapping:prod[mapping[op]].add(op)
        for op in consumers[tid]:
            if op in mapping:cons[mapping[op]].add(op)
        has_original_out=any(all_ops[o].get('op')=='COPY_OUT' for o in consumers[tid])
        for group in set(prod)|set(cons):
            p=profiles[group]
            # 与官方Task边界一致：同核的不同Task仍然有独立DDR读写。
            if cons[group] and not prod[group]:
                p['input_bytes']+=size;p['input_service']+=cost
                for op in cons[group]:input_release[op]=max(input_release[op],cost)
            if prod[group] and (has_original_out or not any(cons.values()) or any(g!=group and cs for g,cs in cons.items())):
                p['output_bytes']+=size;p['output_service']+=cost
                output_reads[group].append((tuple(prod[group]),cost))
    compute_end={};resource_end={}
    for op in topo:
        g=mapping[op];inside=[p for p in preds[op] if mapping[p]==g]
        compute_end[op]=duration[op]+max((compute_end[p] for p in inside),default=0)
        resource_end[op]=duration[op]+max(input_release[op],max((resource_end[p] for p in inside),default=0))
        profiles[g]['compute_pipe_work'][ops[op]['pipe']]+=duration[op]
    compute_path=defaultdict(int);resource_path=defaultdict(int)
    for op in topo:
        g=mapping[op];compute_path[g]=max(compute_path[g],compute_end[op]);resource_path[g]=max(resource_path[g],resource_end[op])
    for g,p in profiles.items():
        for prods,cost in output_reads[g]:resource_path[g]=max(resource_path[g],max(resource_end[op] for op in prods)+cost)
        work=p['compute_pipe_work'];full=dict(work)
        full['PIPE_MTE2']+=p['input_service'];full['PIPE_MTE3']+=p['output_service']
        compute=max(compute_path[g],max(work.values(),default=0))
        copy=p['input_service']+p['output_service']
        p.update(compute_bound=compute,copy_service=copy,dependency_with_boundary=resource_path[g],
                 resource_estimate=max(resource_path[g],copy,max(full.values(),default=0)),
                 additive_estimate=compute+copy)
    return profiles
