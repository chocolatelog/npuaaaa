"""借操作级分核形成A整任务；分组保留核序、边界和完整依赖防环。"""
from dag_priority import build_compute_dag,_topology
from bounded_coarsening import coarsen
from scenario_contract import validate_plan


def group_fixed_mapping(graph,seed,*,block_cap,work_cap):
    """只沿同核连续操作延长任务，不跨越另一个已关闭任务再合并。"""
    validate_plan(graph,seed,'A')
    ops,preds,topo,duration=build_compute_dag(graph)
    mapping={int(k):int(v) for k,v in seed['node_to_subgraph'].items()}
    if len(set(mapping.values()))!=len(ops):raise ValueError('输入须为每算子一个子图')
    op_of={sg:op for op,sg in mapping.items()}
    sequences=[[op_of[sg] for sg in row] for row in seed['core_schedules']]
    affinity={};combined={i:set(ps) for i,ps in preds.items()}
    for row in sequences:
        for a,b in zip(row,row[1:]):
            combined[b].add(a)
            # 正亲和仅允许本核前一个操作所在块，不能与远端前驱合并。
            affinity[a,b]=1
    ordered,_=_topology(combined)
    blocks,audit=coarsen(ordered,combined,affinity,max_ops=block_cap,work=duration,
                         max_work=work_cap,pack_frontiers=False)
    owner={op:sg for sg,block in enumerate(blocks) for op in block}
    schedules=[]
    for row in sequences:
        result=[]
        for op in row:
            sg=owner[op]
            if not result or result[-1]!=sg:result.append(sg)
        if len(result)!=len(set(result)):raise ValueError('分组不是核内连续区间')
        schedules.append(result)
    plan=dict(node_to_subgraph={str(op):owner[op] for op in topo},core_schedules=schedules)
    validate_plan(graph,plan,'A')
    audit['scope']='固定操作级分核与核序，安全收缩连续任务区间'
    return plan,audit


def generate_mapped_block_candidates(graph,parent,num_cores,*,bandwidth=60):
    """两个通信偏好×两个粒度，最多四份不同合法候选。"""
    from op_list_candidates import generate_op_candidates
    from official_protocol import object_digest
    seeds,_=generate_op_candidates(graph,parent,num_cores=num_cores,max_candidates=4,bandwidth=bandwidth)
    total=sum(max(1,o.get('cycles',1)) for o in graph['ops'] if o['op'] not in ('COPY_IN','COPY_OUT'))
    work_cap=max(1,total/(num_cores*4));seen={object_digest(parent)};out=[]
    for seed in seeds:
        if not seed['source'].startswith('op_list_height_'):continue
        for cap in (120,240):
            plan,audit=group_fixed_mapping(graph,seed['plan'],block_cap=cap,work_cap=work_cap)
            key=object_digest(plan)
            if key in seen:continue
            seen.add(key)
            out.append(dict(plan=plan,source=f"mapped_{seed['source']}_cap{cap}",audit=audit))
    return out
