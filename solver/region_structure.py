"""切分区域身份与组外次序约束下的完整列表构造。"""
import heapq
from collections import defaultdict,deque
from itertools import combinations

from task_order_search import replay_tasks


def partition_distance(first,second,weights):
    """加权同组关系分歧；不依赖子图编号。"""
    def labels(partition):
        out={}
        for i,group in enumerate(partition):
            for b in group:
                if b in out:raise ValueError('分区成员重复')
                out[b]=i
        return out
    a,b=labels(first),labels(second)
    if set(a)!=set(b) or not a:raise ValueError('分区距离成员不一致')
    w={v:float(weights[v]) for v in a}
    if any(x<0 for x in w.values()):raise ValueError('负工作量')
    if not sum(w.values()):w=dict.fromkeys(a,1.)
    ma=defaultdict(float);mb=defaultdict(float);joint=defaultdict(float)
    for v in sorted(a):
        ma[a[v]]+=w[v];mb[b[v]]+=w[v];joint[a[v],b[v]]+=w[v]
    value=sum(x*x for x in ma.values())+sum(x*x for x in mb.values())-2*sum(x*x for x in joint.values())
    return max(0.,value/sum(w.values())**2)


def structure_menu(model,parent,roots,num_cores,menu_policy='single'):
    """每区域/结构宽度至多两个分区；宽度来自内部层反链，非硬件加速承诺。"""
    if num_cores not in (2,3,4,5) or not parent.validate(model,num_cores):raise ValueError('结构菜单输入非法')
    if menu_policy not in ('single','diverse','seed_combinations'):raise ValueError('未知结构菜单策略')
    pred=defaultdict(set);succ=defaultdict(set)
    for (a,b),_ in model.block_edges:pred[b].add(a);succ[a].add(b)
    rows=[];seen=set();slots=[];audit={'max_widths':{},'invalid':0,'duplicates':0,'generated':0,'extra_kept':0,'seed_extra_kept':0}
    def append_partition(root,source,partition):
        audit['generated']+=1
        identity=(root,tuple(sorted(tuple(sorted(group)) for group in partition)))
        if identity in seen:audit['duplicates']+=1;return False
        seen.add(identity);child=parent.clone()
        for i,blocks in enumerate(partition):
            if i==0:target=root
            else:
                target=len(child.core_of_sg);child.core_of_sg.append(parent.core_of_sg[root]);child.blocks_in_sg.append(set())
            for b in sorted(blocks):child.move_block(b,target)
        if not child.validate(model,num_cores):audit['invalid']+=1;return False
        rows.append({'sol':child,'source':{**source,'task':root,'region_count':len(partition)}})
        return True
    for root in roots[:2]:
        members=parent.blocks_in_sg[root];topo=sorted(members,key=model.block_pos.get)
        depth={};layers=defaultdict(list);rank={}
        for b in topo:
            depth[b]=1+max((depth[p] for p in pred[b] if p in members),default=-1);layers[depth[b]].append(b)
        for b in reversed(topo):rank[b]=max(model.block_work_m[b],model.block_work_v[b])+max((rank[c] for c in succ[b] if c in members),default=0)
        width_max=max(map(len,layers.values()),default=1);audit['max_widths'][root]=width_max
        menus=[({'kind':'original','width':1},[set(members)])]
        def work(blocks):return max(sum(model.block_work_m[b] for b in blocks),sum(model.block_work_v[b] for b in blocks))
        for width in range(2,min(num_cores,width_max)+1):
            alternatives=[];seed_alternatives=[]
            levels=sorted((level for level in layers if len(layers[level])>=width),
                          key=lambda d:(-len(layers[d]),-sum(rank[b] for b in layers[d]),d))[:3]
            def from_seeds(level,seeds):
                masks={};groups=defaultdict(set)
                seed_bits={b:1<<i for i,b in enumerate(seeds)}
                for b in topo:
                    mask=seed_bits.get(b,0)
                    for p in pred[b]:
                        if p in members:mask|=masks[p]
                    masks[b]=mask;groups[-1 if mask.bit_count()>1 else mask].add(b)
                exclusive=[groups[1<<i] for i in range(width)]
                if any(not group for group in exclusive):raise ValueError('同层反链被错误合并')
                gain=sum(work(group) for group in exclusive)-max(work(group) for group in exclusive)
                partition=([groups[0]] if groups[0] else [])+exclusive+([groups[-1]] if groups[-1] else [])
                return ((-gain,work(groups[0])+work(groups[-1]),tuple(seeds)),
                    {'kind':'branch','width':width,'level':level,'seeds':list(seeds),'ideal_exclusive_gain':gain},partition)
            for level in levels:
                ranked=sorted(layers[level],key=lambda b:(-rank[b],b));primary=tuple(ranked[:width])
                alternatives.append(from_seeds(level,primary))
                if menu_policy=='seed_combinations':
                    for seeds in combinations(ranked[:max(4,width)],width):
                        if seeds!=primary:seed_alternatives.append(from_seeds(level,seeds))
            if alternatives:
                best=min(alternatives,key=lambda x:x[0]);_,source,partition=best;menus.append((source,partition))
                if menu_policy!='single':
                    weights={b:max(model.block_work_m[b],model.block_work_v[b]) for b in members}
                    slots.append({'root':root,'width':width,'kept':[partition],
                                  'options':[x for x in alternatives if x is not best],'seed_options':seed_alternatives,'weights':weights})
            # 双管道累计比例均衡；完整局部拓扑前缀保证每条边方向不被倒置。
            cm=[0.];cv=[0.]
            for b in topo:cm.append(cm[-1]+model.block_work_m[b]);cv.append(cv[-1]+model.block_work_v[b])
            cuts=[0]
            for part in range(1,width):
                def imbalance(i):
                    errors=[]
                    if cm[-1]:errors.append(abs(cm[i]/cm[-1]-part/width))
                    if cv[-1]:errors.append(abs(cv[i]/cv[-1]-part/width))
                    return (max(errors,default=abs(i/len(topo)-part/width)),i)
                cuts.append(min(range(cuts[-1]+1,len(topo)-(width-part)+1),key=imbalance))
            cuts.append(len(topo));partition=[set(topo[a:b]) for a,b in zip(cuts,cuts[1:])]
            menus.append(({'kind':'balanced','width':width,'cuts':cuts},partition))
        for source,partition in menus:
            append_partition(root,source,partition)
    def fill_slots(option_field,counter):
        pending=deque(sorted(slots,key=lambda s:(s['width'],s['root'])))
        while pending and len(rows)<20:
            slot=pending.popleft()
            if not slot[option_field]:continue
            def distance(option):return min(partition_distance(option[2],q,slot['weights']) for q in slot['kept'])
            chosen=min(slot[option_field],key=lambda x:(-distance(x),x[0]));delta=distance(chosen)
            slot[option_field].remove(chosen);_,source,partition=chosen
            if append_partition(slot['root'],{**source,'diversity_distance':delta},partition):
                slot['kept'].append(partition);audit[counter]+=1
            if slot[option_field]:pending.append(slot)
        return sum(len(s[option_field]) for s in pending)
    audit['extra_unselected']=fill_slots('options','extra_kept')
    audit['seed_extra_unselected']=fill_slots('seed_options','seed_extra_kept')
    return rows,audit


def split_region_identity(original,changed,root):
    if len(original)!=len(changed) or root not in original:raise ValueError('区域输入不完整')
    by_old={};by_new={}
    for old,new in zip(original,changed):
        by_old.setdefault(old,set()).add(new);by_new.setdefault(new,set()).add(old)
    if any(len(olds)!=1 for olds in by_new.values()):raise ValueError('新任务跨越原区域边界')
    if any(len(news)!=1 for old,news in by_old.items() if old!=root):raise ValueError('组外任务被拆分')
    return set(by_old[root]),{old:next(iter(news)) for old,news in by_old.items() if old!=root}


def complete_region_orders(durations,preds,base_orders,region,allowed_cores,priority='critical'):
    nodes=set(durations);region=set(region);n=len(base_orders)
    if priority not in ('critical','earliest'):raise ValueError('未知构造优先级')
    if not region<=nodes or not allowed_cores or any(c<0 or c>=n for c in allowed_cores):raise ValueError('区域或核心集合非法')
    flat=[t for q in base_orders for t in q]
    if len(flat)!=len(nodes) or set(flat)!=nodes:raise ValueError('输入任务覆盖不完整')
    owners={t:c for c,q in enumerate(base_orders) for t in q};parents={t:set(preds[t]) for t in nodes}
    if any(not ps<=nodes for ps in parents.values()):raise ValueError('缺失外部前驱')
    for q in base_orders:
        outside=[t for t in q if t not in region]
        for first,second in zip(outside,outside[1:]):parents[second].add(first)
    succ={t:set() for t in nodes};degree={t:len(ps) for t,ps in parents.items()}
    for t,ps in parents.items():
        for p in ps:succ[p].add(t)
    todo=[t for t in nodes if not degree[t]];heapq.heapify(todo);topo=[];remaining=dict(degree)
    while todo:
        t=heapq.heappop(todo);topo.append(t)
        for child in sorted(succ[t]):
            remaining[child]-=1
            if remaining[child]==0:heapq.heappush(todo,child)
    if len(topo)!=len(nodes):return None
    rank={}
    for t in reversed(topo):rank[t]=durations[t]+max((rank[c] for c in succ[t]),default=0)
    ready={t for t in nodes if not degree[t]};orders=[[] for _ in base_orders]
    finish={};start={};placed_owner={};available=[0.]*n
    while ready:
        choices={}
        for task in ready:
            cores=allowed_cores if task in region else (owners[task],)
            options=[]
            for core in cores:
                begin=max(available[core]+(100 if orders[core] else 0),
                    max((finish[p]+(1000 if placed_owner[p]!=core else 0) for p in preds[task]),default=0))
                options.append((begin+durations[task],begin,core))
            choices[task]=min(options)
        if priority=='critical':task=min(ready,key=lambda t:(-rank[t],choices[t][0],t))
        else:task=min(ready,key=lambda t:(choices[t][0],-rank[t],t))
        end,begin,core=choices[task];ready.remove(task)
        orders[core].append(task);placed_owner[task]=core;start[task]=begin;finish[task]=end;available[core]=end
        for child in succ[task]:
            degree[child]-=1
            if degree[child]==0:ready.add(child)
    if len(finish)!=len(nodes):raise ValueError('完整列表构造丢失任务')
    timing=replay_tasks(durations,preds,orders)
    if timing is None or timing['makespan']!=max(finish.values(),default=0):raise ValueError('构造时钟与核序重放不一致')
    if [[t for t in q if t not in region] for q in orders]!=[[t for t in q if t not in region] for q in base_orders]:
        raise ValueError('改变组外核心归属或相对次序')
    return {'orders':orders,'coarse':timing['makespan'],'start':start,'finish':finish,
            'actual_cores':sorted({placed_owner[t] for t in region}),'requested_cores':list(allowed_cores),'priority':priority}


def build_structure_candidates(graph,model,parent,plan,baseline,max_evals=2000,core_policy='tied',menu_policy='single'):
    """按区域/结构轮转，背景任务用原时间线时长，新区域仅用明确标注的粗时长。"""
    from collections import deque
    from scene_a_event import derive_multicore_plan
    from lookahead_place import order_key
    if core_policy not in ('tied','all'):raise ValueError('未知结构/核心解耦策略')
    n=len(plan['core_schedules']);durations={t:r['duration'] for t,r in baseline['tasks'].items()}
    old_preds=derive_multicore_plan(graph,plan)['subgraph_preds']
    critical=set(replay_tasks(durations,old_preds,plan['core_schedules'])['critical'])
    pred=defaultdict(set)
    for (a,b),_ in model.block_edges:pred[b].add(a)
    widths={}
    for root,blocks in enumerate(parent.blocks_in_sg):
        depths={};counts=defaultdict(int)
        for b in sorted(blocks,key=model.block_pos.get):
            depths[b]=1+max((depths[p] for p in pred[b] if p in blocks),default=-1);counts[depths[b]]+=1
        widths[root]=max(counts.values(),default=1)
    roots=sorted((t for t in durations if widths[t]>1),key=lambda t:(t not in critical,-durations[t],-widths[t],t))[:2]
    structures,structure_audit=structure_menu(model,parent,roots,n,menu_policy=menu_policy)
    streams=deque();rows={};audit={'roots':roots,'structure_count':len(structures),'structures':structure_audit,
        'complete_attempts':0,'invalid':0,'duplicates':0,'baseline_skipped':0,'by_region':dict.fromkeys(range(len(roots)),0)}
    for structure in structures:
        child=structure['sol'];source=structure['source'];root=source['task']
        region,outside_map=split_region_identity(parent.sg_of_block,child.sg_of_block,root)
        # 准备一个完整种子计划，区域内部拓扑排列，不拿不完整计划做评分。
        task_preds={t:set() for t in set(child.sg_of_block)}
        for (a,b),_ in model.block_edges:
            x,y=child.sg_of_block[a],child.sg_of_block[b]
            if x!=y:task_preds[y].add(x)
        degree={t:len(task_preds[t]&region) for t in region};todo=[t for t in region if not degree[t]];heapq.heapify(todo);region_order=[]
        while todo:
            t=heapq.heappop(todo);region_order.append(t)
            for c in region:
                if t in task_preds[c]:
                    degree[c]-=1
                    if degree[c]==0:heapq.heappush(todo,c)
        if len(region_order)!=len(region):raise ValueError('已验收分区的区域子图成环')
        seed_orders=[]
        for q in plan['core_schedules']:
            order=[]
            for t in q:order.extend(region_order if t==root else [outside_map[t]])
            seed_orders.append(order)
        seed=model.plan_from(child.sg_of_block,seed_orders)
        preds=derive_multicore_plan(graph,seed)['subgraph_preds']
        _,_,info=model.evaluate(child.sg_of_block,child.core_of_sg,'A',n)
        coarse_durations={t:info['durs'][t] for t in preds}
        for old,new in outside_map.items():coarse_durations[new]=durations[old]
        if source['kind']=='original':coarse_durations[next(iter(region))]=durations[root]
        context={'assignment':child.sg_of_block,'source':source,'region':region,'outside_map':outside_map,
            'region_index':roots.index(root),'preds':preds,'orders':seed_orders,'durations':coarse_durations,
            'bandwidth_floor':(info['original_copy_bytes']+info['total_added_bytes'])/60}
        subset_sizes=range(1,n+1) if core_policy=='all' else (source['width'],)
        subsets=[subset for size in subset_sizes for subset in combinations(range(n),size)]
        choices=((subset,priority) for subset in subsets for priority in ('critical','earliest'))
        streams.append((context,iter(choices)))
    while streams and audit['complete_attempts']<max_evals:
        ctx,choices=streams.popleft()
        try:subset,priority=next(choices)
        except StopIteration:continue
        streams.append((ctx,choices));audit['complete_attempts']+=1;audit['by_region'][ctx['region_index']]+=1
        proposal=complete_region_orders(ctx['durations'],ctx['preds'],ctx['orders'],ctx['region'],subset,priority)
        if proposal is None:audit['invalid']+=1;continue
        key=tuple(ctx['assignment']),order_key(proposal['orders'])
        if key==(tuple(parent.sg_of_block),order_key(plan['core_schedules'])):audit['baseline_skipped']+=1;continue
        origin={'region':ctx['region_index'],'old_task':ctx['source']['task'],'tasks':sorted(ctx['region']),
            'cores':proposal['actual_cores'],'width':len(proposal['actual_cores']),'requested_cores':list(subset),
            'priority':priority,'structure':ctx['source'],'outside_identity':ctx['outside_map']}
        if key in rows:audit['duplicates']+=1;rows[key]['origins'].append(origin);continue
        owners=[0]*(max(ctx['assignment'])+1)
        for c,q in enumerate(proposal['orders']):
            for t in q:owners[t]=c
        rows[key]={'assignment':list(ctx['assignment']),'owners':owners,'orders':proposal['orders'],
            'coarse':max(proposal['coarse'],ctx['bandwidth_floor']),'task_coarse':proposal['coarse'],
            'approximate_bandwidth_floor':ctx['bandwidth_floor'],'origins':[origin]}
    audit.update(legal=len(rows),unfinished_streams=len(streams))
    return list(rows.values()),audit
