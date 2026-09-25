"""固定切分的区域原子联合放置；不引入新硬件或组屏障。"""
from collections import deque
from itertools import combinations,product

from lookahead_place import order_key
from task_order_search import replay_tasks


class RegionState:
    """区域完成事件观察表；每任务释放，不占有或预留任何官方核心。"""
    def __init__(self,owners):
        self.owners=dict(owners);self.pending=set(owners);self.releases=[];self.last_time=0

    def complete(self,task,core,now):
        if task not in self.pending or self.owners[task]!=core or now<self.last_time:
            raise ValueError('区域重复完成、核心错误或事件倒序')
        self.pending.remove(task);self.last_time=now
        self.releases.append({'task':task,'core':core,'time':now,'remaining':len(self.pending)})


def choose_regions(durations,preds,orders,limit=2):
    base=replay_tasks(durations,preds,orders)
    if base is None:raise ValueError('输入计划不合法')
    ancestors={}
    def parents(task):
        if task not in ancestors:
            found=set();stack=list(preds[task])
            while stack:
                p=stack.pop()
                if p not in found:found.add(p);stack.extend(preds[p])
            ancestors[task]=found
        return ancestors[task]
    critical=set(base['critical'])
    pairs=[(a,b) for a,b in combinations(sorted(durations),2)
           if {a,b}&critical and a not in parents(b) and b not in parents(a)]
    pairs.sort(key=lambda p:(-min(durations[t] for t in p),-sum(durations[t] for t in p),p))
    chosen=[];used=set()
    for disjoint in (True,False):
        for pair in pairs:
            if len(chosen)>=limit:return chosen
            if pair in chosen or (disjoint and set(pair)&used):continue
            chosen.append(pair);used.update(pair)
    return chosen


def joint_stream(orders,region,subset):
    members=set(region);base=[[t for t in q if t not in members] for q in orders]
    positions={t:(c,sum(x not in members for x in q[:i])) for c,q in enumerate(orders) for i,t in enumerate(q) if t in members}
    a,b=region
    allocations=[cs for cs in product(subset,repeat=2) if set(cs)==set(subset)]
    allocations.sort(key=lambda cs:(sum(c!=positions[t][0] for c,t in zip(cs,region)),cs))
    for ca,cb in allocations:
        slots_a=sorted(range(len(base[ca])+1),key=lambda i:(abs(i-positions[a][1]),i))
        for ia in slots_a:
            intermediate=[list(q) for q in base];intermediate[ca].insert(ia,a)
            slots_b=sorted(range(len(intermediate[cb])+1),key=lambda i:(abs(i-positions[b][1]),i))
            for ib in slots_b:
                trial=[list(q) for q in intermediate];trial[cb].insert(ib,b)
                yield trial


def generate_joint_candidates(durations,preds,orders,regions,max_evals=2000,mode='adaptive'):
    if mode not in ('adaptive','fixed2'):raise ValueError('未知区域模式')
    n=len(orders);widths=(1,2) if mode=='adaptive' else (2,)
    streams=deque()
    # 核心宽度/子集外循环、区域内循环，使预算逐区域轮转。
    for width in widths:
        for subset in combinations(range(n),width):
            for index,region in enumerate(regions):
                streams.append((index,region,subset,iter(joint_stream(orders,region,subset))))
    rows={};seen=set();base_key=order_key(orders)
    audit={'evaluations':0,'generated':0,'invalid':0,'duplicates':0,'baseline_skipped':0,'by_region':dict.fromkeys(range(len(regions)),0)}
    while streams and audit['evaluations']<max_evals:
        index,region,subset,generator=streams.popleft()
        try:trial=next(generator)
        except StopIteration:continue
        streams.append((index,region,subset,generator));audit['generated']+=1
        key=order_key(trial);origin={'region':index,'tasks':list(region),'cores':list(subset),'width':len(subset)}
        if key==base_key:audit['baseline_skipped']+=1;continue
        if key in seen:
            audit['duplicates']+=1
            if key in rows and origin not in rows[key]['origins']:rows[key]['origins'].append(origin)
            continue
        seen.add(key);audit['evaluations']+=1;audit['by_region'][index]+=1
        result=replay_tasks(durations,preds,trial)
        if result is None:audit['invalid']+=1;continue
        outside=set(durations)-set(region)
        if [[t for t in q if t in outside] for q in trial]!=[[t for t in q if t in outside] for q in orders]:
            raise ValueError('区域动作改变组外相对顺序')
        rows[key]={'orders':trial,'coarse':result['makespan'],'origins':[origin]}
    audit.update(legal=len(rows),unfinished_streams=len(streams),budget_exhausted=bool(streams and audit['evaluations']>=max_evals))
    return list(rows.values()),audit


def select_joint_candidates(rows,limit=12,pool_limit=20):
    def identity(r):return tuple(r.get('assignment',())),order_key(r['orders'])
    def score(r):return (r['coarse'],r.get('features',{}).get('coarse_score',0),identity(r))
    ranked=sorted(rows,key=score);groups=sorted({(o['region'],o['width']) for r in rows for o in r['origins']})
    pool=[];used=set()
    # 每区域/宽度先保留两个，且完整核序去重；其余名额按粗分补足。
    for rank in (0,1):
        for group in groups:
            values=[r for r in ranked if any((o['region'],o['width'])==group for o in r['origins'])]
            if len(values)<=rank:continue
            row=values[rank]
            if rank==1 and 'assignment' in values[0]:
                kinds={o.get('structure',{}).get('kind') for o in values[0]['origins']}
                different=[r for r in values[1:] if any(o.get('structure',{}).get('kind') not in kinds for o in r['origins'])]
                if not different:different=[r for r in values[1:] if r.get('assignment')!=values[0]['assignment']]
                if different:row=different[0]
            key=identity(row)
            if key not in used and len(pool)<pool_limit:pool.append(row);used.add(key)
    for row in ranked:
        key=identity(row)
        if key not in used and len(pool)<pool_limit:pool.append(row);used.add(key)
    selected=[];used=set()
    for group in groups:
        values=[r for r in pool if any((o['region'],o['width'])==group for o in r['origins'])]
        if values:
            row=min(values,key=score);key=identity(row)
            if key not in used and len(selected)<limit:selected.append(row);used.add(key)
    # 结构菜单至少给两种实际生成的切分范式各一个机会；旧固定分区调用不受影响。
    for kind in ('branch','balanced'):
        if any(any(o.get('structure',{}).get('kind')==kind for o in r['origins']) for r in selected):continue
        options=[r for r in pool if any(o.get('structure',{}).get('kind')==kind for o in r['origins'])]
        if options and len(selected)<limit:
            row=min(options,key=score);key=identity(row)
            if key not in used:selected.append(row);used.add(key)
    for row in sorted(pool,key=score):
        key=identity(row)
        if key not in used and len(selected)<limit:selected.append(row);used.add(key)
    return selected,{'pool_size':len(pool),'precise_size':len(selected),'covered_region_widths':groups}


def region_trace(raw,regions):
    tasks={t['task_id']:(core['core_id'],t) for core in raw['per_core_timeline'] for t in core['tasks']}
    result=[]
    for members in regions:
        state=RegionState({t:tasks[t][0] for t in members})
        for t in sorted(members,key=lambda t:(tasks[t][1]['end'],t)):state.complete(t,tasks[t][0],tasks[t][1]['end'])
        end=max(tasks[t][1]['end'] for t in members);outside=[]
        for core in raw['per_core_timeline']:
            queue=core['tasks']
            for first,second in zip(queue,queue[1:]):
                if first['task_id'] in members and second['task_id'] not in members and second['start']<end:
                    outside.append({'core':core['core_id'],'after_region_task':first['task_id'],
                        'outside_task':second['task_id'],'start':second['start'],'region_end':end})
        result.append({'tasks':list(members),'cores':sorted(set(state.owners.values())),'end':end,
                       'releases':state.releases,'outside_before_region_end':outside})
    return result
