"""固定切分的依赖完整前瞻；全图资源重放，未来叶子只提供选择价值。"""
import heapq
import time

from task_order_search import replay_tasks


def order_key(orders):
    return tuple(tuple(queue) for queue in orders)


def bounded_window(root,future,preds,completed,max_nodes=8):
    requested=len(future);future=list(future)
    while True:
        closure=set();stack=[root]+future
        while stack:
            task=stack.pop()
            if task in completed or task in closure:continue
            closure.add(task);stack.extend(sorted(preds.get(task,())))
            if len(closure)>max_nodes:break
        if len(closure)<=max_nodes:
            return future,closure,{'removed_future':requested-len(future),'closure_size':len(closure)}
        if not future:
            return [],set(),{'removed_future':requested,'closure_size':len(closure),'root_closure_exceeded':True}
        future.pop()


def propose_placements(orders,task,preds,durations,locked,feature_batch=None):
    if task in locked:return []
    stripped=[[t for t in q if t!=task] for q in orders]
    candidates=[];seen=set()
    for core,queue in enumerate(stripped):
        prefix=0
        while prefix<len(queue) and queue[prefix] in locked:prefix+=1
        if any(t in locked for t in queue[prefix:]):
            raise ValueError('已执行任务不是核心队列前缀')
        for slot in range(prefix,len(queue)+1):
            trial=[list(q) for q in stripped];trial[core].insert(slot,task)
            state=replay_tasks(durations,preds,trial)
            key=order_key(trial)
            if state is None or key in seen:continue
            seen.add(key);candidates.append({'orders':trial,'core':core,'slot':slot,'coarse':state['makespan']})
    features=feature_batch([r['orders'] for r in candidates]) if feature_batch and candidates else [None]*len(candidates)
    for r,feature in zip(candidates,features):r['features']=feature
    def score(r):return (r['coarse'],(r['features'] or {}).get('coarse_score',0),order_key(r['orders']))
    result=[]
    for core in range(len(orders)):
        rows=[r for r in candidates if r['core']==core]
        if rows:result.append(min(rows,key=score))
    return sorted(result,key=score)


def compact_result(result,orders):
    dm=result['data_movement_bytes']
    return {'orders':[list(q) for q in orders],
            'real':{'makespan':result['makespan'],'added_copy_bytes':dm['added_copy_bytes'],
                    'scheduled_copy_bytes':dm['scheduled_copy_bytes'],
                    'partition_added':dm['partition_added_copy_bytes'],'spill_added':dm['spill_added_copy_bytes']},
            'tasks':{t['task_id']:dict(t) for core in result['per_core_timeline'] for t in core['tasks']}}


class CandidateOracle:
    def __init__(self,graph,plan,baseline,evaluator,max_evals=32,progress=None):
        self.graph,self.plan,self.evaluator=graph,plan,evaluator
        self.baseline=compact_result(baseline,plan['core_schedules'])
        self.cache={order_key(plan['core_schedules']):self.baseline}
        self.max_evals=max_evals;self.progress=progress;self.history=[]
        self.stats={'evaluations':0,'cache_hits':0,'budget_rejections':0,'seconds':0.0}

    def evaluate(self,orders,stage='head'):
        key=order_key(orders)
        if key in self.cache:
            self.stats['cache_hits']+=1
            return self.cache[key]
        if self.stats['evaluations']>=self.max_evals:
            self.stats['budget_rejections']+=1
            return None
        plan={**self.plan,'core_schedules':[list(q) for q in orders]}
        before=time.perf_counter()
        raw=self.evaluator(self.graph,plan,60,{'L1':524288,'UB':131072},1000,100)
        result=compact_result(raw,orders)
        self.stats['seconds']+=time.perf_counter()-before
        for field in ('partition_added','spill_added','added_copy_bytes','scheduled_copy_bytes'):
            if result['real'][field]!=self.baseline['real'][field]:
                raise ValueError('固定切分重排改变搬运不变量')
        self.stats['evaluations']+=1;self.cache[key]=result
        self.history.append({'stage':stage,'orders':result['orders'],'real':result['real']})
        if self.progress and self.stats['evaluations']%8==0:self.progress(self.stats)
        return result


def successor_chain(root,preds,durations,length):
    succ={t:[] for t in preds};degree={t:len(ps) for t,ps in preds.items()}
    for t,ps in preds.items():
        for p in ps:succ[p].append(t)
    queue=[t for t,v in degree.items() if not v];heapq.heapify(queue);topo=[]
    while queue:
        t=heapq.heappop(queue);topo.append(t)
        for child in sorted(succ[t]):
            degree[child]-=1
            if degree[child]==0:heapq.heappush(queue,child)
    if len(topo)!=len(preds):raise ValueError('任务数据依赖成环')
    ranks={}
    for t in reversed(topo):ranks[t]=durations[t]+max((ranks[c] for c in succ[t]),default=0)
    chain=[];current=root
    while len(chain)<length and succ[current]:
        current=min(succ[current],key=lambda t:(-ranks[t],t));chain.append(current)
    return chain


def future_decisions(root,preds,durations,length,tasks,completed,locked,policy='chain'):
    future=successor_chain(root,preds,durations,length)
    if policy=='chain':return future
    if policy!='ready':raise ValueError('未知前瞻窗口策略')
    while len(future)<length:
        available=[t for t in preds if t not in locked|completed|{root}|set(future)
                   and set(preds[t])<=completed|{root}|set(future)]
        if not available:break
        future.append(min(available,key=lambda t:(-durations[t],tasks[t]['start'],t)))
    return future


def decision_window(root,preds,tasks,policy='start'):
    # 离线搜索边界，不是候选执行时刻；等待和外存争用仍由全图重放计算。
    if policy=='start':cut=tasks[root]['start']
    elif policy=='data_ready':cut=max((tasks[p]['end'] for p in preds[root]),default=0)
    else:raise ValueError('未知决策边界策略')
    return (cut,{t for t,r in tasks.items() if r['start']<cut},
            {t for t,r in tasks.items() if r['end']<=cut})


def search_lookahead(oracle,preds,depth=3,max_decisions=3,feature_batch=None,future_policy='chain',cut_policy='start',keep_current=False):
    if depth not in (1,3):raise ValueError('首轮仅比较深度1与3')
    current=oracle.baseline;archive={order_key(current['orders']):current};visited=set();decisions=[]
    for step in range(max_decisions):
        durations={t:r['duration'] for t,r in current['tasks'].items()}
        coarse=replay_tasks(durations,preds,current['orders'])
        if coarse is None:raise ValueError('当前完整计划不合法')
        candidates=[t for t in coarse['critical'] if t not in visited]
        if not candidates:candidates=[t for t in durations if t not in visited]
        if not candidates:break
        root=min(candidates,key=lambda t:(-durations[t],t));visited.add(root)
        cut,locked,completed=decision_window(root,preds,current['tasks'],cut_policy)
        requested=future_decisions(root,preds,durations,depth-1,current['tasks'],completed,locked,future_policy)
        future,closure,note=bounded_window(root,requested,preds,completed)
        if note.get('root_closure_exceeded'):
            decisions.append({'root':root,'skipped':'前驱闭包超限',**note});continue
        heads=[]
        for option in propose_placements(current['orders'],root,preds,durations,locked,feature_batch):
            value=oracle.evaluate(option['orders'],'head')
            if value is None:break
            archive[order_key(value['orders'])]=value
            heads.append({'option':option,'value':value,'shadow_score':value['real']['makespan'],'beam':[value]})
        if keep_current and not any(order_key(h['value']['orders'])==order_key(current['orders']) for h in heads):
            core=next(c for c,q in enumerate(current['orders']) if root in q)
            option={'orders':current['orders'],'core':core,'slot':current['orders'][core].index(root)}
            heads.append({'option':option,'value':current,'shadow_score':current['real']['makespan'],'beam':[current]})
        if not heads:break
        levels=0;budget_shortened=False
        for task in future:
            proposals=[]
            for head_index,head in enumerate(heads):
                for state in head['beam']:
                    shadow_durations={t:r['duration'] for t,r in state['tasks'].items()}
                    options=propose_placements(state['orders'],task,preds,shadow_durations,locked,feature_batch)[:2]
                    proposals.extend((head_index,opt) for opt in options)
            needed={order_key(opt['orders']) for _,opt in proposals}-oracle.cache.keys()
            if len(needed)>oracle.max_evals-oracle.stats['evaluations']:
                budget_shortened=True;break
            grouped={i:[] for i in range(len(heads))}
            for head_index,option in proposals:
                value=oracle.evaluate(option['orders'],'future')
                if value is not None:grouped[head_index].append(value)
            for i,head in enumerate(heads):
                values=grouped[i]
                if values:
                    head['beam']=values
                    head['shadow_score']=min(head['shadow_score'],min(v['real']['makespan'] for v in values))
            levels+=1
        winner=min(heads,key=lambda h:(h['shadow_score'],h['value']['real']['makespan'],order_key(h['value']['orders'])))
        kept_current=order_key(current['orders'])==order_key(winner['value']['orders'])
        before=current['real']['makespan'];current=winner['value']
        decisions.append({'root':root,'cut':cut,'locked_tasks':sorted(locked),'future':future,
            'closure':sorted(closure),'completed_future_levels':levels,'budget_shortened':budget_shortened,
            'before':before,'committed':current['real']['makespan'],'shadow_value':winner['shadow_score'],
            'kept_current':kept_current,
            'chosen_core':winner['option']['core'],'chosen_slot':winner['option']['slot'],**note})
        if oracle.stats['evaluations']>=oracle.max_evals:break
    ranked=sorted(archive.values(),key=lambda r:(r['real']['makespan'],order_key(r['orders'])))
    return ranked,{'depth':depth,'future_policy':future_policy,'cut_policy':cut_policy,'keep_current':keep_current,'decisions':decisions,'oracle':dict(oracle.stats),
                   'archive_size':len(ranked),'history':oracle.history,
                   'scope':'归档仅含当前任务单步完整计划；未来叶子只提供前瞻价值；全图资源从头重放'}
