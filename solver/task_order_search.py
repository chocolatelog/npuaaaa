"""固定子图切分，在完整依赖约束下搜索迁核与核内插入位置。"""
import heapq
import math
import time


def replay_tasks(durations, preds, orders, same_wait=100, cross_wait=1000,
                 bandwidth_floor=0):
    """合并数据依赖与每核顺序；非法覆盖或成环返回空值。"""
    nodes = set(durations)
    flat = [s for order in orders for s in order]
    if not orders or len(flat) != len(nodes) or set(flat) != nodes:
        return None
    if any(not math.isfinite(d) or d < 0 for d in durations.values()):
        return None
    core = {s: c for c, order in enumerate(orders) for s in order}
    succ = {s: {} for s in nodes}
    for s in nodes:
        for p in preds.get(s, ()):
            if p not in nodes:
                return None
            succ[p][s] = cross_wait if core[p] != core[s] else 0
    for order in orders:
        for p, s in zip(order, order[1:]):
            succ[p][s] = max(same_wait, succ[p].get(s, 0))
    indegree = dict.fromkeys(nodes, 0)
    for children in succ.values():
        for s in children:
            indegree[s] += 1
    ready = [s for s in nodes if indegree[s] == 0]
    heapq.heapify(ready)
    start = dict.fromkeys(nodes, 0.0)
    finish, parent = {}, {}
    while ready:
        s = heapq.heappop(ready)
        finish[s] = start[s] + durations[s]
        for child, wait in succ[s].items():
            release = finish[s] + wait
            if release > start[child]:
                start[child], parent[child] = release, s
            indegree[child] -= 1
            if indegree[child] == 0:
                heapq.heappush(ready, child)
    if len(finish) != len(nodes):
        return None
    critical = []
    if finish:
        s = max(sorted(finish), key=finish.get)
        while True:
            critical.append(s)
            if s not in parent:
                break
            s = parent[s]
    task_makespan = max(finish.values(), default=0)
    return {'makespan': max(task_makespan, bandwidth_floor),
            'task_makespan': task_makespan, 'critical': critical,
            'orders': [list(order) for order in orders], 'finish': finish}


def search_orders(durations, preds, orders, same_wait=100, cross_wait=1000,
                  bandwidth_floor=0, beam_width=4, rounds=3, max_evals=4000,
                  seconds=2.0, max_sources=12, keep=2, enable_swaps=False):
    if enable_swaps:
        return _search_with_swaps(durations, preds, orders, same_wait, cross_wait,
                                  bandwidth_floor, beam_width, rounds, max_evals,
                                  seconds, max_sources, keep)
    started = time.perf_counter()
    def evaluate(candidate):
        return replay_tasks(durations, preds, candidate, same_wait, cross_wait,
                            bandwidth_floor)
    def signature(candidate):
        return tuple(tuple(o) for o in candidate)
    def score(state):
        return state['makespan'], state['task_makespan'], signature(state['orders'])
    initial = evaluate(orders)
    if initial is None:
        raise ValueError('输入方案覆盖不完整或数据依赖与核内顺序成环')
    base_sig = signature(orders)
    seen = {base_sig}
    beam, alternatives = [initial], {}
    stats = {'evaluations': 1, 'valid': 1, 'invalid': 0, 'depth': 0,
             'base_makespan': initial['makespan'], 'max_evals': max_evals}
    def exhausted():
        return stats['evaluations'] >= max_evals or (seconds is not None and time.perf_counter() - started >= seconds)
    for depth in range(rounds):
        proposals = list(beam)
        for state in beam:
            sources = sorted(state['critical'], key=lambda s: (-durations[s], s))[:max_sources]
            for task in sources:
                removed = [[s for s in order if s != task] for order in state['orders']]
                for c, order in enumerate(removed):
                    for index in range(len(order) + 1):
                        if exhausted():
                            break
                        candidate = [list(o) for o in removed]
                        candidate[c].insert(index, task)
                        sig = signature(candidate)
                        if sig in seen:
                            continue
                        seen.add(sig)
                        result = evaluate(candidate)
                        stats['evaluations'] += 1
                        if result is None:
                            stats['invalid'] += 1
                            continue
                        stats['valid'] += 1
                        proposals.append(result)
                        alternatives[sig] = result
                    if exhausted():
                        break
                if exhausted():
                    break
            if exhausted():
                break
        # 父状态保留：单层未改进也不会使束内最优变差。
        unique = {signature(s['orders']): s for s in proposals}
        beam = sorted(unique.values(), key=score)[:beam_width]
        stats['depth'] = depth + 1
        if exhausted():
            break
    ranked = sorted((s for s in alternatives.values()
                     if (s['makespan'], s['task_makespan']) <
                        (initial['makespan'], initial['task_makespan'])), key=score)
    stats['seconds'] = time.perf_counter() - started
    stats['budget_exhausted'] = exhausted()
    stats['alternatives'] = len(alternatives)
    stats['predicted_improvements'] = len(ranked)
    return ranked[:keep], stats


def _search_with_swaps(durations, preds, orders, same_wait, cross_wait,
                       bandwidth_floor, beam_width, rounds, max_evals,
                       seconds, max_sources, keep):
    """迁移和交换交错供候选，避免迁移先耗尽共享预算。

    小于等于二十任务枚举全部跨核对；更大方案只展开关键路径及
    最晚结束核心上的长任务，每个源最多十二个负载互补伙伴。
    双方原位置同时替换；完整依赖与核内顺序联合重放裁定合法性。
    """
    started=time.perf_counter()
    if min(beam_width,rounds,max_evals,max_sources,keep)<1:
        raise ValueError('搜索预算必须为正')
    def evaluate(candidate):
        return replay_tasks(durations,preds,candidate,same_wait,cross_wait,bandwidth_floor)
    def sig(candidate):
        return tuple(tuple(o) for o in candidate)
    def score(state):
        return state['makespan'],state['task_makespan'],sig(state['orders'])
    initial=evaluate(orders)
    if initial is None:
        raise ValueError('输入方案覆盖不完整或数据依赖与核内顺序成环')
    stats={'evaluations':1,'valid':1,'invalid':0,'depth':0,'base_makespan':initial['makespan'],
           'max_evals':max_evals,'by_neighborhood':{k:{'evaluations':0,'valid':0,'improvements':0}
                                                  for k in ('migration','swap')}}
    def exhausted():
        return stats['evaluations']>=max_evals or (seconds is not None and time.perf_counter()-started>=seconds)
    def migrations(state):
        for task in sorted(state['critical'],key=lambda s:(-durations[s],s))[:max_sources]:
            removed=[[s for s in order if s!=task] for order in state['orders']]
            for c,order in enumerate(removed):
                for i in range(len(order)+1):
                    child=[list(o) for o in removed];child[c].insert(i,task)
                    yield child
    def swaps(state):
        positions={s:(c,i) for c,order in enumerate(state['orders']) for i,s in enumerate(order)}
        loads=[sum(durations[s] for s in order)+max(0,len(order)-1)*same_wait for order in state['orders']]
        if len(durations)<=20:
            sources=sorted(durations)
        else:
            last=max(range(len(orders)),key=lambda c:max((state['finish'][s] for s in state['orders'][c]),default=0))
            source_set=set(state['critical'])|set(state['orders'][last])
            sources=sorted(source_set,key=lambda s:(-durations[s],s))[:max_sources]
        emitted=set()
        for a in sources:
            ca,ia=positions[a]
            partners=[b for b in sorted(durations) if positions[b][0]!=ca]
            if len(durations)>20:
                def balance(b):
                    cb=positions[b][0]
                    changed=list(loads)
                    changed[ca]+=durations[b]-durations[a]
                    changed[cb]+=durations[a]-durations[b]
                    return max(changed),b
                partners=sorted(partners,key=balance)[:12]
            for b in partners:
                pair=tuple(sorted((a,b)))
                if pair in emitted:
                    continue
                emitted.add(pair)
                cb,ib=positions[b];child=[list(o) for o in state['orders']]
                child[ca][ia],child[cb][ib]=b,a
                yield child
    seen={sig(orders)};beam=[initial];alternatives={}
    # 尚未加入窗口邻域；其预留额度释放给迁移与交换，按 4:3 交错。
    cadence=('swap','migration','migration','swap','migration','swap','migration')
    for depth in range(rounds):
        proposals=list(beam)
        for state in beam:
            generators={'migration':iter(migrations(state)),'swap':iter(swaps(state))}
            active=set(generators)
            while active and not exhausted():
                for kind in cadence:
                    if kind not in active or exhausted():
                        continue
                    while True:
                        try:child=next(generators[kind])
                        except StopIteration:
                            active.discard(kind);break
                        key=sig(child)
                        if key in seen:
                            continue
                        seen.add(key);result=evaluate(child)
                        stats['evaluations']+=1;stats['by_neighborhood'][kind]['evaluations']+=1
                        if result is None:
                            stats['invalid']+=1
                        else:
                            stats['valid']+=1;stats['by_neighborhood'][kind]['valid']+=1
                            result['origin']=kind
                            if score(result)[:2]<score(initial)[:2]:
                                stats['by_neighborhood'][kind]['improvements']+=1
                            proposals.append(result);alternatives[key]=result
                        break
            if exhausted():
                break
        beam=sorted({sig(s['orders']):s for s in proposals}.values(),key=score)[:beam_width]
        stats['depth']=depth+1
        if exhausted():
            break
    ranked=sorted((s for s in alternatives.values() if score(s)[:2]<score(initial)[:2]),key=score)
    stats.update(seconds=time.perf_counter()-started,budget_exhausted=exhausted(),
                 alternatives=len(alternatives),predicted_improvements=len(ranked))
    return ranked[:keep],stats


def refine_plan_orders(graph, plan, baseline, official_evaluator, enable_swaps=False,
                       search_seconds=2.0, max_evals=4000, candidate_evaluator=None,
                       rerank_keep=8):
    """对合法官方基线做固定切分后处理；回调提供官方评估。"""
    from scene_a_event import SceneAEventModel, derive_multicore_plan
    started = time.perf_counter()
    estimate = SceneAEventModel(graph).evaluate(plan)
    model_seconds = time.perf_counter() - started
    view = derive_multicore_plan(graph, plan)
    durations = {s: data['duration'] for s, data in estimate['tasks'].items()}
    candidates, stats = search_orders(durations, view['subgraph_preds'],
                                      plan['core_schedules'],
                                      bandwidth_floor=estimate['total_copy_bytes'] / 60,
                                      enable_swaps=enable_swaps,seconds=search_seconds,max_evals=max_evals,
                                      keep=rerank_keep if candidate_evaluator is not None else 2)
    if abs(stats['base_makespan'] - estimate['makespan']) > 1e-6:
        raise ValueError('轻量任务重放与同方案精评不一致')
    selected, best, records, errors = plan, baseline, [], []
    screened=[]
    if candidate_evaluator is not None:
        from run_all import valid_official
        for candidate in candidates:
            proposed={**plan,'core_schedules':candidate['orders']}
            truth=candidate_evaluator(graph,proposed,'A')
            screened.append({'estimate':candidate['makespan'],'orders':candidate['orders'],
                             'origin':candidate.get('origin','migration'),'fast':truth})
            if not valid_official(truth):
                errors.append({'stage':'replica','result':truth})
                continue
            if any(truth.get(k)!=baseline.get(k) for k in ('added_copy_bytes','scheduled_copy_bytes','spill_added')):
                errors.append({'stage':'replica_traffic_invariant','result':truth})
                continue
            candidate['screened_real']=truth
        candidates=sorted((c for c in candidates if 'screened_real' in c),
                          key=lambda c:(c['screened_real']['makespan'],c['screened_real']['added_copy_bytes'],
                                        tuple(tuple(o) for o in c['orders'])))[:2]
    for candidate in candidates:
        proposed = {**plan, 'core_schedules': candidate['orders']}
        truth = official_evaluator(graph, proposed, 'A')
        record = {'estimate': candidate['makespan'], 'official': truth,
                  'orders': candidate['orders'], 'origin': candidate.get('origin','migration')}
        records.append(record)
        if truth is None or truth.get('error'):
            errors.append({'stage': 'official', 'result': truth})
            continue
        if 'screened_real' in candidate and truth != candidate['screened_real']:
            errors.append({'stage':'replica_mismatch','original':truth,'replica':candidate['screened_real']})
            continue
        changed = [key for key in ('added_copy_bytes', 'scheduled_copy_bytes', 'spill_added')
                   if truth[key] != baseline[key]]
        if changed:
            errors.append({'stage': 'traffic_invariant', 'fields': changed})
            continue
        if (truth['makespan'], truth['added_copy_bytes']) < (best['makespan'], best['added_copy_bytes']):
            selected, best = proposed, truth
    audit = {'baseline_official': baseline, 'final_official': best,
             'search': stats, 'model_seconds': model_seconds, 'errors': errors,
             'candidates': records, 'official_count': len(records),
             'screened_candidates':screened,'fast_count':len(screened),
             'extra_seconds': time.perf_counter() - started}
    return selected, best, audit
