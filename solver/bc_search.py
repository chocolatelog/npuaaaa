"""问题二/三的固定状态预算束搜索。

搜索只生成完整合法方案，代理值只负责排序；调用方仍须把返回候选逐一交给
原官方评测器，并保留父方案作为保底。模块不持有官方可变状态，适合并行消融。
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from copy import deepcopy

from bc_refine import generate_bc_candidates, validate_plan


def balanced_candidates(graph, plan, scene, max_candidates=17):
    """分别保留拆分、换序、迁核、合并名额，避免先生成的家族占满。"""
    from bc_refine import _group_state, _with_split, _with_swap, _metrics
    state = _group_state(graph, plan)
    groups = sorted(state['members'], key=lambda s: (-state['pressure'].get(s,0), s))[:8]
    families = {k:[] for k in ('split','swap','move','merge')}
    seen={plan_id(plan)}
    def add(family, candidate):
        if candidate is None or len(families[family])>=4:return
        key=plan_id(candidate)
        if key not in seen and validate_plan(graph,candidate):
            seen.add(key);families[family].append(candidate)
    for sg in groups:
        members=state['members'][sg]
        for cut in sorted({len(members)//2, len(members)//4, 3*len(members)//4}):
            add('split',_with_split(state,sg,cut))
    for core,row in enumerate(state['schedules']):
        ranked=sorted(range(len(row)-1),key=lambda i: -state['pressure'].get(row[i],0))
        for i in ranked[:12]:
            add('swap',_with_swap(state,core,i))
            candidate=deepcopy(plan);a,b=row[i:i+2]
            candidate['node_to_subgraph']={k:(a if v==b else v) for k,v in candidate['node_to_subgraph'].items()}
            candidate['core_schedules'][core].remove(b)
            add('merge',candidate)
    for sg in groups:
        source=state['core_of'][sg]
        for target in sorted(range(len(state['schedules'])),key=lambda c:(state['load'][c],c)):
            if target==source:continue
            count=len(state['schedules'][target])
            for at in sorted({0,count//2,count}):
                candidate=deepcopy(plan);candidate['core_schedules'][source].remove(sg)
                candidate['core_schedules'][target].insert(at,sg);add('move',candidate)
    plans=[deepcopy(plan)];metrics=[]
    for i in range(4):
        for family,rows in families.items():
            if i<len(rows) and len(plans)<max_candidates:
                p=rows[i];plans.append(p)
                metrics.append({'plan_id':plan_id(p),'action':family,**_metrics(graph,p,scene)})
    return plans, {'candidate_metrics':metrics,'family_counts':{k:len(v) for k,v in families.items()}}


def _cache_replay_metrics(graph, plan):
    """把事件级 FIFO 结果转成候选排序的诊断量。"""
    from cache_replay import replay_fifo_cache
    from resource_state_bc import build_resource_table

    table = build_resource_table(graph, plan, 'C')
    events = table.get('events', ())
    replay = replay_fifo_cache(events, capacity=1048576,
                               ddr_bandwidth=60.0, cache_bandwidth=250.0)
    return {
        'replay_cache_hit_bytes': replay['hit_bytes'],
        'replay_cache_miss_bytes': replay['miss_bytes'],
        'replay_cache_hit_rate': replay['hit_rate'],
        'replay_cache_evictions': replay['evictions'],
        'replay_cache_events': len(replay['events']),
        'replay_mode': replay['mode'],
    }


def plan_id(plan):
    payload = {'node_to_subgraph': plan['node_to_subgraph'],
               'core_schedules': plan['core_schedules']}
    return hashlib.sha256(json.dumps(payload, sort_keys=True,
                                     separators=(',', ':')).encode()).hexdigest()


def _proxy_key(row, scene):
    metrics = row.get('metrics', {})
    invalid = metrics.get('proxy_valid') is False
    lower = float(metrics.get('proxy_lower_bound', 0.0))
    boundary = float(metrics.get('proxy_boundary_cycles', 0.0))
    imbalance = float(metrics.get('load_imbalance', 0.0))
    if scene == 'C':
        # 只作为同层排序信号，不能把命中率当作总时间。
        reuse = float(metrics.get('cache_reuse_bytes', 0.0))
        replay_hits = float(metrics.get('replay_cache_hit_bytes', 0.0))
        replay_evictions = float(metrics.get('replay_cache_evictions', 0.0))
        return (invalid, lower + boundary + imbalance, -replay_hits,
                replay_evictions, -reuse, row['plan_id'])
    return (invalid, lower + boundary + imbalance, row['plan_id'])


def beam_search_candidates(graph, parent_plan, scene, depth=2, width=4,
                           max_states=64, max_candidates=6, seed=0, balanced=False):
    """在 ``max_states`` 固定预算内生成候选，第一项永远是父方案。"""
    if scene not in {'B', 'C'}:
        raise ValueError('束搜索只支持 B/C')
    if depth < 0 or width < 1 or max_states < 1 or max_candidates < 1:
        raise ValueError('束搜索参数必须为正')
    if not validate_plan(graph, parent_plan):
        raise ValueError('父方案不合法')
    parent_id = plan_id(parent_plan)
    states = [{'plan': parent_plan, 'plan_id': parent_id, 'depth': 0,
               'parent_plan_id': None, 'action': 'parent', 'metrics': {}}]
    seen = {parent_id}
    archive = {}
    attempted = 0
    expanded = 0
    levels = []
    # 候选生成器的动作是确定性的；seed 只写入审计，避免伪随机改变合法性。
    for current_depth in range(depth):
        next_rows = []
        for row in states:
            if expanded >= max_states:
                break
            remaining = max(1, min(17, max_states - expanded + 1))
            generator = balanced_candidates if balanced else generate_bc_candidates
            candidates, audit = generator(
                graph, row['plan'], scene, max_candidates=remaining)
            metrics_by_id = {item.get('plan_id'): item
                             for item in audit.get('candidate_metrics', ())}
            for candidate in candidates[1:]:
                attempted += 1
                candidate_id = plan_id(candidate)
                if candidate_id in seen or not validate_plan(graph, candidate):
                    continue
                if expanded >= max_states:
                    break
                seen.add(candidate_id)
                expanded += 1
                action = next((item.get('action') for item in audit.get(
                    'candidate_metrics', ()) if item.get('plan_id') == candidate_id),
                              'unknown')
                candidate_metrics = dict(metrics_by_id.get(candidate_id, {}))
                if scene == 'C':
                    try:
                        candidate_metrics.update(_cache_replay_metrics(
                            graph, candidate))
                    except (KeyError, TypeError, ValueError):
                        candidate_metrics['replay_mode'] = 'unavailable'
                next_rows.append({
                    'plan': candidate, 'plan_id': candidate_id,
                    'depth': current_depth + 1,
                    'parent_plan_id': row['plan_id'], 'action': action,
                    'metrics': candidate_metrics,
                })
                archive[candidate_id] = next_rows[-1]
        next_rows.sort(key=lambda item: _proxy_key(item, scene))
        next_rows = next_rows[:width]
        levels.append({'depth': current_depth + 1,
                       'generated': len(next_rows),
                       'expanded_states': expanded})
        if not next_rows:
            break
        states = next_rows
    selected = [{'plan': parent_plan, 'plan_id': parent_id, 'depth': 0,
                 'parent_plan_id': None, 'action': 'parent', 'metrics': {}}]
    selected.extend(sorted(archive.values(), key=lambda item: _proxy_key(item, scene)))
    unique = {}
    for row in selected:
        unique.setdefault(row['plan_id'], row)
    selected = list(unique.values())[:max_candidates]
    audit = {
        'scene': scene, 'seed': int(seed), 'depth': depth, 'width': width,
        'max_states': max_states, 'expanded_states': expanded,
        'attempted_states': attempted, 'unique_states': len(seen),
        'returned': len(selected), 'parent_plan_id': parent_id,
        'levels': levels,
        'candidates': [{k: row[k] for k in
                       ('plan_id', 'parent_plan_id', 'depth', 'action', 'metrics')}
                       for row in selected],
    }
    return [row['plan'] for row in selected], audit


def refine_official(graph, parent_plan, scene, evaluator, seeds=(),
                    max_states=64, official_limit=6, balanced=True, progress=None):
    """完整原算子方案直接官方裁决；不可投影回块模型不是拒收理由。

    父方案及显式外部种子的官方复核不挤占新增候选六名额，分开报告成本。
    C 可把同批 B 方案作为可行保底，仍记录 C 独立方案与保底来源。
    """
    started=time.perf_counter();seen={};records=[];errors=[]
    n=len(parent_plan['core_schedules'])
    def evaluate(plan, source):
        key=plan_id(plan)
        if key in seen:return seen[key]
        if len(plan['core_schedules'])!=n or not validate_plan(graph,plan):
            errors.append({'plan_id':key,'source':source,'error':'方案非法'});return None
        try:
            value=evaluator(graph,plan,scene)
            if not value or value.get('error') or any(
                isinstance(value.get(k),bool) or not isinstance(value.get(k),(int,float))
                or not math.isfinite(value[k]) or value[k]<0
                for k in ('makespan','added_copy_bytes','scheduled_copy_bytes')) or value['makespan']<=0:
                raise ValueError('官方结果缺失或数值非法')
            seen[key]=value
            records.append({'plan_id':key,'source':source,'official':value})
            return value
        except Exception as exc:
            errors.append({'plan_id':key,'source':source,'error':repr(exc)});return None
    chosen=deepcopy(parent_plan);base=evaluate(chosen,'parent')
    if base is None:raise ValueError('无法获得父方案官方保底')
    best=base
    for seed in seeds:
        truth=evaluate(seed,'external_seed')
        if truth and (truth['makespan'],truth['added_copy_bytes'])<(best['makespan'],best['added_copy_bytes']):
            chosen,best=deepcopy(seed),truth
    seed_best=best
    root_id=plan_id(chosen)
    try:
        candidates,search=beam_search_candidates(graph,chosen,scene,max_states=max_states,
                                                max_candidates=official_limit+1,balanced=balanced)
    except Exception as exc:
        candidates=[];search={'error':repr(exc)}
        errors.append({'source':'candidate_generation','error':repr(exc)})
    evaluated=0
    for plan in candidates:
        if plan_id(plan) in seen:continue
        if evaluated>=official_limit:break
        evaluated+=1;truth=evaluate(plan,'beam_candidate')
        if truth and (truth['makespan'],truth['added_copy_bytes'])<(best['makespan'],best['added_copy_bytes']):
            chosen,best=deepcopy(plan),truth
        if progress:progress(evaluated,official_limit)
    selected_id=plan_id(chosen)
    for row in records:row['selected']=row['plan_id']==selected_id
    return chosen,best,{'baseline_official':base,'seed_best_official':seed_best,
                        'final_official':best,'parent_plan_id':plan_id(parent_plan),
                        'search_root_id':root_id,'selected_plan_id':selected_id,'search':search,
                        'official_records':records,'errors':errors,'candidate_official_count':evaluated,
                        'extra_seconds':time.perf_counter()-started}
