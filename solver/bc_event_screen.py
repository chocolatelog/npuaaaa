"""已验证闭环模型的可选精筛；候选生成、预测、原官方裁决分别负责。"""
import time
from bc_event_model import simulate_bc_plan
from bc_copy_window import generate_critical_copy_candidates
from stage_cache import CachedSceneExpansion


def verify_parent_prediction(parent_prediction,raw,scene):
    """绑定逐操作与缓存观测，不能只用总时间相同判定校准通过。"""
    truth={(c['core_id'],o['op_id']):(o['start'],o['end']) for c in raw['per_core_timeline'] for o in c['ops']}
    prediction={k:(o['start'],o['end']) for k,o in parent_prediction['operations'].items()}
    if parent_prediction['makespan']!=raw['makespan'] or prediction!=truth:
        raise ValueError('闭环模型父方案全起止核验失败')
    if scene=='C' and any(parent_prediction[k]!=raw[k] for k in ('cache_events','cache_stats','cache_final_entries','cache_used_bytes_final')):
        raise ValueError('闭环模型父方案缓存核验失败')


def generate_event_screened_candidates(graph,parent,raw,scene,*,max_expanded=24,max_candidates=6,cache_mib=128):
    if scene not in ('B','C'):raise ValueError('闭环精筛仅支持B/C')
    if type(max_expanded) is not int or not 6<=max_expanded<=48:raise ValueError('闭环展开预算必须6～48')
    if type(cache_mib) is not int or not 0<=cache_mib<=128:raise ValueError('阶段缓存预算必须0～128MiB')
    started=time.perf_counter()
    hardware={k:raw[k] for k in ('bandwidth_bytes_per_cycle','capacity_bytes','cross_core_copy_delay_cycles')}
    if scene=='C':hardware.update({k:raw[k] for k in ('cache_capacity_bytes','cache_bandwidth_bytes_per_cycle')})
    expander=CachedSceneExpansion(cache_mib*1024*1024)
    parent_prediction=simulate_bc_plan(graph,parent,hardware,scene,expander=expander)
    verify_parent_prediction(parent_prediction,raw,scene)
    parent_seconds=time.perf_counter()-started
    def score(plan,tasks,links):
        # 复用本候选已完成合法性检查的展开图，事件模型只读访问。
        result=simulate_bc_plan(graph,plan,hardware,scene,expander=lambda *_:(tasks,links))
        return dict(closed_loop_makespan=result['makespan'],closed_loop_event_seconds=result['event_seconds'],
            closed_loop_events=result['events'],closed_loop_cache_stats=result['cache_stats'],
            scope='候选自主闭环预测，非官方结果；原官方决定是否采纳')
    rows,generation=generate_critical_copy_candidates(graph,parent,raw,max_candidates=max_candidates,
        max_expanded=max_expanded,dependency_closure=True,expander=expander,candidate_scorer=score)
    improved=[r for r in rows if r['source']['closed_loop_makespan']<raw['makespan']]
    generation.update(parent_verified=True,parent_verification_seconds=parent_seconds,
        selected_before_improvement_filter=len(rows),predicted_improving_candidates=len(improved),
        candidates=len(improved),cache_stats=dict(expander.memo.stats),
        resident_serialized_bytes=expander.memo.resident_bytes,cache_budget_mib=cache_mib,
        elapsed_seconds=time.perf_counter()-started)
    return improved,generation
