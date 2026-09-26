"""事件锚点缓存协调：等待首次入缓存、合法依赖闭包、原官方总时间裁决。"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'solver'))


def test_delay_duplicate_read_until_first_copy_completes():
    from cache_event_window import cache_event_targets,generate_cache_event_candidates
    from scenario_contract import CODE
    from evaluation_validation import read_evaluation_config
    from multicore_cut_evaluate_problem_3 import evaluate_problem_3
    config=read_evaluation_config(str(CODE.parent/'data/config.txt'))
    graph=dict(ops=[dict(id=i,op='ADD',pipe='PIPE_V',cycles=1) for i in range(3)],
        tensors=[dict(id=10,pos='UB',size=600),dict(id=20,pos='UB',size=1200)],
        edges=[dict(source=10,target=0),dict(source=10,target=1),dict(source=20,target=2)])
    parent=dict(node_to_subgraph={str(i):i for i in range(3)},core_schedules=[[0],[1,2]])
    def evaluate(plan):return evaluate_problem_3(graph,plan,60,config['capacity'],500,10000,600)
    raw=evaluate(parent)
    targets=cache_event_targets(raw)
    assert any(t['subgraph']==1 and t['reason']=='inflight_duplicate' and t['target_time']>t['time'] for t in targets)
    rows,audit=generate_cache_event_candidates(graph,parent,raw,max_expanded=6)
    assert rows and audit['parent_verified']
    best=min(rows,key=lambda r:r['source']['closed_loop_makespan'])
    official=evaluate(best['plan'])
    assert official['makespan']<raw['makespan']
    assert official['makespan']==best['source']['closed_loop_makespan']
    assert official['cache_stats']['copy_in_hits']>raw['cache_stats']['copy_in_hits']


def test_backward_and_forward_closures_preserve_required_local_order():
    from cache_event_window import move_with_local_closure
    parent=dict(node_to_subgraph={str(i):i for i in range(4)},core_schedules=[[0,1,2,3]])
    preds={0:set(),1:{0},2:set(),3:{1}}
    succ={0:{1},1:{3},2:set(),3:set()}
    advanced,moved=move_with_local_closure(parent,0,3,0,preds,succ)
    assert moved==[0,1,3] and advanced['core_schedules']==[[0,1,3,2]]
    delayed,moved=move_with_local_closure(parent,0,0,2,preds,succ)
    assert moved==[0,1] and delayed['core_schedules']==[[2,0,1,3]]
    assert parent['core_schedules']==[[0,1,2,3]]


def test_eviction_target_uses_actual_eviction_not_merely_past_insert():
    from cache_event_window import cache_event_targets
    raw=dict(cache_capacity_bytes=100,bandwidth_bytes_per_cycle=10,cache_bandwidth_bytes_per_cycle=20,
        per_core_timeline=[dict(core_id=0,ops=[dict(op_id=1,subgraph_id=1,start=0,end=6),dict(op_id=2,subgraph_id=2,start=20,end=26)])],
        cache_events=[dict(event='miss',time=0,tensor_id=10,size_bytes=60,core_id=0,op_id=1),
            dict(event='insert',time=6,tensor_id=10,evicted_tensor_ids=[]),
            dict(event='insert',time=12,tensor_id=20,evicted_tensor_ids=[10]),
            dict(event='miss',time=20,tensor_id=10,size_bytes=60,core_id=0,op_id=2)])
    rows=cache_event_targets(raw)
    assert len(rows)==1 and rows[0]['reason']=='evicted' and rows[0]['target_time']==12
    raw['cache_events'][2]['evicted_tensor_ids']=[]
    assert cache_event_targets(raw)==[]
