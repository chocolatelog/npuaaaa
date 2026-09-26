"""A任务屏障、同核等待和共享DDR必须同时重建，不能复用B/C激活规则。"""
from pathlib import Path
import copy
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'solver'))


def compare(graph,plan,raw,expander=None):
    from a_event_model import simulate_a_plan
    predicted=simulate_a_plan(graph,plan,raw,expander=expander)
    truth={(o['task_id'],o['op_id']):(o['start'],o['end']) for c in raw['per_core_timeline'] for o in c['ops']}
    assert {k:(o['start'],o['end']) for k,o in predicted['operations'].items()}==truth
    assert predicted['task_times']=={o['task_id']:(o['start'],o['end']) for c in raw['per_core_timeline'] for o in c['tasks']}
    assert predicted['makespan']==raw['makespan']
    assert predicted['data_movement_bytes']==raw['data_movement_bytes']


def test_parent_and_output_boundary_split_match_official():
    from test_boundary_release import example
    from boundary_release import generate_boundary_release_candidates
    from multicore_cut_evaluate_problem_1 import evaluate_scene_a
    from a_event_model import CachedAExpansion
    graph,parent,raw=example();saved=copy.deepcopy((graph,parent,raw))
    cache=CachedAExpansion(1024*1024)
    compare(graph,parent,raw,cache)
    for row in generate_boundary_release_candidates(graph,parent,raw)[0]:
        plan=row['plan'];truth=evaluate_scene_a(graph,plan,60,raw['capacity_bytes'],1000,100)
        compare(graph,plan,truth,cache)
    assert (graph,parent,raw)==saved


def test_simultaneous_ddr_reads_and_unrelated_same_core_wait():
    from scenario_contract import CODE
    from evaluation_validation import read_evaluation_config
    from multicore_cut_evaluate_problem_1 import evaluate_scene_a
    config=read_evaluation_config(str(CODE.parent/'data/config.txt'))
    graph=dict(ops=[dict(id=i,op='ADD',pipe='PIPE_V',cycles=2) for i in range(4)],
        tensors=[dict(id=10+i,pos='UB',size=s) for i,s in enumerate((61,121,181,60))],
        edges=[dict(source=10+i,target=i) for i in range(4)])
    parent=dict(node_to_subgraph={str(i):i for i in range(4)},core_schedules=[[0,2],[1,3]])
    raw=evaluate_scene_a(graph,parent,60,config['capacity'],17,7)
    compare(graph,parent,raw)
    core=raw['per_core_timeline'][0]['tasks']
    assert core[1]['start']-core[0]['end']==7
    from a_event_model import simulate_a_plan
    with pytest.raises(ValueError,match='预算'):
        simulate_a_plan(graph,parent,raw,max_events=1)
