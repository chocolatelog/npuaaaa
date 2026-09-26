"""输出祖先闭包拆分：用原官方验证可提前远核释放，保持原算子分核。"""
import copy
from pathlib import Path
import sys
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'solver'))
from boundary_release import generate_boundary_release_candidates


def example():
    from scenario_contract import CODE
    from evaluation_validation import read_evaluation_config
    from multicore_cut_evaluate_problem_1 import evaluate_scene_a
    config=read_evaluation_config(str(CODE.parent/'data/config.txt'))
    graph=dict(ops=[dict(id=i,op='ADD',pipe='PIPE_V',cycles=d) for i,d in enumerate([50,50,5000,5000])],
               tensors=[dict(id=10,pos='UB',size=60)],
               edges=[dict(source=0,target=1),dict(source=1,target=2),dict(source=1,target=10),dict(source=10,target=3)])
    plan=dict(node_to_subgraph={'0':0,'1':0,'2':0,'3':1},core_schedules=[[0],[1]])
    raw=evaluate_scene_a(graph,plan,60,config['capacity'],cross_core_wait=1000,same_core_wait=100)
    return graph,plan,raw


def test_ancestor_cut_releases_remote_work_and_keeps_cores():
    from multicore_cut_evaluate_problem_1 import evaluate_scene_a
    from scenario_contract import validate_plan
    g,p,r=example();before=copy.deepcopy((g,p,r))
    rows,audit=generate_boundary_release_candidates(g,p,r)
    assert rows and audit['hotspots']>0
    results=[]
    for row in rows:
        child=row['plan'];validate_plan(g,child,'A')
        core={group:c for c,order in enumerate(child['core_schedules']) for group in order}
        assert all(core[group]==(1 if op=='3' else 0) for op,group in child['node_to_subgraph'].items())
        assert child['node_to_subgraph']['0']==child['node_to_subgraph']['1']
        assert child['node_to_subgraph']['2']!=child['node_to_subgraph']['1']
        results.append(evaluate_scene_a(g,child,60,r['capacity_bytes'],cross_core_wait=1000,same_core_wait=100)['makespan'])
    assert min(results)<r['makespan']
    assert (g,p,r)==before
    assert generate_boundary_release_candidates(g,p,r)[0]==rows


def test_mismatched_parent_trace_is_rejected():
    g,p,r=example();r['makespan']+=1
    with pytest.raises(ValueError,match='重建'):
        generate_boundary_release_candidates(g,p,r)
