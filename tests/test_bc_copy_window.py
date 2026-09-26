"""B/C关键搬运前移：真实展开图、重建核验与跨流水线合法性。"""
import copy
import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'solver'))


def example():
    from scenario_contract import CODE
    from evaluation_validation import read_evaluation_config
    from multicore_cut_evaluate_problem_2 import evaluate_scene_b
    config=read_evaluation_config(str(CODE.parent/'data/config.txt'))
    graph=dict(ops=[dict(id=i,op='ADD',pipe='PIPE_V',cycles=d) for i,d in enumerate([10,100,10,100])]+[dict(id=4,op='COPY_OUT',pipe='PIPE_MTE3',cycles=1)],
        tensors=[dict(id=t,pos='UB',size=60) for t in (10,11)],
        edges=[dict(source=0,target=10),dict(source=10,target=1),dict(source=3,target=11),dict(source=11,target=4)])
    plan=dict(node_to_subgraph={str(i):i for i in range(4)},core_schedules=[[3,0],[2,1]])
    raw=evaluate_scene_b(graph,plan,60,config['capacity'],cross_core_copy_delay=500)
    return graph,plan,raw


def test_observed_expanded_graph_matches_every_operation():
    from bc_observed_graph import observe_parent
    g,p,r=example();state=observe_parent(g,p,r)
    assert state['replay']['makespan']==r['makespan']
    assert all(state['replay']['starts'][k]==v['start'] for k,v in state['observed'].items())
    bad=copy.deepcopy(r);bad['makespan']+=1
    with pytest.raises(ValueError,match='重建'):observe_parent(g,p,bad)


def test_critical_copy_frontload_improves_remote_release():
    from bc_copy_window import generate_critical_copy_candidates
    from multicore_cut_evaluate_problem_2 import evaluate_scene_b
    g,p,r=example();saved=copy.deepcopy((g,p,r))
    rows,audit=generate_critical_copy_candidates(g,p,r,max_candidates=3)
    assert rows and audit['legal_expanded']>=1
    assert any(row['plan']['core_schedules'][0]==[0,3] for row in rows)
    results=[evaluate_scene_b(g,row['plan'],60,r['capacity_bytes'],cross_core_copy_delay=500)['makespan'] for row in rows]
    assert min(results)<r['makespan']
    assert all(row['plan']['node_to_subgraph']==p['node_to_subgraph'] for row in rows)
    assert (g,p,r)==saved


def test_frontload_dependency_closure_moves_required_producer_together():
    from bc_copy_window import generate_critical_copy_candidates
    from multicore_cut_evaluate_problem_2 import evaluate_scene_b
    g,p,r=example();g['edges'].append(dict(source=2,target=0))
    p['core_schedules']=[[3,2,0],[1]]
    r=evaluate_scene_b(g,p,60,r['capacity_bytes'],cross_core_copy_delay=500)
    rows,audit=generate_critical_copy_candidates(g,p,r,max_candidates=3,dependency_closure=True)
    assert any(row['plan']['core_schedules'][0]==[2,0,3] for row in rows)
    assert any(len(row['source']['moved_subgraphs'])==2 for row in rows)
    assert min(evaluate_scene_b(g,row['plan'],60,r['capacity_bytes'],cross_core_copy_delay=500)['makespan'] for row in rows)<r['makespan']
