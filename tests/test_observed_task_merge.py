"""观测时间回并：输入/输出等待保护、边界字节与不重叠匹配。"""
import copy
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'solver'))


def fixture():
    graph=dict(ops=[dict(id=i,op='ADD',pipe='PIPE_V',cycles=10) for i in range(3)],tensors=[dict(id=10,pos='UB',size=60)],edges=[dict(source=0,target=10),dict(source=10,target=1)])
    plan=dict(node_to_subgraph={str(i):i for i in range(3)},core_schedules=[[0,1],[2]])
    raw=dict(makespan=120,task_same_core_wait_cycles=100,task_cross_core_wait_cycles=1000,bandwidth_bytes_per_cycle=60,
             task_dependencies=[dict(source=0,target=1)],per_core_timeline=[dict(core_id=0,tasks=[dict(task_id=0,subgraph_id=0,start=0,end=10,duration=10),dict(task_id=1,subgraph_id=1,start=110,end=120,duration=10)]),dict(core_id=1,tasks=[dict(task_id=2,subgraph_id=2,start=0,end=10,duration=10)])])
    return graph,plan,raw


def test_pair_matching_is_weighted_and_nonoverlapping():
    from observed_task_merge import path_matching
    assert path_matching([0,1,2,3],{(0,1):2,(1,2):10,(2,3):3})==[(1,2)]
    assert path_matching([0,1,2,3],{(0,1):8,(1,2):10,(2,3):8})==[(0,1),(2,3)]


def test_merge_keeps_mapping_core_and_counts_internal_copy_saving():
    from observed_task_merge import generate_observed_merge_candidates
    from scenario_contract import validate_plan
    from task_resource_profiles import task_resource_profiles
    g,p,raw=fixture();before=copy.deepcopy((g,p,raw))
    rows,audit=generate_observed_merge_candidates(g,p,raw)
    assert len(rows)==1 and audit['eligible_pairs']==1
    out=rows[0]['plan'];validate_plan(g,out,'A')
    assert out['core_schedules']==[[0],[2]]
    assert out['node_to_subgraph']=={'0':0,'1':0,'2':2}
    old=task_resource_profiles(g,p['node_to_subgraph'],60);new=task_resource_profiles(g,out['node_to_subgraph'],60)
    difference=sum(r['input_bytes']+r['output_bytes'] for r in old.values())-sum(r['input_bytes']+r['output_bytes'] for r in new.values())
    assert difference==120==rows[0]['source']['boundary_bytes_saved_estimate']
    assert (g,p,raw)==before
    assert rows==generate_observed_merge_candidates(g,p,raw)[0]


def test_merge_rejects_delaying_an_external_consumer():
    from observed_task_merge import generate_observed_merge_candidates
    g,p,r=fixture();g['edges'].append(dict(source=0,target=2));r['task_dependencies'].append(dict(source=0,target=2))
    r['per_core_timeline'][0]['tasks'][1].update(end=2010,duration=1900)
    r['per_core_timeline'][1]['tasks'][0].update(start=1010,end=1020)
    r['makespan']=2010
    rows,audit=generate_observed_merge_candidates(g,p,r)
    assert not rows and audit['rejections']['delayed_output']==1


def test_merge_rejects_late_extra_input():
    from observed_task_merge import generate_observed_merge_candidates
    g,p,r=fixture();g['edges'].append(dict(source=2,target=1));r['task_dependencies'].append(dict(source=2,target=1))
    r['per_core_timeline'][0]['tasks'][1].update(start=1010,end=1020)
    r['makespan']=1020
    rows,audit=generate_observed_merge_candidates(g,p,r)
    assert not rows and audit['rejections']['new_input_wait']==1
