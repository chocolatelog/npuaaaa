"""场景A资源估计须匹配固定切分边界，不把跨核数当边界量。"""
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'solver'))


def graph():
    return dict(ops=[dict(id=1,op='MATMUL',pipe='PIPE_M',cycles=10),dict(id=2,op='ADD',pipe='PIPE_V',cycles=20),
                     dict(id=3,op='MATMUL',pipe='PIPE_M',cycles=5),dict(id=9,op='COPY_IN',pipe='PIPE_MTE2',cycles=20),
                     dict(id=10,op='COPY_OUT',pipe='PIPE_MTE3',cycles=10)],
                tensors=[dict(id=i,size=size,pos='UB') for i,size in ((101,1200),(102,600),(103,600),(104,30))],
                edges=[dict(source=s,target=t) for s,t in ((9,101),(101,1),(1,102),(102,2),(102,3),(2,103),(103,10),(3,104))])


def test_boundary_bytes_match_official_on_same_and_different_cores():
    from task_resource_profiles import task_resource_profiles
    from scene_a_fast import official
    g=graph();mapping={1:0,2:0,3:1};profiles=task_resource_profiles(g,mapping,60)
    assert profiles[0]['input_bytes']==1200 and profiles[0]['output_bytes']==1200
    assert profiles[1]['input_bytes']==600 and profiles[1]['output_bytes']==30
    for orders in ([[0,1],[]],[[0],[1]]):
        plan=dict(node_to_subgraph={str(k):v for k,v in mapping.items()},core_schedules=orders)
        _,_,traffic,_=official._build_scene_a_tasks(g,plan,60,{'L1':524288,'UB':131072})
        assert sum(p['input_bytes']+p['output_bytes'] for p in profiles.values())==traffic['scheduled_copy_bytes']==3030
        assert traffic['spill_added_copy_bytes']==0


def test_internal_chain_and_boundary_copy_path_are_in_duration():
    from task_resource_profiles import task_resource_profiles
    p=task_resource_profiles(graph(),{1:0,2:0,3:1},60)
    assert p[0]['compute_bound']==30
    assert p[0]['copy_service']==40
    assert p[0]['dependency_with_boundary']==60
    assert p[0]['resource_estimate']==60 and p[0]['additive_estimate']==70
    assert p[1]['resource_estimate']==16
    assert p[0]['scope']=='固定切分忽略溢出与跨任务竞争的资源估计'


def test_direct_dependencies_do_not_invent_tensor_copies_and_invalid_coverage_fails():
    from task_resource_profiles import task_resource_profiles
    g=dict(ops=[dict(id=i,op='ADD',pipe='PIPE_V',cycles=5) for i in (1,2)],tensors=[],edges=[dict(source=1,target=2,data_size=10000)])
    p=task_resource_profiles(g,{1:0,2:1},60)
    assert all(r['copy_service']==0 for r in p.values())
    with pytest.raises(ValueError):task_resource_profiles(g,{1:0},60)
    with pytest.raises(ValueError):task_resource_profiles(g,{1:0,2:1},0)
