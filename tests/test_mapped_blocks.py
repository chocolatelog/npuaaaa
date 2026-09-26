"""固定操作映射后的安全任务分组，防止跨核回路和无界聚合。"""
from pathlib import Path
import sys
import copy
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'solver'))


def graph_fixture():
    return dict(ops=[dict(id=i,op='ADD',pipe='PIPE_V',cycles=10) for i in range(4)],tensors=[],edges=[dict(source=0,target=1),dict(source=1,target=2),dict(source=2,target=3)])


def test_grouping_cannot_merge_across_a_cross_core_roundtrip():
    from mapped_blocks import group_fixed_mapping
    from scenario_contract import validate_plan
    graph=graph_fixture();seed=dict(node_to_subgraph={str(i):i for i in range(4)},core_schedules=[[0,2],[1,3]])
    original=copy.deepcopy(seed)
    plan,audit=group_fixed_mapping(graph,seed,block_cap=240,work_cap=1000)
    validate_plan(graph,plan,'A')
    assert len(set(plan['node_to_subgraph'].values()))==4
    assert audit['cycle_rejections']>=1
    assert seed==original


def test_grouping_preserves_each_core_order_and_size_bounds():
    from mapped_blocks import group_fixed_mapping
    from scenario_contract import validate_plan
    graph=graph_fixture();graph['edges']=[]
    seed=dict(node_to_subgraph={str(i):i for i in range(4)},core_schedules=[[0,2,3],[1],[]])
    plan,audit=group_fixed_mapping(graph,seed,block_cap=2,work_cap=15)
    validate_plan(graph,plan,'A')
    mapping={int(k):v for k,v in plan['node_to_subgraph'].items()}
    for old,new in zip(seed['core_schedules'],plan['core_schedules']):
        members=[[op for op in old if mapping[op]==sg] for sg in new]
        assert [op for row in members for op in row]==old
        assert all(len(row)==1 for row in members)
    assert len(plan['core_schedules'])==3


def test_grouping_rejects_pipeline_legal_but_task_illegal_seed():
    from mapped_blocks import group_fixed_mapping
    graph=graph_fixture();graph['edges']=[dict(source=0,target=1),dict(source=2,target=3)]
    seed=dict(node_to_subgraph={str(i):i for i in range(4)},core_schedules=[[3,0],[1,2]])
    with pytest.raises(ValueError):group_fixed_mapping(graph,seed,block_cap=2,work_cap=100)


def test_generated_family_is_legal_deterministic_and_distinct():
    from mapped_blocks import generate_mapped_block_candidates
    from scenario_contract import validate_plan
    graph=graph_fixture();graph['edges']=[]
    parent=dict(node_to_subgraph={str(i):0 for i in range(4)},core_schedules=[[0],[]])
    rows=generate_mapped_block_candidates(graph,parent,2)
    assert 0<len(rows)<=4
    assert rows==generate_mapped_block_candidates(graph,parent,2)
    for row in rows:validate_plan(graph,row['plan'],'A')
