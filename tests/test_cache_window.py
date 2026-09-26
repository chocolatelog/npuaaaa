"""固定映射缓存窗口的事件归因及依赖安全，不直接把命中率当目标。"""
from pathlib import Path
import sys
import copy
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'solver'))


def raw_fixture():
    return dict(cache_capacity_bytes=100,bandwidth_bytes_per_cycle=10,cache_bandwidth_bytes_per_cycle=20,
        per_core_timeline=[dict(core_id=0,ops=[dict(op_id=100,subgraph_id=0,start=0,end=6)]),
                           dict(core_id=1,ops=[dict(op_id=101,subgraph_id=1,start=1,end=8),dict(op_id=102,subgraph_id=2,start=20,end=26)])],
        cache_events=[dict(event='miss',time=0,tensor_id=10,size_bytes=60,core_id=0,op_id=100),
                      dict(event='miss',time=1,tensor_id=10,size_bytes=60,core_id=1,op_id=101),
                      dict(event='insert',time=6,tensor_id=10,size_bytes=60),
                      dict(event='miss',time=20,tensor_id=10,size_bytes=60,core_id=1,op_id=102)])


def test_cache_window_separates_inflight_from_eviction_and_ignores_first_read():
    from cache_window import cache_window_hotspots
    rows=cache_window_hotspots(raw_fixture())
    assert [(r['subgraph'],r['reason']) for r in rows]==[(1,'inflight_duplicate'),(2,'evicted')]
    assert all(r['potential_saved_cycles']==3 for r in rows)


def test_window_keeps_mapping_and_rejects_dependency_reversal():
    from cache_window import generate_cache_window_candidates
    from scenario_contract import validate_plan
    g=dict(ops=[dict(id=i,op='ADD',pipe='PIPE_V',cycles=2) for i in range(4)],tensors=[],edges=[dict(source=1,target=2)])
    p=dict(node_to_subgraph={str(i):i for i in range(4)},core_schedules=[[0],[1,3,2]])
    before=copy.deepcopy((g,p));rows,audit=generate_cache_window_candidates(g,p,raw_fixture(),window=2)
    assert rows
    assert len(rows)<=6
    for r in rows:
        candidate=r['plan'];assert candidate['node_to_subgraph']==p['node_to_subgraph']
        assert set(candidate['core_schedules'][1])=={1,2,3}
        assert candidate['core_schedules'][1].index(1)<candidate['core_schedules'][1].index(2)
        validate_plan(g,candidate,'A')
    assert audit['rejected']>=1
    assert rows==generate_cache_window_candidates(g,p,raw_fixture(),window=2)[0]
    assert (g,p)==before


def test_window_does_not_generate_unmotivated_candidates_without_repeated_reads():
    from cache_window import generate_cache_window_candidates
    g=dict(ops=[dict(id=0,op='ADD',pipe='PIPE_V',cycles=2)],tensors=[],edges=[])
    p=dict(node_to_subgraph={'0':0},core_schedules=[[0]])
    raw=raw_fixture();raw['cache_events']=raw['cache_events'][:1]
    assert generate_cache_window_candidates(g,p,raw)[0]==[]


def test_many_tensors_requesting_same_move_do_not_starve_other_moves():
    from cache_window import generate_cache_window_candidates
    g=dict(ops=[dict(id=i,op='ADD',pipe='PIPE_V',cycles=2) for i in range(4)],tensors=[],edges=[])
    p=dict(node_to_subgraph={str(i):i for i in range(4)},core_schedules=[[0],[1,2,3]])
    raw=dict(cache_capacity_bytes=100,bandwidth_bytes_per_cycle=10,cache_bandwidth_bytes_per_cycle=20,
             per_core_timeline=[dict(core_id=0,ops=[]),dict(core_id=1,ops=[])],cache_events=[])
    # 多个张量让同一子图向后移一位；另一个子图的独立动作不能被它们挤掉。
    for i in range(31):
        for core in (0,1):
            op_id=100+2*i+core
            raw['per_core_timeline'][core]['ops'].append(dict(op_id=op_id,subgraph_id=0 if core==0 else 1 if i<30 else 2,start=core,end=10))
            raw['cache_events'].append(dict(event='miss',time=core,tensor_id=i,size_bytes=60,core_id=core,op_id=op_id))
    raw['cache_events'].sort(key=lambda e:e['time'])
    rows,audit=generate_cache_window_candidates(g,p,raw,window=1)
    assert len(rows)==2
    assert {r['source']['subgraph'] for r in rows}=={1,2}
    assert audit['attempted']==2


def test_read_boundary_moves_cross_a_real_observed_read_group():
    from cache_window import generate_cache_window_candidates
    g=dict(ops=[dict(id=i,op='ADD',pipe='PIPE_V',cycles=2) for i in range(4)],tensors=[],edges=[])
    p=dict(node_to_subgraph={str(i):i for i in range(4)},core_schedules=[[0],[1,3,2]])
    rows,audit=generate_cache_window_candidates(g,p,raw_fixture(),read_boundary=True,window=2)
    assert len(rows)==2
    assert all(r['source']['distance']==2 for r in rows)
    assert all(r['source']['read_boundary'] for r in rows)
    assert audit['read_boundary'] is True
    # 一格窗口内只有独立计算子图，不能虚构可调整的读入顺序。
    assert generate_cache_window_candidates(g,p,raw_fixture(),read_boundary=True,window=1)[0]==[]
