"""第二、三问廉价特征的边界、寿命、下界和原算子粒度回归。"""
import importlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'solver'))


def api():
    assert importlib.util.find_spec('scene_b_features'), '缺少第二问廉价特征模块'
    return importlib.import_module('scene_b_features').SceneBFeatures


def graph(ops, tensors=(), edges=()):
    return {'ops': [{'id': o, 'op': 'ADD', 'pipe': 'PIPE_V', 'cycles': 1} for o in ops],
            'tensors': [{'id': t, 'pos': p, 'size': s} for t, p, s in tensors],
            'edges': [dict(source=e[0], target=e[1], **({'data_size': e[2]} if len(e)>2 else {})) for e in edges]}


def plan(rows):
    return {'node_to_subgraph': {str(o): o for row in rows for o in row}, 'core_schedules': rows}


def test_fanout_core_dedup_and_incremental_migration():
    cls = api()
    g = graph([0, 1, 2, 3], [(100, 'UB', 6000)], [(0,100),(100,1),(100,2),(100,3)])
    p = plan([[0], [1,2], [3]])
    f = cls(g).evaluate(p)
    assert f['boundary_bytes'] == 24000
    assert f['boundary_service_cycles'] == 400
    index = importlib.import_module('tensor_index').TensorIndex(g)
    counts = index.core_counts(p)
    assert counts[100]['consumers'] == {1: 2, 2: 1}
    assert index.migration_delta(p, [1], 0)['boundary_bytes_delta'] == 0
    assert index.migration_delta(p, [1,2], 0)['boundary_bytes_delta'] == -12000


@pytest.mark.parametrize('size,bytes_,cycles', [(6000, 12000, 200), (0, 0, 2)])
def test_direct_edges_count_zero_byte_requests(size, bytes_, cycles):
    f = api()(graph([0,1], edges=[(0,1,size)])).evaluate(plan([[0],[1]]))
    assert f['boundary_bytes'] == bytes_
    assert f['boundary_service_cycles'] == cycles
    assert f['mandatory_write_cycles'] == cycles // 2


def test_allocate_inputs_and_outputs_before_free_and_order_matters():
    cls = api()
    g = graph(range(4), [(100,'UB',80*1024),(101,'DDR',32*1024),(102,'UB',32*1024)],
              [(0,100),(100,3),(101,1),(1,102),(102,2)])
    old = cls(g).evaluate(plan([[0,1,2,3]]))
    new = cls(g).evaluate(plan([[0,3,1,2]]))
    assert old['per_pool']['UB']['peak'] == 144*1024
    assert old['per_pool']['UB']['overage'] == 16*1024
    assert old['per_pool']['UB']['area'] == 16*1024
    assert new['per_pool']['UB']['peak'] == 80*1024
    assert old['boundary_bytes'] == new['boundary_bytes']
    assert old['per_pool']['L1']['peak'] == 0
    assert old['local_orders'] == {0:[0,1,2,3]}
    assert old['hotspot_windows'][0]['ops']
    assert old['area_unit'] == '字节×序位置'
    json.dumps(old, allow_nan=False)


def test_two_pools_are_independent_and_coarse_group_is_not_a_single_op():
    g = graph(range(130), [(1000,'L1',100000),(1001,'UB',100000)], [(0,1000),(1000,129),(1,1001),(1001,128)])
    p = {'node_to_subgraph': {str(o):0 for o in range(130)}, 'core_schedules': [[0]]}
    f = api()(g).evaluate(p)
    assert f['per_pool']['UB']['peak'] == f['per_pool']['L1']['peak'] == 100000
    assert f['per_pool']['UB']['overage'] == f['per_pool']['L1']['overage'] == 0
    assert f['local_orders'][0] == list(range(130))
    assert all(len(w['ops']) <= 32 for w in f['hotspot_windows'])


def test_scene_c_excludes_all_reads_from_lower_bound():
    cls = api(); g = graph([0], [(100,'DDR',60000)], [(100,0)])
    b = cls(g).evaluate(plan([[0]]), scene='B')
    c = cls(g).evaluate(plan([[0]]), scene='C')
    assert b['lower_bound'] == 1000
    assert c['lower_bound'] == c['compute_lower_bound'] == 1
    assert c['mandatory_write_cycles'] == 0


def test_original_copy_and_pure_copy_tensors_and_multiple_producers():
    cls=api()
    g=graph([0,1,2],[(100,'UB',6000),(101,'UB',2000)],[(0,100),(1,100),(100,2),(100,4),(3,101),(101,4)])
    g['ops'] += [{'id':3,'op':'COPY_IN','pipe':'PIPE_MTE2','cycles':1},
                 {'id':4,'op':'COPY_OUT','pipe':'PIPE_MTE3','cycles':1}]
    f=cls(g).evaluate(plan([[0,2],[1]]))
    assert f['boundary_bytes'] == 24000
    assert f['original_copy_bytes'] == 10000
    assert f['partition_added_bytes'] == 14000
    assert f['mandatory_write_cycles'] == 300


def test_boundary_invariant_under_refinement_and_plan_state_supported():
    cls=api();g=graph(range(4),[(100,'UB',6000)],[(0,100),(100,1),(100,2)])
    p=plan([[0,1],[2,3]])
    state=importlib.import_module('plan_state').from_plan(g,p)
    coarse={'node_to_subgraph':{'0':0,'1':0,'2':1,'3':1},'core_schedules':[[0],[1]]}
    assert cls(g).evaluate(state)['boundary_bytes'] == cls(g).evaluate(coarse)['boundary_bytes'] == 12000


def test_no_official_expansion_in_cheap_features(monkeypatch):
    cls=api()
    import multicore_cut_evaluate_problem_2 as official
    def forbidden(*args, **kwargs):
        raise AssertionError('廉价层禁止官方展开')
    monkeypatch.setattr(official,'_build_scene_b_tasks',forbidden)
    assert cls(graph([0])).evaluate(plan([[0]]))['lower_bound'] == 1


def test_tensor_dependency_and_non_topological_ids_preserve_critical_path():
    g = graph([9, 2, 5], [(100, 'UB', 0)], [(9, 100), (100, 2), (2, 5)])
    for op in g['ops']:
        op['cycles'] = 10
    p = plan([[9], [2], [5]])
    f = api()(g).evaluate(p, scene='C')
    assert f['lower_bound'] == 30
    coarse = {'node_to_subgraph': {'9': 0, '2': 0, '5': 0}, 'core_schedules': [[0]]}
    assert api()(g).evaluate(coarse)['local_orders'] == {0: [9, 2, 5]}


def test_feature_failure_is_explicit_and_never_zero_cost(monkeypatch):
    from bc_refine import _metrics
    def fail(*a, **kw):
        raise ValueError('故意构造的特征计算失败')
    monkeypatch.setattr(api(), 'evaluate', fail)
    metrics = _metrics(graph([0]), plan([[0]]), 'B')
    assert metrics['proxy_valid'] is False
    assert metrics['proxy_error']
    from bc_search import _proxy_key
    failed = {'plan_id': 'a', 'metrics': metrics}
    valid = {'plan_id': 'b', 'metrics': {'proxy_lower_bound': 100000}}
    assert _proxy_key(failed, 'B') > _proxy_key(valid, 'B')
