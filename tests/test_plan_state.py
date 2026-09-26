"""原算子方案：覆盖、不可变性、收缩环、真实核心与局部细化。"""
import importlib
import importlib.util
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'solver'))


def api():
    assert importlib.util.find_spec('plan_state'), '缺少原算子完整方案模块'
    return importlib.import_module('plan_state')


def chain(size=3):
    return {'ops': [{'id': i, 'op': 'ADD', 'pipe': 'PIPE_V', 'cycles': 1}
                    for i in range(size)], 'tensors': [],
            'edges': [{'source': i, 'target': i+1, 'data_size': 0}
                      for i in range(size-1)]}


def test_immutable_roundtrip_and_exact_order_identity():
    m = api()
    graph = chain(); graph['edges'] = []
    raw = {'node_to_subgraph': {'0': 8, '1': 9, '2': 10},
           'core_schedules': [[8, 9], [10], []]}
    state = m.from_plan(graph, raw, config_digest='fixed')
    assert state.to_plan() == raw
    raw['core_schedules'][0].reverse()
    assert state.orders[0] == (8, 9)
    other = m.from_plan(graph, raw, config_digest='fixed')
    assert other.plan_id != state.plan_id
    assert state.sg_members == {8: (0,), 9: (1,), 10: (2,)}
    changed = state.to_plan(); changed['node_to_subgraph']['0'] = 99
    assert state.mapping[0] == 8


def test_noncontiguous_contraction_rejected_and_core_cycle_allowed():
    m = api(); graph = chain()
    with pytest.raises(ValueError, match='cycle'):
        m.from_plan(graph, {'node_to_subgraph': {'0': 0, '1': 1, '2': 0},
                            'core_schedules': [[0], [1]]})
    valid = m.from_plan(graph, {'node_to_subgraph': {'0': 0, '1': 1, '2': 2},
                              'core_schedules': [[0, 2], [1]]})
    assert valid.orders == ((0, 2), (1,))


def test_hotspot_split_preserves_both_outside_segments():
    m = api(); graph = chain(130)
    root = m.from_plan(graph, {'node_to_subgraph': {str(i): 4 for i in range(130)},
                               'core_schedules': [[4], []]})
    state = m.refine_window(graph, root, set(range(60, 65)), 1)
    groups = [state.sg_members[s] for s in state.orders[0]]
    assert groups == [tuple(range(60))] + [(i,) for i in range(60, 65)] + [tuple(range(65, 130))]
    assert len(state.mapping) == 130 and root.orders == ((4,), ())
    assert state.orders[0][0] == 4


def test_hotspot_split_obeys_given_local_sequence():
    m = api(); graph = chain(4); graph['edges'] = []
    root = m.from_plan(graph, {'node_to_subgraph': {str(i): 0 for i in range(4)},
                             'core_schedules': [[0]]})
    state = m.refine_window(graph, root, {1, 3}, 1, local_orders={0: [2, 3, 1, 0]})
    assert [state.sg_members[s] for s in state.orders[0]] == [(2,), (3,), (1,), (0,)]


def test_legal_slots_use_transitive_dependencies():
    m = api(); graph = chain(3)
    root = m.from_plan(graph, {'node_to_subgraph': {'0': 0, '1': 1, '2': 2},
                              'core_schedules': [[0, 2], [1]]})
    assert m.legal_slots(graph, root, 1, 0) == (1,)
    changed = m.move_group(graph, root, (1,), 0, 1)
    assert changed.orders == ((0, 1, 2), ())
    with pytest.raises(ValueError):
        m.move_group(graph, root, (1,), 0, 0)


def test_copy_nodes_are_excluded_but_contracted_dependencies_preserved():
    m = api()
    graph = {'ops': [
        {'id': 0, 'op': 'ADD', 'pipe': 'PIPE_V', 'cycles': 1},
        {'id': 1, 'op': 'COPY_OUT', 'pipe': 'PIPE_MTE3', 'cycles': 1},
        {'id': 2, 'op': 'COPY_IN', 'pipe': 'PIPE_MTE2', 'cycles': 1},
        {'id': 3, 'op': 'ADD', 'pipe': 'PIPE_V', 'cycles': 1}],
        'tensors': [], 'edges': [{'source': a, 'target': b} for a,b in [(0,1),(1,2),(2,3)]]}
    root = m.from_plan(graph, {'node_to_subgraph': {'0': 0, '3': 1}, 'core_schedules': [[0,1]]})
    assert m.legal_slots(graph, root, 1, 0) == (1,)


def test_invalid_inputs_are_not_silently_repaired():
    m=api();graph=chain()
    with pytest.raises(ValueError):
        m.from_plan(graph, {'node_to_subgraph': {'0': 0}, 'core_schedules': [[0]]})
    root=m.from_plan(graph, {'node_to_subgraph': {str(i): 0 for i in range(3)}, 'core_schedules': [[0]]})
    with pytest.raises(ValueError):m.refine_window(graph,root,{1},0)
    with pytest.raises(ValueError):m.refine_window(graph,root,{9},1)


def test_budget_restore_keeps_unique_and_logical_counts():
    assert importlib.util.find_spec('budget_state'), '缺少共享预算账本'
    cls=importlib.import_module('budget_state').BudgetState
    b=cls(max_attempts=4,max_states=2,max_exact=1)
    assert b.attempt() and b.score('a') and b.score('b')
    assert b.score('a') and not b.score('c')
    assert b.exact('a') and not b.exact('b')
    saved=b.to_dict();r=cls.from_dict(saved)
    assert r.to_dict()==saved
    assert not r.score('c') and not r.exact('a')
    assert r.attempt() and r.attempt() and r.attempt() and not r.attempt()
