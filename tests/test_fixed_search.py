"""固定轮数模式必须独立于墙钟预算，并能终止无可用邻域的情况。"""
import inspect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'solver'))
from pipeline import solve_case


def test_fixed_search_is_independent_of_wall_clock_budget():
    assert 'search_rounds' in inspect.signature(solve_case).parameters, '缺少完整流程的固定轮数模式'
    graph = {'ops': [{'id': i, 'op': 'MATMUL', 'pipe': 'PIPE_M', 'cycles': 10+i}
                     for i in (1, 2, 3, 4)], 'tensors': [], 'edges': []}
    a = solve_case(graph, 2, 'A', seed=41, time_budget=.00001, search_rounds=2, block_ops_cap=1)
    b = solve_case(graph, 2, 'A', seed=41, time_budget=1000, search_rounds=2, block_ops_cap=1)
    assert a['plan'] == b['plan']
    assert a['real'] == b['real']
    assert a['est'] == b['est']
    assert a['log']['n_evals'] == b['log']['n_evals']
    assert a['log']['search_mode'] == 'fixed_rounds'


def test_fixed_search_rejects_unbounded_postprocessing():
    assert 'search_rounds' in inspect.signature(solve_case).parameters, '缺少完整流程的固定轮数模式'
    import pytest
    with pytest.raises(ValueError, match='固定轮数'):
        solve_case({}, 2, 'A', search_rounds=2, event_rerank=True)
