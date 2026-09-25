"""计数式完成判定必须保持完整事件结果与官方一致。"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'solver'))
import scene_a_event
from multicore_cut_evaluate_problem_1 import evaluate_scene_a


@pytest.mark.parametrize('orders', [[[0, 1], []], [[0], [1]], [[], [0, 1]]])
def test_counter_replica_matches_complete_official_result(orders):
    from scene_a_fast import evaluate_scene_a_fast
    graph = {'ops': [
        {'id': 0, 'op': 'MATMUL', 'pipe': 'PIPE_M', 'cycles': 10},
        {'id': 1, 'op': 'MATMUL', 'pipe': 'PIPE_M', 'cycles': 10},
        {'id': 2, 'op': 'VEC', 'pipe': 'PIPE_V', 'cycles': 7}],
        'tensors': [{'id': 100, 'pos': 'UB', 'size': 60}],
        'edges': [{'source': 0, 'target': 100}, {'source': 100, 'target': 2}]}
    plan = {'node_to_subgraph': {'0': 0, '1': 0, '2': 1}, 'core_schedules': orders}
    args = (graph, plan, 60, {'L1': 524288, 'UB': 131072}, 1000, 100)
    assert evaluate_scene_a_fast(*args) == evaluate_scene_a(*args)


def test_counter_replica_keeps_official_function_untouched():
    import multicore_cut_evaluate_problem_1 as official
    before = official.evaluate_scene_a
    from scene_a_fast import evaluate_scene_a_fast
    assert official.evaluate_scene_a is before
    assert evaluate_scene_a_fast is not before
    assert evaluate_scene_a_fast.__globals__ is not before.__globals__
