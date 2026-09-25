"""独立局部图构建必须遵循官方边界语义。"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'solver'))
sys.path.insert(0, str(ROOT / '通用神经网络处理器下的多核调度问题附件/code'))
from scene_a_event import SceneAEventModel
import multicore_cut_evaluate_problem_1 as official


def test_local_graphs_match_official_boundaries_and_ddr_pool(monkeypatch):
    graph = {'ops': [
        {'id': 1, 'op': 'COPY_IN', 'pipe': 'PIPE_MTE2', 'cycles': 1},
        {'id': 2, 'op': 'MATMUL', 'pipe': 'PIPE_M', 'cycles': 10},
        {'id': 3, 'op': 'ADD', 'pipe': 'PIPE_V', 'cycles': 10},
        {'id': 4, 'op': 'COPY_OUT', 'pipe': 'PIPE_MTE3', 'cycles': 1}],
        'tensors': [{'id': 10001+i, 'pos': p, 'size': 60}
                    for i, p in enumerate(['DDR', 'L1', 'DDR', 'UB', 'DDR'])],
        'edges': [{'source': a, 'target': b} for a, b in [
            (10001,1), (1,10002), (10002,2), (2,10003),
            (10003,3), (3,10004), (10004,4), (4,10005)]]}
    plan = {'node_to_subgraph': {'2': 0, '3': 1}, 'core_schedules': [[0], [1]]}
    captured = []
    original = official.step1_schedule
    def capture(g):
        captured.append(g)
        return original(g)
    monkeypatch.setattr(official, 'step1_schedule', capture)
    official._build_scene_a_tasks(graph, plan, 60, {'L1': 524288, 'UB': 131072})
    local = SceneAEventModel(graph).local_graphs(plan)
    assert [g for _, g in local] == captured
    assert all(next(t for t in g['tensors'] if t['id'] == 10003)['pos'] == 'UB'
               for _, g in local)


def test_no_copy_graph_keeps_independent_cores_parallel():
    graph = {'ops': [{'id': i, 'op': 'MATMUL', 'pipe': 'PIPE_M', 'cycles': 10}
                     for i in (1, 2)], 'tensors': [], 'edges': []}
    plan = {'node_to_subgraph': {'1': 0, '2': 1}, 'core_schedules': [[0], [1]]}
    result = SceneAEventModel(graph).evaluate(plan)
    assert result['makespan'] == 10
    assert result['total_copy_bytes'] == 0
