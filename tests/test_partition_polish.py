"""切分精化需要保持覆盖，拒绝收缩环，并按真实边界字节计收益。"""
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'solver'))
from solution import Sol


def test_split_uses_workload_and_preserves_blocks():
    from partition_polish import split_solution
    model = SimpleNamespace(blocks=[[0], [1], [2], [3]], block_pos=[0, 1, 2, 3],
                            block_work_m=[1, 1, 10, 1], block_work_v=[0]*4,
                            block_edges=[((0, 1), 1), ((1, 2), 1), ((2, 3), 1)])
    parent = Sol([0, 0, 0, 0], [0])
    child = split_solution(model, parent, 0, 0.5, 1, 2)
    assert child.sg_of_block == [0, 0, 1, 1]
    assert child.core_of_sg == [0, 1]
    assert parent.sg_of_block == [0, 0, 0, 0]
    assert child.validate(model, 2)


def test_merge_rejects_alternative_path_cycle():
    from partition_polish import merge_solution
    model = SimpleNamespace(blocks=[[0], [1], [2]],
                            block_edges=[((0, 1), 1), ((1, 2), 1), ((0, 2), 1)])
    parent = Sol([0, 1, 2], [0, 1, 0])
    assert merge_solution(model, parent, 0, 2, 2) is None
    assert merge_solution(model, parent, 0, 1, 2).validate(model, 2)
    assert parent.sg_of_block == [0, 1, 2]


def test_merge_savings_preserves_shared_output_copy():
    from partition_polish import boundary_savings
    from scene_a_event import SceneAEventModel
    graph = {'ops': [{'id': i, 'op': 'MATMUL', 'pipe': 'PIPE_M', 'cycles': 10}
                     for i in range(3)],
             'tensors': [{'id': 100, 'pos': 'UB', 'size': 60}],
             'edges': [{'source': 0, 'target': 100},
                       {'source': 100, 'target': 1}, {'source': 100, 'target': 2}]}
    # 合并 0/1 后仍要给子图 2 输出，只省一次输入的 60 字节，不能误计为 120。
    assert boundary_savings(SceneAEventModel(graph), {0: 0, 1: 1, 2: 2}, 0, 1) == 60


def test_partition_polish_can_use_fixed_counts_without_wall_clock(monkeypatch):
    import partition_polish
    from model import Model
    from pipeline import real_evaluate
    graph = {'ops': [{'id': i, 'op': 'MATMUL', 'pipe': 'PIPE_M', 'cycles': 10}
                     for i in range(4)], 'tensors': [], 'edges': []}
    plan = {'node_to_subgraph': {str(i): 0 for i in range(4)}, 'core_schedules': [[0], []]}
    observed = []
    def bounded_search(*args, **kwargs):
        observed.append((kwargs['seconds'], kwargs['max_evals']))
        return [], {'evaluations': 0}
    monkeypatch.setattr(partition_polish, 'search_orders', bounded_search)
    selected, real, audit = partition_polish.polish_partition(
        graph, plan, real_evaluate(graph, plan, 'A'), real_evaluate,
        seconds=None, local_search_seconds=None, model=Model(graph, block_ops_cap=1))
    assert observed and all(row == (None, 400) for row in observed)
    assert audit['search']['soft_budget_seconds'] is None
    assert not audit['search']['soft_budget_exhausted']
    assert audit['official_count'] <= 2
