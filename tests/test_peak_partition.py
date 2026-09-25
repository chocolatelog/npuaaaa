"""峰值边界必须按完整块生成前驱闭合集，不得偷偷拆写原块。"""
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'solver'))


def test_cut_excludes_straddling_block_and_closes_predecessors():
    from peak_partition import closed_prefix
    model = SimpleNamespace(blocks=[[1, 4], [2], [3]])
    positions = {1: 0, 2: 1, 3: 2, 4: 3}
    prefix, stats = closed_prefix(model, {0, 1, 2}, positions, 2, {1: {0}})
    # 块1在边界前，但其前驱块0跨边界；闭包需整体纳入并明确记录扩张。
    assert prefix == {0, 1}
    assert stats == {'straddling_blocks': 1, 'closure_added_blocks': 1, 'prefix_blocks': 2}


def test_cut_keeps_noncontiguous_complete_blocks():
    from peak_partition import closed_prefix
    model = SimpleNamespace(blocks=[[1, 4], [2], [3]])
    prefix, stats = closed_prefix(model, {0, 1, 2}, {1: 0, 2: 1, 3: 2, 4: 3}, 3, {})
    assert prefix == {1, 2} and stats['straddling_blocks'] == 1


def test_no_spill_generates_no_peak_candidates():
    from peak_partition import generate_peak_candidates
    from model import Model
    from solution import Sol
    graph = {'ops': [{'id': i, 'op': 'MATMUL', 'pipe': 'PIPE_M', 'cycles': 10} for i in (1, 2)],
             'tensors': [], 'edges': []}
    model = Model(graph, block_ops_cap=1)
    parent = Sol([0, 0], [0])
    plan = model.plan_from(parent.sg_of_block, [[0], []])
    from scene_a_event import SceneAEventModel
    evaluator = SceneAEventModel(graph)
    candidates, audit = generate_peak_candidates(model, parent, plan, 2, evaluator, evaluator.evaluate(plan))
    assert not candidates and audit['legal'] == 0 and audit['target_tasks'] == []
