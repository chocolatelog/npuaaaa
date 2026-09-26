import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'solver'))


def test_operation_split_inside_block_is_not_lossless_for_block_proxy():
    from pipeline import _project_plan_to_blocks

    class FakeModel:
        blocks = [[1, 2]]

        @staticmethod
        def plan_from(_groups, _orders):
            raise AssertionError('内部拆分不应调用块级重建')

    plan = {
        'node_to_subgraph': {'1': 0, '2': 1},
        'core_schedules': [[0], [1]],
    }
    assert _project_plan_to_blocks(FakeModel(), plan) is None


def test_pipeline_selects_officially_better_original_operation_plan(monkeypatch):
    import json
    import pipeline
    import bc_refine
    from plan_state import topological_ops
    root = Path(__file__).resolve().parents[1]
    graph = json.loads((root/'通用神经网络处理器下的多核调度问题附件/data/case_001.json').read_text(encoding='utf-8'))
    order = topological_ops(graph)
    fine = {'node_to_subgraph': {str(op): i for i, op in enumerate(order)},
            'core_schedules': [list(range(len(order))), []]}
    monkeypatch.setattr(bc_refine, 'generate_bc_candidates',
                        lambda *a, **kw: ([fine], {'candidate_metrics': []}))
    def official(g, plan, scene, **kw):
        return {'makespan': 1 if plan == fine else 1000000, 'added_copy_bytes': 0,
                'scheduled_copy_bytes': 100, 'partition_added': 0, 'spill_added': 0}
    monkeypatch.setattr(pipeline, 'real_evaluate', official)
    result = pipeline.solve_case(graph, 2, 'B', time_budget=.01, seed=7, search_rounds=1, bc_refine=True)
    assert result['plan'] == fine
    assert result['real']['makespan'] == 1
    assert result['log']['proxy_representation'] == 'original_operations'
