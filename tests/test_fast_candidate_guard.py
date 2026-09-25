"""副本只能筛选候选，最终方案必须由原官方确认。"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'solver'))
import pipeline
from construct_refine import ConstructCandidate
from solution import Sol


@pytest.mark.parametrize('mode,expected', [
    ('match', 50), ('wrong_time', 100), ('wrong_bytes', 100), ('error', 100)])
def test_fast_candidate_final_official_guard(monkeypatch, mode, expected):
    graph = {'ops': [
        {'id': 0, 'op': 'MATMUL', 'pipe': 'PIPE_M', 'cycles': 10},
        {'id': 1, 'op': 'MATMUL', 'pipe': 'PIPE_M', 'cycles': 10}],
        'tensors': [], 'edges': []}
    candidates = [ConstructCandidate('heft', 'fine', 'A', 'R0', Sol([0, 1], [0, 0]))]
    monkeypatch.setattr(pipeline, 'build_construct_pool', lambda *a, **k: candidates)
    for name in ('tabu_search', 'sa_search', 'aco_search', 'mopso_search'):
        monkeypatch.setattr(pipeline, name, lambda *a, **k: (None, None, None, None))
    calls = []
    def official(graph, plan, scene):
        parallel = all(len(order) == 1 for order in plan['core_schedules'])
        calls.append(parallel)
        if parallel and mode == 'error':
            return {'error': '故意模拟最终验证失败'}
        return {'makespan': (80 if mode == 'wrong_time' else 50) if parallel else 100,
                'added_copy_bytes': 1 if parallel and mode == 'wrong_bytes' else 0,
                'scheduled_copy_bytes': 0, 'spill_added': 0, 'partition_added': 0}
    def replica(*args, **kwargs):
        return {'makespan': 50, 'added_copy_bytes': 0,
                'scheduled_copy_bytes': 0, 'spill_added': 0, 'partition_added': 0}
    monkeypatch.setattr(pipeline, 'real_evaluate', official)
    monkeypatch.setattr(pipeline, 'fast_evaluate', replica, raising=False)
    result = pipeline.solve_case(graph, 2, 'A', block_ops_cap=1, time_budget=0,
                                 task_order_refine=True, fast_candidate_eval=True)
    assert result['real']['makespan'] == expected
    assert calls == [False, True]
    assert result['log']['fast_candidate_eval']['final_verified'] == (mode == 'match')
    assert result['log']['fast_candidate_eval']['fallback_used'] == (mode != 'match')
    assert result['plan']['core_schedules'] == ([[0], [1]] if mode == 'match' else [[0, 1], []])
    assert result['log']['final_est'] == list(result['est'])
    selected = [r for r in result['log']['official_candidates']
                if r['plan_id'] == result['log']['selected_plan_id']]
    assert any(r['evaluation_source'] == 'original_official' for r in selected)
    assert any(r['evaluation_source'] == 'counter_replica'
               for r in result['log']['official_candidates'])


def test_fast_wrapper_matches_real_metrics():
    graph = {'ops': [{'id': 0, 'op': 'MATMUL', 'pipe': 'PIPE_M', 'cycles': 10}],
             'tensors': [], 'edges': []}
    plan = {'node_to_subgraph': {'0': 0}, 'core_schedules': [[0], []]}
    assert pipeline.fast_evaluate(graph, plan, 'A') == pipeline.real_evaluate(graph, plan, 'A')


def test_unchanged_plan_reuses_official_baseline(monkeypatch):
    graph = {'ops': [{'id': 0, 'op': 'MATMUL', 'pipe': 'PIPE_M', 'cycles': 10}],
             'tensors': [], 'edges': []}
    monkeypatch.setattr(pipeline, 'build_construct_pool', lambda *a, **k: [
        ConstructCandidate('heft', 'fine', 'A', 'R0', Sol([0], [0]))])
    for name in ('tabu_search', 'sa_search', 'aco_search', 'mopso_search'):
        monkeypatch.setattr(pipeline, name, lambda *a, **k: (None, None, None, None))
    calls = []
    def official(*args):
        calls.append(1)
        return {'makespan': 10, 'added_copy_bytes': 0, 'scheduled_copy_bytes': 0,
                'partition_added': 0, 'spill_added': 0}
    monkeypatch.setattr(pipeline, 'real_evaluate', official)
    result = pipeline.solve_case(graph, 2, 'A', time_budget=0, fast_candidate_eval=True)
    assert calls == [1]
    assert result['log']['fast_candidate_eval']['validation_reused_baseline']
