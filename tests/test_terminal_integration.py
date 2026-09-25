"""末端交换必须接在完整流程的原官方确认之后，失败不改变保底。"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'solver'))
import pipeline
from construct_refine import ConstructCandidate
from solution import Sol


def setup_small(monkeypatch):
    graph = {'ops': [
        {'id': i, 'op': 'MATMUL', 'pipe': 'PIPE_M', 'cycles': 10}
        for i in (0, 1)], 'tensors': [], 'edges': []}
    monkeypatch.setattr(pipeline, 'build_construct_pool', lambda *a, **k: [
        ConstructCandidate('heft', 'fine', 'A', 'R0', Sol([0, 1], [0, 0]))])
    for name in ('tabu_search', 'sa_search', 'aco_search', 'mopso_search'):
        monkeypatch.setattr(pipeline, name, lambda *a, **k: (None, None, None, None))
    return graph


def metric(makespan=100, **kw):
    return dict(makespan=makespan, added_copy_bytes=0, scheduled_copy_bytes=0,
                spill_added=0, partition_added=0, **kw)


@pytest.mark.parametrize('mode', ['match', 'mismatch', 'error'])
def test_terminal_swap_guard_and_binding(monkeypatch, mode):
    graph = setup_small(monkeypatch)
    def official(graph, plan, scene):
        if all(len(o) == 1 for o in plan['core_schedules']):
            return {'error': '模拟失败'} if mode == 'error' else metric(60 if mode == 'mismatch' else 50)
        return metric()
    monkeypatch.setattr(pipeline, 'real_evaluate', official)
    monkeypatch.setattr(pipeline, 'fast_evaluate', lambda *a, **k: metric(50))
    result = pipeline.solve_case(graph, 2, 'A', search_rounds=1, block_ops_cap=1,
                                 terminal_swaps=True)
    assert result['real'] == metric(50 if mode == 'match' else 100)
    assert result['plan']['core_schedules'] == ([[0], [1]] if mode == 'match' else [[0, 1], []])
    audit = result['log']['terminal_swaps']
    assert audit['search_seconds_limit'] is None
    assert audit['fast_count'] <= 8 and audit['official_count'] <= 2
    assert bool(audit['errors']) == (mode != 'match')
    from run_all import plan_digest
    assert result['log']['selected_plan_id'] == plan_digest(result['plan'])
    selected = [r for r in result['log']['official_candidates']
                if r['plan_id'] == result['log']['selected_plan_id']]
    assert any(r['evaluation_source'] == 'original_official' for r in selected)
    assert result['log']['final_est'] == list(result['est'])


def test_terminal_runs_after_partition_reservoir_and_original_guard(monkeypatch):
    graph = setup_small(monkeypatch)
    events = []
    def official(graph, plan, scene):
        parallel = all(len(o) == 1 for o in plan['core_schedules'])
        events.append('verify' if parallel else 'base')
        return metric(50 if parallel else 100)
    monkeypatch.setattr(pipeline, 'real_evaluate', official)
    import partition_polish, terminal_refine, task_order_search
    monkeypatch.setattr(terminal_refine, 'recommend_challengers',
                        lambda *a: ([], {'official_count': 0}))
    def partition(graph, plan, real, evaluator, **kw):
        events.append('partition')
        return {**plan, 'core_schedules': [[0], [1]]}, metric(50), {'candidates': []}
    def reservoir(graph, ctx, plan, real, *args):
        events.append('reservoir')
        return plan, real, {'candidates': []}
    def terminal(graph, plan, baseline, evaluator, **kw):
        events.append('terminal')
        assert baseline == metric(50) and plan['core_schedules'] == [[0], [1]]
        assert evaluator is official and kw['enable_swaps']
        return plan, baseline, {'candidates': [], 'official_count': 0}
    monkeypatch.setattr(partition_polish, 'polish_partition', partition)
    monkeypatch.setattr(terminal_refine, 'refine_reservoir_branch', reservoir)
    monkeypatch.setattr(task_order_search, 'refine_plan_orders', terminal)
    result = pipeline.solve_case(graph, 2, 'A', block_ops_cap=1, time_budget=0,
        event_rerank=True, partition_polish=True, construct_reservoir=True,
        fast_candidate_eval=True, terminal_swaps=True, terminal_cache=True)
    assert events == ['base', 'partition', 'reservoir', 'verify', 'terminal']
    assert result['real'] == metric(50)
    assert result['log']['terminal_swaps']['cache_stats']['hits'] == 0


def test_terminal_exception_preserves_current_plan_and_default_is_off(monkeypatch):
    graph = setup_small(monkeypatch)
    monkeypatch.setattr(pipeline, 'real_evaluate', lambda *a: metric())
    import task_order_search
    def broken(*a, **kw):
        raise ValueError('模拟精化失败')
    monkeypatch.setattr(task_order_search, 'refine_plan_orders', broken)
    base = pipeline.solve_case(graph, 2, 'A', search_rounds=1, block_ops_cap=1)
    result = pipeline.solve_case(graph, 2, 'A', search_rounds=1, block_ops_cap=1, terminal_swaps=True)
    assert 'terminal_swaps' not in base['log']
    assert result['plan'] == base['plan'] and result['est'] == base['est'] and result['real'] == base['real']
    assert '模拟精化失败' in result['log']['terminal_swaps']['error']
    with pytest.raises(ValueError, match='terminal'):
        pipeline.solve_case(graph, 2, 'A', terminal_cache=True)


def test_terminal_cache_initializes_only_for_first_screened_candidate(monkeypatch):
    import scene_a_fast
    calls = []
    original = scene_a_fast.build_counter_evaluator
    def counted(*a, **kw):
        calls.append(1)
        return original(*a, **kw)
    monkeypatch.setattr(scene_a_fast, 'build_counter_evaluator', counted)
    evaluate, cache = pipeline._cached_candidate_evaluator()
    assert calls == [], '无入围候选时不能编译或填充模板'
    graph = {'ops': [{'id': 0, 'op': 'MATMUL', 'pipe': 'PIPE_M', 'cycles': 10}],
             'tensors': [], 'edges': []}
    plan = {'node_to_subgraph': {'0': 0}, 'core_schedules': [[0], []]}
    assert evaluate(graph, plan, 'A') == pipeline.real_evaluate(graph, plan, 'A')
    assert evaluate(graph, plan, 'A') == pipeline.real_evaluate(graph, plan, 'A')
    assert calls == [1] and cache.stats['hits'] > 0
