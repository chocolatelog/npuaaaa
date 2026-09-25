"""保证最终代理指标属于官方选中的方案，而不是另一个搜索候选。"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'solver'))
import pipeline
from model import Model
from solution import Sol
from construct_refine import ConstructCandidate


def test_official_winner_keeps_its_own_proxy_metrics(monkeypatch):
    graph = {'ops': [
        {'id': 0, 'op': 'MATMUL', 'pipe': 'PIPE_M', 'cycles': 10},
        {'id': 1, 'op': 'MATMUL', 'pipe': 'PIPE_M', 'cycles': 10}],
        'tensors': [], 'edges': []}
    candidates = [
        ConstructCandidate('heft', 'fine', 'A', 'R0', Sol([0, 1], [0, 1])),
        ConstructCandidate('chain', 'coarse', 'A', 'R0', Sol([0, 0], [0]))]
    monkeypatch.setattr(pipeline, 'build_construct_pool', lambda *a, **k: candidates)
    for name in ('tabu_search', 'sa_search', 'aco_search', 'mopso_search'):
        monkeypatch.setattr(pipeline, name, lambda *a, **k: (None, None, None, None))
    def official(graph, plan, scene):
        merged = len(set(plan['node_to_subgraph'].values())) == 1
        return {'makespan': 1 if merged else 100, 'added_copy_bytes': 0}
    monkeypatch.setattr(pipeline, 'real_evaluate', official)
    result = pipeline.solve_case(graph, 2, 'A', block_ops_cap=1, time_budget=0)
    assert len(set(result['plan']['node_to_subgraph'].values())) == 1
    model = Model(graph, block_ops_cap=1)
    expected = model.evaluate([0, 0], [0], 'A', 2,
                             orders_override=result['plan']['core_schedules'])
    assert result['est'] == pytest.approx(expected[:2])
    assert result['log']['final_est'] == pytest.approx(expected[:2])


@pytest.mark.parametrize('extra_makespan,expected', [(50, 50), (100.2, 100)])
def test_terminal_challenger_uses_official_strict_guard(monkeypatch, extra_makespan, expected):
    graph = {'ops': [
        {'id': 0, 'op': 'MATMUL', 'pipe': 'PIPE_M', 'cycles': 10},
        {'id': 1, 'op': 'MATMUL', 'pipe': 'PIPE_M', 'cycles': 10}],
        'tensors': [], 'edges': []}
    # 切分完全相同、核分配不同；原构造候选过滤会遗漏第二个方案。
    candidates = [
        ConstructCandidate('heft', 'fine', 'A', 'R0', Sol([0, 1], [0, 1])),
        ConstructCandidate('chain', 'coarse', 'A', 'R0', Sol([0, 1], [0, 0]))]
    monkeypatch.setattr(pipeline, 'build_construct_pool', lambda *a, **k: candidates)
    for name in ('tabu_search', 'sa_search', 'aco_search', 'mopso_search'):
        monkeypatch.setattr(pipeline, name, lambda *a, **k: (None, None, None, None))
    def official(graph, plan, scene):
        same_core = any(len(order) == 2 for order in plan['core_schedules'])
        return {'makespan': extra_makespan if same_core else 100,
                'added_copy_bytes': 0 if same_core else 1000}
    monkeypatch.setattr(pipeline, 'real_evaluate', official)
    result = pipeline.solve_case(graph, 2, 'A', block_ops_cap=1, time_budget=0,
                                 event_rerank=True)
    audit = result['log']['event_rerank']
    assert audit['baseline_official']['makespan'] == 100
    assert audit['official_count'] == 1
    assert result['real']['makespan'] == expected
    same_core = any(len(order) == 2 for order in result['plan']['core_schedules'])
    assert same_core == (extra_makespan < 100)
    model = Model(graph, block_ops_cap=1)
    expected_proxy = model.evaluate([0, 1], [0, 0] if same_core else [0, 1], 'A', 2,
                                    orders_override=result['plan']['core_schedules'])
    assert result['est'] == pytest.approx(expected_proxy[:2])


@pytest.mark.parametrize('split_makespan,expected', [(50, 50), (100.2, 100)])
def test_partition_postprocess_binds_changed_partition(monkeypatch, split_makespan, expected):
    graph = {'ops': [
        {'id': 0, 'op': 'MATMUL', 'pipe': 'PIPE_M', 'cycles': 10},
        {'id': 1, 'op': 'MATMUL', 'pipe': 'PIPE_M', 'cycles': 10}],
        'tensors': [], 'edges': []}
    candidates = [ConstructCandidate('heft', 'coarse', 'A', 'R0', Sol([0, 0], [0]))]
    monkeypatch.setattr(pipeline, 'build_construct_pool', lambda *a, **k: candidates)
    for name in ('tabu_search', 'sa_search', 'aco_search', 'mopso_search'):
        monkeypatch.setattr(pipeline, name, lambda *a, **k: (None, None, None, None))
    def official(graph, plan, scene):
        split = len(set(plan['node_to_subgraph'].values())) == 2
        return {'makespan': split_makespan if split else 100, 'added_copy_bytes': 0,
                'scheduled_copy_bytes': 0, 'spill_added': 0}
    monkeypatch.setattr(pipeline, 'real_evaluate', official)
    result = pipeline.solve_case(graph, 2, 'A', block_ops_cap=1, time_budget=0,
                                 partition_polish=True)
    assert result['real']['makespan'] == expected
    assert not result['log']['partition_polish'].get('error')
    assert result['log']['selected_plan_id'] in {
        row['plan_id'] for row in result['log']['official_candidates']}
    assert len(set(result['plan']['node_to_subgraph'].values())) == (2 if split_makespan < 100 else 1)
    assert result['est'][0] == (10 if split_makespan < 100 else 20)


def test_reservoir_preserves_completed_incumbent_path(monkeypatch):
    graph = {'ops': [
        {'id': 0, 'op': 'MATMUL', 'pipe': 'PIPE_M', 'cycles': 10},
        {'id': 1, 'op': 'MATMUL', 'pipe': 'PIPE_M', 'cycles': 10}],
        'tensors': [], 'edges': []}
    def build(model, n, scene, ctx, return_candidates):
        ctx.construct_reservoir = [ConstructCandidate('chain', 'coarse', 'A', 'R0', Sol([0, 0], [0]))]
        return [ConstructCandidate('heft', 'fine', 'A', 'R0', Sol([0, 1], [0, 0]))]
    monkeypatch.setattr(pipeline, 'build_construct_pool', build)
    for name in ('tabu_search', 'sa_search', 'aco_search', 'mopso_search'):
        monkeypatch.setattr(pipeline, name, lambda *a, **k: (None, None, None, None))
    def official(graph, plan, scene):
        merged = len(set(plan['node_to_subgraph'].values())) == 1
        parallel = all(len(order) == 1 for order in plan['core_schedules'])
        return {'makespan': 75 if merged else 50 if parallel else 100,
                'added_copy_bytes': 0, 'scheduled_copy_bytes': 0, 'spill_added': 0}
    monkeypatch.setattr(pipeline, 'real_evaluate', official)
    result = pipeline.solve_case(graph, 2, 'A', block_ops_cap=1, time_budget=0,
                                 event_rerank=True, task_order_refine=True,
                                 construct_reservoir=True)
    # 旁路中间解 75 虽优于初始 100，却劣于原路径精修后的 50，必须保留 50。
    assert result['real']['makespan'] == 50
    assert result['log']['construct_reservoir']['baseline_official']['makespan'] == 50


@pytest.mark.parametrize('split_makespan,expected', [(50, 50), (100.2, 100)])
def test_order_postprocess_keeps_selected_plan_metrics(monkeypatch, split_makespan, expected):
    graph = {'ops': [
        {'id': 0, 'op': 'MATMUL', 'pipe': 'PIPE_M', 'cycles': 10},
        {'id': 1, 'op': 'MATMUL', 'pipe': 'PIPE_M', 'cycles': 10}],
        'tensors': [], 'edges': []}
    candidates = [ConstructCandidate('heft', 'fine', 'A', 'R0', Sol([0, 1], [0, 0]))]
    monkeypatch.setattr(pipeline, 'build_construct_pool', lambda *a, **k: candidates)
    for name in ('tabu_search', 'sa_search', 'aco_search', 'mopso_search'):
        monkeypatch.setattr(pipeline, name, lambda *a, **k: (None, None, None, None))
    def official(graph, plan, scene):
        split = all(len(order) == 1 for order in plan['core_schedules'])
        return {'makespan': split_makespan if split else 100, 'added_copy_bytes': 0,
                'scheduled_copy_bytes': 0, 'spill_added': 0}
    monkeypatch.setattr(pipeline, 'real_evaluate', official)
    result = pipeline.solve_case(graph, 2, 'A', block_ops_cap=1, time_budget=0,
                                 task_order_refine=True)
    assert result['real']['makespan'] == expected
    assert result['log']['task_order_refine']['baseline_official']['makespan'] == 100
    assert result['log']['selected_plan_id'] in {
        row['plan_id'] for row in result['log']['official_candidates']}
    plan = result['plan']
    cores = [0, 0]
    for c, order in enumerate(plan['core_schedules']):
        for s in order:
            cores[s] = c
    expected_proxy = Model(graph, block_ops_cap=1).evaluate(
        [0, 1], cores, 'A', 2, orders_override=plan['core_schedules'])
    assert result['est'] == pytest.approx(expected_proxy[:2])
