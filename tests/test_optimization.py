import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'solver'))

from diagnostics import schedule_diagnostics  # noqa: E402
from model import Model  # noqa: E402
from solution import Context, Sol  # noqa: E402
from run_all import stable_seed  # noqa: E402
from run_all import load_warm_candidates  # noqa: E402


def _toy_graph():
    # Task 3 releases the long task 4 on the other core. The starting order
    # delays this release behind two independent tasks.
    return {
        'ops': [
            {'id': 1, 'op': 'COMPUTE', 'pipe': 'PIPE_M', 'cycles': 10},
            {'id': 2, 'op': 'COMPUTE', 'pipe': 'PIPE_M', 'cycles': 10},
            {'id': 3, 'op': 'COMPUTE', 'pipe': 'PIPE_M', 'cycles': 10},
            {'id': 4, 'op': 'COMPUTE', 'pipe': 'PIPE_M', 'cycles': 100},
        ],
        'tensors': [{'id': 100, 'pos': 'L1', 'size': 60}],
        'edges': [{'source': 3, 'target': 100},
                  {'source': 100, 'target': 4}],
    }


def test_lower_bound_and_diagnostics_hold_for_valid_plan():
    model = Model(_toy_graph(), block_ops_cap=10)
    plan = model.plan_from([0, 1, 2, 3], [[0, 1, 2], [3]])
    diag = schedule_diagnostics(model, plan, 'A', 2)
    assert diag['lower_bounds']['overall'] <= diag['proxy_makespan']
    assert diag['critical_subgraphs'][-1] == 3
    assert len(diag['proxy_core_idle']) == 2


def test_critical_window_exact_search_moves_release_task_earlier():
    pytest.importorskip('ortools')
    from order_refine import refine_orders_exact

    model = Model(_toy_graph(), block_ops_cap=10)
    sol = Sol([0, 1, 2, 3], [0, 0, 0, 1])
    original = [[0, 1, 2], [3]]
    orders, before, after, stats = refine_orders_exact(
        Context(model, 2, 'A'), sol, original,
        window=3, max_windows=1, seconds_per_window=1.0)
    assert stats['windows'] == 1
    assert after < before
    assert orders[0][0] == 2
    assert model.evaluate(sol.sg_of_block, sol.core_of_sg, 'A', 2,
                          use_cache=False, orders_override=orders)[0] == after


def test_search_seed_is_stable_and_task_specific():
    assert stable_seed('case_001', 'A', 2) == stable_seed('case_001', 'A', 2)
    assert stable_seed('case_001', 'A', 2) != stable_seed('case_001', 'B', 2)


def test_search_call_deadline_is_bounded_by_stage_and_global_limit():
    from pipeline import _bounded_deadline

    assert _bounded_deadline(20.0, 3.0, now=10.0) == 13.0
    assert _bounded_deadline(11.0, 3.0, now=10.0) == 11.0


def test_warm_portfolio_loads_cross_scene_and_pads_lower_cores(tmp_path):
    import json

    base = {'node_to_subgraph': {'1': 0}}
    plans = {
        'case_001_A_N5.json': ({'1': 0}, [[0], [], [], [], []]),
        'case_001_B_N5.json': ({'1': 0}, [[], [0], [], [], []]),
        'case_001_A_N2.json': ({'1': 0, '2': 1}, [[0, 1], []]),
    }
    for name, (mapping, schedules) in plans.items():
        (tmp_path / name).write_text(json.dumps({
            **base, 'node_to_subgraph': mapping,
            'core_schedules': schedules}), encoding='utf-8')

    rows = load_warm_candidates(
        'case_001', 'A', 5, [str(tmp_path)], portfolio=True)
    assert len(rows) == 3
    assert all(len(row['plan']['core_schedules']) == 5 for row in rows)
    padded = next(row for row in rows if row['source'].endswith('A:N2'))
    assert padded['plan']['core_schedules'][2:] == [[], [], []]


def test_legacy_n5_constructs_are_valid_and_only_enabled_for_five_cores():
    from pipeline import _legacy_n5_constructs

    model = Model(_toy_graph(), block_ops_cap=1)
    rows = _legacy_n5_constructs(model, 5, 'A')
    assert {row.paradigm for row in rows} == {
        'legacy_strip10', 'legacy_strip40', 'legacy_chain_uncapped'}
    assert all(row.sol.validate(model, 5) for row in rows)
    assert _legacy_n5_constructs(model, 4, 'A') == []


def test_legacy_cycle_repair_merges_cyclic_quotient_components():
    from pipeline import _repair_quotient_cycles

    class QuotientModel:
        block_edges = [(0, 1), (1, 2)]

    repaired = _repair_quotient_cycles(
        QuotientModel(), Sol([0, 1, 0], [0, 1]))
    assert repaired.sg_of_block == [0, 0, 0]


def test_mcts_search_returns_valid_schedule():
    from mcts_schedule import mcts_search, _normalized_reward

    model = Model(_toy_graph(), block_ops_cap=10)
    ctx = Context(model, 2, 'A', seed=3)
    root = Sol([0, 1, 2, 3], [0, 0, 1, 1])
    best, fitness, makespan, added, stats = mcts_search(
        ctx, root, time_budget=0.05, max_depth=2, branching=6,
        rng=__import__('random').Random(3))
    assert best.validate(model, 2)
    assert fitness >= 0
    assert makespan >= 0
    assert added >= 0
    assert stats['iterations'] >= 1
    assert stats['evaluated_states'] <= 128
    assert -1.0 <= stats['reward_min'] <= stats['reward_max'] <= 1.0
    assert _normalized_reward(100.0, 90.0) == pytest.approx(0.1)
    assert _normalized_reward(100.0, 300.0) == -1.0


def test_mcts_stops_after_neighborhood_is_exhausted():
    from mcts_schedule import mcts_search

    model = Model(_toy_graph(), block_ops_cap=10)
    ctx = Context(model, 2, 'A', seed=5)
    root = Sol([0, 1, 2, 3], [0, 0, 1, 1])
    _best, _fitness, _makespan, _added, stats = mcts_search(
        ctx, root, time_budget=2.0, max_depth=3, branching=8,
        max_evals=8, stall_limit=12,
        rng=__import__('random').Random(5))
    assert stats['evaluated_states'] <= 8
    assert stats['iterations'] < 1000
    assert stats['stop_reason'] in {'max_evals', 'stalled'}


def test_structural_beam_search_returns_valid_candidates():
    from beam_schedule import beam_search

    model = Model(_toy_graph(), block_ops_cap=10)
    ctx = Context(model, 2, 'A', seed=7)
    root = Sol([0, 1, 2, 3], [0, 0, 1, 1])
    pool = []
    best, fitness, makespan, added, stats = beam_search(
        ctx, root, time_budget=0.2, max_depth=3, width=4, branching=12,
        max_evals=24, candidate_pool=pool, candidate_cap=8)
    assert best.validate(model, 2)
    assert all(row[3].validate(model, 2) for row in pool)
    assert fitness >= 0 and makespan >= 0 and added >= 0
    assert 1 <= stats['evaluated_states'] <= 24
    assert stats['depth'] >= 1


def test_wavefront_skips_non_five_core():
    from wavefront_schedule import generate_candidates

    class Ctx:
        num_cores = 4

    candidates, stats = generate_candidates(Ctx(), object())
    assert candidates == []
    assert stats['status'] == 'skipped'
