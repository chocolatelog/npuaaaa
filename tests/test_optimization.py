import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'solver'))

from diagnostics import schedule_diagnostics  # noqa: E402
from model import Model  # noqa: E402
from solution import Context, Sol  # noqa: E402
from run_all import stable_seed  # noqa: E402


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


def test_mcts_search_returns_valid_schedule():
    from mcts_schedule import mcts_search

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
