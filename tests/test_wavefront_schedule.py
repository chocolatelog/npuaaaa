"""Deterministic unit coverage for the migration-only wavefront generator."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'solver'))

from solution import Sol
from wavefront_schedule import generate_candidates


class IndependentTasks:
    """A small deterministic load model; no official speedup claims."""

    def __init__(self, durations):
        self.durations = list(durations)
        self.blocks = [[i] for i in range(len(durations))]
        self.block_edges = []

    def evaluate(self, groups, cores, scene, num_cores, **kwargs):
        assert groups == list(range(len(self.blocks)))
        orders = [[] for _ in range(num_cores)]
        loads = [0.0] * num_cores
        ends = [0.0] * len(groups)
        for group, core in enumerate(cores):
            orders[core].append(group)
            loads[core] += self.durations[group]
            ends[group] = loads[core]
        return max(loads), 0.0, {
            'orders': orders,
            'durs': list(self.durations),
            'end_time': ends,
            'sg_preds': [[] for _ in groups],
        }


@pytest.mark.parametrize('limit', [1, 2, 8, 24])
def test_migrations_are_valid_unique_bounded_and_preserve_root(limit):
    model = IndependentTasks([8, 7, 6, 5, 4, 3])
    ctx = SimpleNamespace(model=model, num_cores=5, scene='B')
    root = Sol(list(range(6)), [0] * 6)
    original = root.clone()

    rows, stats = generate_candidates(ctx, root, max_actions=limit)

    assert 0 < len(rows) <= limit
    assert stats['status'] == 'ok'
    signatures = set()
    for makespan, added, child, action in rows:
        assert child.validate(model, 5)
        assert child.sg_of_block == original.sg_of_block
        assert sum(a != b for a, b in zip(
            child.core_of_sg, original.core_of_sg)) == 1
        assert action[0] == 'move_core'
        assert action[2] == 0 and action[3] != 0
        assert (makespan, added) < (sum(model.durations), 0.0)
        signatures.add((tuple(child.sg_of_block), tuple(child.core_of_sg)))
    assert len(signatures) == len(rows)
    assert root.sg_of_block == original.sg_of_block
    assert root.core_of_sg == original.core_of_sg
    assert root.blocks_in_sg == original.blocks_in_sg
    # Mutating a returned child must not mutate the input's nested sets.
    rows[0][2].blocks_in_sg[0].clear()
    assert root.blocks_in_sg == original.blocks_in_sg


def test_balanced_loads_do_not_generate_moves():
    model = IndependentTasks([10] * 5)
    ctx = SimpleNamespace(model=model, num_cores=5, scene='B')
    root = Sol(list(range(5)), list(range(5)))
    rows, stats = generate_candidates(ctx, root)
    assert rows == []
    assert stats['status'] == 'balanced'


def test_non_five_core_path_does_not_evaluate_a_model():
    ctx = SimpleNamespace(num_cores=4)
    rows, stats = generate_candidates(ctx, object())
    assert rows == []
    assert stats == {'status': 'skipped', 'reason': 'requires_5_cores'}


def test_real_model_migration_smoke_preserves_a_valid_partition():
    from model import Model
    from solution import Context

    graph = {
        'ops': [
            {'id': i, 'op': 'COMPUTE', 'pipe': 'PIPE_M', 'cycles': 10000}
            for i in range(1, 7)
        ],
        'tensors': [],
        'edges': [],
    }
    model = Model(graph, block_ops_cap=1)
    root = Sol(list(range(len(model.blocks))), [0] * len(model.blocks))
    rows, stats = generate_candidates(Context(model, 5, 'B'), root, max_actions=4)
    assert stats['status'] == 'ok'
    assert rows
    assert all(row[2].validate(model, 5) for row in rows)
    assert root.core_of_sg == [0] * len(model.blocks)
