import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SOLVER = ROOT / "solver"
sys.path.insert(0, str(SOLVER))

from construct_refine import (  # noqa: E402
    ConstructCandidate,
    build_construct_candidates,
    _evaluate,
    _coarsen_solution,
    _refine_boundary,
    _refine_order,
    resolve_granularity,
)
from solution import Sol  # noqa: E402
from construct import chain_construct  # noqa: E402


class TinyModel:
    eligible = list(range(900))
    blocks = [list(range(100)), list(range(100, 200)), list(range(200, 300))]
    block_work_m = [10.0, 20.0, 30.0]
    block_work_v = [10.0, 20.0, 30.0]
    block_edges = [(0, 1), (1, 2)]


class ChainModel:
    blocks = [[0], [1], [2], [3], [4], [5]]
    block_work_m = [1.0] * 6
    block_work_v = [1.0] * 6
    block_pos = list(range(6))
    block_topo = list(range(6))
    block_edges = [((i, i + 1), 1.0) for i in range(5)]
    tg = []


def _fake_builder(model, cores, scene, **kwargs):
    return [0, 1, 2], [0, 1, 0]


def test_granularity_scales_with_graph_and_core_count():
    profile = resolve_granularity(TinyModel(), 3, "balanced")
    assert profile.target_ops == 128
    assert profile.target_strips == 9
    assert profile.name == "balanced"


def test_build_candidates_has_twelve_r0_candidates(monkeypatch):
    import construct_refine

    for name in ("heft_construct", "strip_construct", "chain_construct", "netbenefit_construct"):
        monkeypatch.setattr(construct_refine, name, _fake_builder)
    items = build_construct_candidates(TinyModel(), 2, "A", ctx=None)
    assert len(items) == 12
    assert all(isinstance(item, ConstructCandidate) for item in items)
    assert {item.refine_level for item in items} == {"R0"}
    assert {item.paradigm for item in items} == {"heft", "strip", "chain", "netbenefit"}
    assert {item.granularity for item in items} == {"coarse", "balanced", "fine"}
    assert all(isinstance(item.sol, Sol) for item in items)


def test_legacy_pool_returns_solutions(monkeypatch):
    import pipeline

    monkeypatch.setattr(pipeline, "build_construct_candidates", lambda *args, **kwargs: [
        ConstructCandidate("heft", "coarse", "A", "R0", Sol([0], [0]), None, "x")
    ])
    pool = pipeline.build_construct_pool(TinyModel(), 1, "A")
    assert len(pool) == 1
    assert isinstance(pool[0], Sol)


class FakeContext:
    num_cores = 2

    def evaluate(self, sol):
        # 结构变化必须能被精化层观察到；spill 使用真实 info 字段名。
        mk = float(sum(sol.sg_of_block) + len(sol.core_of_sg))
        return mk, 10.0, {
            "spill_sig": {"L1p": 7.0, "L1w": 3.0, "UBp": 0.0, "UBw": 0.0}
        }


def test_evaluate_reads_spill_signal_and_runs_refinement(monkeypatch):
    import construct_refine

    for name in ("heft_construct", "strip_construct", "chain_construct", "netbenefit_construct"):
        monkeypatch.setattr(construct_refine, name, _fake_builder)
    ctx = FakeContext()
    items = build_construct_candidates(TinyModel(), 2, "A", ctx=ctx,
                                       max_base=12, max_r1=8, max_r2=6,
                                       max_r3=4, final_seeds=6)
    assert any(item.refine_level != "R0" for item in items)
    assert all(item.metrics["spill_bytes"] == 10.0 for item in items)


def test_boundary_refinement_uses_graph_edge_and_changes_partition():
    parent = ConstructCandidate("heft", "balanced", "A", "R1",
                                Sol([0, 1, 2], [0, 0, 1]))
    child = _refine_boundary(parent, TinyModel(), "A", FakeContext())
    assert child.sol.sg_of_block != parent.sol.sg_of_block
    assert child.sol.sg_of_block[0] == child.sol.sg_of_block[1]


def test_order_refinement_swaps_independent_subgraphs():
    model = TinyModel()
    model.block_edges = [(0, 1)]
    parent = ConstructCandidate("heft", "balanced", "A", "R2",
                                Sol([0, 1, 2], [0, 1, 0]))
    child = _refine_order(parent, model, "A", FakeContext())
    assert child.sol.core_of_sg != parent.sol.core_of_sg


def test_pipeline_seed_selection_preserves_structural_diversity():
    import pipeline

    scored = []
    for i, paradigm in enumerate(("heft", "strip", "chain", "netbenefit")):
        scored.append((float(i), Sol([0, 1], [0, 1]), 10.0 + i, 0.0,
                       ConstructCandidate(paradigm, "balanced", "A", "R0",
                                           Sol([0, 1], [0, 1]))))
    seeds = pipeline._select_seed_solutions(scored, 4)
    assert len(seeds) == 4
    assert {x[4].paradigm for x in seeds} == {"heft", "strip", "chain", "netbenefit"}


def test_chain_construct_respects_subgraph_cap():
    sg, cores = chain_construct(ChainModel(), 2, "A", max_sg_ops=1,
                                max_subgraphs=2)
    assert len(cores) <= 2
    assert len(set(sg)) <= 2


def test_solution_validation_checks_coverage_and_quotient_dag():
    model = TinyModel()
    model.block_edges = [((0, 1), 1.0), ((1, 2), 1.0)]
    valid = Sol([0, 0, 1], [0, 1])
    cyclic = Sol([0, 1, 0], [0, 1])
    assert valid.validate(model, 2)
    assert not cyclic.validate(model, 2)


def test_coarsen_solution_limits_all_paradigms():
    sol = Sol(list(range(6)), [0, 0, 0, 0, 0, 0])
    out = _coarsen_solution(ChainModel(), sol, 2)
    assert out.num_used_sg() <= 2
