"""分支切分必须分离共享汇合，保持全局拓扑与完整块覆盖。"""
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'solver'))


def diamond():
    from solution import Sol
    model = SimpleNamespace(blocks=[[0], [1], [2], [3]], block_topo=[0, 1, 2, 3],
        block_pos={0: 0, 1: 1, 2: 2, 3: 3},
        block_edges=[((0, 1), 1), ((0, 2), 1), ((1, 3), 1), ((2, 3), 1)],
        block_work_m=[1, 100, 100, 1], block_work_v=[0, 0, 0, 0])
    return model, Sol([0, 0, 0, 0], [0])


def test_diamond_keeps_join_outside_exclusive_branches():
    from branch_partition import generate_branch_candidates
    model, parent = diamond()
    candidates, audit = generate_branch_candidates(model, parent, [0], 2)
    assert candidates and audit['legal'] > 0
    for row in candidates:
        child = row['sol']
        assert child.num_used_sg() == 4 and child.validate(model, 2)
        assert len(set(child.sg_of_block)) == 4
        assert child.core_of_sg[child.sg_of_block[1]] != child.core_of_sg[child.sg_of_block[2]]
        assert row['source']['shared_work'] == 1
        assert row['source']['exclusive_work'] == 200
    assert parent.sg_of_block == [0, 0, 0, 0]


def test_chain_does_not_fabricate_parallel_branches():
    from branch_partition import generate_branch_candidates
    model, parent = diamond()
    model.block_edges = [((0, 1), 1), ((1, 2), 1), ((2, 3), 1)]
    candidates, audit = generate_branch_candidates(model, parent, [0], 2)
    assert not candidates and audit['no_wide_level'] == 1


def test_region_members_need_not_be_contiguous_in_one_topological_order():
    from branch_partition import generate_branch_candidates
    from solution import Sol
    model = SimpleNamespace(blocks=[[i] for i in range(6)], block_topo=list(range(6)),
        block_pos={i: i for i in range(6)},
        block_edges=[((0, 1), 1), ((0, 2), 1), ((1, 3), 1), ((2, 4), 1), ((3, 5), 1), ((4, 5), 1)],
        block_work_m=[1, 10, 10, 10, 10, 1], block_work_v=[0]*6)
    rows, _ = generate_branch_candidates(model, Sol([0]*6, [0]), [0], 2)
    assert any(r['sol'].sg_of_block[1] == r['sol'].sg_of_block[3]
               and r['sol'].sg_of_block[1] != r['sol'].sg_of_block[2] for r in rows)


def test_branch_candidates_are_deterministic():
    from branch_partition import generate_branch_candidates
    model, parent = diamond()
    a, aa = generate_branch_candidates(model, parent, [0], 2)
    b, bb = generate_branch_candidates(model, parent, [0], 2)
    assert aa == bb
    assert [(r['sol'].sg_of_block, r['sol'].core_of_sg, r['source']) for r in a] == [
        (r['sol'].sg_of_block, r['sol'].core_of_sg, r['source']) for r in b]


def test_external_predecessor_and_successor_stay_in_their_tasks():
    from branch_partition import generate_branch_candidates
    from solution import Sol
    model = SimpleNamespace(blocks=[[i] for i in range(6)], block_topo=list(range(6)),
        block_pos={i: i for i in range(6)},
        block_edges=[((0, 1), 1), ((1, 2), 1), ((1, 3), 1), ((2, 4), 1), ((3, 4), 1), ((4, 5), 1)],
        block_work_m=[10, 1, 100, 100, 1, 10], block_work_v=[0]*6)
    parent = Sol([0, 1, 1, 1, 1, 2], [1, 0, 1])
    rows, _ = generate_branch_candidates(model, parent, [1], 2)
    assert rows
    for row in rows:
        child = row['sol']
        assert child.validate(model, 2)
        for block in (0, 5):
            group = child.sg_of_block[block]
            assert child.blocks_in_sg[group] == {block} and child.core_of_sg[group] == 1
