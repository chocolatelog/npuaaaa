"""完整块边界移动的拓扑、覆盖和候选隔离反例。"""
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'solver'))
from solution import Sol


def chain():
    return SimpleNamespace(blocks=[[1],[2],[3]], block_edges=[((0,1),1),((1,2),1)])


def test_atomic_move_rejects_cycle_and_closure_can_empty_source():
    from hotspot_partition import move_region, necessary_region
    model, parent = chain(), Sol([0,0,1],[0,1])
    assert move_region(model,parent,{0},1,2) is None
    region, foreign = necessary_region(model,parent,{0},1)
    assert region == {0,1} and not foreign
    child = move_region(model,parent,region,1,2)
    assert child.sg_of_block == [0,0,0] and child.core_of_sg == [1]
    assert child.blocks_in_sg == [{0,1,2}] and child.validate(model,2)
    assert parent.sg_of_block == [0,0,1] and parent.blocks_in_sg == [{0,1},{2}]


def test_region_does_not_silently_absorb_third_task():
    from hotspot_partition import necessary_region, move_region
    model, parent = chain(), Sol([0,1,2],[0,1,0])
    region, foreign = necessary_region(model,parent,{0},2)
    assert foreign == {1}
    assert move_region(model,parent,{0},2,2) is None


def test_noncontiguous_independent_consumers_move_as_whole_blocks():
    from hotspot_partition import move_region
    model = SimpleNamespace(blocks=[[1,4],[2],[3]],block_edges=[])
    parent = Sol([0,0,1],[0,1])
    child = move_region(model,parent,{0},1,2)
    assert child.blocks_in_sg == [{1},{0,2}]
    assert parent.blocks_in_sg == [{0,1},{2}]
