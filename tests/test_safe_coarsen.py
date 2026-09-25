"""保守收缩必须保持完整拓扑合法性，不能为凑数量强制串行。"""
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'solver'))
from solution import Sol


def test_alternate_path_edge_cannot_be_contracted():
    from coarsen_safe import admissible_pairs
    model=SimpleNamespace(block_edges=[((0,1),1),((1,2),1),((0,2),1)])
    sol=Sol([0,1,2],[0,1,0])
    assert set(admissible_pairs(model,sol)) == {(0,1),(1,2)}


def test_disconnected_heavy_tasks_keep_parallelism_and_soft_target():
    from coarsen_safe import safe_coarsen
    model=SimpleNamespace(blocks=[[0],[1]],block_edges=[],block_work_m=[100,100],
                          block_work_v=[0,0],tg=[])
    sol=Sol([0,1],[0,1])
    value,audit=safe_coarsen(model,sol,2,target=1,ops_cap=2)
    assert value.sg_of_block == sol.sg_of_block and value.core_of_sg == sol.core_of_sg
    assert not audit['target_reached']


def test_merge_keeps_full_coverage_and_explicit_core_ownership():
    from coarsen_safe import safe_coarsen
    model=SimpleNamespace(blocks=[[0],[1]],block_edges=[((0,1),6000)],
                          block_work_m=[1,2],block_work_v=[0,0],
                          tg=[([0],[1],False,'UB',6000)])
    original=Sol([0,1],[0,1])
    value,audit=safe_coarsen(model,original,2,target=1,ops_cap=2)
    assert value.validate(model,2) and value.num_used_sg()==1
    assert original.sg_of_block==[0,1]
    assert audit['target_reached'] and audit['candidate_count']>0
