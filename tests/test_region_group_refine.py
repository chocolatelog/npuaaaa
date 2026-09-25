"""区域动作遵守官方任务粒度、完整依赖与逐任务释放。"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'solver'))


def test_region_table_releases_each_core_without_barrier():
    from region_group_refine import RegionState
    import pytest
    region=RegionState({0:0,1:1})
    region.complete(0,0,10)
    assert region.pending=={1} and region.releases==[{'task':0,'core':0,'time':10,'remaining':1}]
    with pytest.raises(ValueError):region.complete(0,0,11)
    region.complete(1,1,20)
    assert not region.pending and region.releases[-1]['remaining']==0


def test_strict_chain_and_single_task_cannot_form_parallel_region():
    from region_group_refine import choose_regions
    assert choose_regions({0:100},{0:set()},[[0],[]])==[]
    assert choose_regions({0:10,1:20,2:30},{0:set(),1:{0},2:{1}},[[0,1],[2]])==[]


def test_joint_candidates_cover_atomic_exchange_and_preserve_outside_order():
    from region_group_refine import generate_joint_candidates
    from task_order_search import replay_tasks
    orders=[[0,2,3],[1]];preds={0:set(),1:set(),2:{0,1},3:{2}}
    rows,audit=generate_joint_candidates({t:10 for t in preds},preds,orders,[(0,1)],max_evals=2000)
    assert any(r['orders']==[[1,2,3],[0]] for r in rows)
    for row in rows:
        assert [[t for t in q if t not in {0,1}] for q in row['orders']]==[[2,3],[]]
        assert replay_tasks(dict.fromkeys(preds,10),preds,row['orders']) is not None
    assert audit['invalid']>0 and orders==[[0,2,3],[1]]
    _,small=generate_joint_candidates(dict.fromkeys(preds,10),preds,orders,[(0,1)],max_evals=2)
    assert small['evaluations']==2 and small['unfinished_streams']>0
