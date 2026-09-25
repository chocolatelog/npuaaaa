"""共享外存服务守恒、同刻取整、版本失效与候选隔离。"""
import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'solver'))


def test_join_advances_old_members_before_sharing():
    from scene_a_replay import DDRResource
    pool=DDRResource();pool.add((0,1),10,0)
    version,ends=pool.forecast()
    assert ends=={(0,1):10}
    pool.add((1,1),6,4)
    assert pool.remaining=={(0,1):6.0,(1,1):6.0}
    assert pool.forecast()[1]=={(0,1):16,(1,1):16}
    assert not pool.is_current(version)


def test_finish_and_remove_do_not_give_past_bandwidth_to_survivor():
    from scene_a_replay import DDRResource
    pool=DDRResource();pool.add((0,1),10,0);pool.add((1,1),20,0)
    assert pool.forecast()[1]=={(0,1):20,(1,1):30}
    pool.remove((0,1),20)
    assert pool.remaining=={(1,1):10.0}
    assert pool.forecast()[1]=={(1,1):30}


def test_snapshot_branch_isolation_and_rounding():
    from scene_a_replay import DDRResource
    pool=DDRResource();pool.add((0,1),0.25,0);pool.add((1,1),0.25,0)
    snapshot=pool.snapshot();other=DDRResource.from_snapshot(snapshot)
    assert not other.is_current(pool.forecast()[0])
    assert other.forecast()[1]=={(0,1):1,(1,1):1}
    other.remove((0,1),1)
    assert pool.snapshot()==snapshot and (0,1) in pool.remaining
    with pytest.raises(ValueError):other.advance(0)


def test_rounding_progress_reclaims_service_after_fractional_completion():
    from scene_a_replay import DDRResource
    pool=DDRResource();pool.add((0,1),0.25,0);pool.add((1,1),2,0)
    assert pool.forecast()[1]=={(0,1):1,(1,1):3}
    pool.advance(1)
    assert pool.remaining=={(0,1):0.0,(1,1):1.25}
    pool.remove((0,1),1)
    assert pool.forecast()[1]=={(1,1):3}


@pytest.mark.parametrize('orders', [[[0,1],[]], [[0],[1]], [[],[0,1]]])
@pytest.mark.parametrize('unified', [False,True])
def test_resource_replay_full_official_fields_and_function_isolation(orders,unified):
    from scene_a_replay import build_resource_evaluator
    import multicore_cut_evaluate_problem_1 as official
    original=official.evaluate_scene_a
    graph={'ops':[{'id':0,'op':'MATMUL','pipe':'PIPE_M','cycles':10},
                  {'id':1,'op':'MATMUL','pipe':'PIPE_M','cycles':10},
                  {'id':2,'op':'VEC','pipe':'PIPE_V','cycles':7}],
           'tensors':[{'id':100,'pos':'UB','size':60}],
           'edges':[{'source':0,'target':100},{'source':100,'target':2}]}
    plan={'node_to_subgraph':{'0':0,'1':0,'2':1},'core_schedules':orders}
    params=(graph,plan,60,{'L1':524288,'UB':131072},1000,100)
    replay=build_resource_evaluator(unified=unified)
    assert replay(*params)==original(*params)
    assert original is official.evaluate_scene_a
    assert replay.__globals__ is not original.__globals__
