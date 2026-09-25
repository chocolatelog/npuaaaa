"""E06下一子项：未完成前驱计数、固定队首与重复完成防护。"""
import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'solver'))


def fixture_state():
    from scene_a_replay import TaskCoreState
    tasks={0:{'core_id':0,'pred_tasks':[],'seq':[1]},
           1:{'core_id':1,'pred_tasks':[0],'seq':[2]},
           2:{'core_id':0,'pred_tasks':[],'seq':[3]},
           3:{'core_id':1,'pred_tasks':[0,2],'seq':[4]}}
    return TaskCoreState(tasks,{0:[0,2],1:[1,3]},2,1000,100)


def test_cross_core_release_wait_and_queue_head():
    state=fixture_state()
    assert state.release_time(1) is None
    with pytest.raises(ValueError):state.activate(2,0)
    state.activate(0,0);state.complete_op(0,1);state.complete(0,10)
    assert state.pending_predecessors[1]==0 and state.pending_predecessors[3]==1
    assert state.release_time(1)==1010 and state.release_time(2)==110
    with pytest.raises(ValueError):state.activate(1,1009)
    state.activate(1,1010)
    assert state.core_active[1]==1


def test_duplicate_completion_cannot_unlock_successor_twice():
    state=fixture_state();state.activate(0,0)
    with pytest.raises(ValueError):state.complete(0,1)
    state.complete_op(0,1)
    with pytest.raises(ValueError):state.complete_op(0,1)
    state.complete(0,10)
    with pytest.raises(ValueError):state.complete(0,10)
    assert state.pending_predecessors[3]==1
    assert state.completed_count==1


def test_tables_do_not_share_mutable_state_and_preserve_active_order():
    first=fixture_state();second=fixture_state();first.activate(0,0)
    assert second.core_active[0] is None
    assert first.active_tasks_in_original_order()==[0]
    assert first.task_version[0]>second.task_version[0]
