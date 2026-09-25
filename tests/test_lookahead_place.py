"""前瞻必须补链外前驱、保留固定前缀且严格计费。"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'solver'))


def test_window_closes_external_parents_and_reduces_depth_if_over_limit():
    from lookahead_place import bounded_window
    preds={0:set(),1:{0},2:set(),3:{1,2}}
    future,closure,note=bounded_window(0,[1,3],preds,set(),8)
    assert future==[1,3] and closure=={0,1,2,3}
    future,closure,note=bounded_window(0,[1,3],preds,set(),3)
    assert future==[1] and closure=={0,1} and note['removed_future']==1
    future,closure,note=bounded_window(0,[1,3],preds,{2},3)
    assert future==[1,3] and closure=={0,1,3}


def test_ready_frontier_supplies_independent_future_tasks_without_unlocking_blocked_parents():
    from lookahead_place import future_decisions
    preds={0:set(),1:set(),2:set(),3:{4},4:set()}
    durations={0:20,1:15,2:10,3:100,4:1}
    tasks={t:{'start':0,'duration':durations[t]} for t in preds}
    assert future_decisions(0,preds,durations,2,tasks,set(),set(),'chain')==[]
    assert future_decisions(0,preds,durations,2,tasks,set(),set(),'ready')==[1,2]
    assert future_decisions(0,preds,durations,2,tasks,set(),{1},'ready')==[2,4]


def test_data_ready_window_unlocks_queued_independent_work_but_keeps_running_prefix():
    from lookahead_place import decision_window
    tasks={0:{'start':0,'end':10},1:{'start':20,'end':30},
           2:{'start':5,'end':25},3:{'start':100,'end':120}}
    preds={0:set(),1:set(),2:set(),3:{0}}
    assert decision_window(3,preds,tasks,'start')==(100,{0,1,2},{0,1,2})
    assert decision_window(3,preds,tasks,'data_ready')==(10,{0,2},{0})
    assert decision_window(1,preds,tasks,'data_ready')==(0,set(),set())


def test_placements_preserve_locked_prefix_coverage_and_dependency():
    from lookahead_place import propose_placements
    from task_order_search import replay_tasks
    orders=[[0,1],[2,3]];preds={0:set(),1:{0},2:set(),3:{1,2}}
    candidates=propose_placements(orders,1,preds,{t:10 for t in preds},{0,2})
    assert orders==[[0,1],[2,3]]
    assert len(candidates)<=2
    for r in candidates:
        assert r['orders'][0][0]==0 and r['orders'][1][0]==2
        assert replay_tasks({t:10 for t in preds},preds,r['orders']) is not None
    assert propose_placements(orders,0,preds,{t:10 for t in preds},{0,2})==[]


def test_oracle_reuses_exact_plan_and_does_not_exceed_budget():
    from lookahead_place import CandidateOracle
    from types import SimpleNamespace
    plan={'node_to_subgraph':{'0':0,'1':1},'core_schedules':[[0],[1]]}
    raw={'makespan':10,'data_movement_bytes':{'added_copy_bytes':0,'scheduled_copy_bytes':0,
        'partition_added_copy_bytes':0,'spill_added_copy_bytes':0},
        'per_core_timeline':[{'core_id':0,'tasks':[{'task_id':0,'start':0,'end':10,'duration':10}]},
                             {'core_id':1,'tasks':[{'task_id':1,'start':0,'end':10,'duration':10}]}]}
    calls=[]
    def evaluator(*args):calls.append(1);return raw
    oracle=CandidateOracle({},plan,raw,evaluator,max_evals=1)
    assert oracle.evaluate([[0],[1]]) is not None and not calls
    assert oracle.evaluate([[0,1],[]]) is not None and len(calls)==1
    assert oracle.evaluate([[0,1],[]]) is not None and len(calls)==1
    assert oracle.evaluate([[],[0,1]]) is None
    assert oracle.stats['evaluations']==1


def test_proposed_head_is_not_an_executed_prefix(monkeypatch):
    import lookahead_place as module
    durations={0:1000,1:1,2:1};preds={0:set(),1:set(),2:{0}}
    plan={'node_to_subgraph':{'0':0,'1':1,'2':2},'core_schedules':[[0,2],[1]]}
    def raw(orders):
        timing=module.replay_tasks(durations,preds,orders)
        return {'makespan':timing['makespan'],
                'data_movement_bytes':{'added_copy_bytes':0,'scheduled_copy_bytes':0,
                    'partition_added_copy_bytes':0,'spill_added_copy_bytes':0},
                'per_core_timeline':[{'core_id':c,'tasks':[{'task_id':t,'start':timing['finish'][t]-durations[t],
                    'end':timing['finish'][t],'duration':durations[t]} for t in q]} for c,q in enumerate(orders)]}
    oracle=module.CandidateOracle({},plan,raw(plan['core_schedules']),lambda g,p,*args:raw(p['core_schedules']))
    original=module.propose_placements
    def forced(orders,task,*args,**kwargs):
        if task==0:return [{'orders':[[2],[1,0]],'core':1,'slot':1,'coarse':2102}]
        return original(orders,task,*args,**kwargs)
    monkeypatch.setattr(module,'propose_placements',forced)
    candidates,audit=module.search_lookahead(oracle,preds,depth=3,max_decisions=1)
    assert candidates and audit['decisions'][0]['completed_future_levels']==1


def test_keep_current_prevents_forced_bad_move_without_extra_evaluation(monkeypatch):
    import lookahead_place as module
    plan={'core_schedules':[[0],[1]]}
    def raw(orders,makespan):
        return {'makespan':makespan,'data_movement_bytes':{'added_copy_bytes':0,
            'scheduled_copy_bytes':0,'partition_added_copy_bytes':0,'spill_added_copy_bytes':0},
            'per_core_timeline':[{'tasks':[{'task_id':t,'start':i*10,'end':(i+1)*10,'duration':10}
                for i,t in enumerate(q)]} for q in orders]}
    oracle=module.CandidateOracle({},plan,raw(plan['core_schedules'],10),
        lambda g,p,*args:raw(p['core_schedules'],20))
    monkeypatch.setattr(module,'propose_placements',lambda *args:[{'orders':[[0,1],[]],'core':0,'slot':0,'coarse':9}])
    _,audit=module.search_lookahead(oracle,{0:set(),1:set()},depth=1,max_decisions=1,keep_current=True)
    assert audit['decisions'][0]['committed']==10
    assert audit['decisions'][0]['kept_current'] is True
    assert oracle.stats['evaluations']==1
