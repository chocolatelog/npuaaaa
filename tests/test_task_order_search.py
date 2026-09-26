"""固定切分迁核、顺序搜索的合法性与有效性。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'solver'))


def test_replay_rejects_cycle_created_by_core_order():
    from task_order_search import replay_tasks
    durations = {0: 10, 1: 10, 2: 100}
    preds = {0: set(), 1: {0}, 2: {1}}
    assert replay_tasks(durations, preds, [[2, 0], [1]]) is None
    valid = replay_tasks(durations, preds, [[0, 2], [1]])
    assert valid['makespan'] == 2120


def test_search_migrates_tasks_and_preserves_coverage():
    from task_order_search import search_orders
    durations = {0: 9, 1: 8, 2: 3, 3: 2}
    preds = {s: set() for s in durations}
    orders = [[0, 1], [2, 3]]
    candidates, stats = search_orders(durations, preds, orders, same_wait=0,
                                       cross_wait=0, seconds=10)
    assert candidates[0]['makespan'] == 11
    assert sorted(s for o in candidates[0]['orders'] for s in o) == [0, 1, 2, 3]
    assert orders == [[0, 1], [2, 3]]
    assert stats['evaluations'] <= 4000


def test_replay_rejects_missing_and_duplicate_tasks():
    from task_order_search import replay_tasks
    assert replay_tasks({0: 10, 1: 10}, {0: set(), 1: set()}, [[0, 0], [1]]) is None
    assert replay_tasks({0: 10, 1: 10}, {0: set(), 1: set()}, [[0], []]) is None


def test_swaps_cross_a_single_migration_local_optimum():
    import inspect
    from task_order_search import search_orders
    assert 'enable_swaps' in inspect.signature(search_orders).parameters
    durations={0:8,1:7,2:6,3:5}
    preds={s:set() for s in durations}
    orders=[[0,1],[2,3]]
    old,_=search_orders(durations,preds,orders,same_wait=0,cross_wait=0,
                        rounds=1,seconds=None,max_evals=100,enable_swaps=False)
    new,stats=search_orders(durations,preds,orders,same_wait=0,cross_wait=0,
                            rounds=1,seconds=None,max_evals=100,enable_swaps=True)
    assert not old
    assert new[0]['makespan']==13
    assert stats['by_neighborhood']['swap']['evaluations']>0
    assert stats['evaluations']<=100
    assert orders==[[0,1],[2,3]]


def test_swaps_respect_combined_precedence_and_core_order():
    import inspect
    from task_order_search import search_orders, replay_tasks
    assert 'enable_swaps' in inspect.signature(search_orders).parameters
    durations={0:8,1:7,2:6,3:5}
    preds={0:set(),1:{0},2:{1},3:{2}}
    candidates,stats=search_orders(durations,preds,[[0,1],[2,3]],
                                  rounds=2,seconds=None,max_evals=100,enable_swaps=True)
    assert stats['invalid']>0
    assert all(replay_tasks(durations,preds,c['orders']) is not None for c in candidates)


def test_window_neighborhood_reorders_local_core_window():
    from task_order_search import search_orders

    durations = {0: 5, 1: 7, 2: 6, 3: 7}
    preds = {0: set(), 1: {0}, 2: set(), 3: set()}
    orders = [[0, 2], [1, 3]]
    candidates, stats = search_orders(
        durations, preds, orders, same_wait=0, cross_wait=10,
        rounds=1, seconds=None, max_evals=100, max_sources=0,
        enable_window=True, window_size=2, window_candidates=32)

    assert stats['by_neighborhood']['window']['evaluations'] > 0
    assert candidates
    assert candidates[0]['orders'] == [[0, 2], [3, 1]]
    assert orders == [[0, 2], [1, 3]]


def test_window_neighborhood_respects_size_and_budget():
    from task_order_search import search_orders

    durations = {i: 1 for i in range(8)}
    preds = {i: set() for i in durations}
    orders = [list(range(4)), list(range(4, 8))]
    _candidates, stats = search_orders(
        durations, preds, orders, same_wait=0, cross_wait=0,
        rounds=3, seconds=None, max_evals=9, max_sources=0,
        enable_window=True, window_size=2, window_candidates=5)

    assert stats['evaluations'] <= 9
    assert stats['by_neighborhood']['window']['evaluations'] <= 5


def test_pipeline_exposes_window_refine_switch():
    import inspect
    import pipeline

    assert 'terminal_window' in inspect.signature(pipeline.solve_case).parameters


def test_reranking_checks_original_official_before_accepting(monkeypatch):
    import inspect
    import task_order_search as search
    assert 'candidate_evaluator' in inspect.signature(search.refine_plan_orders).parameters
    graph={'ops':[{'id':i,'op':'MATMUL','pipe':'PIPE_M','cycles':10} for i in (0,1)],
           'tensors':[],'edges':[]}
    plan={'node_to_subgraph':{'0':0,'1':1},'core_schedules':[[0,1],[]]}
    base={'makespan':120,'added_copy_bytes':0,'scheduled_copy_bytes':0,'spill_added':0,'partition_added':0}
    monkeypatch.setattr(search,'search_orders',lambda *a,**k: (
        [{'makespan':10,'orders':[[0],[1]],'origin':'swap'}],{'base_makespan':120}))
    fake={**base,'makespan':1}
    calls=[]
    def official(*args):
        calls.append(1)
        return {**base,'makespan':10}
    selected,best,audit=search.refine_plan_orders(graph,plan,base,official,
        enable_swaps=True,candidate_evaluator=lambda *a:fake)
    assert calls==[1]
    assert selected==plan and best==base
    assert any(e['stage']=='replica_mismatch' for e in audit['errors'])
