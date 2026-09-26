import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'solver'))


def test_beam_search_keeps_parent_and_valid_candidates():
    from bc_search import beam_search_candidates, plan_id
    from bc_refine import validate_plan

    graph = json.loads((ROOT / '通用神经网络处理器下的多核调度问题附件' /
                        'data' / 'case_001.json').read_text(encoding='utf-8'))
    parent = json.loads((ROOT / 'tests' / 'fixtures' /
                         'case_001_B_N2.json').read_text(encoding='utf-8'))
    plans, audit = beam_search_candidates(graph, parent, 'C', depth=2,
                                          width=3, max_states=12,
                                          max_candidates=6, seed=7)
    assert plans
    assert plan_id(plans[0]) == audit['parent_plan_id']
    assert all(validate_plan(graph, plan) for plan in plans)
    assert audit['expanded_states'] <= 12
    assert all('replay_mode' in row['metrics']
               for row in audit['candidates'][1:])


def test_beam_keeps_shallow_best_and_deduplicates_transpositions(monkeypatch):
    import bc_search as m
    from copy import deepcopy
    graph = {'ops':[{'id':i,'op':'ADD','cycles':1} for i in (1,2,3)],'tensors':[],'edges':[]}
    def p(order):return {'node_to_subgraph':{'1':1,'2':2,'3':3},'core_schedules':[order]}
    root, a, b, c = p([1,2,3]), p([2,1,3]), p([3,1,2]), p([3,2,1])
    visits = []
    def gen(g, parent, scene, max_candidates):
        key = m.plan_id(parent); visits.append(key)
        children = [a,b] if key == m.plan_id(root) else [c,c]
        plans = [parent] + children
        return plans, {'candidate_metrics':[{'plan_id':m.plan_id(x),'proxy_lower_bound':1 if x==a else 10,'action':'swap'} for x in plans]}
    monkeypatch.setattr(m,'generate_bc_candidates',gen)
    saved=deepcopy(root)
    plans,audit=m.beam_search_candidates(graph,root,'B',depth=3,width=2,max_states=6,max_candidates=4)
    assert a in plans
    assert len(visits)==len(set(visits))
    assert root==saved
    assert len({m.plan_id(x) for x in plans})==len(plans)


def test_validator_rejects_cycle_created_only_by_core_orders():
    from bc_refine import validate_plan
    g={'ops':[{'id':i,'op':'ADD'} for i in range(4)],'tensors':[],
       'edges':[{'source':0,'target':1},{'source':2,'target':3}]}
    p={'node_to_subgraph':{str(i):i for i in range(4)},'core_schedules':[[3,0],[1,2]]}
    assert not validate_plan(g,p)


def test_official_guard_accepts_fine_split_and_keeps_faster_seed(monkeypatch):
    import bc_search as m
    graph={'ops':[{'id':i,'op':'ADD'} for i in (1,2)],'edges':[],'tensors':[]}
    parent={'node_to_subgraph':{'1':0,'2':0},'core_schedules':[[0],[]]}
    seed={'node_to_subgraph':{'1':0,'2':1},'core_schedules':[[0],[1]]}
    finer={'node_to_subgraph':{'1':0,'2':1},'core_schedules':[[0,1],[]]}
    def candidates(*args,**kwargs):return [seed,finer], {'candidates':[]}
    monkeypatch.setattr(m,'beam_search_candidates',candidates)
    def evaluate(g,p,s):return {'makespan':{m.plan_id(parent):100,m.plan_id(seed):80,m.plan_id(finer):70}[m.plan_id(p)],'added_copy_bytes':0,'scheduled_copy_bytes':0}
    chosen,truth,audit=m.refine_official(graph,parent,'C',evaluate,seeds=[seed])
    assert chosen==finer and truth['makespan']==70
    assert audit['baseline_official']['makespan']==100
    assert audit['seed_best_official']['makespan']==80
    assert audit['final_official']['makespan']==70


def test_official_guard_rejects_nan_and_exception(monkeypatch):
    import bc_search as m
    g={'ops':[{'id':1,'op':'ADD'}],'edges':[],'tensors':[]}
    p={'node_to_subgraph':{'1':0},'core_schedules':[[0],[]]}
    q={'node_to_subgraph':{'1':0},'core_schedules':[[],[0]]}
    monkeypatch.setattr(m,'beam_search_candidates',lambda *a,**kw:([p,q],{}))
    def ev(g,x,s): return {'makespan':100 if x==p else float('nan'),'added_copy_bytes':0,'scheduled_copy_bytes':0}
    chosen,truth,audit=m.refine_official(g,p,'B',ev)
    assert chosen==p and truth['makespan']==100
    assert audit['errors']
