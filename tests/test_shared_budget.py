"""共同预算不能被重复来源、真实成绩或空家族改变。"""
import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'solver'))

def candidate(pid,assignment,coarse):
    return {'plan_id':pid,'assignment':assignment,'coarse':coarse,'features':{'boundary_bytes':10}}

def test_six_slots_deduplicate_and_prefer_different_partitions():
    from shared_budget import select_shared
    pools={'schedule':[candidate('a',[0,0],1),candidate('b',[0,0],2)],
           'partition':[candidate('a',[0,0],1),candidate('c',[0,1],2),candidate('d',[1,0],3),candidate('e',[0,0],4)],
           'region':[candidate('f',[0,1],1),candidate('g',[0,0],2)]}
    chosen,audit=select_shared(pools)
    assert len(chosen)==6 and len({r['plan_id'] for r in chosen})==6
    assert [r['plan_id'] for r in chosen]==['a','b','c','e','f','g']
    assert audit['counts']==dict.fromkeys(pools,2)

def test_empty_family_transfers_by_minimum_count_and_truth_never_changes_selection():
    from shared_budget import select_shared
    pools={'schedule':[],'partition':[candidate(str(i),[0,i%2],i+1) for i in range(5)],
           'region':[candidate('r'+str(i),[0,i%2],i+1) for i in range(5)]}
    before,audit=select_shared(pools)
    for rows in pools.values():
        for row in rows:row['official']={'makespan':-999};row['resource']={'makespan':0}
    after,_=select_shared(pools)
    assert [r['plan_id'] for r in before]==[r['plan_id'] for r in after]
    assert audit['counts']=={'schedule':0,'partition':3,'region':3}

def test_unknown_family_and_nonfinite_coarse_are_rejected():
    from shared_budget import select_shared
    with pytest.raises(ValueError):select_shared({'unknown':[]})
    with pytest.raises(ValueError):select_shared({'schedule':[candidate('a',[0],float('nan'))],'partition':[],'region':[]})

def test_full_official_requires_full_hash_and_same_plan(tmp_path):
    import gzip,json
    from evidence_index import digest,file_sha
    from baseline_evidence import read_full_official
    plan=tmp_path/'plan.json';plan.write_text('{}')
    full={'makespan':10,'data_movement_bytes':{'added_copy_bytes':3,'scheduled_copy_bytes':4,
        'partition_added_copy_bytes':1,'spill_added_copy_bytes':2}}
    trace=tmp_path/'trace.gz';trace.write_bytes(gzip.compress(json.dumps(full).encode()))
    metrics={'makespan':10,'added_copy_bytes':3,'scheduled_copy_bytes':4,'partition_added':1,'spill_added':2}
    row={'source_plan':str(plan),'source_sha256':file_sha(plan),'full_result_equal':True,
         'result_sha256':{'official':digest(full),'resource':digest(full)},'real':metrics,'artifacts':{str(trace):file_sha(trace)}}
    assert read_full_official(row)==(metrics,str(trace))
    row['result_sha256']['official']='wrong'
    with pytest.raises(ValueError):read_full_official(row)

def test_unified_scores_cache_local_tasks_but_replay_each_queue_and_ignore_old_truth():
    from shared_budget import unify_scores
    from model import Model
    from shared_candidate_pool import make_candidate
    from copy import deepcopy
    graph={'ops':[{'id':1,'op':'MATMUL','pipe':'PIPE_M','cycles':100},
                  {'id':2,'op':'MATMUL','pipe':'PIPE_M','cycles':80}],'edges':[],'tensors':[]}
    model=Model(graph,block_ops_cap=1)
    serial=make_candidate(model,model.plan_from([0,1],[[0,1],[]]),1,'serial')
    parallel=make_candidate(model,model.plan_from([0,1],[[0],[1]]),999,'parallel')
    pools={'schedule':[serial,parallel],'partition':[],'region':[deepcopy(parallel)]}
    before=deepcopy(pools);after,audit=unify_scores(graph,pools)
    assert pools==before and audit['local_models']==1 and audit['queue_replays']==3
    assert [r['coarse'] for r in after['schedule']]==[280,100]
    assert after['region'][0]['coarse']==100
    for family in pools:
        for r in pools[family]:r['coarse']=-900;r['resource']={'makespan':1}
    poisoned,_=unify_scores(graph,pools)
    assert [[r['coarse'] for r in poisoned[f]] for f in pools]==[[r['coarse'] for r in after[f]] for f in pools]
    assert all(a['plan']==b['plan'] for f in pools for a,b in zip(after[f],before[f]))
    relabeled=make_candidate(model,model.plan_from([1,0],[[0],[1]]),123,'labels_changed')
    _,different=unify_scores(graph,{'schedule':[parallel,relabeled],'partition':[],'region':[]})
    assert different['local_models']==2

def test_total_copy_tie_break_counts_spill_but_never_overrides_time():
    from shared_budget import select_shared
    def row(pid,time,boundary,spill):
        return {'plan_id':pid,'assignment':[0,0],'coarse':time,'features':{'boundary_bytes':boundary},
                'unified_local':{'spill_bytes':spill,'bandwidth_floor':(boundary+spill)/60}}
    pools={'schedule':[row('low_boundary',100,10,90),row('low_total',100,30,10),row('slower',101,1,0)],
           'partition':[],'region':[]}
    old,_=select_shared(pools);new,_=select_shared(pools,tie_break='total_copy')
    assert old[0]['plan_id']=='low_boundary' and new[0]['plan_id']=='low_total'
    assert new[1]['plan_id']=='low_boundary' and new[2]['plan_id']=='slower'
    del pools['schedule'][0]['unified_local']
    with pytest.raises(ValueError):select_shared(pools,tie_break='total_copy')
