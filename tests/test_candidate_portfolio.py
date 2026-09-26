"""外部/少核候选只能经合法性和同场景官方评估替换父方案。"""
import copy
import json
from pathlib import Path
import sys
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'solver'))
from candidate_portfolio import lift_plan, select_official_portfolio


def graph():
    return {'ops': [{'id': i, 'op': 'ADD', 'pipe': 'PIPE_V', 'cycles': 10} for i in (1,2)],
            'tensors': [], 'edges': []}


def test_lifting_preserves_identity_and_does_not_mutate_parent():
    plan={'node_to_subgraph': {'1': 7, '2': 12}, 'core_schedules': [[7],[12]]}
    before=copy.deepcopy(plan)
    result=lift_plan(plan,5)
    assert result==dict(node_to_subgraph=plan['node_to_subgraph'],core_schedules=[[7],[12],[],[],[]])
    result['core_schedules'][0].append(99)
    assert plan==before
    with pytest.raises(ValueError):lift_plan(plan,1)


def test_duplicate_invalid_timeout_and_worse_candidates_keep_official_best():
    parent={'node_to_subgraph': {'1':0,'2':1},'core_schedules':[[0,1],[]]}
    better={'node_to_subgraph': {'1':0,'2':1},'core_schedules':[[0],[1]]}
    invalid={'node_to_subgraph': {'1':0},'core_schedules':[[0],[]]}
    timeout={'node_to_subgraph': {'1':3,'2':4},'core_schedules':[[3],[4]]}
    worse={'node_to_subgraph': {'1':5,'2':6},'core_schedules':[[5],[6]]}
    base={'makespan':100,'added_copy_bytes':0}
    calls=[]
    def evaluate(g,p,s):
        calls.append(p)
        if p==timeout:raise RuntimeError('timeout')
        return {'makespan':80 if p==better else 120,'added_copy_bytes':1}
    candidates=[{'plan':p,'source':str(i)} for i,p in enumerate([parent,invalid,timeout,better,better,worse])]
    chosen,best,audit=select_official_portfolio(graph(),parent,base,'A',candidates,evaluate)
    assert chosen==better and best['makespan']==80
    assert len(calls)==3
    assert len(audit['errors'])==2
    assert audit['duplicate_count']==2
    assert audit['final_official']==best


def test_tie_in_time_never_accepts_more_copy_and_wrong_core_count_rejected():
    parent={'node_to_subgraph': {'1':0,'2':1},'core_schedules':[[0,1],[]]}
    equal={'node_to_subgraph': {'1':0,'2':1},'core_schedules':[[0],[1]]}
    too_many=lift_plan(parent,3)
    chosen,real,audit=select_official_portfolio(graph(),parent,{'makespan':100,'added_copy_bytes':0},'A',
        [{'plan':equal,'source':'tie'},{'plan':too_many,'source':'bad_core'}],
        lambda *args: {'makespan':100,'added_copy_bytes':1})
    assert chosen==parent and real['added_copy_bytes']==0 and len(audit['errors'])==1


def test_runner_portfolio_mode_uses_target_core_and_resumes(tmp_path,monkeypatch):
    import run_saved_refine as runner
    attachment=tmp_path/'attachment'; source=tmp_path/'source';extra=tmp_path/'H';out=tmp_path/'out'
    runner.write_json(attachment/'data/case_001.json',graph())
    (attachment/'code').mkdir();(attachment/'data/config.txt').write_text('fixture')
    monkeypatch.setattr(runner,'ATTACHMENT',attachment)
    parent={'node_to_subgraph':{'1':0,'2':1},'core_schedules':[[0,1],[],[]]}
    seed={'node_to_subgraph':{'1':0,'2':1},'core_schedules':[[0],[1]]}
    runner.write_json(source/'case_001_A_N3.json',parent)
    runner.write_json(extra/'case_001_A_N2.json',seed)
    calls=[]
    def official(job,folder,plans,**kw):
        assert job['N']==3
        path=plans/'case_001_A_N3.json';p=json.loads(path.read_text(encoding='utf8'))
        assert len(p['core_schedules'])==3
        calls.append(p)
        return dict(status='official_success',record_id=str(len(calls)),plan_sha256=runner.digest(path),
            real=dict(makespan=100 if p==parent else 70,added_copy_bytes=0))
    monkeypatch.setattr(runner,'evaluate_job',official)
    monkeypatch.setattr(runner,'verified_record',lambda r:r['status']=='official_success')
    settings=dict(input_dir=str(source),output_dir=str(out),portfolio_dir=[str(extra)],
        portfolio_only=True,include_lower_cores=True,timeout=20)
    job=dict(case='case_001',N=3,plan_scene='A',kind='problem_1')
    row,reused=runner.run_task(job,settings,'frozen',{})
    assert not reused and row['real']['makespan']==70 and len(calls)==2
    again,reused=runner.run_task(job,settings,'frozen',{})
    assert reused and len(calls)==2 and again==row


def test_portfolio_analysis_does_not_invent_proxy_error(tmp_path,monkeypatch):
    import analyze_saved_refine as analysis
    import run_saved_refine as runner
    path=tmp_path/'plans/case_001_A_N2.json'
    runner.write_json(path,{'saved':'plan'})
    metrics=dict(makespan=80,added_copy_bytes=0,scheduled_copy_bytes=0,spill_added=0,partition_added=0)
    row=dict(case='case_001',N=2,plan_scene='A',elapsed=1,baseline={**metrics,'makespan':100},
        real=metrics,official_record={},output_plan_sha256=runner.digest(path),
        audit=dict(candidates=[dict(official=metrics)],errors=[]))
    runner.write_json(tmp_path/'checkpoints/case_001_A_N2.json',row)
    monkeypatch.setattr(analysis,'verified_record',lambda r:True)
    result=analysis.analyze(tmp_path)
    assert result['completed']==1 and result['proxy_official_pairs']==[]
