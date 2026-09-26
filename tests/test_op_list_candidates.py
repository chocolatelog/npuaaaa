"""操作级生成器：完整依赖、原拓扑顺序、资源插空与输入防护。"""
import copy
from pathlib import Path
import sys
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'solver'))


def test_calendar_fits_holes_not_only_tail():
    import op_list_candidates as m
    assert m._fit([(10,20),(30,40)],0,8)==0
    assert m._fit([(10,20),(30,40)],12,9)==20
    assert m._fit([(10,20),(30,40)],12,11)==40


def test_copy_contraction_preserves_raw_topology_and_all_precedence():
    from dag_priority import build_compute_dag
    graph=dict(ops=[dict(id=i,op='COPY_IN' if i==100 else 'ADD',pipe='PIPE_V',cycles=3) for i in (1,2,100)],
               tensors=[],edges=[dict(source=100,target=1)])
    ops,preds,topo,durations=build_compute_dag(graph)
    assert topo==[2,1]  # 原图堆先取2再取100；收缩后重排则错成1、2。
    graph['edges']=[dict(source=2,target=100),dict(source=100,target=1)]
    ops,preds,topo,durations=build_compute_dag(graph)
    assert preds[1]=={2} and topo==[2,1] and durations[1]==3


def test_generated_plans_are_deterministic_legal_and_do_not_mutate_parent():
    from op_list_candidates import generate_op_candidates
    from scenario_contract import validate_plan
    graph=dict(ops=[dict(id=i,op='ADD',pipe='PIPE_V' if i%2 else 'PIPE_M',cycles=10+i) for i in range(1,8)],
               tensors=[dict(id=20,size=128,pos='UB')],edges=[dict(source=1,target=20),dict(source=20,target=4),dict(source=2,target=5),dict(source=5,target=7)])
    parent=dict(node_to_subgraph={str(i):0 for i in range(1,8)},core_schedules=[[0],[]])
    original=copy.deepcopy((graph,parent))
    rows,audit=generate_op_candidates(graph,parent,num_cores=2,max_candidates=12,bandwidth=60)
    assert rows and len(rows)<=12
    assert all(validate_plan(graph,r['plan'],'B') for r in rows)
    assert rows==generate_op_candidates(graph,parent,num_cores=2,max_candidates=12,bandwidth=60)[0]
    assert (graph,parent)==original
    assert audit['configurations']==12
    comm,comm_audit=generate_op_candidates(graph,parent,num_cores=2,max_candidates=12,bandwidth=60,communication_rank=True,cross_delay=100)
    assert comm_audit['configurations']==2
    assert all(validate_plan(graph,r['plan'],'B') for r in comm)
    with pytest.raises(ValueError):generate_op_candidates(graph,parent,num_cores=2,communication_rank=True)
    with pytest.raises(ValueError):generate_op_candidates(graph,parent,num_cores=0)
    with pytest.raises(ValueError):generate_op_candidates(graph,parent,bandwidth=float('nan'))
    with pytest.raises(ValueError):generate_op_candidates(graph,parent,max_candidates=-1)


def test_runner_uses_B_contract_without_A_task_barriers(tmp_path,monkeypatch):
    import run_saved_refine as runner
    from scenario_contract import validate_plan
    graph=dict(ops=[dict(id=i,op='ADD',pipe='PIPE_M' if i in (0,1) else 'PIPE_V',cycles=10) for i in range(4)],
               tensors=[],edges=[dict(source=0,target=1),dict(source=2,target=3)])
    parent=dict(node_to_subgraph={str(i):i for i in range(4)},core_schedules=[[3,0],[1,2]])
    validate_plan(graph,parent,'B')
    with pytest.raises(ValueError):validate_plan(graph,parent,'A')
    attachment=tmp_path/'attachment';source=tmp_path/'source';out=tmp_path/'out'
    runner.write_json(attachment/'data/case_001.json',graph)
    (attachment/'data/config.txt').write_text('fixture')
    (attachment/'code').mkdir()
    runner.write_json(source/'case_001_B_N2.json',parent)
    monkeypatch.setattr(runner,'ATTACHMENT',attachment)
    calls=[]
    def evaluate(job,folder,plans,**kw):
        path=plans/'case_001_B_N2.json';calls.append(job)
        return dict(status='official_success',record_id='test',plan_sha256=runner.digest(path),real=dict(makespan=20,added_copy_bytes=0))
    monkeypatch.setattr(runner,'evaluate_job',evaluate)
    settings=dict(input_dir=str(source),output_dir=str(out),portfolio_only=True,portfolio_dir=[],timeout=20)
    row,_=runner.run_task(dict(case='case_001',N=2,kind='problem_2',plan_scene='B'),settings,'test',{})
    assert row['real']['makespan']==20 and len(calls)==1


def test_communication_height_prioritizes_heavy_transfer_chain():
    from op_list_candidates import _schedule
    ops={i:dict(id=i,op='ADD',pipe='PIPE_V') for i in range(4)}
    preds={0:set(),1:set(),2:{0},3:{1}}
    durations={0:20,1:1,2:20,3:1};traffic={(1,3):1000}
    old=_schedule(ops,preds,list(ops),durations,2,0,traffic,'height')
    new=_schedule(ops,preds,list(ops),durations,2,0,traffic,'communication_height')
    def core(plan,op):
        return next(c for c,row in enumerate(plan['core_schedules']) if plan['node_to_subgraph'][str(op)] in row)
    assert core(old,0)==0
    assert core(new,1)==0
    assert _schedule(ops,preds,list(ops),durations,1,0,traffic,'communication_height')==_schedule(ops,preds,list(ops),durations,1,0,traffic,'height')
