import copy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'solver'))
import run_oplist_full as full


def make_graph():
    return {'ops': [{'id':1,'op':'COMPUTE','pipe':'PIPE_M','cycles':10}],
            'tensors':[], 'edges':[]}


@pytest.mark.parametrize('scene', ['A','B'])
@pytest.mark.parametrize('cores', [2,3,4,5])
def test_task_passes_scene_and_cores_and_preserves_baseline(tmp_path,monkeypatch,scene,cores):
    baseline, output = tmp_path/'baseline',tmp_path/'output'
    baseline.mkdir()
    output.mkdir()
    plan = {'node_to_subgraph':{'1':0},'core_schedules':[[0]]+[[] for _ in range(cores-1)]}
    candidate = copy.deepcopy(plan)
    candidate['core_schedules'][0],candidate['core_schedules'][1]=[],[0]
    source = baseline/f'case_001_{scene}_N{cores}.json'
    original = json.dumps(plan)
    source.write_text(original)
    monkeypatch.setattr(full,'load_graph',lambda _:make_graph())
    def evaluate(graph, supplied, actual_scene, timeout):
        assert actual_scene == scene
        assert len(supplied['core_schedules']) == cores
        return {'makespan':100 if supplied['core_schedules'][0] else 80,'added_copy_bytes':0}
    def generate(graph, supplied, num_cores, max_candidates):
        assert num_cores == cores and max_candidates == 12
        assert supplied == plan
        return [dict(plan=candidate,source='test')],{'status':'ok'}
    monkeypatch.setattr(full,'evaluate',evaluate)
    monkeypatch.setattr(full,'generate_op_candidates',generate)
    row=full.run_task(dict(case='case_001',scene=scene,N=cores,
        baseline_dirs=[str(baseline)],output_dir=str(output),singlecore=100,
        evaluation_seconds=1,generation_seconds=1,max_candidates=12))
    assert row['real']['makespan']==80 and row['status']=='official'
    assert row['candidate_trace'][0]['accepted']
    assert row['certificate']['num_cores']==cores
    assert row['scene']==scene and row['N']==cores
    assert source.read_text()==original
    assert json.loads((output/source.name).read_text())==candidate
    assert row['plan_sha256']==full.signature(candidate)


def test_unverified_baseline_is_retained_without_fake_metrics(tmp_path,monkeypatch):
    baseline, output=tmp_path/'baseline',tmp_path/'output'
    baseline.mkdir()
    output.mkdir()
    plan={'node_to_subgraph':{'1':0},'core_schedules':[[0],[]]}
    name='case_001_A_N2.json'
    (baseline/name).write_text(json.dumps(plan))
    monkeypatch.setattr(full,'load_graph',lambda _:make_graph())
    monkeypatch.setattr(full,'evaluate',lambda *args:{'error':'timeout'})
    monkeypatch.setattr(full,'generate_op_candidates',lambda *args,**kwargs:pytest.fail('no verified baseline'))
    row=full.run_task(dict(case='case_001',scene='A',N=2,baseline_dirs=[str(baseline)],
        output_dir=str(output),singlecore=100,evaluation_seconds=1,generation_seconds=1,max_candidates=12))
    assert row['status']=='unverified_input_retained'
    assert row['real'] is None and row['speedup'] is None and row['certificate'] is None
    assert json.loads((output/name).read_text())==plan


def test_summary_separates_historical_subset_and_large_official_population():
    def row(case,large,speedup):
        return dict(case=case,scene='B',N=5,large=large,status='official',
                    baseline={'makespan':100,'added_copy_bytes':0},
                    real={'makespan':80,'added_copy_bytes':0},
                    baseline_speedup=speedup*.8,speedup=speedup)
    rows=[row('case_001',False,4),row('case_003',True,2)]
    keys=[('case_001','B',5),('case_003','B',5)]
    reference={keys[0]:{'real':{'makespan':120,'added_copy_bytes':0}},keys[1]:{'real':None}}
    result=full.summarize(rows,keys,reference)
    assert result['all_official']['mean_speedup']==3
    assert result['paired_historical']['mean_speedup']==4
    assert result['large_official_tasks']==1 and result['all_official_complete']
    assert result['versus_historical']['wins']==1
    assert result['versus_historical']['mean_paired_gain_percent']==50
    with pytest.raises(ValueError):
        full.summarize(rows+rows,keys,reference)
    rows[1]=dict(case='case_003',scene='B',N=5,status='failed',real=None)
    result=full.summarize(rows,keys,reference)
    assert result['unverified_tasks']==1 and not result['all_official_complete']
    assert result['by_scene_and_cores']['B_N5']['paired_complete']


def test_large_evaluation_uses_isolated_process(monkeypatch):
    graph={'ops':[{}]*12001}
    calls=[]
    def isolated(operation,payload,timeout):
        calls.append((operation,payload,timeout))
        return {'error':'timeout'}
    monkeypatch.setattr(full,'isolated_call',isolated)
    assert full.evaluate(graph,{},'A',3)=={'error':'timeout'}
    assert calls==[('evaluate',(graph,{},'A'),3)]


def test_isolated_child_failure_is_reported():
    result=full.isolated_call('invalid',None,10)
    assert 'unknown isolated operation' in result['error']
