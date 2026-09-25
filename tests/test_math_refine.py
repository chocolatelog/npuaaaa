import copy
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'solver'))
from bound_certificate import build_compute_dag, compute_certificate
from run_math_refine import official_key, summarize


def graph(pipes, cycles=100):
    return {'ops':[{'id':i+1,'op':'COMPUTE','pipe':p,'cycles':cycles}
                   for i,p in enumerate(pipes)], 'tensors':[], 'edges':[]}


def test_compute_bound_respects_parallel_pipes_and_ceil():
    g=graph(['PIPE_M']*5+['PIPE_V']*5)
    original=copy.deepcopy(g)
    c=compute_certificate(g,5,100,100)
    assert c['overall']==100
    assert c['speedup_upper_bound']==1
    assert c['certified'] and not c['attainment_proven']
    assert g==original
    assert compute_certificate(graph(['PIPE_M'],101),5)['pipe_work_bounds']['PIPE_M']==21


def test_dependency_witness_does_not_charge_optional_communication():
    g=graph(['PIPE_M','PIPE_M'],10)
    g['edges']=[{'source':1,'target':2}]
    c=compute_certificate(g,5,20,20)
    assert c['dependency_path']==20
    assert c['critical_path_ops']==[1,2]


def test_copy_contraction_preserves_precedence_but_omits_copy_duration():
    g=graph(['PIPE_M','PIPE_V'],10)
    g['ops'].append({'id':3,'op':'COPY_IN','pipe':'PIPE_MTE2','cycles':999})
    g['tensors']=[{'id':100,'pos':'L1','size':60}]
    g['edges']=[{'source':1,'target':3},{'source':3,'target':100},{'source':100,'target':2}]
    ops,preds,topo,durations=build_compute_dag(g)
    assert set(ops)=={1,2} and preds[2]=={1}
    assert compute_certificate(g,5)['overall']==20


@pytest.mark.parametrize('fault',['duplicate','cycle','pipe','endpoint','contradiction'])
def test_invalid_certificates_fail_closed(fault):
    g=graph(['PIPE_M','PIPE_V'])
    if fault=='duplicate':
        g['ops'].append(dict(g['ops'][0]))
    elif fault=='cycle':
        g['edges']=[{'source':1,'target':2},{'source':2,'target':1}]
    elif fault=='pipe':
        g['ops'][0]['pipe']='UNKNOWN'
    elif fault=='endpoint':
        g['edges']=[{'source':1,'target':99}]
    with pytest.raises(ValueError):
        compute_certificate(g,5,incumbent_makespan=50 if fault=='contradiction' else None)


def test_official_gate_rejects_errors_and_nonfinite_objectives():
    for value in [None,{'error':'bad'},{'makespan':float('nan'),'added_copy_bytes':0},
                  {'makespan':0,'added_copy_bytes':0}]:
        with pytest.raises(ValueError):
            official_key(value)
    assert official_key({'makespan':100,'added_copy_bytes':3})==(100,3)
    with pytest.raises(ValueError):
        summarize([],1)


def test_local_cp_parallel_resources_and_no_input_mutation():
    pytest.importorskip('ortools')
    from pipe_local_cp import _solve,_view,generate_local_candidates
    g=graph(['PIPE_M','PIPE_V'])
    p={'node_to_subgraph':{'1':0,'2':1},'core_schedules':[[0,1],[],[],[],[]]}
    original=copy.deepcopy(p)
    ops,preds,topo,durations=build_compute_dag(g)
    result,stats=_solve(p,ops,preds,durations,5,1,2,0,1)
    assert stats['status']=='OPTIMAL' and stats['objective']==100
    assert 'not official' in stats['bound_scope']
    assert p==original
    _view(result,ops,preds,5)
    candidates,summary=generate_local_candidates(g,p,max_solves=2,seconds_per_solve=0.3)
    assert summary['solves']<=2
    assert p==original
    for row in candidates:
        _view(row['plan'],ops,preds,5)
    assert generate_local_candidates(g,p,max_solves=0)[1]['solves']==0


def test_local_split_preserves_original_coverage_and_precedence():
    from pipe_local_cp import _split,_view
    g=graph(['PIPE_M']*8,10)
    g['edges']=[{'source':i,'target':i+1} for i in range(1,8)]
    p={'node_to_subgraph':{str(i):0 for i in range(1,9)},'core_schedules':[[0],[],[],[],[]]}
    ops,preds,topo,durations=build_compute_dag(g)
    result=_split(p,ops,preds,topo,durations,5,4,1)
    _view(result,ops,preds,5)
    assert len(set(result['node_to_subgraph'].values()))==4
    assert set(result['node_to_subgraph'])==set(p['node_to_subgraph'])
    assert set(p['node_to_subgraph'].values())=={0}


def test_runner_rejects_bad_candidates_and_keeps_official_best(tmp_path,monkeypatch):
    pytest.importorskip('ortools')
    import json
    import model
    import pipeline
    import pipe_local_cp
    from run_math_refine import refine_one
    g=graph(['PIPE_M','PIPE_V'],10)
    root={'node_to_subgraph':{'1':0,'2':1},'core_schedules':[[0,1],[],[],[],[]]}
    source=tmp_path/'source'
    output=tmp_path/'output'
    source.mkdir()
    output.mkdir()
    path=source/'case_001_B_N5.json'
    original=json.dumps(root)
    path.write_text(original)
    candidate_plans=[]
    for core in (1,2,3):
        p=copy.deepcopy(root)
        p['core_schedules']=[[] for _ in range(5)]
        p['core_schedules'][core]=[0,1]
        candidate_plans.append(p)
    def evaluate(_graph,p,_scene):
        core=next(i for i,r in enumerate(p['core_schedules']) if r)
        if core==1:
            return {'error':'invalid candidate'}
        return {'makespan':{0:100,2:120,3:80}[core],'added_copy_bytes':0}
    monkeypatch.setattr(model,'load_graph',lambda path:g)
    monkeypatch.setattr(pipeline,'real_evaluate',evaluate)
    monkeypatch.setattr(pipe_local_cp,'generate_local_candidates',lambda *a,**kw:(
        [dict(plan=p,source=f'test_{i}',stats={}) for i,p in enumerate(candidate_plans)],
        {'solves':3}))
    task=dict(case='case_001',baseline_dirs=[str(source)],output_dir=str(output),
              singlecore=100,rounds=1,max_official=3,seconds_per_solve=1,
              max_solves=3,movable_limit=2,seed=1)
    result=refine_one(task)
    assert result['real']['makespan']==80
    assert result['candidate_trace'][0]['error']
    assert not result['candidate_trace'][1]['accepted']
    assert result['candidate_trace'][2]['accepted']
    assert path.read_text()==original
    assert json.loads((output/path.name).read_text())==candidate_plans[2]
    # The hard official-call cap still preserves the original plan.
    task['max_official']=1
    result=refine_one(task)
    assert len(result['candidate_trace'])==1
    assert result['real']['makespan']==100


def test_op_list_preserves_coverage_and_dependency_without_mutation():
    from op_list_candidates import generate_op_candidates, _fit
    from pipe_local_cp import _view
    g=graph(['PIPE_M','PIPE_V','PIPE_M','PIPE_V'],20)
    g['edges']=[{'source':1,'target':3},{'source':2,'target':4}]
    plan={'node_to_subgraph':{str(i):0 for i in range(1,5)},
          'core_schedules':[[0],[],[],[],[]]}
    original=copy.deepcopy(plan)
    ops,preds,topo,durations=build_compute_dag(g)
    candidates,stats=generate_op_candidates(g,plan,max_candidates=6)
    assert 0<len(candidates)<=6
    for row in candidates:
        _view(row['plan'],ops,preds,5)
        assert len(set(row['plan']['node_to_subgraph'].values()))==4
    assert plan==original
    assert _fit([(0,10),(20,30)],5,5)==10
    assert _fit([(0,10),(20,30)],5,15)==30


@pytest.mark.parametrize('scene,cores', [('A', 2), ('B', 3), ('A', 4), ('B', 5)])
def test_refine_forwards_scene_and_core_count(tmp_path, monkeypatch, scene, cores):
    import json
    import model
    import pipeline
    import op_list_candidates
    from run_math_refine import refine_one
    g = graph(['PIPE_M', 'PIPE_V'], 10)
    p = {'node_to_subgraph': {'1': 0, '2': 1},
         'core_schedules': [[0, 1]] + [[] for _ in range(cores-1)]}
    source, output = tmp_path/'source', tmp_path/'output'
    source.mkdir()
    output.mkdir()
    name = f'case_001_{scene}_N{cores}.json'
    (source/name).write_text(json.dumps(p))
    observed = []
    monkeypatch.setattr(model, 'load_graph', lambda path: g)
    def evaluate(g, plan, actual_scene):
        observed.append(actual_scene)
        assert len(plan['core_schedules']) == cores
        return dict(makespan=100, added_copy_bytes=0)
    def generate(g, plan, **kwargs):
        assert kwargs['num_cores'] == cores
        return [], {}
    monkeypatch.setattr(pipeline, 'real_evaluate', evaluate)
    monkeypatch.setattr(op_list_candidates, 'generate_op_candidates', generate)
    task = dict(case='case_001', scene=scene, N=cores, baseline_dirs=[str(source)],
                output_dir=str(output), singlecore=100, rounds=1, max_official=12,
                seconds_per_solve=1, max_solves=6, movable_limit=8, seed=0, generator='op-list')
    result = refine_one(task)
    assert observed == [scene]
    assert (result['scene'], result['N']) == (scene, cores)
    assert (output/name).is_file()


@pytest.mark.parametrize('changed', [False, True])
def test_large_rerun_does_not_accept_proxy_only_changes(tmp_path, monkeypatch, changed):
    import json
    import pipeline
    import time
    from run_math_refine import rerun_large
    source, output = tmp_path/'source', tmp_path/'output'
    source.mkdir()
    output.mkdir()
    p = {'node_to_subgraph': {'1': 0}, 'core_schedules': [[0], []]}
    candidate = copy.deepcopy(p)
    if changed:
        candidate['core_schedules'] = [[], [0]]
    name = 'case_001_A_N2.json'
    (source/name).write_text(json.dumps(p))
    monkeypatch.setattr(pipeline, 'solve_case', lambda *a, **kw: dict(
        plan=candidate, est=(25, 0), real=None, log={'large_case_fallback': True}))
    task = dict(case='case_001', scene='A', N=2, seed=0,
                baseline_dirs=[str(source)], output_dir=str(output))
    result = rerun_large(task, graph(['PIPE_M']), time.monotonic())
    assert json.loads((output/name).read_text()) == p
    assert result['real'] is None and result['speedup'] is None
    assert result['fallback_matches_incumbent'] is not changed


def test_summary_separates_proxy_results_from_official_means():
    valid = dict(case='case_001', scene='B', N=5,
                 real=dict(makespan=20, added_copy_bytes=0),
                 baseline=dict(makespan=25, added_copy_bytes=0),
                 speedup=5, baseline_speedup=4,
                 baseline_trace=[], candidate_trace=[], elapsed=1)
    proxy = dict(case='case_002', scene='B', N=5, evaluation_status='proxy_only',
                 real=None, baseline=None, speedup=None, baseline_speedup=None,
                 baseline_trace=[], candidate_trace=[], elapsed=1)
    result = summarize([valid, proxy], 2)
    assert result['tasks'] == 2 and result['official_tasks'] == 1
    assert result['proxy_only_tasks'] == 1
    assert result['mean_speedup'] == 5 and result['wins'] == 1
    assert result['by_scene_and_cores']['B_N5']['tasks'] == 1
