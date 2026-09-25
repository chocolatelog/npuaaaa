"""资源评测不能升级为原官方证据，跨图/方案/语义必须隔离。"""
import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'solver'))

def item(kind='resource',plan_id='plan',graph_sha='graph',value=100,source='one'):
    return {'case':'case_005','scene':'A','N':2,'graph_sha256':graph_sha,'plan_id':plan_id,
            'kind':kind,'contract':'resource-v1' if kind=='resource' else 'official-v1',
            'metrics':{'makespan':value,'added_copy_bytes':0,'scheduled_copy_bytes':0,'partition_added':0,'spill_added':0},
            'plan_path':'saved.json','provenance':{'binding':source},'timeline_level':'metrics_only','trace':[]}

def test_resource_is_not_official_and_input_plan_contract_are_separate():
    from evidence_index import EvidenceIndex
    index=EvidenceIndex({'resource':'resource-v1','official':'official-v1'});index.add_many([item()])
    assert index.get('case_005','A',2,'graph','plan','resource') is not None
    assert index.get('case_005','A',2,'graph','plan','official') is None
    assert index.get('case_005','A',2,'other','plan','resource') is None
    assert index.get('case_005','A',2,'graph','other','resource') is None
    with pytest.raises(ValueError):index.add_many([{**item(),'contract':'old-version'}])

def test_same_metric_different_plan_is_not_duplicate_and_provenance_is_preserved():
    from evidence_index import EvidenceIndex
    index=EvidenceIndex({'resource':'resource-v1','official':'official-v1'})
    index.add_many([item(),item(source='two'),item(plan_id='another')])
    assert len(index.records)==2
    value=index.get('case_005','A',2,'graph','plan','resource')
    assert [p['binding'] for p in value['sources']]==['one','two']
    value['metrics']['makespan']=999
    assert index.get('case_005','A',2,'graph','plan','resource')['metrics']['makespan']==100

def test_conflicting_batch_is_atomic_and_metrics_only_never_claims_full_timeline():
    from evidence_index import EvidenceIndex
    index=EvidenceIndex({'resource':'resource-v1','official':'official-v1'});index.add_many([item()])
    with pytest.raises(ValueError):index.add_many([item(plan_id='other'),item(value=101)])
    assert len(index.records)==1
    assert index.get('case_005','A',2,'graph','plan','resource')['timeline_level']=='metrics_only'
    index.add_many([{**item(source='region'),'timeline_level':'region_summary','trace':[{'region':0}]}])
    assert index.get('case_005','A',2,'graph','plan','resource')['timeline_level']=='region_summary'


def test_original_official_and_exact_resource_disagreement_is_rejected():
    from evidence_index import EvidenceIndex
    index=EvidenceIndex({'resource':'resource-v1','official':'official-v1'});index.add_many([item()])
    with pytest.raises(ValueError):index.add_many([item(kind='official',value=101)])
    assert index.get('case_005','A',2,'graph','plan','official') is None


def test_evaluation_lookup_requires_matching_runtime_but_read_only_lookup_does_not(monkeypatch):
    import evidence_index as module
    index=module.EvidenceIndex({'resource':'resource-v1','official':'official-v1'},environment={'python':'expected'})
    index.add_many([item()]);monkeypatch.setattr(module,'runtime_environment',lambda:{'python':'other'})
    assert index.get('case_005','A',2,'graph','plan','resource')
    with pytest.raises(ValueError):index.lookup_for_evaluation('case_005','A',2,'graph','plan','resource')
    monkeypatch.setattr(module,'runtime_environment',lambda:{'python':'expected'})
    assert index.lookup_for_evaluation('case_005','A',2,'graph','plan','resource')

def test_unbound_row_or_changed_plan_artifact_is_rejected(tmp_path):
    from evidence_index import verified_rows
    import json,hashlib
    plan=tmp_path/'plan.json';plan.write_text('{}')
    row={'id':'one','fingerprint':'fp','artifacts':{str(plan):hashlib.sha256(plan.read_bytes()).hexdigest()}}
    row['binding']=hashlib.sha256(json.dumps(row,sort_keys=True).encode()).hexdigest()
    log=tmp_path/'rows.jsonl';log.write_text(json.dumps(row)+'\n')
    assert len(verified_rows(log,'fp'))==1
    plan.write_text('{"changed":true}')
    with pytest.raises(ValueError):verified_rows(log,'fp')
