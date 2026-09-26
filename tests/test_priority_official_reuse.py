"""省去重复重放不省掉原官方；复用仅限完整匹配的有效凭据。"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'solver'))
import audit_priority_candidates as audit


def test_reuse_validates_context_and_receipt_before_avoiding_evaluation(tmp_path,monkeypatch):
    monkeypatch.setattr(audit,'input_context',lambda *a:({'run':'same'},))
    key=audit.object_digest({'run':'same'})
    record=dict(status='official_success',record_id='old')
    monkeypatch.setattr(audit,'verified_record',lambda r:r.get('record_id')=='old')
    calls=[]
    monkeypatch.setattr(audit,'evaluate_job',lambda *a,**k:calls.append(1) or {'record_id':'new'})
    actual,reused=audit.evaluate_with_history({},tmp_path,tmp_path,{key:record})
    assert actual is record and reused and not calls
    actual,reused=audit.evaluate_with_history({},tmp_path,tmp_path,{key:{'record_id':'broken'}})
    assert actual['record_id']=='new' and not reused and len(calls)==1
    actual,reused=audit.evaluate_with_history({},tmp_path,tmp_path,{'other':record})
    assert not reused and len(calls)==2
