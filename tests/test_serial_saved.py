"""串行阶段不得凭一份旧汇总或部分官方结果启动下一阶段。"""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'solver'))
import run_serial_saved as serial


def test_gate_rejects_stale_incomplete_and_unverified_summary(tmp_path, monkeypatch):
    folder = tmp_path/'official'; folder.mkdir()
    ledger = folder/'results.jsonl'
    ledger.write_text(json.dumps({'job_id':'one','ok':True})+'\n')
    summary = {'source_sha256': serial.digest(ledger), 'expected':1,
               'official_success':1, 'pending':0, 'failed_or_invalid':0}
    path = folder/'summary.json'
    path.write_text(json.dumps(summary))
    monkeypatch.setattr(serial,'verified_record',lambda r:r['ok'])
    assert serial.complete_summary(tmp_path,1) == summary
    assert serial.complete_summary(tmp_path,500) is None
    summary['pending']=1; path.write_text(json.dumps(summary))
    assert serial.complete_summary(tmp_path,1) is None
    summary['pending']=0;path.write_text(json.dumps(summary))
    ledger.write_text(json.dumps({'job_id':'one','ok':False})+'\n')
    assert serial.complete_summary(tmp_path,1) is None
    summary['source_sha256']=serial.digest(ledger);path.write_text(json.dumps(summary))
    assert serial.complete_summary(tmp_path,1) is None
