"""新增嵌套诊断必须能跨JSON序列化继续复用，不能触发重复实验。"""
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'solver'))


def test_integer_diagnostic_keys_roundtrip_and_tamper_rejected():
    from experiment_branch_partition import seal_result, verified
    result = {'status':'official_success','real':{'makespan':1,'added_copy_bytes':0,'scheduled_copy_bytes':1},
              'artifacts':{},'diagnostic':{'reload_by_op':{2:1,10:2}}}
    sealed = seal_result(result,'fingerprint')
    saved = json.loads(json.dumps(sealed))
    assert verified(saved,'fingerprint')
    saved['diagnostic']['reload_by_op']['10'] = 99
    assert not verified(saved,'fingerprint')
