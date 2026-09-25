"""审计必须包括被粗筛排掉的同核原位置，并保持已执行前缀。"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'solver'))


def test_all_positions_include_original_and_filter_dependency_cycles():
    from audit_lookahead_coverage import all_positions
    orders=[[0,1],[2,3]];preds={0:set(),1:{0},2:set(),3:{1,2}}
    rows=all_positions(orders,1,preds,dict.fromkeys(preds,10),{0,2})
    assert any(r['orders']==orders for r in rows)
    assert len(rows)==2
    assert all(r['orders'][0][0]==0 and r['orders'][1][0]==2 for r in rows)
    assert all(r['orders'][1]!=[2,3,1] for r in rows)
    assert orders==[[0,1],[2,3]]
