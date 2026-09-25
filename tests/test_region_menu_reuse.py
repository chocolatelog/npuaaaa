"""跨轮次完整方案复用不得丢失来源或容忍互相冲突的指标。"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'solver'))


def test_merge_reuse_preserves_each_origin_and_rejects_conflicting_same_plan():
    import pytest
    from experiment_region_menu import merge_prior_rows
    a={'id':'case_005_N2','source_sha256':'same','baseline':{'makespan':100},'binding':'first',
       'evaluated':[{'plan_id':'p','resource':{'makespan':90}}],'checks':[]}
    b={**a,'binding':'second','evaluated':[{'plan_id':'q','resource':{'makespan':80}}]}
    merged=merge_prior_rows(a,b)
    assert [(r['plan_id'],r['evidence_binding']) for r in merged['evaluated']]==[('p','first'),('q','second')]
    with pytest.raises(ValueError):merge_prior_rows(a,{**b,'source_sha256':'other'})
    with pytest.raises(ValueError):merge_prior_rows(a,{**b,'evaluated':[{'plan_id':'p','resource':{'makespan':89}}]})
    assert 'evidence_binding' not in a['evaluated'][0]
