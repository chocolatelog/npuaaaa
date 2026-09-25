"""统一预算必须保护来源、区分成员分区、释放空来源预算。"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'solver'))
from solution import Sol


def candidates():
    rows=[]
    for origin in ('split','branch','merge','hotspot'):
        for i in range(4):
            # 不同成员分区；刻意让分支全体粗分远差于其他来源。
            code=len(rows)+1
            assignment=[(code>>b)&1 for b in range(6)]
            rows.append({'sol':Sol(assignment,[0,1]),
                         'source':{'origin':origin,'task':i,'kind':origin},
                         'features':{'compute_load':10+code,'boundary_bytes':100-code,
                                     'coarse_score':1000+code if origin=='branch' else code}})
    return rows


def test_source_quotas_and_fixed_budget():
    from partition_fusion import select_fusion_candidates
    selected,audit=select_fusion_candidates(candidates())
    assert len(selected)==12
    assert all(sum(r['source']['origin']==o for r in selected)>=2 for o in ('split','branch','merge','hotspot'))
    assert audit['selected']==12


def test_partition_relabel_or_other_placement_does_not_consume_budget_twice():
    from partition_fusion import select_fusion_candidates, member_key
    rows=candidates()
    first=rows[0]
    rows.append({'sol':Sol([1-s for s in first['sol'].sg_of_block],[1,0]),
                 'source':{'origin':'hotspot','task':0,'kind':'aggregate'},
                 'features':dict(first['features'],coarse_score=-10)})
    selected,audit=select_fusion_candidates(rows)
    assert len(selected)==len({member_key(r['sol']) for r in selected})==12
    assert audit['member_duplicates']==1


def test_missing_sources_release_budget():
    from partition_fusion import select_fusion_candidates
    rows=[r for r in candidates() if r['source']['origin']=='branch']
    selected,audit=select_fusion_candidates(rows)
    assert len(selected)==4
    assert audit['available_by_origin']=={'split':0,'branch':4,'merge':0,'hotspot':0}
