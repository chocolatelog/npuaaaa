"""闭环精筛必须在真实搬运反例上保持预测和原官方一致。"""
from pathlib import Path
import sys
import copy
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'solver'))


def test_closed_screen_selects_real_improvement_and_rejects_bad_parent():
    from test_bc_copy_window import example
    from bc_event_screen import generate_event_screened_candidates
    from multicore_cut_evaluate_problem_2 import evaluate_scene_b
    g,p,r=example()
    rows,audit=generate_event_screened_candidates(g,p,r,'B',max_expanded=12,max_candidates=3)
    assert rows and audit['parent_verified']
    assert all(x['source']['closed_loop_makespan']<r['makespan'] for x in rows)
    for row in rows:
        actual=evaluate_scene_b(g,row['plan'],60,r['capacity_bytes'],cross_core_copy_delay=500)
        assert actual['makespan']==row['source']['closed_loop_makespan']
    bad=copy.deepcopy(r);bad['makespan']+=1
    with pytest.raises(ValueError,match='父方案'):
        generate_event_screened_candidates(g,p,bad,'B',max_expanded=12,max_candidates=3)


def test_scene_c_screen_matches_official_cache_semantics():
    from test_bc_copy_window import example
    from bc_event_screen import generate_event_screened_candidates
    from multicore_cut_evaluate_problem_3 import evaluate_problem_3,read_cache_config
    from scenario_contract import CODE
    g,p,b=example();config=read_cache_config(CODE.parent/'data/config.txt')
    def evaluate(plan):
        return evaluate_problem_3(g,plan,60,b['capacity_bytes'],500,config['cache_capacity_bytes'],config['cache_bandwidth_bytes_per_cycle'])
    r=evaluate(p)
    rows,audit=generate_event_screened_candidates(g,p,r,'C',max_expanded=12,max_candidates=3)
    assert audit['parent_verified']
    for row in rows:
        assert row['source']['closed_loop_makespan']==evaluate(row['plan'])['makespan']
