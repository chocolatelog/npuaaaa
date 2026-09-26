"""拆分结构保持固定，迁尾只变归属和插入位置；动态模型与原官方一致。"""
from pathlib import Path
import copy
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'solver'))


def test_tail_candidates_keep_all_outside_relative_orders_and_match_official():
    from test_boundary_release import example
    from boundary_tail_refine import generate_boundary_tail_candidates
    from multicore_cut_evaluate_problem_1 import evaluate_scene_a
    from scenario_contract import validate_plan
    graph,parent,raw=example();before=copy.deepcopy((graph,parent,raw))
    rows,audit=generate_boundary_tail_candidates(graph,parent,raw,max_expanded=12)
    assert rows and audit['parent_verified'] and audit['scored']<=12
    assert any(r['source']['mode']=='tail_migration' for r in rows)
    for row in rows:
        plan=row['plan'];source=row['source'];validate_plan(graph,plan,'A')
        tail=source['tail_group']
        assert [[g for g in q if g!=tail] for q in plan['core_schedules']]==parent['core_schedules']
        assert set(plan['node_to_subgraph'])==set(parent['node_to_subgraph'])
        assert all(g==parent['node_to_subgraph'][op] or g==tail for op,g in plan['node_to_subgraph'].items())
        official=evaluate_scene_a(graph,plan,60,raw['capacity_bytes'],1000,100)
        assert official['makespan']==source['closed_loop_makespan']
    assert (graph,parent,raw)==before
