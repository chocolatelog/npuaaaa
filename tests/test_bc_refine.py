import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'solver'))
DATA = ROOT / '通用神经网络处理器下的多核调度问题附件' / 'data'


def _graph():
    return json.loads((DATA / 'case_001.json').read_text(encoding='utf-8'))


def _plan():
    return json.loads((ROOT / 'tests' / 'fixtures' / 'case_001_B_N2.json').read_text(encoding='utf-8'))


def test_bc_candidates_keep_operation_coverage_and_are_topological():
    from bc_refine import generate_bc_candidates, validate_plan

    graph, plan = _graph(), _plan()
    candidates, audit = generate_bc_candidates(graph, plan, 'B', max_candidates=6)
    assert candidates
    assert audit['generated'] >= len(candidates)
    assert all(validate_plan(graph, candidate) for candidate in candidates)
    eligible = {str(op['id']) for op in graph['ops'] if op['op'] not in {'COPY_IN', 'COPY_OUT'}}
    for candidate in candidates:
        assert set(candidate['node_to_subgraph']) == eligible


def test_bc_refine_has_c_and_b_reuse_signals():
    from bc_refine import generate_bc_candidates

    graph, plan = _graph(), _plan()
    b_candidates, b_audit = generate_bc_candidates(graph, plan, 'B', max_candidates=4)
    c_candidates, c_audit = generate_bc_candidates(graph, plan, 'C', max_candidates=4)
    assert b_audit['scene'] == 'B'
    assert c_audit['scene'] == 'C'
    assert all('lifetime_pressure' in row for row in b_audit['candidate_metrics'])
    assert all('cache_reuse_bytes' in row for row in c_audit['candidate_metrics'])
    assert all('cache_hit_potential_bytes' in row for row in c_audit['candidate_metrics'])


def test_same_core_split_is_explicitly_recorded():
    from bc_refine import generate_bc_candidates

    graph, plan = _graph(), _plan()
    candidates, audit = generate_bc_candidates(graph, plan, 'B', max_candidates=12)
    assert audit['split_candidates'] >= 1
    assert any(row['action'] == 'split' for row in audit['candidate_metrics'])
