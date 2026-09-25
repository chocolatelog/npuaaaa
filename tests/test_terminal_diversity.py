"""预算受限时优先覆盖不同范式及不同切分。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'solver'))


def test_shortlist_covers_paradigms_before_duplicate_family():
    from terminal_refine import diverse_shortlist
    rows = []
    for i, (paradigm, granularity, cost) in enumerate([
        ('heft', 'fine', 1), ('heft', 'balanced', 2), ('heft', 'coarse', 3),
        ('strip', 'fine', 10), ('chain', 'balanced', 20), ('netbenefit', 'fine', 30)]):
        rows.append({'plan_id': str(i), 'source': f'reservoir:{paradigm}:{granularity}:R0',
                     'mk': cost, 'ad': cost})
    selected = diverse_shortlist(rows, max_precise=4)
    assert {r['source'].split(':')[1] for r in selected} == {'heft', 'strip', 'chain', 'netbenefit'}


def test_official_slots_cover_distinct_partitions():
    from terminal_refine import diverse_finalists
    p = {'node_to_subgraph': {'0': 0, '1': 1}, 'core_schedules': [[0], [1]]}
    q = {'node_to_subgraph': {'0': 0, '1': 1}, 'core_schedules': [[1], [0]]}
    r = {'node_to_subgraph': {'0': 0, '1': 0}, 'core_schedules': [[0], []]}
    rows = [{'plan': plan, 'plan_id': str(i)} for i, plan in enumerate([p, q, r])]
    assert [x['plan_id'] for x in diverse_finalists(rows, 2)] == ['0', '2']


def test_official_slots_prioritize_unverified_partitions():
    from terminal_refine import diverse_finalists
    p = {'node_to_subgraph': {'0': 0, '1': 1}, 'core_schedules': [[0], [1]]}
    q = {'node_to_subgraph': {'0': 0, '1': 0}, 'core_schedules': [[0], []]}
    rows = [{'plan': plan, 'plan_id': str(i)} for i, plan in enumerate([p, q])]
    verified = {tuple(sorted(p['node_to_subgraph'].items()))}
    assert diverse_finalists(rows, 1, verified)[0]['plan_id'] == '1'
