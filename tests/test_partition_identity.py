"""分区多样性以算子成员为准，子图标签不能消耗额外名额。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'solver'))


def test_partition_key_ignores_labels_core_placement_and_sequence():
    from partition_identity import partition_key
    a = {'node_to_subgraph': {'1': 0, '2': 0, '3': 1}, 'core_schedules': [[0], [1]]}
    b = {'node_to_subgraph': {'3': 20, '2': 7, '1': 7}, 'core_schedules': [[20, 7], []]}
    c = {'node_to_subgraph': {'1': 0, '2': 1, '3': 1}, 'core_schedules': [[0], [1]]}
    assert partition_key(a) == partition_key(b)
    assert partition_key(a) != partition_key(c)


def test_distinct_shortlist_keeps_first_ranked_placement_then_next_partition():
    from partition_identity import select_distinct_partitions
    rows = [
        {'name': 'best', 'plan': {'node_to_subgraph': {'1': 0, '2': 1, '3': 1}}},
        {'name': 'duplicate', 'plan': {'node_to_subgraph': {'1': 5, '2': 4, '3': 4}}},
        {'name': 'different', 'plan': {'node_to_subgraph': {'1': 0, '2': 0, '3': 1}}}]
    assert [r['name'] for r in select_distinct_partitions(rows, lambda r: r['plan'], 2)] == ['best', 'different']
