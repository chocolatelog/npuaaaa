"""只规范化比较用分区键；不改方案编号、核心顺序或官方模拟输入。"""
from collections import defaultdict


def partition_key(plan):
    groups = defaultdict(list)
    for op, label in plan['node_to_subgraph'].items():
        groups[label].append(int(op))
    return tuple(sorted(tuple(sorted(members)) for members in groups.values()))


def select_distinct_partitions(ranked, get_plan, limit):
    if limit < 1:
        return []
    selected, seen = [], set()
    for row in ranked:
        key = partition_key(get_plan(row))
        if key in seen:
            continue
        seen.add(key); selected.append(row)
        if len(selected) >= limit:
            break
    return selected
