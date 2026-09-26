"""受控的确定性树搜索候选器。

这是 AlphaGo（阿尔法围棋）式“先验＋价值＋树遍历”在本题单人确定性优化
上的小型适配：先验来自动作类型，价值来自 B/C 廉价资源表，所有状态都由
官方评估器在调用方统一裁决。没有训练网络，也不改变硬件语义。
"""
from __future__ import annotations

import hashlib
import json
import math

from bc_refine import generate_bc_candidates, validate_plan


def _id(plan):
    normalized = {'node_to_subgraph': {str(k): int(v)
                                       for k, v in plan['node_to_subgraph'].items()},
                  'core_schedules': [[int(x) for x in row]
                                     for row in plan['core_schedules']]}
    return hashlib.sha256(json.dumps(normalized, sort_keys=True,
                                     separators=(',', ':')).encode()).hexdigest()


def _prior(action):
    return {'cache_cluster_swap': 1.0, 'cache_cluster_move': 0.8,
            'split': 0.7, 'pressure_swap': 0.6, 'move_group': 0.5,
            'parent': 0.0}.get(action, 0.4)


def mcts_candidates(graph, plan, scene, max_states=32, depth=2, width=3,
                    exploration=0.7):
    if scene not in {'B', 'C'}:
        raise ValueError('树搜索只支持场景 B/C')
    if max_states < 1 or depth < 0 or width < 1:
        raise ValueError('树搜索预算参数非法')
    root_id = _id(plan)
    states = {root_id: {'plan': plan, 'depth': 0, 'visits': 0,
                        'value': 0.0, 'parent': None, 'action': 'parent'}}
    order = [root_id]
    expanded = 0
    edge_count = 0
    while expanded < max_states and order:
        parent_id = max(
            order,
            key=lambda pid: (
                -states[pid]['depth'],
                states[pid]['value'] / max(1, states[pid]['visits'])
                + exploration * math.sqrt(math.log(expanded + 2) /
                                           max(1, states[pid]['visits'])),
                pid),
        )
        parent = states[parent_id]
        if parent['depth'] >= depth:
            order.remove(parent_id)
            continue
        children, audit = generate_bc_candidates(
            graph, parent['plan'], scene, max_candidates=width + 1)
        metrics = {row.get('plan_id'): row for row in audit.get('candidate_metrics', ())}
        parent['visits'] += 1
        expanded += 1
        for child in children[1:]:
            if expanded >= max_states:
                break
            child_id = _id(child)
            edge_count += 1
            if child_id in states:
                continue
            if not validate_plan(graph, child):
                continue
            row = metrics.get(child_id, {})
            score = float(row.get('proxy_score', 0.0))
            state = {'plan': child, 'depth': parent['depth'] + 1,
                     'visits': 1, 'value': -score,
                     'parent': parent_id, 'action': row.get('action', 'unknown')}
            states[child_id] = state
            order.append(child_id)
    # 根方案始终排第一，其他方案按访问价值和稳定摘要排序。
    rest = sorted((state for pid, state in states.items() if pid != root_id),
                  key=lambda item: (-item['visits'], -item['value'],
                                    _id(item['plan'])))
    result = [states[root_id]['plan']] + [item['plan'] for item in rest]
    result = result[:max_states]
    return result, {
        'scene': scene,
        'max_states': int(max_states),
        'depth': int(depth),
        'width': int(width),
        'expanded': int(expanded),
        'edge_count': int(edge_count),
        'unique_states': len(result),
        'root_plan_id': root_id,
        'actions': [states[_id(item)]['action'] for item in result],
    }
