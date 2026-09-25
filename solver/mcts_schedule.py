"""Bounded PUCT search for core assignment.

The search reuses the deterministic proxy evaluator as its value function and
critical-path/load signals as its policy prior. Rewards are normalized around
the root so the value and exploration terms remain on comparable scales.
"""
from __future__ import annotations

import math
import random
import time


class _Edge:
    __slots__ = ("action", "prior", "child")

    def __init__(self, action, prior, child):
        self.action = action
        self.prior = prior
        self.child = child


class _Node:
    __slots__ = ("sol", "signature", "children", "unexpanded", "visits",
                 "value_sum", "fitness", "mk", "added")

    def __init__(self, sol, fitness, mk, added):
        self.sol = sol
        self.signature = _signature(sol)
        self.children = []
        self.unexpanded = None
        self.visits = 0
        self.value_sum = 0.0
        self.fitness = fitness
        self.mk = mk
        self.added = added


def _signature(sol):
    return (tuple(sol.sg_of_block), tuple(sol.core_of_sg))


def _normalized_reward(root_fitness, leaf_fitness):
    scale = max(1.0, abs(float(root_fitness)))
    reward = (float(root_fitness) - float(leaf_fitness)) / scale
    return max(-1.0, min(1.0, reward))


def _critical_subgraphs(ctx, sol, limit=8):
    scores = []
    for s, blocks in enumerate(sol.blocks_in_sg):
        if not blocks:
            continue
        rank = max((ctx.brank[b] for b in blocks), default=0.0)
        work = sum(ctx.bdur[b] for b in blocks)
        traffic = sum(ctx._block_out_traffic(b) for b in blocks)
        scores.append((rank + 0.05 * work + 0.000001 * traffic, s))
    scores.sort(reverse=True)
    return [s for _score, s in scores[:max(1, limit)]]


def _actions(ctx, sol, branching=16):
    """Return normalized priors for core moves and pairwise core swaps."""
    critical = _critical_subgraphs(ctx, sol, limit=min(10, branching))
    loads = [0.0] * ctx.num_cores
    for s, blocks in enumerate(sol.blocks_in_sg):
        if blocks and s < len(sol.core_of_sg):
            loads[sol.core_of_sg[s]] += sum(ctx.bdur[b] for b in blocks)
    actions = []
    for s in critical:
        old = sol.core_of_sg[s]
        block_work = sum(ctx.bdur[b] for b in sol.blocks_in_sg[s])
        for c in sorted(range(ctx.num_cores), key=lambda x: (loads[x], x)):
            if c == old:
                continue
            prior = 1.0 + block_work / max(1.0, loads[c] + 1.0)
            actions.append((prior, ("move_core", s, c)))
    for i, s in enumerate(critical):
        for t in critical[i + 1:]:
            if sol.core_of_sg[s] == sol.core_of_sg[t]:
                continue
            srank = max((ctx.brank[b] for b in sol.blocks_in_sg[s]),
                        default=0.0)
            trank = max((ctx.brank[b] for b in sol.blocks_in_sg[t]),
                        default=0.0)
            actions.append((0.5 + abs(srank - trank),
                            ("swap_core", s, t)))
    actions.sort(key=lambda item: (-item[0], item[1]))
    unique = []
    seen = set()
    for prior, action in actions:
        if action in seen:
            continue
        seen.add(action)
        unique.append((prior, action))
        if len(unique) >= branching:
            break
    total = sum(prior for prior, _action in unique) or 1.0
    return [(prior / total, action) for prior, action in unique]


def _apply(sol, action):
    child = sol.clone()
    if action[0] == "move_core":
        _kind, s, c = action
        if s >= len(child.core_of_sg) or child.core_of_sg[s] == c:
            return None
        child.move_sg_core(s, c)
    else:
        _kind, s, t = action
        if s >= len(child.core_of_sg) or t >= len(child.core_of_sg):
            return None
        child.core_of_sg[s], child.core_of_sg[t] = (
            child.core_of_sg[t], child.core_of_sg[s])
    child.compact()
    return child


def mcts_search(ctx, root_sol, time_budget=1.0, deadline=None,
                rng=None, archive=None, max_depth=3, branching=16,
                c_puct=1.2, max_evals=128, stall_limit=256,
                candidate_pool=None, candidate_cap=32):
    """Search core assignments and return the best proxy solution.

    Duplicate states share node statistics through a transposition table. A
    finite evaluation budget and a stall limit prevent the search from
    spinning after its reachable neighborhood is exhausted.
    """
    rng = rng or random.Random()
    started = time.time()
    end = deadline if deadline is not None else started + time_budget
    root = root_sol.clone()
    root.compact()
    f0, mk0, ad0 = ctx.eval_fitness(root)
    root.compact()
    root_node = _Node(root, f0, mk0, ad0)
    nodes = {root_node.signature: root_node}
    if candidate_pool is not None:
        candidate_pool.append((f0, mk0, ad0, root.clone()))
    best = root.clone()
    best_f, best_mk, best_ad = f0, mk0, ad0
    iterations = 0
    evaluated_states = 1
    transposition_hits = 0
    duplicate_actions = 0
    stalled = 0
    expanded_edges = 0
    reward_min = 0.0
    reward_max = 0.0
    stop_reason = "time_budget"

    while True:
        if time.time() >= end:
            stop_reason = "time_budget"
            break
        if max_evals is not None and evaluated_states >= max_evals:
            stop_reason = "max_evals"
            break
        if stalled >= max(1, stall_limit):
            stop_reason = "stalled"
            break

        iterations += 1
        node = root_node
        path = [node]
        path_signatures = {node.signature}
        depth = 0
        added_state = False

        while depth < max_depth:
            if node.unexpanded is None:
                node.unexpanded = _actions(ctx, node.sol, branching)

            next_node = None
            while node.unexpanded:
                idx = (0 if rng.random() < 0.75
                       else rng.randrange(len(node.unexpanded)))
                prior, action = node.unexpanded.pop(idx)
                child_sol = _apply(node.sol, action)
                if child_sol is None:
                    duplicate_actions += 1
                    continue
                signature = _signature(child_sol)
                if signature in path_signatures:
                    duplicate_actions += 1
                    continue
                child = nodes.get(signature)
                if child is None:
                    f_new, mk_new, ad_new = ctx.eval_fitness(child_sol)
                    child_sol.compact()
                    signature = _signature(child_sol)
                    child = nodes.get(signature)
                    if child is None:
                        child = _Node(child_sol, f_new, mk_new, ad_new)
                        nodes[signature] = child
                        evaluated_states += 1
                        added_state = True
                        if archive is not None:
                            archive.insert(mk_new, ad_new, child_sol)
                        if candidate_pool is not None:
                            candidate_pool.append(
                                (f_new, mk_new, ad_new, child_sol.clone()))
                            candidate_pool.sort(key=lambda item: item[0])
                            del candidate_pool[max(1, candidate_cap):]
                        if f_new < best_f:
                            best, best_f, best_mk, best_ad = (
                                child_sol.clone(), f_new, mk_new, ad_new)
                    else:
                        transposition_hits += 1
                else:
                    transposition_hits += 1
                node.children.append(_Edge(action, prior, child))
                expanded_edges += 1
                next_node = child
                break

            if next_node is None:
                available = [edge for edge in node.children
                             if edge.child.signature not in path_signatures]
                if not available:
                    break
                parent_visits = max(1, node.visits)

                def puct(edge):
                    child = edge.child
                    q = child.value_sum / max(1, child.visits)
                    u = (c_puct * edge.prior * math.sqrt(parent_visits) /
                         (1 + child.visits))
                    return q + u, -child.fitness

                next_node = max(available, key=puct).child

            node = next_node
            path.append(node)
            path_signatures.add(node.signature)
            depth += 1
            if added_state:
                break

        reward = _normalized_reward(f0, node.fitness)
        reward_min = min(reward_min, reward)
        reward_max = max(reward_max, reward)
        for visited in path:
            visited.visits += 1
            visited.value_sum += reward

        if added_state:
            stalled = 0
        else:
            stalled += 1

    return best, best_f, best_mk, best_ad, {
        "iterations": iterations,
        "states": len(nodes),
        "evaluated_states": evaluated_states,
        "expanded_edges": expanded_edges,
        "transposition_hits": transposition_hits,
        "duplicate_actions": duplicate_actions,
        "stalled_iterations": stalled,
        "stop_reason": stop_reason,
        "root_children": len(root_node.children),
        "reward_min": reward_min,
        "reward_max": reward_max,
        "best_fitness": best_f,
        "elapsed": time.time() - started,
    }


__all__ = ["mcts_search"]
