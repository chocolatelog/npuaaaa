"""AlphaGo-style local tree search for core assignment.

This is deliberately a bounded PUCT search over legal schedule perturbations,
not a full AlphaZero trainer.  The existing proxy evaluator supplies the leaf
value, while HEFT/critical-path signals provide the policy prior.
"""
from __future__ import annotations

import math
import random
import time


class _Node:
    __slots__ = ("sol", "parent", "action", "prior", "children",
                 "unexpanded", "visits", "value_sum", "fitness", "mk",
                 "added")

    def __init__(self, sol, parent=None, action=None,
                 fitness=None, mk=None, added=None, prior=0.0):
        self.sol = sol
        self.parent = parent
        self.action = action
        self.prior = prior
        self.children = []
        self.unexpanded = None
        self.visits = 0
        self.value_sum = 0.0
        self.fitness = fitness
        self.mk = mk
        self.added = added


def _signature(sol):
    return (tuple(sol.sg_of_block), tuple(sol.core_of_sg))


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
    """Return prioritized core moves and pairwise core swaps."""
    critical = _critical_subgraphs(ctx, sol, limit=min(10, branching))
    loads = [0.0] * ctx.num_cores
    for s, blocks in enumerate(sol.blocks_in_sg):
        if blocks and s < len(sol.core_of_sg):
            loads[sol.core_of_sg[s]] += sum(ctx.bdur[b] for b in blocks)
    actions = []
    for s in critical:
        old = sol.core_of_sg[s]
        block_work = sum(ctx.bdur[b] for b in sol.blocks_in_sg[s])
        for c in sorted(range(ctx.num_cores),
                        key=lambda x: (loads[x], x)):
            if c == old:
                continue
            # Prior favors moving critical work to a less loaded core; the
            # tree policy still explores every action through PUCT.
            prior = 1.0 + block_work / max(1.0, loads[c] + 1.0)
            actions.append((prior, ("move_core", s, c)))
    for i, s in enumerate(critical):
        for t in critical[i + 1:]:
            if sol.core_of_sg[s] == sol.core_of_sg[t]:
                continue
            prior = 0.5 + abs(ctx.brank[next(iter(sol.blocks_in_sg[s]))]
                               - ctx.brank[next(iter(sol.blocks_in_sg[t]))])
            actions.append((prior, ("swap_core", s, t)))
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
                c_puct=1.2):
    """Search legal core assignments and return the best proxy solution.

    The return shape matches the other searchers:
    ``(best_sol, best_fitness, best_makespan, best_added, stats)``.
    """
    rng = rng or random.Random()
    end = deadline if deadline is not None else time.time() + time_budget
    root = root_sol.clone()
    root.compact()
    f0, mk0, ad0 = ctx.eval_fitness(root)
    root_node = _Node(root, fitness=f0, mk=mk0, added=ad0)
    best = root.clone()
    best_f, best_mk, best_ad = f0, mk0, ad0
    seen = {_signature(root)}
    iterations = 0

    while time.time() < end:
        iterations += 1
        node = root_node
        path = [node]
        depth = 0
        # Selection: PUCT maximizes the reward, which is negative fitness.
        while depth < max_depth:
            if node.unexpanded is None:
                node.unexpanded = _actions(ctx, node.sol, branching)
            if node.unexpanded:
                break
            if not node.children:
                break
            parent_visits = max(1, node.visits)
            node = max(
                node.children,
                key=lambda child: (
                    child.value_sum / max(1, child.visits)
                    + c_puct * child.prior * math.sqrt(parent_visits) /
                    (1 + child.visits),
                    -child.fitness),
            )
            path.append(node)
            depth += 1

        if depth < max_depth and node.unexpanded:
            # Prefer high-prior actions, with light randomized tie breaking.
            idx = 0 if rng.random() < 0.75 else rng.randrange(len(node.unexpanded))
            prior, action = node.unexpanded.pop(idx)
            child_sol = _apply(node.sol, action)
            if child_sol is not None:
                sig = _signature(child_sol)
                if sig in seen:
                    child_sol = None
                else:
                    seen.add(sig)
            if child_sol is None:
                continue
            f_new, mk_new, ad_new = ctx.eval_fitness(child_sol)
            child = _Node(child_sol, node, action, f_new, mk_new, ad_new,
                          prior=prior)
            node.children.append(child)
            node = child
            path.append(node)
            if archive is not None:
                archive.insert(mk_new, ad_new, child_sol)
            if f_new < best_f:
                best, best_f, best_mk, best_ad = (
                    child_sol.clone(), f_new, mk_new, ad_new)

        # Short rollout from the expanded node.  The rollout is intentionally
        # shallow because each leaf uses the full proxy simulation.
        rollout = node.sol.clone()
        rollout_best = node
        for _ in range(max(0, max_depth - len(path))):
            choices = _actions(ctx, rollout, branching=6)
            if not choices:
                break
            _prior, action = rng.choice(choices[:min(3, len(choices))])
            child_sol = _apply(rollout, action)
            if child_sol is None or _signature(child_sol) in seen:
                break
            seen.add(_signature(child_sol))
            f_new, mk_new, ad_new = ctx.eval_fitness(child_sol)
            rollout = child_sol
            if archive is not None:
                archive.insert(mk_new, ad_new, child_sol)
            if f_new < best_f:
                best, best_f, best_mk, best_ad = (
                    child_sol.clone(), f_new, mk_new, ad_new)
            rollout_best = _Node(child_sol, node, action, f_new, mk_new, ad_new)

        reward = -float(rollout_best.fitness)
        for visited in path:
            visited.visits += 1
            visited.value_sum += reward

    return best, best_f, best_mk, best_ad, {
        "iterations": iterations,
        "states": len(seen),
        "root_children": len(root_node.children),
        "best_fitness": best_f,
    }


__all__ = ["mcts_search"]
