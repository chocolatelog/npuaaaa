"""Deterministic beam search over scheduling neighborhoods."""
from __future__ import annotations

import time

from mcts_schedule import _actions, _apply, _signature


def _record(pool, row, cap):
    if pool is None:
        return
    pool.append(row)
    pool.sort(key=lambda item: item[0])
    del pool[max(1, cap):]


def _select_diverse(rows, width):
    selected = []
    deferred = []
    bins = set()
    for row in sorted(rows, key=lambda item: item[0]):
        sol = row[4]
        counts = [0] * max(1, max(sol.core_of_sg, default=0) + 1)
        for core in sol.core_of_sg:
            if core >= len(counts):
                counts.extend([0] * (core - len(counts) + 1))
            counts[core] += 1
        key = (len(sol.core_of_sg), tuple(counts))
        if key not in bins and len(selected) < width:
            bins.add(key)
            selected.append(row)
        else:
            deferred.append(row)
    for row in deferred:
        if len(selected) >= width:
            break
        selected.append(row)
    return selected


def beam_search(ctx, root_sol, time_budget=2.0, deadline=None, max_depth=4,
                width=8, branching=24, max_evals=128, archive=None,
                candidate_pool=None, candidate_cap=32,
                structural_actions=True):
    """Return the best proxy solution found with an evaluation-bounded beam."""
    started = time.time()
    end = deadline if deadline is not None else started + time_budget
    root = root_sol.clone()
    root.compact()
    f0, mk0, ad0 = ctx.eval_fitness(root)
    root.compact()
    seen = {_signature(root)}
    frontier = [(f0, mk0, ad0, 0.0, root)]
    best, best_f, best_mk, best_ad = root.clone(), f0, mk0, ad0
    _record(candidate_pool, (f0, mk0, ad0, root.clone()), candidate_cap)
    evaluations = 1
    generated = 0
    depth_done = 0
    stop_reason = "completed"

    for depth in range(max_depth):
        if time.time() >= end:
            stop_reason = "time_budget"
            break
        proposals = []
        for _fit, _mk, _ad, _prior, parent in frontier:
            for prior, action in _actions(
                    ctx, parent, branching, structural=structural_actions):
                if time.time() >= end:
                    stop_reason = "time_budget"
                    break
                if evaluations >= max_evals:
                    stop_reason = "max_evals"
                    break
                child = _apply(ctx, parent, action)
                if child is None or _signature(child) in seen:
                    continue
                f_new, mk_new, ad_new = ctx.eval_fitness(child)
                evaluations += 1
                child.compact()
                signature = _signature(child)
                if signature in seen:
                    continue
                seen.add(signature)
                generated += 1
                row = (f_new, mk_new, ad_new, prior, child)
                proposals.append(row)
                _record(candidate_pool,
                        (f_new, mk_new, ad_new, child.clone()), candidate_cap)
                if archive is not None:
                    archive.insert(mk_new, ad_new, child)
                if f_new < best_f:
                    best, best_f, best_mk, best_ad = (
                        child.clone(), f_new, mk_new, ad_new)
            if stop_reason != "completed":
                break
        if not proposals:
            if stop_reason == "completed":
                stop_reason = "exhausted"
            break
        frontier = _select_diverse(proposals, width)
        depth_done = depth + 1
        if stop_reason != "completed":
            break

    return best, best_f, best_mk, best_ad, {
        "evaluated_states": evaluations,
        "states": len(seen),
        "generated": generated,
        "depth": depth_done,
        "frontier": len(frontier),
        "stop_reason": stop_reason,
        "best_fitness": best_f,
        "elapsed": time.time() - started,
    }


__all__ = ["beam_search"]
