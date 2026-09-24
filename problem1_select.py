#!/usr/bin/env python3
"""Select a best-known Problem-1 plan with the official evaluator.

The scheduling problem is combinatorial, so this tool does not claim a proof
of global optimality.  It evaluates a bounded candidate set using the official
scene-A simulator and chooses the feasible plan with the smallest makespan,
then the smallest added DDR traffic.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path

from problem1_heft import generate_problem1_plan
from greedy_sa_scheduler import (
    build_dependencies,
    group_graph,
    load_graph_from_object,
    topological_order,
)


def _plan_mapping(plan):
    return {int(node): int(group)
            for node, group in plan["node_to_subgraph"].items()}


def _plan_cores(plan):
    group_core = {}
    for core, groups in enumerate(plan["core_schedules"]):
        for group in groups:
            group_core[int(group)] = core
    return group_core


def _plan_key(plan, real_result=None):
    if real_result is not None:
        traffic = real_result["data_movement_bytes"]
        return (real_result["makespan"], traffic["added_copy_bytes"],
                traffic["scheduled_copy_bytes"])
    return None


def _rebuild_plan(graph, mapping, group_core, cores):
    used = sorted(set(mapping.values()))
    compact = {old: new for new, old in enumerate(used)}
    compact_mapping = {node: compact[group] for node, group in mapping.items()}
    compact_core = {compact[group]: group_core[group] for group in used}
    graph, ops, tensors, eligible = load_graph_from_object(graph)
    preds, succs, _ = build_dependencies(graph, ops, tensors, eligible)
    order = topological_order(eligible, preds, succs)
    group_order = group_graph(compact_mapping, len(used), preds, order)
    if group_order is None:
        return None
    schedules = [[] for _ in range(cores)]
    for group in group_order:
        schedules[compact_core[group]].append(group)
    return {
        "node_to_subgraph": {
            str(node): compact_mapping[node] for node in sorted(compact_mapping)
        },
        "core_schedules": schedules,
    }


def _proxy_components(graph, plan, cores):
    """Cheap operation-level proxy used only to shortlist tabu neighbors."""
    graph, ops, tensors, eligible = load_graph_from_object(graph)
    preds, succs, weights = build_dependencies(graph, ops, tensors, eligible)
    mapping = _plan_mapping(plan)
    group_core = _plan_cores(plan)
    core_load = [0] * cores
    pipe_load = [dict() for _ in range(cores)]
    cut_bytes = 0
    cross_bytes = 0
    cross_edges = 0
    group_loads = [0] * (max(mapping.values(), default=-1) + 1)
    for node in eligible:
        cycles = max(1, int(ops[node].get("cycles", 0)))
        group = mapping[node]
        core = group_core[group]
        core_load[core] += cycles
        group_loads[group] += cycles
        pipe = str(ops[node].get("pipe", "PIPE_M"))
        pipe_load[core][pipe] = pipe_load[core].get(pipe, 0) + cycles
        for pred in preds[node]:
            size = weights.get((pred, node), 0)
            if mapping[pred] != group:
                cut_bytes += size
            if group_core[mapping[pred]] != core:
                cross_bytes += size
                cross_edges += 1
    proxy_makespan = max(
        0.75 * max(loads.values(), default=0) + 0.25 * core_load[core]
        for core, loads in enumerate(pipe_load)
    )
    cross_sync = cross_bytes / 60.0 + 1000.0 * cross_edges
    imbalance = max(core_load, default=0) - sum(core_load) / max(1, cores)
    max_group = max(group_loads, default=1)
    # This is a shortlist score, not the official objective and not used for
    # final candidate selection.
    score = (proxy_makespan / max(1, sum(core_load))
             + cross_sync / max(1, sum(core_load))
             + cut_bytes / max(60, sum(tensor.get("size", 0)
                                       for tensor in tensors.values()))
             + imbalance / max(1, sum(core_load))
             + max_group / max(1, sum(core_load)))
    return score


def _quotient_edges(graph, mapping):
    graph, ops, tensors, eligible = load_graph_from_object(graph)
    preds, succs, weights = build_dependencies(graph, ops, tensors, eligible)
    result = {}
    for src in eligible:
        for dst in succs[src]:
            a, b = mapping[src], mapping[dst]
            if a == b:
                continue
            result[(a, b)] = result.get((a, b), 0) + weights.get((src, dst), 0)
    return result


def _tabu_neighbors(graph, plan, cores, rng, max_neighbors=48):
    mapping = _plan_mapping(plan)
    group_core = _plan_cores(plan)
    groups = sorted(set(mapping.values()))
    quotient_edges = _quotient_edges(graph, mapping)
    candidates = []

    # Merge highly communicating adjacent groups; keep the merged group on
    # the source core and compact labels afterward.
    merge_edges = sorted(quotient_edges, key=lambda edge: (-quotient_edges[edge], edge))
    for source, target in merge_edges[:12]:
        moved = {node: (source if group == target else group)
                 for node, group in mapping.items()}
        cores_after = dict(group_core)
        cores_after[source] = group_core[source]
        plan_after = _rebuild_plan(graph, moved, cores_after, cores)
        if plan_after is not None:
            candidates.append((f"merge:{source}:{target}", plan_after))

    # Move a small boundary slice between adjacent groups.  This is cheaper
    # than a full arbitrary node move and preserves the topological-block
    # structure that makes the quotient DAG easy to validate.
    group_order = sorted(groups)
    for left, right in zip(group_order, group_order[1:]):
        left_nodes = sorted(node for node, group in mapping.items() if group == left)
        right_nodes = sorted(node for node, group in mapping.items() if group == right)
        if len(left_nodes) > 2:
            moved = dict(mapping)
            for node in left_nodes[-max(1, len(left_nodes) // 5):]:
                moved[node] = right
            plan_after = _rebuild_plan(graph, moved, group_core, cores)
            if plan_after is not None:
                candidates.append((f"boundary:left_to_right:{left}:{right}", plan_after))
        if len(right_nodes) > 2:
            moved = dict(mapping)
            for node in right_nodes[:max(1, len(right_nodes) // 5)]:
                moved[node] = left
            plan_after = _rebuild_plan(graph, moved, group_core, cores)
            if plan_after is not None:
                candidates.append((f"boundary:right_to_left:{right}:{left}", plan_after))

    # Split one of the largest groups by a balanced deterministic topological
    # cut.  Both sides inherit the old core before HEFT-like core reassignment
    # is applied by rebuilding the schedule.
    graph_view, ops, tensors, eligible = load_graph_from_object(graph)
    preds, succs, _ = build_dependencies(graph_view, ops, tensors, eligible)
    topo = topological_order(eligible, preds, succs)
    group_load = {group: 0 for group in groups}
    for node, group in mapping.items():
        group_load[group] += max(1, int(ops[node].get("cycles", 0)))
    for source in sorted(groups, key=lambda group: (-group_load[group], group))[:8]:
        members = [node for node in topo if mapping[node] == source]
        if len(members) < 2:
            continue
        split_at = len(members) // 2
        new_group = max(groups) + 1
        moved = dict(mapping)
        for node in members[split_at:]:
            moved[node] = new_group
        cores_after = dict(group_core)
        cores_after[new_group] = group_core[source]
        plan_after = _rebuild_plan(graph, moved, cores_after, cores)
        if plan_after is not None:
            candidates.append((f"split:{source}", plan_after))

    # Exchange core placement of two groups; schedules are reconstructed from
    # the quotient DAG, so the resulting plan keeps every core order legal.
    for a in groups:
        for b in groups:
            if b <= a or group_core[a] == group_core[b]:
                continue
            cores_after = dict(group_core)
            cores_after[a], cores_after[b] = cores_after[b], cores_after[a]
            plan_after = _rebuild_plan(graph, mapping, cores_after, cores)
            if plan_after is not None:
                candidates.append((f"swap_core:{a}:{b}", plan_after))
    # Deduplicate after collecting every move type.  Truncating before the
    # type split would systematically discard split/boundary moves on graphs
    # with many core-pair exchanges.
    deduplicated = []
    signatures = set()
    for move, candidate in candidates:
        signature = json.dumps(candidate, sort_keys=True)
        if signature in signatures:
            continue
        signatures.add(signature)
        deduplicated.append((move, candidate))
    merges = [item for item in deduplicated if item[0].startswith("merge:")]
    splits = [item for item in deduplicated if item[0].startswith("split:")]
    swaps = [item for item in deduplicated if item[0].startswith("swap_core:")]
    rng.shuffle(merges)
    rng.shuffle(splits)
    rng.shuffle(swaps)
    # Reserve capacity for all move types so core exchanges cannot crowd out
    # the structural merge/split neighborhoods.
    boundaries = [item for item in deduplicated
                  if item[0].startswith("boundary:")]
    rng.shuffle(boundaries)
    per_type = max(1, max_neighbors // 4)
    balanced = merges[:per_type]
    balanced += splits[:per_type]
    balanced += boundaries[:per_type]
    balanced += swaps[:max_neighbors - len(balanced)]
    return balanced[:max_neighbors]


def tabu_refine(graph, initial_plan, initial_key, evaluator_root,
                config_path, cores, seed=0, iterations=4,
                shortlist=3, tabu_tenure=4, aspiration=0.01,
                worsening_limit=0.05, max_evaluations=None):
    """Proxy-shortlisted tabu search; official evaluator decides acceptance.

    A bounded worsening move is allowed to escape a local basin.  The global
    best is always selected using the exact evaluator key, so exploration never
    changes the final acceptance criterion.
    """
    rng = random.Random(seed)
    current_plan = initial_plan
    current_key = tuple(initial_key)
    best = {"plan": initial_plan, "key": current_key, "result": None}
    tabu_until = {}
    history = []
    evaluation_count = 0
    for step in range(iterations):
        neighbors = _tabu_neighbors(graph, current_plan, cores, rng)
        ranked_all = sorted(
            ((_proxy_components(graph, plan, cores), move, plan)
             for move, plan in neighbors),
            key=lambda item: (item[0], item[1]),
        )
        # Keep at least one structural move from every neighborhood family;
        # otherwise core swaps dominate the proxy and the tabu search never
        # changes the partition.
        families = {}
        for item in ranked_all:
            family = item[1].split(":", 1)[0]
            families.setdefault(family, item)
        ranked = list(families.values())
        ranked += [item for item in ranked_all if item not in ranked]
        ranked = ranked[:max(shortlist, len(families))]
        evaluated = []
        for proxy, move, plan in ranked:
            if max_evaluations is not None and evaluation_count >= max_evaluations:
                break
            signature = json.dumps(plan, sort_keys=True)
            if tabu_until.get(signature, -1) > step:
                continue
            result, key = evaluate_candidate(graph, plan, evaluator_root, config_path)
            evaluation_count += 1
            evaluated.append((key, proxy, move, plan, result, signature))
        if not evaluated:
            history.append({"iteration": step, "evaluated": 0, "accepted": False})
            continue
        candidate = min(evaluated, key=lambda item: item[0])
        key, proxy, move, plan, result, signature = candidate
        current_ms = current_key[0]
        accepted = key < current_key
        if not accepted and key[0] <= current_ms * (1.0 + worsening_limit):
            # Aspiration is based on the current neighborhood's proxy rank;
            # this permits a bounded uphill move only when it opens a clearly
            # better proxy basin than the best exact candidate seen here.
            accepted = proxy < min(item[0] for item in ranked) * (1.0 - aspiration)
        if accepted:
            tabu_until[json.dumps(current_plan, sort_keys=True)] = step + tabu_tenure
            current_plan, current_key = plan, key
        if key < tuple(best["key"]):
            best = {"plan": plan, "key": key, "result": result}
        history.append({
            "iteration": step,
            "evaluated": len(evaluated),
            "move": move,
            "proxy_score": proxy,
            "candidate_key": key,
            "accepted": accepted,
            "tabu_candidates_skipped": len(ranked) - len(evaluated),
        })
    best["history"] = history
    best["evaluation_count"] = evaluation_count
    return best


def _official_modules(evaluator_root):
    code_dir = str(Path(evaluator_root) / "code")
    if code_dir not in sys.path:
        sys.path.insert(0, code_dir)
    from evaluation_validation import read_evaluation_config
    from multicore_cut_evaluate_problem_1 import evaluate_scene_a, read_scene_a_config
    return read_evaluation_config, read_scene_a_config, evaluate_scene_a


def candidate_groups(graph, cores, explicit=None):
    if explicit:
        return sorted(set(int(value) for value in explicit))
    # Three nearby partitions are enough for ordinary graphs.  Very large
    # graphs are expensive in the official Step1/2/3 simulator, so narrow the
    # search while retaining one alternative to the default 2K partition.
    n_ops = sum(op.get("op") not in {"COPY_IN", "COPY_OUT"}
                for op in graph.get("ops", []))
    base = {
        2: (4, 6, 8),
        3: (6, 8, 12),
        4: (8, 10, 12),
        5: (10, 12, 15),
    }[cores]
    if n_ops > 12000:
        return (2 * cores,)
    if n_ops > 5000:
        return (2 * cores, base[-1])
    return base


def candidate_orders(graph, cores, explicit=None):
    """Return deterministic topological-order variants for Metis repair."""
    if explicit:
        result = []
        for value in explicit:
            if value.startswith("random:"):
                result.append(("random", int(value.split(":", 1)[1])))
            else:
                result.append((value, None))
        return result
    n_ops = sum(op.get("op") not in {"COPY_IN", "COPY_OUT"}
                for op in graph.get("ops", []))
    if n_ops > 12000:
        return [("critical", None), ("random", 1)]
    if n_ops > 5000:
        return [("critical", None), ("id", None), ("random", 1), ("random", 3)]
    return [
        ("critical", None),
        ("id", None),
        ("level", None),
        ("random", 1),
        ("random", 3),
    ]


def evaluate_candidate(graph, plan, evaluator_root, config_path):
    read_config, read_scene_config, evaluate_scene_a = _official_modules(
        evaluator_root)
    settings = read_config(str(config_path))
    waits = read_scene_config(str(config_path))
    result = evaluate_scene_a(
        graph,
        plan,
        bandwidth=settings["bandwidth"],
        capacity=settings["capacity"],
        cross_core_wait=waits["task_cross_core_wait_cycles"],
        same_core_wait=waits["task_same_core_wait_cycles"],
    )
    traffic = result["data_movement_bytes"]
    return result, (
        result["makespan"],
        traffic["added_copy_bytes"],
        traffic["scheduled_copy_bytes"],
    )


def select_plan(graph, cores, evaluator_root, config_path, groups=None,
                orders=None, seed=0, local_rounds=1,
                tabu_iterations=0, tabu_shortlist=3,
                tabu_max_evaluations=None):
    candidates = []
    failures = []
    for group_count in candidate_groups(graph, cores, groups):
        for order_mode, order_seed in candidate_orders(graph, cores, orders):
            try:
                plan = generate_problem1_plan(
                    graph,
                    cores,
                    seed,
                    group_count,
                    local_rounds,
                    evaluator_root,
                    config_path,
                    order_mode=order_mode,
                    order_seed=order_seed,
                )
                result, key = evaluate_candidate(
                    graph, plan, evaluator_root, config_path)
                candidates.append({
                    "groups": group_count,
                    "order_mode": order_mode,
                    "order_seed": order_seed,
                    "key": key,
                    "result": result,
                    "plan": plan,
                })
            except Exception as exc:  # retain other candidates if one is invalid
                failures.append({
                    "groups": group_count,
                    "order_mode": order_mode,
                    "order_seed": order_seed,
                    "error": str(exc),
                })
    if not candidates:
        raise RuntimeError(f"all Problem-1 candidates failed: {failures}")
    best = min(candidates, key=lambda item: item["key"])
    tabu_result = None
    if tabu_iterations > 0:
        refined = tabu_refine(
            graph,
            best["plan"],
            best["key"],
            evaluator_root,
            config_path,
            cores,
            seed=seed,
            iterations=tabu_iterations,
            shortlist=tabu_shortlist,
            max_evaluations=tabu_max_evaluations,
        )
        tabu_result = refined
        if tuple(refined["key"]) < tuple(best["key"]):
            best = {
                "groups": len(set(refined["plan"]["node_to_subgraph"].values())),
                "order_mode": "tabu",
                "order_seed": seed,
                "key": refined["key"],
                "result": refined["result"],
                "plan": refined["plan"],
            }
    return best, candidates, failures, tabu_result


def main():
    parser = argparse.ArgumentParser(
        description="Choose a Problem-1 plan using official makespan evaluation")
    parser.add_argument("graph", type=Path)
    parser.add_argument("-n", "--num-cores", type=int, required=True)
    parser.add_argument("--groups", type=int, nargs="+")
    parser.add_argument(
        "--orders", nargs="+",
        help="order modes (critical/id/level/lowcritical/random:N)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--local-rounds", type=int, default=1)
    parser.add_argument("--tabu-iterations", type=int, default=0)
    parser.add_argument("--tabu-shortlist", type=int, default=3)
    parser.add_argument("--tabu-max-evaluations", type=int)
    parser.add_argument("--evaluator-root", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    args = parser.parse_args()
    if args.num_cores not in (2, 3, 4, 5):
        parser.error("num-cores must be one of 2, 3, 4 or 5")
    evaluator_root = args.evaluator_root or args.graph.parent.parent
    config_path = args.config or evaluator_root / "data" / "config.txt"
    graph = json.loads(args.graph.read_text(encoding="utf-8"))
    best, candidates, failures, tabu_result = select_plan(
        graph, args.num_cores, evaluator_root, config_path,
        groups=args.groups, orders=args.orders, seed=args.seed,
        local_rounds=args.local_rounds,
        tabu_iterations=args.tabu_iterations,
        tabu_shortlist=args.tabu_shortlist,
        tabu_max_evaluations=args.tabu_max_evaluations,
    )
    args.output.write_text(
        json.dumps(best["plan"], ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    report = {
        "graph": str(args.graph),
        "cores": args.num_cores,
        "selected_groups": best["groups"],
        "selected_order_mode": best["order_mode"],
        "selected_order_seed": best["order_seed"],
        "selected_key": best["key"],
        "candidates": [
            {
                "groups": item["groups"],
                "order_mode": item["order_mode"],
                "order_seed": item["order_seed"],
                "key": item["key"],
            }
            for item in candidates
        ],
        "failures": failures,
        "tabu_history": tabu_result["history"] if tabu_result else [],
        "global_optimum_claim": False,
    }
    if args.report:
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
