#!/usr/bin/env python3
"""Problem-1 HEFT scheduler with official Step3 task durations.

The A-question evaluator treats every subgraph as one Task.  This wrapper
keeps the communication-aware Metis partition from :mod:`aco_scheduler`, but
assigns its Tasks to cores with a HEFT list schedule.  The estimated Task
duration is obtained from the official Step1/2/3 code, so the processor
selection sees the same local memory and pipe effects as the final evaluator.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from aco_scheduler import AntColony
from greedy_sa_scheduler import (
    build_dependencies,
    group_graph,
    load_graph_from_object,
    criticality,
    topological_order,
)


def _load_official_code(evaluator_root: Path):
    """Import the supplied evaluator without modifying the user's project."""
    code_dir = evaluator_root / "code"
    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))
    from evaluation_validation import read_evaluation_config
    from multicore_cut_evaluate_problem_1 import _build_scene_a_tasks
    return read_evaluation_config, _build_scene_a_tasks


def _compact_mapping(mapping):
    used = sorted(set(mapping.values()))
    remap = {old: new for new, old in enumerate(used)}
    return {node: remap[group] for node, group in mapping.items()}


def _topological_groups(group_preds, group_succs):
    indegree = {group: len(preds) for group, preds in group_preds.items()}
    ready = sorted(group for group, degree in indegree.items() if degree == 0)
    order = []
    while ready:
        group = ready.pop(0)
        order.append(group)
        for child in sorted(group_succs[group]):
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
                ready.sort()
    if len(order) != len(group_preds):
        raise ValueError("subgraph quotient graph contains a cycle")
    return order


def _exact_task_work(graph, mapping, group_count, cores, evaluator_root,
                     config_path):
    """Get official Step3 makespan for each subgraph.

    All groups are temporarily placed on core 0 only to build the independent
    local Task graphs.  Core placement affects release delays, not Step3's
    local pipe schedule, so this gives the same duration for every candidate
    core assignment.
    """
    read_evaluation_config, build_scene_tasks = _load_official_code(
        evaluator_root)
    graph_view, ops, tensors, eligible = load_graph_from_object(graph)
    preds, succs, weights = build_dependencies(
        graph_view, ops, tensors, eligible)
    topo = topological_order(eligible, preds, succs)
    group_order = group_graph(mapping, group_count, preds, topo)
    if group_order is None:
        raise ValueError("operation partition has a cyclic quotient graph")
    temporary_plan = {
        "node_to_subgraph": {
            str(node): int(group) for node, group in mapping.items()
        },
        "core_schedules": [group_order] + [[] for _ in range(cores - 1)],
    }
    config = read_evaluation_config(str(config_path))
    tasks, _, _, view = build_scene_tasks(
        graph,
        temporary_plan,
        config["bandwidth"],
        config["capacity"],
    )
    work = [float(tasks[group]["step3"]["makespan"])
            for group in range(group_count)]
    return work, view, preds, succs, weights, group_order


def _assign_heft(mapping, group_count, cores, work, view, preds, succs,
                 weights):
    """HEFT processor selection with 100/1000-cycle release delays."""
    group_preds = {group: set() for group in range(group_count)}
    group_succs = {group: set() for group in range(group_count)}
    communication = {}
    for source, children in succs.items():
        source_group = mapping[source]
        for target in children:
            target_group = mapping[target]
            if source_group == target_group:
                continue
            group_preds[target_group].add(source_group)
            group_succs[source_group].add(target_group)
            communication[(source_group, target_group)] = (
                communication.get((source_group, target_group), 0)
                + weights.get((source, target), 0)
            )

    # Prefer the official quotient view when available; it has already
    # contracted COPY nodes and passed the evaluator's DAG checks.
    if view.get("subgraph_preds"):
        group_preds = {
            group: set(view["subgraph_preds"].get(group, ()))
            for group in range(group_count)
        }
        group_succs = {
            group: set(view["subgraph_succs"].get(group, ()))
            for group in range(group_count)
        }
    topo = _topological_groups(group_preds, group_succs)

    rank = [0.0] * group_count
    for group in reversed(topo):
        rank[group] = work[group] + max(
            (
                rank[child] + communication.get((group, child), 0) / 60.0
                for child in group_succs[group]
            ),
            default=0.0,
        )

    group_core = [-1] * group_count
    end_time = [0.0] * group_count
    core_end = [0.0] * cores
    core_used = [False] * cores
    for group in sorted(topo, key=lambda item: (-rank[item], item)):
        best = None
        for core in range(cores):
            start = core_end[core] + (100.0 if core_used[core] else 0.0)
            for pred in group_preds[group]:
                pred_core = group_core[pred]
                if pred_core < 0:
                    raise ValueError("HEFT predecessor was not assigned")
                if pred_core == core:
                    delay = 100.0
                else:
                    delay = 1000.0 + communication.get((pred, group), 0) / 60.0
                start = max(start, end_time[pred] + delay)
            candidate = (start + work[group], start, core)
            if best is None or candidate < best:
                best = candidate
        finish, _, core = best
        group_core[group] = core
        end_time[group] = finish
        core_end[core] = finish
        core_used[core] = True

    schedules = [[] for _ in range(cores)]
    for group in topo:
        schedules[group_core[group]].append(group)
    return group_core, schedules


def _set_order(colony, mode, seed):
    """Choose the topological linear extension used before Metis labeling."""
    if mode == "critical":
        return
    base = topological_order(colony.eligible, colony.preds, colony.succs)
    if mode == "id":
        order = base
    elif mode == "lowcritical":
        rank = criticality(base, colony.succs, colony.ops)
        order = topological_order(
            colony.eligible, colony.preds, colony.succs,
            {node: -value for node, value in rank.items()},
        )
    elif mode == "level":
        level = {}
        for node in base:
            level[node] = 0 if not colony.preds[node] else 1 + max(
                level[parent] for parent in colony.preds[node]
            )
        order = topological_order(
            colony.eligible, colony.preds, colony.succs,
            {node: -value for node, value in level.items()},
        )
    elif mode == "random":
        rng = random.Random(seed)
        order = topological_order(
            colony.eligible, colony.preds, colony.succs,
            {node: rng.random() for node in colony.eligible},
        )
    else:
        raise ValueError(f"unknown order mode: {mode}")
    colony.order = order
    colony.search.order = order


def generate_problem1_plan(graph, cores, seed, groups, local_rounds,
                           evaluator_root, config_path,
                           order_mode="critical", order_seed=None):
    colony = AntColony(
        graph,
        cores,
        ants=2,
        iterations=2,
        seed=seed,
        scene="A",
        group_count=groups,
        local_rounds=local_rounds,
    )
    _set_order(colony, order_mode, seed if order_seed is None else order_seed)
    seed_solution = colony._guided_metis_solution()
    if seed_solution is None:
        raise RuntimeError("Metis partition could not be constructed")
    mapping = _compact_mapping(seed_solution[0])
    group_count = max(mapping.values(), default=-1) + 1
    work, view, preds, succs, weights, _ = _exact_task_work(
        graph, mapping, group_count, cores, evaluator_root, config_path)
    _, schedules = _assign_heft(
        mapping, group_count, cores, work, view, preds, succs, weights)
    return {
        "node_to_subgraph": {
            str(node): int(group) for node, group in sorted(mapping.items())
        },
        "core_schedules": schedules,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Problem-1 HEFT plan using official Step3 task durations")
    parser.add_argument("graph", type=Path)
    parser.add_argument("-n", "--num-cores", type=int, required=True)
    parser.add_argument("--groups", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--local-rounds", type=int, default=1)
    parser.add_argument(
        "--order-mode",
        choices=["critical", "id", "lowcritical", "level", "random"],
        default="critical",
    )
    parser.add_argument("--order-seed", type=int)
    parser.add_argument("--evaluator-root", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    args = parser.parse_args()
    if args.num_cores < 1 or args.local_rounds < 0:
        parser.error("num-cores must be positive; local-rounds cannot be negative")
    evaluator_root = args.evaluator_root or args.graph.parent.parent
    config_path = args.config or evaluator_root / "data" / "config.txt"
    groups = args.groups or min(2 * args.num_cores, 64)
    with args.graph.open(encoding="utf-8") as handle:
        graph = json.load(handle)
    plan = generate_problem1_plan(
        graph,
        args.num_cores,
        args.seed,
        groups,
        args.local_rounds,
        evaluator_root,
        config_path,
        order_mode=args.order_mode,
        order_seed=args.order_seed,
    )
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(plan, handle, ensure_ascii=False, indent=2, sort_keys=True)
    print(json.dumps({
        "output": str(args.output),
        "cores": args.num_cores,
        "groups": len(set(plan["node_to_subgraph"].values())),
        "algorithm": "metis+official-step3-heft",
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
