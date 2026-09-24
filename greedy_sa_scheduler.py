#!/usr/bin/env python3
"""Greedy + simulated annealing multi-core plan generator.

The competition evaluator owns the detailed cache, pipe, spill and event
simulation.  This file only searches the two fields accepted by that
evaluator: node_to_subgraph and core_schedules.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict, deque
from pathlib import Path


COPY_TYPES = {"COPY_IN", "COPY_OUT"}


def load_graph(path: Path):
    with path.open(encoding="utf-8") as handle:
        graph = json.load(handle)
    ops = {int(op["id"]): op for op in graph["ops"]}
    tensors = {int(t["id"]): t for t in graph["tensors"]}
    eligible = {op_id for op_id, op in ops.items() if op.get("op") not in COPY_TYPES}
    return graph, ops, tensors, eligible


def build_dependencies(graph, ops, tensors, eligible):
    """Build an eligible-op DAG, contracting COPY operations."""
    full_succ = defaultdict(set)
    tensor_producers = defaultdict(list)
    tensor_consumers = defaultdict(list)
    for edge in graph["edges"]:
        src, dst = int(edge["source"]), int(edge["target"])
        if src in ops and dst in ops:
            full_succ[src].add(dst)
        elif src in ops and dst in tensors:
            tensor_producers[dst].append(src)
        elif src in tensors and dst in ops:
            tensor_consumers[src].append(dst)

    edge_weights = {}
    for tensor_id, producers in tensor_producers.items():
        for src in producers:
            for dst in tensor_consumers.get(tensor_id, ()):
                full_succ[src].add(dst)
                # Several tensors can connect the same pair of operations;
                # their communication cost is additive.
                edge_weights[(src, dst)] = (
                    edge_weights.get((src, dst), 0)
                    + int(tensors[tensor_id].get("size", 0))
                )

    succs = {op_id: set() for op_id in eligible}
    weights = {}
    excluded = set(ops) - set(eligible)
    reach_cache = {}

    def excluded_reach(node, visiting=None):
        """Return eligible descendants of one excluded op, cached once.

        Values are accumulated over independent tensor paths. Along one COPY
        chain the largest boundary tensor is retained, matching the logical
        tensor transfer represented by that chain.
        """
        if node in reach_cache:
            return reach_cache[node]
        visiting = set() if visiting is None else visiting
        if node in visiting:
            return {}
        visiting.add(node)
        result = defaultdict(int)
        for nxt in full_succ[node]:
            boundary = edge_weights.get((node, nxt), 0)
            if nxt in eligible:
                result[nxt] += boundary
            elif nxt in excluded:
                for dest, child_weight in excluded_reach(nxt, visiting).items():
                    result[dest] += max(boundary, child_weight)
        visiting.remove(node)
        reach_cache[node] = dict(result)
        return reach_cache[node]

    for src in sorted(eligible):
        for dst in full_succ[src]:
            boundary = edge_weights.get((src, dst), 0)
            if dst in eligible:
                succs[src].add(dst)
                weights[(src, dst)] = weights.get((src, dst), 0) + boundary
            elif dst in excluded:
                for dest, path_weight in excluded_reach(dst).items():
                    if dest != src:
                        succs[src].add(dest)
                        # Distinct tensor paths represent distinct transfers.
                        weights[(src, dest)] = weights.get((src, dest), 0) + max(boundary, path_weight)

    preds = {op_id: set() for op_id in eligible}
    for src, children in succs.items():
        for dst in children:
            preds[dst].add(src)
    return preds, succs, weights


def topological_order(nodes, preds, succs, priority=None):
    indegree = {node: len(preds[node]) for node in nodes}
    ready = [node for node in nodes if indegree[node] == 0]
    order = []
    while ready:
        if priority is None:
            node = min(ready)
        else:
            node = max(ready, key=lambda value: (priority.get(value, 0), -value))
        ready.remove(node)
        order.append(node)
        for child in succs[node]:
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
    if len(order) != len(nodes):
        raise ValueError("eligible operation graph contains a cycle")
    return order


def criticality(order, succs, ops):
    rank = {}
    for node in reversed(order):
        own = max(1, int(ops[node].get("cycles", 0)))
        rank[node] = own + max((rank[child] for child in succs[node]), default=0)
    return rank


def group_graph(mapping, group_count, preds, order):
    group_preds = [set() for _ in range(group_count)]
    group_min_pos = [len(order) + 1] * group_count
    position = {node: index for index, node in enumerate(order)}
    for node, group in mapping.items():
        group_min_pos[group] = min(group_min_pos[group], position[node])
        for pred in preds[node]:
            pred_group = mapping[pred]
            if pred_group != group:
                group_preds[group].add(pred_group)
    indegree = [len(items) for items in group_preds]
    ready = [group for group, degree in enumerate(indegree) if degree == 0]
    result = []
    while ready:
        group = min(ready, key=lambda item: (group_min_pos[item], item))
        ready.remove(group)
        result.append(group)
        for child, dependencies in enumerate(group_preds):
            if group in dependencies:
                indegree[child] -= 1
                if indegree[child] == 0:
                    ready.append(child)
    if len(result) != group_count:
        return None
    return result


DEFAULT_LAMBDA_WEIGHTS = {
    # Objectives are all minimised after normalization:
    # makespan, cross-core synchronization/traffic, cut traffic, imbalance,
    # and cache pressure.  The fifth component is only active in scene C.
    "A": (0.55, 0.25, 0.10, 0.10, 0.00),
    "B": (0.45, 0.35, 0.10, 0.10, 0.00),
    "C": (0.40, 0.20, 0.05, 0.05, 0.30),
}


class WeightedTchebycheff:
    """Normalized weighted Chebyshev scalarization for minimization.

    A linear weighted sum can hide a poor objective whenever another term is
    numerically much larger.  The Chebyshev form keeps the worst normalized
    deviation visible while the small ``rho`` term breaks ties smoothly.
    """

    def __init__(self, scales, weights=None, rho=0.05):
        # Every component is a non-negative cost with natural ideal value 0;
        # this is the z* point in the model description.
        self.scales = tuple(max(float(value), 1e-12) for value in scales)
        raw = tuple(weights or (1.0,) * len(self.scales))
        if len(raw) != len(self.scales) or any(value < 0 for value in raw):
            raise ValueError("Tchebycheff weights must be non-negative and match objectives")
        if not any(raw):
            raise ValueError("at least one Tchebycheff weight must be positive")
        self.weights = raw
        self.rho = max(0.0, float(rho))

    def normalized(self, values):
        if len(values) != len(self.scales):
            raise ValueError("objective dimension does not match Tchebycheff scales")
        return tuple(max(0.0, float(value)) / scale
                     for value, scale in zip(values, self.scales))

    def __call__(self, values):
        normalized = self.normalized(values)
        weighted = [weight * value
                    for weight, value in zip(self.weights, normalized)]
        return max(weighted) + self.rho * sum(weighted)


class ParetoArchive:
    """Small deterministic archive of non-dominated minimization solutions."""

    @staticmethod
    def dominates(left, right):
        return (all(a <= b + 1e-12 for a, b in zip(left, right))
                and any(a < b - 1e-12 for a, b in zip(left, right)))

    def __init__(self, max_size=128):
        self.max_size = max(1, int(max_size))
        self.items = []

    def add(self, mapping, group_core, values, score):
        values = tuple(float(value) for value in values)
        if any(self.dominates(item["objectives"], values)
               for item in self.items):
            return False
        self.items = [item for item in self.items
                      if not self.dominates(values, item["objectives"])]
        self.items.append({
            "objectives": values,
            "score": float(score),
            "mapping": {str(node): int(group) for node, group in mapping.items()},
            "group_core": [int(core) for core in group_core],
        })
        if len(self.items) > self.max_size:
            self.items.sort(key=lambda item: (item["score"], item["objectives"]))
            self.items = self.items[:self.max_size]
        return True

    def sorted_items(self):
        return sorted(self.items, key=lambda item: (item["score"], item["objectives"]))


class Search:
    def __init__(self, graph, ops, eligible, preds, succs, weights, cores, seed,
                 scene, lambda_weights=None, tchebycheff_rho=0.05):
        self.graph = graph
        self.ops = ops
        self.eligible = eligible
        self.preds = preds
        self.succs = succs
        self.weights = weights
        self.cores = cores
        self.rng = random.Random(seed)
        self.scene = scene
        raw_weights = lambda_weights or DEFAULT_LAMBDA_WEIGHTS.get(
            scene, DEFAULT_LAMBDA_WEIGHTS["B"])
        total_cycles = sum(max(1, int(ops[node].get("cycles", 0)))
                           for node in eligible)
        total_bytes = sum(max(0, int(tensor.get("size", 0)))
                          for tensor in graph.get("tensors", []))
        wait = 500.0 if scene in {"B", "C"} else 1000.0
        self.objective_names = (
            "proxy_makespan", "cross_sync", "cut_transfer",
            "imbalance", "cache_pressure")
        self.scalarizer = WeightedTchebycheff(
            scales=(
                max(1.0, total_cycles),
                max(1.0, total_bytes / 60.0 + len(weights) * wait),
                max(1.0, total_bytes / 60.0),
                max(1.0, total_cycles),
                max(1.0, 1048576.0 / 250.0),
            ),
            weights=raw_weights,
            rho=tchebycheff_rho,
        )
        self.lambda_weights = tuple(raw_weights)
        self.pareto_archive = ParetoArchive()
        self.order = []
        self.cycles = {}
        self.max_groups = min(len(eligible), max(cores, 2 * cores))
        self.pipe_by_node = {
            node: str(ops[node].get("pipe", "PIPE_M")) for node in eligible
        }
        graph_tensors = {
            int(tensor["id"]): tensor for tensor in graph.get("tensors", [])
        }
        graph_succ = defaultdict(list)
        for edge in graph.get("edges", []):
            graph_succ[int(edge["source"])].append(int(edge["target"]))
        self.ddr_input_consumers = defaultdict(list)
        # Follow tensor -> COPY_IN -> tensor -> operation paths so problem 3
        # can recognize repeated DDR inputs even though the original graph
        # does not connect DDR tensors directly to eligible operations.
        for tensor_id, tensor in graph_tensors.items():
            if tensor.get("pos") != "DDR":
                continue
            queue = deque(graph_succ[tensor_id])
            seen = set()
            while queue:
                node = queue.popleft()
                if node in seen:
                    continue
                seen.add(node)
                if node in eligible:
                    self.ddr_input_consumers[tensor_id].append(node)
                    continue
                queue.extend(graph_succ[node])
        self.tensor_sizes = {
            int(tensor["id"]): int(tensor.get("size", 0))
            for tensor in graph.get("tensors", [])
        }

    def make_order(self):
        topo = topological_order(self.eligible, self.preds, self.succs)
        rank = criticality(topo, self.succs, self.ops)
        self.order = topological_order(self.eligible, self.preds, self.succs, rank)
        self.cycles = {node: max(1, int(self.ops[node].get("cycles", 0))) for node in self.eligible}

    def initial_state(self):
        group_count = self.max_groups
        # Contiguous topological blocks guarantee an acyclic quotient graph.
        # Subsequent moves are accepted only when this invariant remains true.
        mapping = {
            node: min(group_count - 1, index * group_count // len(self.order))
            for index, node in enumerate(self.order)
        }
        group_core = [index % self.cores for index in range(group_count)]
        self.rebalance_cores(mapping, group_core, group_count)
        return mapping, group_core

    def incremental_score(self, node, group, mapping, group_count):
        group_core = group % self.cores
        score = self.cycles[node] / 1000.0
        for pred in self.preds[node]:
            if pred not in mapping:
                continue
            pred_group = mapping[pred]
            if pred_group != group:
                size = self.weights.get((pred, node), 0)
                score += size / 60.0
                if pred_group % self.cores != group_core:
                    score += 500.0 if self.scene in {"B", "C"} else 1000.0
        return score + 0.01 * group

    def rebalance_cores(self, mapping, group_core, group_count):
        loads = [0] * self.cores
        group_loads = [0] * group_count
        for node, group in mapping.items():
            group_loads[group] += self.cycles[node]
        for group in sorted(range(group_count), key=lambda item: group_loads[item], reverse=True):
            core = min(range(self.cores), key=lambda item: loads[item])
            group_core[group] = core
            loads[core] += group_loads[group]

    def objective_components(self, mapping, group_core):
        group_count = max(mapping.values(), default=-1) + 1
        if group_count <= 0:
            return None
        if any(group not in mapping.values() for group in range(group_count)):
            return None
        if group_graph(mapping, group_count, self.preds, self.order) is None:
            return None

        core_loads = [0] * self.cores
        pipe_loads = [defaultdict(int) for _ in range(self.cores)]
        cut_bytes = 0
        cross_bytes = 0
        cross_edges = 0
        for node, group in mapping.items():
            core = group_core[group]
            cycles = self.cycles[node]
            core_loads[core] += cycles
            pipe_loads[core][self.pipe_by_node[node]] += cycles
            for pred in self.preds[node]:
                size = self.weights.get((pred, node), 0)
                if mapping[pred] != group:
                    cut_bytes += size
                if group_core[mapping[pred]] != group_core[group]:
                    cross_bytes += size
                    cross_edges += 1
        max_load = max(core_loads, default=0)
        mean_load = sum(core_loads) / max(1, self.cores)
        imbalance = max_load - mean_load
        pipe_spans = [max(loads.values(), default=0) for loads in pipe_loads]
        pipe_span = max(pipe_spans, default=0)
        # A task cannot finish before its busiest pipe finishes. Keep a small
        # total-load term because dependencies prevent perfect overlap.
        proxy_makespan = 0.75 * pipe_span + 0.25 * max_load
        cross_sync = cross_bytes / 60.0 + (
            500.0 if self.scene in {"B", "C"} else 1000.0
        ) * cross_edges
        cache_pressure = 0.0
        if self.scene == "C":
            # A DDR input consumed by several cores is a likely L2 reuse
            # opportunity. Reward reuse when the shared working set fits in
            # the 1 MiB cache; otherwise penalize only the excess pressure.
            cache_reuse_bytes = 0
            cache_working_set = 0
            for tensor_id, consumers in self.ddr_input_consumers.items():
                consumer_cores = {
                    group_core[mapping[node]] for node in consumers if node in mapping
                }
                if len(consumer_cores) > 1:
                    cache_working_set += self.tensor_sizes[tensor_id]
                    cache_reuse_bytes += self.tensor_sizes[tensor_id] * (len(consumer_cores) - 1)
            cache_capacity = 1048576
            # Lower is better: shared bytes that can be reused are cheaper,
            # while exceeding the 1 MiB FIFO cache is penalised explicitly.
            cache_pressure = max(
                0.0,
                (cache_working_set - 0.75 * cache_reuse_bytes) / 250.0,
            ) + max(0.0, cache_working_set - cache_capacity) / 250.0
        return (
            float(proxy_makespan),
            float(cross_sync),
            float(cut_bytes / 60.0),
            float(imbalance),
            float(cache_pressure),
        )

    def objective(self, mapping, group_core):
        components = self.objective_components(mapping, group_core)
        if components is None:
            return float("inf")
        return self.scalarizer(components)

    def anneal(self, iterations, start_temp, cooling):
        mapping, group_core = self.initial_state()
        current = self.objective(mapping, group_core)
        best = (dict(mapping), list(group_core), current)
        for step in range(iterations):
            candidate_mapping = dict(mapping)
            candidate_core = list(group_core)
            if self.rng.random() < 0.78:
                node = self.rng.choice(self.order)
                source = candidate_mapping[node]
                target = self.rng.randrange(self.max_groups)
                if target == source or sum(1 for item in candidate_mapping.values() if item == source) <= 1:
                    continue
                candidate_mapping[node] = target
            else:
                group = self.rng.randrange(self.max_groups)
                candidate_core[group] = self.rng.randrange(self.cores)
            candidate = self.objective(candidate_mapping, candidate_core)
            temperature = max(1e-6, start_temp * (cooling ** step))
            delta = candidate - current
            if delta <= 0 or self.rng.random() < math.exp(-delta / temperature):
                mapping, group_core, current = candidate_mapping, candidate_core, candidate
                if current < best[2]:
                    best = (dict(mapping), list(group_core), current)
        return best[0], best[1], best[2]

    def to_plan(self, mapping, group_core):
        used = sorted(set(mapping.values()))
        compact = {old: new for new, old in enumerate(used)}
        compact_mapping = {node: compact[group] for node, group in mapping.items()}
        compact_core = [group_core[group] for group in used]
        group_order = group_graph(compact_mapping, len(used), self.preds, self.order)
        if group_order is None:
            raise ValueError("best state has cyclic subgraph quotient")
        schedules = [[] for _ in range(self.cores)]
        for group in group_order:
            schedules[compact_core[group]].append(group)
        return {
            "node_to_subgraph": {str(node): group for node, group in sorted(compact_mapping.items())},
            "core_schedules": schedules,
        }


def generate_plan(graph, cores, iterations, seed, scene):
    graph, ops, tensors, eligible = load_graph_from_object(graph)
    preds, succs, weights = build_dependencies(graph, ops, tensors, eligible)
    search = Search(graph, ops, eligible, preds, succs, weights, cores, seed, scene)
    search.make_order()
    mapping, group_core, _ = search.anneal(iterations, start_temp=3000.0, cooling=0.995)
    return search.to_plan(mapping, group_core)


def load_graph_from_object(graph):
    ops = {int(op["id"]): op for op in graph["ops"]}
    tensors = {int(t["id"]): t for t in graph["tensors"]}
    eligible = {op_id for op_id, op in ops.items() if op.get("op") not in COPY_TYPES}
    return graph, ops, tensors, eligible


def main():
    parser = argparse.ArgumentParser(description="Generate a greedy + simulated annealing multicore plan")
    parser.add_argument("graph", type=Path)
    parser.add_argument("-n", "--num-cores", type=int, required=True)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scene", choices=["A", "B", "C"], default="B")
    parser.add_argument("-o", "--output", type=Path, required=True)
    args = parser.parse_args()
    if args.num_cores < 1 or args.iterations < 1:
        parser.error("num-cores and iterations must be positive")
    with args.graph.open(encoding="utf-8") as handle:
        graph = json.load(handle)
    plan = generate_plan(graph, args.num_cores, args.iterations, args.seed, args.scene)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(plan, handle, indent=2, sort_keys=True)
    print(json.dumps({"output": str(args.output), "cores": args.num_cores,
                      "iterations": args.iterations, "seed": args.seed,
                      "subgraphs": len(set(plan["node_to_subgraph"].values()))}, ensure_ascii=False))


if __name__ == "__main__":
    main()
