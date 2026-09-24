#!/usr/bin/env python3
"""Ant-colony search for the competition's multicore plan format.

An ant constructs an operation-to-subgraph mapping in topological order. The
pheromone table stores the historical quality of assigning an operation to a
subgraph. Core placement is rebalanced after construction and the resulting
plan is checked by the official evaluator.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path

try:
    import pymetis
except ImportError:  # pragma: no cover - runtime fallback
    pymetis = None

# OR-Tools 9.15 currently aborts during import on this macOS/Python 3.13
# runtime because of a protobuf descriptor conflict. Keep the hook disabled
# in-process; _solve_core_assignment falls back to deterministic balancing.
cp_model = None

from greedy_sa_scheduler import (
    COPY_TYPES,
    DEFAULT_LAMBDA_WEIGHTS,
    Search,
    build_dependencies,
    criticality,
    group_graph,
    load_graph_from_object,
    topological_order,
)


class AntColony:
    def __init__(self, graph, cores, ants, iterations, seed, scene,
                 group_count=None, local_rounds=2, lambda_weights=None,
                 tchebycheff_rho=0.05, alpha=1.1, beta=2.0,
                 evaporation=0.18):
        graph, ops, tensors, eligible = load_graph_from_object(graph)
        preds, succs, weights = build_dependencies(graph, ops, tensors, eligible)
        topo = topological_order(eligible, preds, succs)
        rank = criticality(topo, succs, ops)
        self.graph = graph
        self.ops = ops
        self.eligible = eligible
        self.preds = preds
        self.succs = succs
        self.weights = weights
        self.order = topological_order(eligible, preds, succs, rank)
        self.cores = cores
        self.ants = ants
        self.iterations = iterations
        self.rng = random.Random(seed)
        self.scene = scene
        self.group_count = group_count or min(len(eligible), max(cores, 2 * cores))
        self.local_rounds = max(0, local_rounds)
        self.alpha = max(0.0, float(alpha))
        self.beta = max(0.0, float(beta))
        self.evaporation = min(0.99, max(0.0, float(evaporation)))
        self.lambda_weights = tuple(lambda_weights or DEFAULT_LAMBDA_WEIGHTS[scene])
        self.search = Search(graph, ops, eligible, preds, succs, weights,
                             cores, seed, scene,
                             lambda_weights=self.lambda_weights,
                             tchebycheff_rho=tchebycheff_rho)
        self.search.order = self.order
        self.search.cycles = {
            node: max(1, int(ops[node].get("cycles", 0)))
            for node in eligible
        }
        self.pheromone = {
            node: [1.0] * self.group_count for node in self.order
        }
        self.pareto_archive = self.search.pareto_archive

    def _record_solution(self, mapping, group_core, score):
        components = self.search.objective_components(mapping, group_core)
        if components is not None:
            self.pareto_archive.add(mapping, group_core, components, score)

    def _metis_labels(self):
        """Return a communication-aware partition label for each operation."""
        if pymetis is None or self.group_count <= 1:
            return None
        index = {node: i for i, node in enumerate(self.order)}
        adjacency = [[] for _ in self.order]
        edge_seen = set()
        for src in self.order:
            for dst in self.succs[src]:
                if dst not in index:
                    continue
                a, b = index[src], index[dst]
                if a == b or (a, b) in edge_seen or (b, a) in edge_seen:
                    continue
                edge_seen.add((a, b))
                adjacency[a].append(b)
                adjacency[b].append(a)
        try:
            _, membership = pymetis.part_graph(
                self.group_count,
                adjacency=adjacency,
                vweights=[max(1, self.search.cycles[node]) for node in self.order],
            )
        except Exception:
            return None
        return {node: int(membership[index[node]]) for node in self.order}

    def _guided_metis_solution(self):
        """Use Metis labels as preferences, while constructing an acyclic quotient."""
        labels = self._metis_labels()
        if labels is None:
            return None
        mapping = {}
        adjacency = [set() for _ in range(self.group_count)]

        def reachable(start, target):
            if start == target:
                return True
            stack, seen = [start], set()
            while stack:
                current = stack.pop()
                if current in seen:
                    continue
                seen.add(current)
                stack.extend(adjacency[current])
                if target in adjacency[current]:
                    return True
            return False

        for node in self.order:
            preferred = labels[node]
            valid = []
            for group in range(self.group_count):
                bad = False
                for pred in self.preds[node]:
                    if pred not in mapping:
                        continue
                    pred_group = mapping[pred]
                    if pred_group != group and reachable(group, pred_group):
                        bad = True
                        break
                if not bad:
                    valid.append(group)
            if not valid:
                valid = list(range(self.group_count))
            group = preferred if preferred in valid else min(
                valid,
                key=lambda candidate: self.heuristic(node, candidate, mapping),
            )
            for pred in self.preds[node]:
                if pred in mapping and mapping[pred] != group:
                    adjacency[mapping[pred]].add(group)
            mapping[node] = group

        used = sorted(set(mapping.values()))
        if len(used) < self.group_count:
            # Keep the search state compact if a partition bucket was empty.
            remap = {old: new for new, old in enumerate(used)}
            mapping = {node: remap[group] for node, group in mapping.items()}
        actual_groups = max(mapping.values()) + 1
        group_core = [index % self.cores for index in range(actual_groups)]
        self._solve_core_assignment(mapping, group_core, actual_groups)
        score = self.search.objective(mapping, group_core)
        return mapping, group_core, score

    def _solve_core_assignment(self, mapping, group_core, group_count):
        """Optimize a small group-to-core assignment with CP-SAT."""
        # Problem 1 is dominated by task release delays (1000 cycles across
        # cores and 100 cycles between tasks on one core).  A pure load
        # balancing assignment can therefore put a critical chain on
        # different cores and lose more to synchronization than it gains from
        # parallelism.  Use a lightweight HEFT-style processor selection for
        # scene A; scenes B/C keep the communication/cache-oriented balancing
        # path below.
        if self.scene == "A":
            self._heft_core_assignment(mapping, group_core, group_count)
            return
        if cp_model is None or group_count == 0:
            self.search.rebalance_cores(mapping, group_core, group_count)
            return
        model = cp_model.CpModel()
        core_vars = [model.NewIntVar(0, self.cores - 1, f'core_{g}')
                     for g in range(group_count)]
        loads = [model.NewIntVar(0, sum(self.search.cycles.values()), f'load_{k}')
                 for k in range(self.cores)]
        group_loads = [sum(self.search.cycles[node] for node, group in mapping.items()
                           if group == g) for g in range(group_count)]
        for k in range(self.cores):
            terms = []
            for g, load in enumerate(group_loads):
                flag = model.NewBoolVar(f'g{g}_k{k}')
                model.Add(core_vars[g] == k).OnlyEnforceIf(flag)
                model.Add(core_vars[g] != k).OnlyEnforceIf(flag.Not())
                terms.append(load * flag)
            model.Add(loads[k] == sum(terms))
        max_load = model.NewIntVar(0, sum(self.search.cycles.values()), 'max_load')
        for load in loads:
            model.Add(max_load >= load)
        objective_terms = [max_load * 100]
        for src in mapping:
            for dst in self.succs[src]:
                if dst not in mapping or mapping[src] == mapping[dst]:
                    continue
                same = model.NewBoolVar(f'same_{mapping[src]}_{mapping[dst]}_{src}')
                model.Add(core_vars[mapping[src]] == core_vars[mapping[dst]]).OnlyEnforceIf(same)
                model.Add(core_vars[mapping[src]] != core_vars[mapping[dst]]).OnlyEnforceIf(same.Not())
                objective_terms.append((self.weights.get((src, dst), 1) // 16 + 1) * (1 - same))
        model.Minimize(sum(objective_terms))
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 0.15
        solver.parameters.num_search_workers = 1
        if solver.Solve(model) in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            for group in range(group_count):
                group_core[group] = solver.Value(core_vars[group])

    def _heft_core_assignment(self, mapping, group_core, group_count):
        """Assign Metis subgraphs to cores with a task-level HEFT heuristic.

        The official problem-1 simulator executes one subgraph as one Task.
        This routine estimates each Task's local work from its pipe loads and
        inserts the official synchronization costs when evaluating a core.
        It is deliberately deterministic and only changes the group-to-core
        assignment; the operation partition and the final topological order
        remain unchanged.
        """
        if group_count <= 0:
            return

        # Build the quotient DAG and aggregate boundary tensor traffic.
        group_preds = [set() for _ in range(group_count)]
        group_succs = [set() for _ in range(group_count)]
        boundary_bytes = {}
        pipe_work = [defaultdict(int) for _ in range(group_count)]
        total_work = [0] * group_count
        for node, group in mapping.items():
            cycles = self.search.cycles[node]
            total_work[group] += cycles
            pipe_work[group][self.search.pipe_by_node[node]] += cycles
            for child in self.succs[node]:
                if child not in mapping:
                    continue
                child_group = mapping[child]
                if child_group == group:
                    continue
                group_succs[group].add(child_group)
                group_preds[child_group].add(group)
                boundary_bytes[(group, child_group)] = (
                    boundary_bytes.get((group, child_group), 0)
                    + self.weights.get((node, child), 0)
                )

        # Estimate the duration of a Task.  The max-pipe term reflects the
        # evaluator's overlap of MTE and compute pipes; the total-work term
        # prevents a Task with many lightly used pipes from looking free.
        group_work = []
        for group in range(group_count):
            pipe_span = max(pipe_work[group].values(), default=0)
            local = 0.75 * pipe_span + 0.25 * total_work[group]
            incoming = sum(
                size for (src, dst), size in boundary_bytes.items() if dst == group
            )
            outgoing = sum(
                size for (src, dst), size in boundary_bytes.items() if src == group
            )
            # Each boundary is represented by a copy in problem 1.  Charging
            # half of the input and output traffic keeps this an estimate
            # rather than double-counting both sides of the same transfer.
            local += 0.5 * (incoming + outgoing) / 60.0
            group_work.append(max(1.0, local))

        # Upward rank: critical-path groups are placed first, as in HEFT.
        rank = [0.0] * group_count
        topo = []
        indegree = [len(group_preds[group]) for group in range(group_count)]
        ready = [group for group, degree in enumerate(indegree) if degree == 0]
        while ready:
            group = min(ready)
            ready.remove(group)
            topo.append(group)
            for child in sorted(group_succs[group]):
                indegree[child] -= 1
                if indegree[child] == 0:
                    ready.append(child)
        if len(topo) != group_count:
            # The caller normally guarantees an acyclic quotient.  Retain a
            # safe fallback if a future partitioner violates that contract.
            self.search.rebalance_cores(mapping, group_core, group_count)
            return
        for group in reversed(topo):
            rank[group] = group_work[group] + max(
                (
                    rank[child]
                    + boundary_bytes.get((group, child), 0) / 60.0
                    for child in group_succs[group]
                ),
                default=0.0,
            )

        order = sorted(topo, key=lambda group: (-rank[group], group))
        core_end = [0.0] * self.cores
        core_used = [False] * self.cores
        end_time = [0.0] * group_count
        for group in order:
            best = None
            for core in range(self.cores):
                ready_time = core_end[core] + (100.0 if core_used[core] else 0.0)
                for pred in group_preds[group]:
                    pred_core = group_core[pred]
                    communication = boundary_bytes.get((pred, group), 0) / 60.0
                    delay = 100.0 if pred_core == core else 1000.0 + communication
                    ready_time = max(ready_time, end_time[pred] + delay)
                finish = ready_time + group_work[group]
                candidate = (finish, ready_time, core, group)
                if best is None or candidate < best:
                    best = candidate
            _, start, core, _ = best
            group_core[group] = core
            end_time[group] = best[0]
            core_end[core] = best[0]
            core_used[core] = True

    @staticmethod
    def _reachable(adjacency, start, target):
        if start == target:
            return True
        stack = [start]
        seen = set()
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            for child in adjacency[current]:
                if child == target:
                    return True
                stack.append(child)
        return False

    def candidate_groups(self, node, mapping, adjacency):
        # Only edges from already assigned predecessors to the current node
        # can be added. The maintained quotient graph makes this O(E_group)
        # instead of rebuilding the operation-sized graph for every candidate.
        valid = []
        for group in range(self.group_count):
            creates_cycle = False
            for pred in self.preds[node]:
                if pred not in mapping:
                    continue
                pred_group = mapping[pred]
                if pred_group != group and self._reachable(adjacency, group, pred_group):
                    creates_cycle = True
                    break
            if not creates_cycle:
                valid.append(group)
        return valid or list(range(self.group_count))

    def heuristic(self, node, group, mapping):
        cycles = self.search.cycles[node]
        load = sum(self.search.cycles[item] for item, item_group in mapping.items()
                   if item_group == group)
        communication = 0.0
        for pred in self.preds[node]:
            if pred in mapping and mapping[pred] != group:
                communication += self.weights.get((pred, node), 0) / 60.0
        # A small load term encourages balanced subgraphs while communication
        # remains the dominant local signal.
        return 1.0 / (1.0 + cycles / 1000.0 + load / 10000.0 + communication)

    def choose_group(self, node, mapping, adjacency):
        valid = self.candidate_groups(node, mapping, adjacency)
        values = []
        for group in valid:
            tau = self.pheromone[node][group] ** self.alpha
            eta = self.heuristic(node, group, mapping) ** self.beta
            values.append(max(1e-15, tau * eta))
        total = sum(values)
        draw = self.rng.random() * total
        cumulative = 0.0
        for group, value in zip(valid, values):
            cumulative += value
            if cumulative >= draw:
                return group
        return valid[-1]

    def construct_ant(self):
        mapping = {}
        adjacency = [set() for _ in range(self.group_count)]

        def assign(node, group):
            for pred in self.preds[node]:
                if pred in mapping:
                    pred_group = mapping[pred]
                    if pred_group != group:
                        adjacency[pred_group].add(group)
            mapping[node] = group

        # Seed each group with one early topological operation so that the
        # eventual solution has no empty group IDs.
        anchors = min(self.group_count, len(self.order))
        for index, node in enumerate(self.order[:anchors]):
            assign(node, index)
        for node in self.order[anchors:]:
            assign(node, self.choose_group(node, mapping, adjacency))
        group_core = [index % self.cores for index in range(self.group_count)]
        self.search.rebalance_cores(mapping, group_core, self.group_count)
        score = self.search.objective(mapping, group_core)
        return mapping, group_core, score

    def local_improve(self, mapping, group_core, score):
        """Apply best-improvement moves around an elite ant solution."""
        current_mapping = dict(mapping)
        current_core = list(group_core)
        current_score = score
        for _ in range(self.local_rounds):
            best_move = None
            # Critical-path and high-cycle nodes have the largest influence;
            # sampling all nodes on small graphs keeps the result deterministic.
            candidates = self.order
            if len(candidates) > 1200:
                candidates = sorted(
                    candidates,
                    key=lambda node: self.search.cycles[node],
                    reverse=True,
                )[:1200]
            for node in candidates:
                source = current_mapping[node]
                target_groups = {
                    current_mapping[pred]
                    for pred in self.preds[node]
                    if pred in current_mapping
                }
                target_groups.update(
                    current_mapping[child]
                    for child in self.succs[node]
                    if child in current_mapping
                )
                target_groups.update(range(self.group_count))
                for target in target_groups:
                    if target == source:
                        continue
                    if sum(1 for value in current_mapping.values() if value == source) <= 1:
                        continue
                    candidate_mapping = dict(current_mapping)
                    candidate_mapping[node] = target
                    if group_graph(candidate_mapping, self.group_count,
                                   self.preds, self.order) is None:
                        continue
                    candidate_score = self.search.objective(candidate_mapping, current_core)
                    if candidate_score + 1e-9 < current_score:
                        if best_move is None or candidate_score < best_move[2]:
                            best_move = (candidate_mapping, list(current_core), candidate_score)

            # Reassign one subgraph to the least loaded core. This captures a
            # core-placement move without changing the operation partition.
            for group in range(self.group_count):
                old_core = current_core[group]
                for target_core in range(self.cores):
                    if target_core == old_core:
                        continue
                    candidate_core = list(current_core)
                    candidate_core[group] = target_core
                    candidate_score = self.search.objective(current_mapping, candidate_core)
                    if candidate_score + 1e-9 < current_score:
                        if best_move is None or candidate_score < best_move[2]:
                            best_move = (dict(current_mapping), candidate_core, candidate_score)
            if best_move is None:
                break
            current_mapping, current_core, current_score = best_move
        return current_mapping, current_core, current_score

    def update_pheromone(self, solutions, global_best):
        evaporation = self.evaporation
        for node in self.pheromone:
            for group in range(self.group_count):
                self.pheromone[node][group] *= 1.0 - evaporation
                self.pheromone[node][group] = max(0.05, self.pheromone[node][group])

        ranked = sorted(solutions, key=lambda item: item[2])[:max(1, self.ants // 3)]
        deposits = ranked + [global_best]
        for mapping, _, score in deposits:
            if not math.isfinite(score):
                continue
            # Scores are now normalized Chebyshev values (typically O(1));
            # the old 100000-scale deposit would saturate pheromones.
            amount = 1.0 / (score + 1.0)
            for node, group in mapping.items():
                self.pheromone[node][group] += amount

    def run(self):
        best = None
        metis_seed = self._guided_metis_solution()
        if metis_seed is not None:
            mapping, group_core, score = metis_seed
            self._record_solution(mapping, group_core, score)
            if self.local_rounds:
                mapping, group_core, score = self.local_improve(
                    mapping, group_core, score
                )
                self._record_solution(mapping, group_core, score)
            best = (dict(mapping), list(group_core), score)
        for _ in range(self.iterations):
            solutions = [self.construct_ant() for _ in range(self.ants)]
            for mapping, group_core, score in solutions:
                self._record_solution(mapping, group_core, score)
            elite_count = max(1, self.ants // 3)
            elite_indices = sorted(
                range(len(solutions)), key=lambda index: solutions[index][2]
            )[:elite_count]
            for index in elite_indices:
                mapping, group_core, score = solutions[index]
                solutions[index] = self.local_improve(mapping, group_core, score)
                self._record_solution(*solutions[index])
            local_best = min(solutions, key=lambda item: item[2])
            if best is None or local_best[2] < best[2]:
                best = (dict(local_best[0]), list(local_best[1]), local_best[2])
            self.update_pheromone(solutions, best)
        if best is None:
            raise RuntimeError("ant colony produced no solution")
        return self.search.to_plan(best[0], best[1]), best[2]

    def write_pareto_archive(self, path):
        """Write non-dominated objective vectors and their plans as JSON."""
        payload = {
            "objective_names": list(self.search.objective_names),
            "lambda_weights": list(self.lambda_weights),
            "normalization_scales": list(self.search.scalarizer.scales),
            "solutions": self.pareto_archive.sorted_items(),
        }
        with Path(path).open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Generate a multicore plan with ant-colony search")
    parser.add_argument("graph", type=Path)
    parser.add_argument("-n", "--num-cores", type=int, required=True)
    parser.add_argument("--ants", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scene", choices=["A", "B", "C"], default="B")
    parser.add_argument("--groups", type=int)
    parser.add_argument("--local-rounds", type=int, default=2)
    parser.add_argument(
        "--lambdas",
        help="five non-negative Chebyshev weights: makespan,cross,cut,imbalance,cache",
    )
    parser.add_argument("--rho", type=float, default=0.05,
                        help="small augmented-Chebyshev tie-break coefficient")
    parser.add_argument("--alpha", type=float, default=1.1,
                        help="pheromone exponent")
    parser.add_argument("--beta", type=float, default=2.0,
                        help="heuristic exponent")
    parser.add_argument("--evaporation", type=float, default=0.18,
                        help="pheromone evaporation rate")
    parser.add_argument("--pareto-output", type=Path,
                        help="optional JSON output for the non-dominated archive")
    parser.add_argument("-o", "--output", type=Path, required=True)
    args = parser.parse_args()
    if args.num_cores < 1 or args.ants < 1 or args.iterations < 1 or args.local_rounds < 0:
        parser.error("num-cores, ants and iterations must be positive; local-rounds cannot be negative")
    if args.rho < 0 or args.alpha < 0 or args.beta < 0 or not 0 <= args.evaporation < 1:
        parser.error("rho, alpha and beta must be non-negative; evaporation must be in [0, 1)")
    lambda_weights = None
    if args.lambdas:
        try:
            lambda_weights = tuple(float(value) for value in args.lambdas.split(","))
        except ValueError:
            parser.error("--lambdas must be comma-separated numbers")
        if len(lambda_weights) != 5 or any(value < 0 for value in lambda_weights):
            parser.error("--lambdas requires five non-negative values")
    with args.graph.open(encoding="utf-8") as handle:
        graph = json.load(handle)
    colony = AntColony(graph, args.num_cores, args.ants, args.iterations,
                        args.seed, args.scene, args.groups, args.local_rounds,
                        lambda_weights=lambda_weights,
                        tchebycheff_rho=args.rho, alpha=args.alpha,
                        beta=args.beta, evaporation=args.evaporation)
    plan, score = colony.run()
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(plan, handle, indent=2, sort_keys=True)
    if args.pareto_output:
        colony.write_pareto_archive(args.pareto_output)
    print(json.dumps({"output": str(args.output), "cores": args.num_cores,
                      "ants": args.ants, "iterations": args.iterations,
                      "seed": args.seed, "subgraphs": len(set(plan["node_to_subgraph"].values())),
                      "proxy_score": score,
                      "scalarization": "weighted_tchebycheff",
                      "pareto_solutions": len(colony.pareto_archive.items)},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
