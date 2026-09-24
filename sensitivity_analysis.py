#!/usr/bin/env python3
"""One-factor-at-a-time sensitivity analysis for the ACO parameters."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aco_scheduler import AntColony
from greedy_sa_scheduler import DEFAULT_LAMBDA_WEIGHTS
from pareto_metrics import compute_hypervolume


def _run(graph, cores, scene, seed, ants, iterations, local_rounds,
         lambdas, alpha, beta, evaporation):
    colony = AntColony(
        graph, cores, ants, iterations, seed, scene,
        local_rounds=local_rounds,
        lambda_weights=lambdas,
        alpha=alpha,
        beta=beta,
        evaporation=evaporation,
    )
    plan, score = colony.run()
    archive = {
        "objective_names": list(colony.search.objective_names),
        "lambda_weights": list(lambdas),
        "normalization_scales": list(colony.search.scalarizer.scales),
        "solutions": colony.pareto_archive.sorted_items(),
    }
    hv, _, _ = compute_hypervolume(archive)
    first = archive["solutions"][0] if archive["solutions"] else None
    return {
        "score": score,
        "objectives": first["objectives"] if first else None,
        "pareto_size": len(archive["solutions"]),
        "hypervolume": hv,
        "subgraphs": len(set(plan["node_to_subgraph"].values())),
    }


def main():
    parser = argparse.ArgumentParser(description="ACO parameter sensitivity analysis")
    parser.add_argument("graph", type=Path)
    parser.add_argument("-n", "--num-cores", type=int, required=True)
    parser.add_argument("--scene", choices=["A", "B", "C"], default="B")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ants", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--local-rounds", type=int, default=1)
    parser.add_argument("-o", "--output", type=Path, required=True)
    args = parser.parse_args()
    with args.graph.open(encoding="utf-8") as handle:
        graph = json.load(handle)

    base = DEFAULT_LAMBDA_WEIGHTS[args.scene]
    experiments = [("baseline", "baseline", base, 1.1, 2.0, 0.18)]
    for values in (
        (0.70, 0.15, 0.10, 0.05, 0.00),
        (0.40, 0.35, 0.15, 0.10, 0.00),
        (0.35, 0.25, 0.10, 0.05, 0.25),
    ):
        experiments.append(("lambda", str(values), values, 1.1, 2.0, 0.18))
    for value in (0.8, 1.1, 1.4):
        experiments.append(("alpha", str(value), base, value, 2.0, 0.18))
    for value in (1.5, 2.0, 2.5):
        experiments.append(("beta", str(value), base, 1.1, value, 0.18))
    for value in (0.10, 0.18, 0.30):
        experiments.append(("evaporation", str(value), base, 1.1, 2.0, value))

    rows = []
    for parameter, value, lambdas, alpha, beta, evaporation in experiments:
        if parameter == "lambda":
            current_lambdas = lambdas
        else:
            current_lambdas = base
        result = _run(
            graph, args.num_cores, args.scene, args.seed, args.ants,
            args.iterations, args.local_rounds, current_lambdas,
            alpha, beta, evaporation,
        )
        rows.append({
            "parameter": parameter,
            "value": value,
            "lambda_weights": list(current_lambdas),
            "alpha": alpha,
            "beta": beta,
            "evaporation": evaporation,
            **result,
        })
    args.output.write_text(
        json.dumps({"scene": args.scene, "rows": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output), "experiments": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
