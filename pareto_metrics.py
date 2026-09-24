#!/usr/bin/env python3
"""Evaluate a saved ACO Pareto archive with the pymoo HV indicator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def compute_hypervolume(archive, reference=None):
    try:
        import numpy as np
        from pymoo.indicators.hv import HV
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(
            "pymoo is required; install it with `python -m pip install pymoo`"
        ) from exc
    scales = np.asarray(archive["normalization_scales"], dtype=float)
    points = np.asarray([
        item["objectives"] for item in archive.get("solutions", [])
    ], dtype=float)
    if points.size == 0:
        return 0.0, [], None
    points = points / scales
    weights = np.asarray(archive.get("lambda_weights", [1.0] * points.shape[1]))
    # Do not let an intentionally inactive objective (for example cache
    # pressure in scenes A/B) collapse the HV to an almost-zero number.
    active = (weights > 0) & ((points.max(axis=0) - points.min(axis=0)) > 1e-12)
    if not active.any():
        active = np.ones(points.shape[1], dtype=bool)
    points = points[:, active]
    if reference is None:
        reference = points.max(axis=0) * 1.10 + 1e-9
    else:
        reference = np.asarray(reference, dtype=float)[active]
    # pymoo's one-line HV call for a minimization Pareto front.
    value = float(HV(ref_point=reference)(points))
    return value, points.tolist(), reference.tolist()


def main():
    parser = argparse.ArgumentParser(description="Compute Pareto HV with pymoo")
    parser.add_argument("archive", type=Path)
    parser.add_argument("--reference", help="comma-separated normalized reference point")
    args = parser.parse_args()
    archive = json.loads(args.archive.read_text(encoding="utf-8"))
    reference = None
    if args.reference:
        reference = [float(value) for value in args.reference.split(",")]
    value, points, used_reference = compute_hypervolume(archive, reference)
    print(json.dumps({
        "hypervolume": value,
        "points": len(points),
        "reference_point": used_reference,
        "objective_names": archive.get("objective_names", []),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
