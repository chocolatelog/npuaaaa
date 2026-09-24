#!/usr/bin/env python3
"""Batch official evaluation for Problem 1 best-known plan selection."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from pathlib import Path


def _baseline(data_dir, case):
    path = data_dir / f"case_{case}_singlecore_log.txt"
    if not path.exists():
        return None
    match = re.search(r"makespan:\s*(\d+)", path.read_text(encoding="utf-8"))
    return int(match.group(1)) if match else None


def main():
    parser = argparse.ArgumentParser(description="Run global Problem-1 tests")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cases", nargs="+", help="case ids; default all 001..100")
    parser.add_argument("--cores", nargs="+", type=int, default=[2, 3, 4, 5])
    parser.add_argument("--tabu-iterations", type=int, default=0)
    parser.add_argument("--tabu-shortlist", type=int, default=3)
    parser.add_argument("--tabu-max-evaluations", type=int)
    parser.add_argument("--quick", action="store_true",
                        help="use only the default candidate for large graphs")
    parser.add_argument("--resume", action="store_true",
                        help="skip case/core pairs already present in results.json")
    parser.add_argument("--max-ops", type=int,
                        help="only run graphs with at most this many non-COPY ops")
    parser.add_argument("--max-seconds", type=float,
                        help="stop launching new jobs after this wall-clock budget")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases = args.cases or [f"{index:03d}" for index in range(1, 101)]
    evaluator_root = args.data_dir.parent
    results_path = args.output_dir / "results.json"
    rows = []
    completed = set()
    if args.resume and results_path.exists():
        try:
            rows = json.loads(results_path.read_text(encoding="utf-8"))
            completed = {(str(row.get("case")), int(row.get("cores")))
                         for row in rows if "cores" in row}
        except (json.JSONDecodeError, OSError, TypeError, ValueError):
            rows = []
    started_global = time.time()
    for case in cases:
        graph = args.data_dir / f"case_{case}.json"
        if not graph.exists():
            rows.append({"case": case, "ok": False, "error": "missing graph"})
            continue
        baseline = _baseline(args.data_dir, case)
        for cores in args.cores:
            if (str(case), int(cores)) in completed:
                print(json.dumps({"case": case, "cores": cores,
                                  "skipped": "resume"}, ensure_ascii=False),
                      flush=True)
                continue
            if args.max_seconds is not None and time.time() - started_global >= args.max_seconds:
                print(json.dumps({"stopped": "max-seconds", "runs": len(rows)},
                                  ensure_ascii=False), flush=True)
                results_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2),
                                        encoding="utf-8")
                return
            if args.max_ops is not None:
                try:
                    obj = json.loads(graph.read_text(encoding="utf-8"))
                    n_ops = sum(op.get("op") not in {"COPY_IN", "COPY_OUT"}
                                for op in obj.get("ops", []))
                    if n_ops > args.max_ops:
                        rows.append({"case": case, "cores": cores,
                                     "ok": False, "skipped": "max-ops",
                                     "n_ops": n_ops})
                        continue
                except (OSError, json.JSONDecodeError):
                    pass
            output = args.output_dir / f"case_{case}_k{cores}.json"
            report = args.output_dir / f"case_{case}_k{cores}_report.json"
            command = [
                "python", str(args.workspace / "problem1_select.py"), str(graph),
                "-n", str(cores), "-o", str(output), "--report", str(report),
                "--tabu-iterations", str(args.tabu_iterations),
                "--tabu-shortlist", str(args.tabu_shortlist),
            ]
            if args.tabu_max_evaluations is not None:
                command += ["--tabu-max-evaluations", str(args.tabu_max_evaluations)]
            if args.quick:
                command += ["--groups", str(2 * cores), "--orders", "critical"]
            started = time.time()
            process = subprocess.run(
                command, cwd=args.workspace, capture_output=True, text=True,
            )
            if process.returncode:
                rows.append({
                    "case": case, "cores": cores, "ok": False,
                    "error": process.stderr[-1000:],
                    "seconds": round(time.time() - started, 2),
                })
                continue
            details = json.loads(report.read_text(encoding="utf-8"))
            key = details["selected_key"]
            rows.append({
                "case": case, "cores": cores, "ok": True,
                "groups": details["selected_groups"],
                "order_mode": details["selected_order_mode"],
                "order_seed": details["selected_order_seed"],
                "makespan": key[0],
                "added_copy_bytes": key[1],
                "baseline": baseline,
                "speedup": baseline / key[0] if baseline else None,
                "candidate_count": len(details["candidates"]),
                "seconds": round(time.time() - started, 2),
            })
            print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
            (args.output_dir / "results.json").write_text(
                json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
            )
    (args.output_dir / "results.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "output": str(args.output_dir / "results.json"),
        "runs": len(rows),
        "successes": sum(row.get("ok", False) for row in rows),
        "failures": sum(not row.get("ok", False) for row in rows),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
