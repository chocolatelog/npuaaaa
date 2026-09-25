#!/usr/bin/env python3
"""Batch runner for all contest cases."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--first", type=int, default=1)
    ap.add_argument("--last", type=int, default=100)
    ap.add_argument("--iterations", type=int, default=80)
    ap.add_argument("--cores", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20260923)
    ap.add_argument("--problem3", action="store_true")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    data_dir = root / "_数据解压" / "data"
    code_dir = root / "_数据解压" / "code"
    config = data_dir / "config.txt"
    solver = root / "mamba_alns_solver.py"
    records = []

    for case_id in range(args.first, args.last + 1):
        stem = f"case_{case_id:03d}"
        graph = data_dir / f"{stem}.json"
        plan = root / f"{stem}_multicore_res.json"
        if not graph.exists():
            records.append({"case": stem, "status": "missing"})
            continue
        cmd = [
            sys.executable, str(solver), str(graph), "-o", str(plan),
            "-n", str(args.cores), "--iterations", str(args.iterations),
            "--seed", str(args.seed + case_id), "--code-dir", str(code_dir),
            "--config", str(config), "--exact-problem", "2",
        ]
        p = subprocess.run(cmd, cwd=str(root), capture_output=True, text=True)
        summary_path = root / f"{stem}_multicore_res_summary.json"
        rec = {"case": stem, "returncode": p.returncode}
        if summary_path.exists():
            try:
                s = json.loads(summary_path.read_text(encoding="utf-8"))
                rec.update({
                    "status": "ok" if s.get("exact_makespan") is not None else "failed",
                    "makespan_problem2": s.get("exact_makespan"),
                    "subgraphs": s.get("subgraph_count"),
                    "fallback": s.get("fallback_to_greedy"),
                    "cross_core_bytes": s.get("surrogate_meta", {}).get("cross_core_bytes"),
                })
            except Exception as exc:
                rec.update({"status": "summary_error", "error": str(exc)})
        else:
            rec.update({"status": "solver_error", "stderr": p.stderr[-1000:]})
        records.append(rec)
        print(json.dumps(rec, ensure_ascii=False), flush=True)

    out = root / f"batch_summary_{args.first:03d}_{args.last:03d}.json"
    out.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    ok = [r for r in records if r.get("status") == "ok"]
    print(json.dumps({"output": str(out), "total": len(records), "ok": len(ok), "failed": len(records)-len(ok)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
