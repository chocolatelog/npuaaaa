#!/usr/bin/env python3
"""Batch problem-3 FIFO Cache evaluation for generated plans."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    data_dir = root / "_数据解压" / "data"
    code_dir = root / "_数据解压" / "code"
    config = data_dir / "config.txt"
    records = []
    for i in range(1, 101):
        stem = f"case_{i:03d}"
        graph = data_dir / f"{stem}.json"
        plan = root / f"{stem}_multicore_res.json"
        cmd = [sys.executable, str(code_dir / "multicore_cut_evaluate_problem_3.py"),
               str(graph), str(plan), "--config", str(config)]
        p = subprocess.run(cmd, cwd=str(root), capture_output=True, text=True)
        result_path = data_dir / f"{stem}_problem_3_res.json"
        rec = {"case": stem, "returncode": p.returncode}
        if p.returncode == 0 and result_path.exists():
            try:
                d = json.loads(result_path.read_text(encoding="utf-8"))
                stats = d.get("cache_stats", {})
                rec.update({
                    "status": "ok",
                    "makespan_problem3": d.get("makespan"),
                    "cache_hits": stats.get("hits"),
                    "cache_accesses": stats.get("accesses"),
                    "cache_hit_bytes": stats.get("hit_bytes"),
                    "cache_hit_rate": stats.get("hit_rate"),
                })
            except Exception as exc:
                rec.update({"status": "result_parse_error", "error": str(exc)})
        else:
            rec.update({"status": "failed", "stderr": p.stderr[-2000:], "stdout": p.stdout[-1000:]})
        records.append(rec)
        print(json.dumps(rec, ensure_ascii=False), flush=True)
    out = root / "batch_problem3_summary_001_100.json"
    out.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    ok = [r for r in records if r.get("status") == "ok"]
    print(json.dumps({"output": str(out), "total": len(records), "ok": len(ok), "failed": len(records)-len(ok)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
