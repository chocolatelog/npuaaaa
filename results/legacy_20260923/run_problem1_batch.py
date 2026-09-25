#!/usr/bin/env python3
"""Batch problem-1 evaluation for generated plans."""

from __future__ import annotations

import json
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
        cmd = [sys.executable, str(code_dir / "multicore_cut_evaluate_problem_1.py"),
               str(graph), str(plan), "--config", str(config)]
        p = subprocess.run(cmd, cwd=str(root), capture_output=True, text=True)
        result_path = data_dir / f"{stem}_problem_1_res.json"
        rec = {"case": stem, "returncode": p.returncode}
        if p.returncode == 0 and result_path.exists():
            d = json.loads(result_path.read_text(encoding="utf-8"))
            rec.update({"status": "ok", "makespan_problem1": d.get("makespan")})
        else:
            rec.update({"status": "failed", "stderr": p.stderr[-2000:]})
        records.append(rec)
        print(json.dumps(rec, ensure_ascii=False), flush=True)
    out = root / "batch_problem1_summary_001_100.json"
    out.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    ok = [r for r in records if r.get("status") == "ok"]
    print(json.dumps({"output": str(out), "total": len(records), "ok": len(ok), "failed": len(records)-len(ok)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
