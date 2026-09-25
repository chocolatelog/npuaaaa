#!/usr/bin/env python3
"""Batch official single-core baseline evaluation."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    data = root / "_数据解压" / "data"
    code = root / "_数据解压" / "code"
    config = data / "config.txt"
    rows = []
    for i in range(1, 101):
        stem = f"case_{i:03d}"
        graph = data / f"{stem}.json"
        result = data / f"{stem}_singlecore_res.json"
        cmd = [sys.executable, str(code / "singlecore_evaluate.py"), str(graph),
               "--config", str(config), "-o", str(result)]
        p = subprocess.run(cmd, cwd=str(root), capture_output=True, text=True)
        row = {"case": stem, "returncode": p.returncode}
        if p.returncode == 0 and result.exists():
            d = json.loads(result.read_text(encoding="utf-8"))
            row.update({"status": "ok", "makespan_singlecore": d.get("makespan")})
        else:
            row.update({"status": "failed", "stderr": p.stderr[-2000:]})
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    out = root / "batch_singlecore_summary_001_100.json"
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    ok = [r for r in rows if r.get("status") == "ok"]
    print(json.dumps({"output": str(out), "total": len(rows), "ok": len(ok), "failed": len(rows)-len(ok)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
