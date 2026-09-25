"""Small, reproducible Problem 3 probe.

Evaluates existing Problem 2 (B) plans and operation-list candidates under the
official Problem 3 evaluator. It deliberately does not modify the 800-task
archive or claim a full-run result.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "通用神经网络处理器下的多核调度问题附件" / "data"
PLAN_DIR = ROOT / "results" / "plans_oplist_full800"
sys.path.insert(0, str(HERE))

from model import load_graph
from op_list_candidates import generate_op_candidates
from pipeline import real_evaluate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="case_001,case_002,case_003,case_004,case_005")
    ap.add_argument("--cores", type=int, default=5)
    ap.add_argument("--candidate-count", type=int, default=6)
    ap.add_argument("--output", default=str(ROOT / "results" / "problem3_probe"))
    args = ap.parse_args()
    cases = [x.strip() for x in args.cases.split(",") if x.strip()]
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in cases:
        graph = load_graph(str(DATA / (case + ".json")))
        base_path = PLAN_DIR / f"{case}_B_N{args.cores}.json"
        if not base_path.is_file():
            raise FileNotFoundError(base_path)
        base = json.loads(base_path.read_text(encoding="utf-8"))
        candidates = [("B_warm_start", base)]
        generated, _stats = generate_op_candidates(
            graph, base, num_cores=args.cores, max_candidates=args.candidate_count
        )
        candidates.extend((item["source"], item["plan"]) for item in generated)
        seen = set()
        for source, plan in candidates:
            signature = json.dumps(plan, sort_keys=True, separators=(",", ":"))
            if signature in seen:
                continue
            seen.add(signature)
            result = real_evaluate(graph, plan, "C", verify_cap=12000)
            row = {
                "case": case,
                "cores": args.cores,
                "source": source,
                "makespan": result.get("makespan") if result else None,
                "added_copy_bytes": result.get("added_copy_bytes") if result else None,
                "cache_hit_rate": result.get("cache_hit_rate") if result else None,
                "error": result.get("error") if isinstance(result, dict) else None,
            }
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False))
    json_path = out_dir / "summary.json"
    csv_path = out_dir / "summary.csv"
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else
                                ["case", "cores", "source", "makespan", "added_copy_bytes", "cache_hit_rate", "error"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved {json_path}")
    print(f"saved {csv_path}")


if __name__ == "__main__":
    main()
