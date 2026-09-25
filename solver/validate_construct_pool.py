"""构造池分层验证入口，支持进度显示和 JSONL 断点续跑。"""
import argparse
import json
import os
import sys
from pathlib import Path

from tqdm import tqdm

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "通用神经网络处理器下的多核调度问题附件" / "data"
sys.path.insert(0, str(HERE))


def _load_tasks(cases, cores, scenes):
    from model import load_graph
    rows = []
    sizes = []
    for case in cases:
        graph = load_graph(str(DATA / f"{case}.json"))
        sizes.append((len(graph["ops"]), case))
    sizes.sort()
    groups = {"small": [], "medium": [], "large": []}
    for i, (_n, case) in enumerate(sizes):
        groups["small" if i < len(sizes) / 3 else "medium" if i < len(sizes) * 2 / 3 else "large"].append(case)
    for group, group_cases in groups.items():
        for case in group_cases:
            for scene in scenes:
                for core in cores:
                    rows.append((group, case, scene, core))
    return rows


def run(args):
    from model import Model, load_graph
    from solution import Context
    from construct_refine import build_construct_candidates

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
                done.add((row["case"], row["scene"], row["cores"]))
            except (ValueError, KeyError):
                continue
    cases = [f"case_{i:03d}" for i in range(args.start, args.end + 1)]
    tasks = [t for t in _load_tasks(cases, args.cores, args.scenes)
             if (t[1], t[2], t[3]) not in done]
    with out.open("a", encoding="utf-8") as fp:
        for group, case, scene, cores in tqdm(tasks, desc="构造池验证"):
            graph = load_graph(str(DATA / f"{case}.json"))
            eligible = sum(o["op"] not in ("COPY_IN", "COPY_OUT") for o in graph["ops"])
            block_cap = max(24, min(240, eligible // 400))
            model = Model(graph, block_ops_cap=block_cap)
            ctx = Context(model, cores, scene)
            items = build_construct_candidates(model, cores, scene, ctx=ctx)
            row = {"case": case, "scene": scene, "cores": cores, "group": group,
                   "n_ops": len(graph["ops"]), "n_candidates": len(items),
                   "levels": {x: sum(i.refine_level == x for i in items) for x in ("R0", "R1", "R2", "R3")},
                   "seeds": [{"paradigm": i.paradigm, "granularity": i.granularity,
                              "refine_level": i.refine_level, "makespan": i.metrics.get("makespan") if i.metrics else None,
                              "target_ops": i.target_ops, "target_strips": i.target_strips,
                              "signature": i.signature} for i in items]}
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")
            fp.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=30)
    parser.add_argument("--cores", default="4")
    parser.add_argument("--scenes", default="A,B,C")
    parser.add_argument("--output", default=str(ROOT / "results" / "construct_validation.jsonl"))
    args = parser.parse_args()
    args.cores = [int(x) for x in args.cores.split(",")]
    args.scenes = [x.strip() for x in args.scenes.split(",") if x.strip()]
    run(args)


if __name__ == "__main__":
    main()
