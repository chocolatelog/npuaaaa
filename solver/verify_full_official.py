"""Officially evaluate plans skipped by the solver's large-case fast path.

The JSONL output is resumable and stays separate from the solver log.
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def verify_task(task):
    from model import load_graph
    from pipeline import real_evaluate
    from run_all import DATA

    case, scene, cores = task['case'], task['scene'], task['N']
    graph = load_graph(os.path.join(DATA, case + '.json'))
    plan_path = os.path.join(task['plans_dir'],
                             f'{case}_{scene}_N{cores}.json')
    with open(plan_path, encoding='utf-8') as stream:
        plan = json.load(stream)
    started = time.time()
    result = real_evaluate(graph, plan, scene, verify_cap=10 ** 9)
    return {'case': case, 'scene': scene, 'N': cores,
            'real': result, 'elapsed': round(time.time() - started, 2)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--solve-log', required=True)
    parser.add_argument('--plans-dir', required=True)
    parser.add_argument('--log-file', required=True)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()

    with open(args.solve_log, encoding='utf-8') as stream:
        solve_rows = [json.loads(line) for line in stream]
    if len(solve_rows) != 800:
        parser.error(f'expected 800 solver results; found {len(solve_rows)}')
    done = set()
    if os.path.exists(args.log_file):
        with open(args.log_file, encoding='utf-8') as stream:
            for line in stream:
                row = json.loads(line)
                done.add((row['case'], row['scene'], row['N']))
    tasks = []
    for row in solve_rows:
        key = row['case'], row['scene'], row['N']
        if row['real'] is None and key not in done:
            tasks.append({**dict(zip(('case', 'scene', 'N'), key)),
                          'plans_dir': args.plans_dir})
    print(f'{len(tasks)} large-case plans to officially verify '
          f'({len(done)} already done)', flush=True)
    with open(args.log_file, 'a', encoding='utf-8') as stream:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(verify_task, task): task for task in tasks}
            for index, future in enumerate(as_completed(futures), start=1):
                task = futures[future]
                try:
                    row = future.result()
                except Exception as exc:
                    row = {k: task[k] for k in ('case', 'scene', 'N')}
                    row['real'] = {'error': repr(exc)}
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
                stream.flush()
                real = row['real'] or {}
                print(f'[{index}/{len(tasks)}] {row["case"]} '
                      f'{row["scene"]} N={row["N"]} '
                      f'real={real.get("makespan", real.get("error"))}',
                      flush=True)


if __name__ == '__main__':
    main()
