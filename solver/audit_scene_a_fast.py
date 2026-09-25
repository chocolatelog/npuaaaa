"""原官方与计数式副本逐字段配对；单进程交替执行顺序测量耗时。"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / '通用神经网络处理器下的多核调度问题附件/code').is_dir())
sys.path.insert(0, str(ROOT / 'solver'))
import scene_a_fast
from run_all import ensure_run_manifest, parse_cases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cases', default='1,7,15,23,42,44,49,63,67,78,82,97')
    parser.add_argument('--source-dir', default='results/plans_three_stage_v1')
    parser.add_argument('--output', default='results/fast_replica_control48_v1.jsonl')
    args = parser.parse_args()
    cases = parse_cases(args.cases)
    tasks = [(c, n) for c in cases for n in (2, 3, 4, 5)]
    source = Path(args.source_dir).resolve()
    data = ROOT / '通用神经网络处理器下的多核调度问题附件/data'
    files = [Path(__file__), Path(scene_a_fast.__file__)] + list((ROOT / 'solver').glob('*.py'))
    files += list(scene_a_fast.CODE.glob('*.py'))
    files += [data / f'{c}.json' for c in cases]
    files += [source / f'{c}_A_N{n}.json' for c, n in tasks]
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fingerprint = ensure_run_manifest(out, files, vars(args))
    rows = [json.loads(s) for s in out.read_text(encoding='utf-8').splitlines()] if out.exists() else []
    done = {(r['case'], r['N']) for r in rows if r['full_result_equal']}
    print(f'[{len(done)}/{len(tasks)}] 原官方/计数式副本完整结果核验，单进程', flush=True)
    with out.open('a', encoding='utf-8') as f:
        for index, (case, n) in enumerate(tasks):
            if (case, n) in done:
                continue
            graph = json.loads((data / f'{case}.json').read_text(encoding='utf-8'))
            plan = json.loads((source / f'{case}_A_N{n}.json').read_text(encoding='utf-8'))
            params = (graph, plan, 60, {'L1': 524288, 'UB': 131072}, 1000, 100)
            functions = [('official', scene_a_fast.official.evaluate_scene_a),
                         ('replica', scene_a_fast.evaluate_scene_a_fast)]
            if index % 2:
                functions.reverse()
            results, timing = {}, {}
            for label, function in functions:
                started = time.perf_counter()
                results[label] = function(*params)
                timing[label] = time.perf_counter() - started
            equal = results['official'] == results['replica']
            mismatch_keys = [key for key in results['official']
                             if results['official'][key] != results['replica'].get(key)]
            hashes = {key: hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
                      for key, value in results.items()}
            row = {'case': case, 'N': n, 'full_result_equal': equal,
                   'mismatch_keys': mismatch_keys, 'result_sha256': hashes,
                   'makespan': results['official']['makespan'],
                   'official_seconds': timing['official'], 'replica_seconds': timing['replica'],
                   'speed_ratio': timing['official'] / timing['replica'],
                   'first_evaluator': functions[0][0], 'fingerprint': fingerprint}
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
            f.flush()
            rows.append(row)
            print(f'[{len(rows)}/{len(tasks)}] {case}/N{n}: 完整一致={equal}，'
                  f'{timing["official"]:.3f}->{timing["replica"]:.3f} 秒', flush=True)
            if not equal:
                raise ValueError(f'完整结果不一致，停止核验：{case}/N{n}: {mismatch_keys}')
    official_seconds = sum(r['official_seconds'] for r in rows)
    replica_seconds = sum(r['replica_seconds'] for r in rows)
    summary = {'count': len(rows), 'mismatches': sum(not r['full_result_equal'] for r in rows),
               'official_seconds': official_seconds, 'replica_seconds': replica_seconds,
               'total_speed_ratio': official_seconds / replica_seconds,
               'median_speed_ratio': statistics.median(r['speed_ratio'] for r in rows),
               'scope': '固定保存方案完整事件结果比较；计数式副本不替代最终原官方复核'}
    out.with_suffix('.summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
