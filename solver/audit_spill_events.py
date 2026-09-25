"""固定方案逐子图对照官方溢出字节，逐条落盘，可中断后续跑。"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

from spill_events import count_spill_events

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / '通用神经网络处理器下的多核调度问题附件' / 'code'
sys.path.insert(0, str(CODE))
import multicore_cut_evaluate_problem_1 as official


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cases', default='44,67')
    parser.add_argument('--plans', default=str(ROOT / 'results/plans_case044_067_instrumented_v2'))
    parser.add_argument('--output', default=str(ROOT / 'results/spill_events_audit_v1.jsonl'))
    args = parser.parse_args()
    fingerprint = hashlib.sha256(''.join(digest(p) for p in (
        Path(__file__).resolve(), Path(__file__).with_name('spill_events.py'),
        CODE / 'schedule_step2.py', CODE / 'multicore_cut_evaluate_problem_1.py')).encode()).hexdigest()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(s) for s in out.read_text(encoding='utf-8').splitlines()] if out.exists() else []
    if any(row['source_hash'] != fingerprint for row in rows):
        raise ValueError('代码指纹变化，请用新的输出文件，不能混入旧断点')
    done = {(r['case'], r['N']): r for r in rows}
    tasks = [(f'case_{int(c):03}', n) for c in args.cases.split(',') for n in (2, 3, 4, 5)]
    original_step2 = official.step2_spill_insertion
    for index, (case, n) in enumerate(tasks, 1):
        pp = Path(args.plans) / f'{case}_A_N{n}.json'
        gp = ROOT / '通用神经网络处理器下的多核调度问题附件/data' / f'{case}.json'
        if (case, n) in done:
            if done[case, n]['plan_sha256'] != digest(pp) or done[case, n]['graph_sha256'] != digest(gp):
                raise ValueError('输入方案或图变化，不能使用旧断点')
            print(f'[{index}/{len(tasks)}] {case}/N{n} 已完成，跳过', flush=True)
            continue
        graph, plan = json.loads(gp.read_text(encoding='utf-8')), json.loads(pp.read_text(encoding='utf-8'))
        task_rows = []
        def checked_step2(task_graph, sequence, capacity):
            start = time.perf_counter()
            estimate = count_spill_events(task_graph, sequence, capacity)
            count_seconds = time.perf_counter() - start
            start = time.perf_counter()
            result = original_step2(task_graph, sequence, capacity)
            truth_seconds = time.perf_counter() - start
            truth = sum(s['size'] * (1 + int(s['spill_out_copies_data'])) for s in result['spill_records'])
            task_rows.append({'ops': len(task_graph['ops']), 'prediction': estimate,
                              'official_spill_bytes': truth,
                              'official_evictions': len(result['spill_records']),
                              'count_seconds': count_seconds, 'official_step2_seconds': truth_seconds})
            return result
        start = time.perf_counter()
        official.step2_spill_insertion = checked_step2
        try:
            result = official.evaluate_scene_a(graph, plan, 60.0, {'L1': 524288, 'UB': 131072}, 1000, 100)
        finally:
            official.step2_spill_insertion = original_step2
        row = {'case': case, 'N': n, 'scene': 'A', 'source_hash': fingerprint,
               'plan_sha256': digest(pp), 'graph_sha256': digest(gp),
               'official_makespan': result['makespan'],
               'official_spill_bytes': result['data_movement_bytes']['spill_added_copy_bytes'],
               'event_spill_bytes': sum(r['prediction']['spill_bytes'] for r in task_rows),
               'mismatched_tasks': sum(r['prediction']['spill_bytes'] != r['official_spill_bytes'] for r in task_rows),
               'task_rows': task_rows, 'elapsed_seconds': time.perf_counter() - start}
        with out.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
            stream.flush()
        rows.append(row)
        print(f'[{index}/{len(tasks)}] {case}/N{n}: 不一致子图={row["mismatched_tasks"]}, '
              f'事件字节={row["event_spill_bytes"]}, 官方字节={row["official_spill_bytes"]}', flush=True)
    positive = [r for r in rows if r['official_spill_bytes'] > 0]
    report = {'scope': '场景A；固定方案与官方同一子图、同一算子访问顺序的字节核验，尚未验证搜索性能',
              'rows': len(rows), 'mismatched_tasks': sum(r['mismatched_tasks'] for r in rows),
              'mae_bytes': statistics.mean(abs(r['event_spill_bytes'] - r['official_spill_bytes']) for r in rows),
              'positive_mape_ratio': statistics.mean(abs(r['event_spill_bytes']/r['official_spill_bytes'] - 1) for r in positive) if positive else None,
              'source_hash': fingerprint,
              'count_seconds': sum(t['count_seconds'] for r in rows for t in r['task_rows']),
              'official_step2_seconds': sum(t['official_step2_seconds'] for r in rows for t in r['task_rows'])}
    out.with_suffix('.summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
