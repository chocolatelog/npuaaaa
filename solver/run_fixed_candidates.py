"""固定候选个数重放：冻结输入清单、官方完整结果、诊断轨迹与断点。"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

from run_all import ensure_run_manifest, run_input_files, append_run_row, plan_digest

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / '通用神经网络处理器下的多核调度问题附件/data'
CODE = DATA.parent / 'code'


def load_candidates(path, limit=0):
    path = Path(path).resolve()
    rows = json.loads(path.read_text(encoding='utf-8'))
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('固定候选标识重复')
    if limit < 0:
        raise ValueError('候选数量不能为负')
    rows = rows[:limit] if limit else rows
    for row in rows:
        p = Path(row['plan_path'])
        row['plan_path'] = str(p.resolve() if p.is_absolute() else (path.parent / p).resolve())
    return rows


def compact_official(result):
    dm = result['data_movement_bytes']
    return {'makespan': result['makespan'], 'added_copy_bytes': dm['added_copy_bytes'],
            'scheduled_copy_bytes': dm['scheduled_copy_bytes'],
            'partition_added': dm.get('partition_added_copy_bytes'),
            'spill_added': dm.get('spill_added_copy_bytes')}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidates', required=True, help='固定输入清单')
    parser.add_argument('--output', required=True)
    parser.add_argument('--candidate-limit', type=int, default=0,
                        help='只评清单前几个候选，0 为全部；计入实验指纹')
    parser.add_argument('--max-tasks', type=int, default=0,
                        help='本次处理几个待评候选，0 为全部；不改变固定清单')
    args = parser.parse_args()
    if args.max_tasks < 0:
        parser.error('分批上限不能为负')
    rows = load_candidates(args.candidates, args.candidate_limit)
    out = Path(args.output).resolve()
    traces = out.parent / (out.stem + '_traces')
    traces.mkdir(parents=True, exist_ok=True)
    files = run_input_files(sorted({r['case'] for r in rows}))
    files += [Path(args.candidates).resolve()] + [Path(r['plan_path']) for r in rows]
    fingerprint = ensure_run_manifest(out, files, {
        'mode': 'fixed_candidates', 'candidates': str(Path(args.candidates).resolve()),
        'candidate_limit': args.candidate_limit, 'output': str(out),
        'order': [r['id'] for r in rows]})
    old, warnings = {}, []
    if out.exists():
        for n, line in enumerate(out.read_text(encoding='utf-8').splitlines(), 1):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
                old[r['id']] = r
            except (ValueError, KeyError):
                warnings.append(n)
    done = set()
    for key, r in old.items():
        trace = Path(r.get('trace_path', ''))
        if (r.get('status') == 'official_success' and r.get('fingerprint') == fingerprint
                and trace.is_file() and hashlib.sha256(trace.read_bytes()).hexdigest() == r.get('trace_sha256')):
            truth = compact_official(json.loads(trace.read_text(encoding='utf-8')))
            if truth == r.get('real'):
                done.add(key)
    print(f'[{len(done)}/{len(rows)}] 固定候选重放；损坏日志行 {warnings}', flush=True)
    sys.path.insert(0, str(CODE))
    from multicore_cut_evaluate_problem_1 import evaluate_scene_a
    graphs, executed = {}, 0
    for row in rows:
        if row['id'] in done:
            continue
        if args.max_tasks and executed >= args.max_tasks:
            break
        started = time.perf_counter()
        record = {'id': row['id'], 'case': row['case'], 'N': row['N'],
                  'fingerprint': fingerprint, 'attempt': old.get(row['id'], {}).get('attempt', 0)+1}
        try:
            case = row['case']
            if case not in graphs:
                graphs[case] = json.loads((DATA / f'{case}.json').read_text(encoding='utf-8'))
            graph = graphs[case]
            if len(graph['ops']) > 12000:
                raise ValueError('固定候选诊断入口不隐式启动超大图官方评估，请按 E10 单独执行')
            plan = json.loads(Path(row['plan_path']).read_text(encoding='utf-8'))
            if len(plan['core_schedules']) != row['N']:
                raise ValueError('清单核数与方案不一致')
            truth = evaluate_scene_a(graph, plan, 60, {'L1': 524288, 'UB': 131072}, 1000, 100)
            real = compact_official(truth)
            trace = traces / (hashlib.sha256(row['id'].encode()).hexdigest()[:16] + '.json')
            trace.write_text(json.dumps(truth, ensure_ascii=False), encoding='utf-8')
            match = all(real.get(k) == v for k, v in row.get('expected_real', {}).items())
            record.update(status='official_success' if match else 'mismatch', real=real,
                          expected_match=match, plan_id=plan_digest(plan),
                          plan_sha256=hashlib.sha256(Path(row['plan_path']).read_bytes()).hexdigest(),
                          trace_path=str(trace), trace_sha256=hashlib.sha256(trace.read_bytes()).hexdigest())
            if match:
                done.add(row['id'])
        except Exception as exc:
            record.update(status='failed', error=repr(exc))
        record['seconds'] = time.perf_counter() - started
        append_run_row(out, record)
        old[row['id']] = record
        executed += 1
        print(f'[{len(done)}/{len(rows)}] {row["id"]}: {record["status"]}, '
              f'耗时 {record["seconds"]:.3f} 秒', flush=True)
    result = {'count': len(rows), 'official_success': len(done), 'remaining': len(rows)-len(done),
              'executed_this_batch': executed, 'fingerprint': fingerprint,
              'scope': '固定输入、固定候选数量；不宣称完整限时搜索轨迹确定性。'}
    out.with_suffix('.summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False))
    if any(r.get('status') in ('mismatch', 'failed') for r in old.values()):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
