"""官方评估：内容指纹复用、独立尝试、三组对照和统一清单。"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from official_protocol import evaluate_job, read_ledger, verified_record, ATTACHMENT, SCENES
from run_all import append_run_row, parse_cases

ROOT = Path(__file__).resolve().parents[1]


def build_jobs(cases, cores, problems, three_way=False, include_singlecore=True):
    jobs = []
    for case in cases:
        if include_singlecore:
            jobs.append({'case': case, 'kind': 'singlecore', 'N': 1, 'plan_scene': None})
        for n in cores:
            for kind in problems:
                scenes = ['B', 'C'] if three_way and kind == 'problem_3' else [SCENES[kind]]
                jobs += [{'case': case, 'kind': kind, 'N': n, 'plan_scene': scene} for scene in scenes]
    return jobs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cases', default='1-100')
    parser.add_argument('--cores', default='2,3,4,5')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--skip-singlecore', action='store_true')
    parser.add_argument('--problems', default='1,2,3')
    parser.add_argument('--three-way', action='store_true', help='问题三额外评估同一 B 方案，分离硬件与算法收益')
    parser.add_argument('--plans-dir', default=str(ROOT/'results/plans'))
    parser.add_argument('--output-dir', default=str(ROOT/'results/official_verified'))
    parser.add_argument('--timeout', type=int, default=3600)
    parser.add_argument('--max-tasks', type=int, default=0)
    args = parser.parse_args()
    cores = list(dict.fromkeys(map(int, args.cores.split(','))))
    problems = list(dict.fromkeys('problem_'+p for p in args.problems.split(',')))
    if (any(n not in (2,3,4,5) for n in cores) or any(p not in SCENES or p == 'singlecore' for p in problems)
            or args.workers < 1 or args.timeout < 1 or args.max_tasks < 0):
        parser.error('核数、问题编号、并发数、超时或分批上限非法')
    jobs = build_jobs(parse_cases(args.cases), cores, problems, args.three_way, not args.skip_singlecore)
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    ledger = out/'results.jsonl'
    # 必须记录预期集合，汇总才能识别尚未执行的任务；参数变化使用新目录。
    from run_all import ensure_run_manifest
    ensure_run_manifest(ledger, [Path(__file__).resolve(), Path(__file__).with_name('official_protocol.py')],
                        {'jobs': jobs, 'plans_dir': str(Path(args.plans_dir).resolve()), 'timeout': args.timeout})
    previous, warnings = read_ledger(ledger)
    known = {r['record_id'] for r in previous.values()}
    # 工作进程重新计算输入内容指纹后才判断可复用；不能仅凭旧清单跳过。
    if args.max_tasks:
        from official_protocol import input_context, object_digest, job_key
        pending = []
        for job in jobs:
            old = previous.get(job_key(job), {})
            try:
                current = object_digest(input_context(job, args.plans_dir, ATTACHMENT)[0])
                done = old.get('fingerprint') == current and verified_record(old)
            except (OSError, ValueError, KeyError):
                done = False
            if not done:
                pending.append(job)
        jobs = pending[:args.max_tasks]
    print(f'[0/{len(jobs)}] 官方核验；并发 {args.workers}；损坏日志行 {warnings}', flush=True)
    failures, executed, reused = 0, 0, 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(evaluate_job, job, str(out), str(Path(args.plans_dir).resolve()),
                               timeout=args.timeout): job for job in jobs}
        for i, future in enumerate(as_completed(futures), 1):
            row = future.result()
            if row['record_id'] not in known:
                append_run_row(ledger, row)
                known.add(row['record_id'])
                executed += 1
            else:
                reused += 1
            failures += row['status'] != 'official_success'
            print(f"[{i}/{len(jobs)}] {row['job_id']}：{row['status']}；新增 {executed}，复用 {reused}", flush=True)
            if row['status'] != 'official_success':
                print(row.get('error'), flush=True)
    from aggregate import write_reports
    write_reports(ledger, out)
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
