"""等待冻结问题一结束，依次执行问题二、问题三和严格三组对照。"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

from official_protocol import (ATTACHMENT, read_ledger, verified_record, input_context,
                               object_digest, digest)
from run_all import ensure_run_manifest, append_run_row, parse_cases
from run_saved_refine import write_json


def complete_summary(folder, expected):
    path = Path(folder)/'official/summary.json'
    try:
        summary = json.loads(path.read_text(encoding='utf-8'))
        ledger = Path(folder)/'official/results.jsonl'
        if summary['source_sha256'] != digest(ledger):
            return None
        if summary['expected'] != expected or summary['official_success'] != expected:
            return None
        if summary['pending'] or summary['failed_or_invalid']:
            return None
        records, warnings = read_ledger(ledger)
        if warnings or len(records) != expected or not all(verified_record(r) for r in records.values()):
            return None
        return summary
    except (OSError, ValueError, KeyError, TypeError):
        return None


def record_stage(project, label, directory):
    from analyze_saved_refine import analyze
    analysis = analyze(directory)
    now = datetime.now().strftime('%Y-%m-%d %H:%M')
    summary = json.loads((Path(directory)/'official/summary.json').read_text(encoding='utf-8'))
    text = f'\n\n## 串行全量阶段结果：{label}（{now}）\n\n'
    text += f'证据目录：`{directory}`。官方有效{summary["official_success"]}项，待评{summary["pending"]}项。所有数字来自内容指纹核验后的同一官方清单。\n\n'
    text += '| 核数 | 官方案例数 | 平均加速比 |\n|---|---:|---:|\n'
    for group in summary['groups']:
        text += f'| {group["N"]} | {group["comparable_count"]} | {group["mean_speedup"]:.6f} |\n'
    text += '\n父方案配对：\n\n|核数|改善|退化|平均时间改善|平均新增溢出字节|平均任务秒数|\n|---|---:|---:|---:|---:|---:|\n'
    for g in analysis['groups']:
        text += f'|{g["N"]}|{g["wins"]}|{g["losses"]}|{100*g["mean_reduction_fraction"]:.4f}%|{g["spill_added_delta"]:.2f}|{g["mean_seconds"]:.2f}|\n'
    text += '\n本阶段为固定保存方案精化；保底来源和搜索新增收益见逐任务记录。没有采集显存或进程全程峰值，不宣称显卡加速。问题二/三大图未运行新搜索时明确记录，官方评定仍覆盖全部100例。完整候选池召回、大图多候选校准和全事件显卡评分仍未完成；不得因全量结束就标记这些模块完成。\n'
    marker = f'证据目录：`{directory}`。'
    for name in ('实验计划_问题一.md', '实验记录_主分支_main-online-shared-v1.0.md'):
        path = Path(project)/name
        old = path.read_text(encoding='utf-8')
        if marker in old:
            continue
        lines = old.splitlines()
        for i, line in enumerate(lines[:18]):
            if line.startswith('更新时间：'):
                lines[i] = '更新时间：'+now
        path.write_text('\n'.join(lines)+text, encoding='utf-8')
    print(text, flush=True)


def run(args):
    project, out = Path(args.project).resolve(), Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    solver = Path(__file__).parent
    gate = json.loads(Path(args.gate).read_text(encoding='utf-8'))
    if not gate.get('accepted_for_full_validation'):
        raise ValueError('代表验收未通过，禁止进入全量')
    for name, sha in gate['algorithm_hashes'].items():
        if digest(solver/name) != sha:
            raise ValueError('待运行算法与验收版本不一致: '+name)
    def status(stage, **extra):
        write_json(out/'serial_status.json', {'stage': stage, 'updated_at': datetime.now().isoformat(timespec='minutes'), **extra})
    def launch(arguments):
        print('执行：'+subprocess.list2cmdline([sys.executable, '-X', 'utf8', *arguments]), flush=True)
        subprocess.run([sys.executable, '-X', 'utf8', *arguments], cwd=solver.parent, check=True)
    status('等待问题一官方全量完成')
    while complete_summary(args.p1_dir, 500) is None:
        time.sleep(20)
    record_stage(project, '问题一最佳保存方案加末端精化', args.p1_dir)
    history = str(project/'results/abc_full_official_r04_p23_full/results.jsonl')
    b_dir, c_dir = out/'problem_2', out/'problem_3'
    shared = ['--cases', '1-100', '--cores', '2,3,4,5', '--workers', args.workers,
              '--refine-op-limit', '12000', '--reuse-ledger', history]
    try:
        status('问题二全量运行中')
        launch([str(solver/'run_saved_refine.py'), '--scene', 'B', *shared,
                '--input-dir', str(project/'results/p23_r02_1_full_plans'),
                '--seed-dir', str(project/'results/abc_full_plans_r04'), '--output-dir', str(b_dir)])
        if complete_summary(b_dir, 500) is None:
            raise ValueError('问题二官方全量不完整，停止启动问题三')
        record_stage(project, '问题二完整原算子精化', b_dir)
        status('问题三全量运行中')
        launch([str(solver/'run_saved_refine.py'), '--scene', 'C', *shared,
                '--input-dir', str(project/'results/abc_full_plans_r04'),
                '--seed-dir', str(b_dir/'plans'), '--output-dir', str(c_dir)])
        if complete_summary(c_dir, 500) is None:
            raise ValueError('问题三官方全量不完整，停止三组归因')
        record_stage(project, '问题三独立候选加B方案保底', c_dir)
        status('问题三三组对照运行中')
        plans, triad = out/'three_way_plans', out/'three_way'
        plans.mkdir(parents=True, exist_ok=True)
        for folder in (b_dir/'plans', c_dir/'plans'):
            for path in folder.glob('*.json'):
                shutil.copyfile(path, plans/path.name)
        import evaluate_official, official_protocol
        jobs = evaluate_official.build_jobs(parse_cases('1-100'), [2,3,4,5], ['problem_2','problem_3'], True)
        ledger = triad/'results.jsonl'
        ensure_run_manifest(ledger, [Path(evaluate_official.__file__), Path(official_protocol.__file__)],
                            {'jobs':jobs, 'plans_dir':str(plans), 'timeout':3600})
        index = {}
        for path in (Path(history), b_dir/'official/results.jsonl', c_dir/'official/results.jsonl', ledger):
            records, warnings = read_ledger(path)
            if warnings:
                raise ValueError('官方清单损坏')
            index.update({r['fingerprint']: r for r in records.values() if r.get('status') == 'official_success'})
        previous, _ = read_ledger(ledger)
        known = {r['record_id'] for r in previous.values()}
        for job in jobs:
            identity = object_digest(input_context(job, plans, ATTACHMENT)[0])
            row = index.get(identity)
            if row and row['record_id'] not in known and verified_record(row):
                append_run_row(ledger, row); known.add(row['record_id'])
        launch([str(solver/'evaluate_official.py'), '--cases','1-100','--cores','2,3,4,5',
                '--problems','2,3','--three-way','--workers',args.workers,'--max-tasks','1300',
                '--plans-dir',str(plans),'--output-dir',str(triad)])
        result = json.loads((triad/'summary.json').read_text(encoding='utf-8'))
        if result['official_success'] != 1300 or len(result['three_way']) != 400:
            raise ValueError('三组对照未达到400组')
        import statistics
        now = datetime.now().strftime('%Y-%m-%d %H:%M')
        note = f'\n\n## 问题三三组官方对照完成（{now}）\n\n证据：`{triad}`。1300份有效官方结果，组成400组同图、同核数的B/B、B/C、C/C对照，外加100个单核分母。\n\n'
        note += '| 核数 | 组数 | 平均硬件加速比 | 平均算法加速比 |\n|---|---:|---:|---:|\n'
        for n in (2,3,4,5):
            pairs = [r for r in result['three_way'] if r['N'] == n]
            note += f'| {n} | {len(pairs)} | {statistics.mean(r["hardware_speedup"] for r in pairs):.6f} | {statistics.mean(r["algorithm_speedup"] for r in pairs):.6f} |\n'
        note += '\n算法比值包含B方案保底选择；C搜索本身的新增收益必须另看逐任务seed_best_official与final_official，不能把保底换方案的收益都归给缓存新算法。下一步按退化/持平案例、溢出、代理排序区分瓶颈；临时协作组、树搜索、大图校准和完整GPU事件评分不因本轮结束而自动完成。\n'
        for name in ('实验计划_问题一.md','实验记录_主分支_main-online-shared-v1.0.md'):
            path = project/name
            if f'证据：`{triad}`。' not in path.read_text(encoding='utf-8'):
                with path.open('a',encoding='utf-8') as handle:
                    handle.write(note)
        print(note, flush=True)
        status('三问官方全量及三组对照完成', three_way_count=400)
    except Exception as exc:
        status('失败，保留检查点等待修复', error=repr(exc))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', required=True)
    parser.add_argument('--p1-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--workers', default='auto')
    parser.add_argument('--gate', required=True)
    args = parser.parse_args()
    folder = Path(args.output_dir); folder.mkdir(parents=True, exist_ok=True)
    with (folder/'serial.lock').open('a+b') as lock:
        import msvcrt
        if lock.tell() == 0:
            lock.write(b'0'); lock.flush()
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        run(args)
