"""异机串行重评一/二/三；共享分母、完整三组、进度及内容指纹断点恢复。"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from validated_bundle import verify_files
from runtime_resources import initialize_worker_threads, collect_runtime_environment, resolve_workers
from official_protocol import ATTACHMENT, digest, read_ledger, input_context, object_digest, verified_record, job_key
from run_all import ensure_run_manifest, append_run_row, parse_cases
from run_saved_refine import write_json
from online_shared import process_lock
import evaluate_official
import official_protocol


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only',action='store_true',help='仅核验包内容和环境，不重复运行官方实验')
    parser.add_argument('--workers',default='auto');parser.add_argument('--output-dir',default='results/reproduction_r37')
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1];out=Path(args.output_dir).resolve()
    manifest_path=root/'bundle_manifest.json'
    bundle=json.loads(manifest_path.read_text(encoding='utf8'))
    verify_files(root,bundle['files'],lambda i,n:print(f'包内容核验[{i}/{n}]',flush=True))
    initialize_worker_threads()
    resolution=resolve_workers(args.workers)
    out.mkdir(parents=True,exist_ok=True)
    with process_lock(out/'run.lock'):
        write_json(out/'runtime.json',collect_runtime_environment(include_torch=True))
        expected={job_key(row):row for row in bundle['expected']}
        if len(expected)!=1300:raise ValueError('包内必须为1200多核+100唯一单核预期指标')
        print(f'1200份多核方案摘要已核验；推荐并发{resolution.workers}；历史指标仅作对照',flush=True)
        if args.check_only:
            write_json(out/'preflight.json',dict(bundle_sha256=digest(manifest_path),files=len(bundle['files']),plans=1200,
                       official_evaluation_started=False,scope='内容与环境预检，不是异机性能复现'))
            return
        plans=root/'plans';history={}
        for stage,kinds,three_way in [('problem_1',['problem_1'],False),('problem_2',['problem_2'],False),('problem_3',['problem_2','problem_3'],True)]:
            folder=out/stage;ledger=folder/'results.jsonl'
            jobs=evaluate_official.build_jobs(parse_cases('1-100'),[2,3,4,5],kinds,three_way)
            ensure_run_manifest(ledger,[Path(evaluate_official.__file__).resolve(),Path(official_protocol.__file__).resolve()],
                                dict(jobs=jobs,plans_dir=str(plans.resolve()),timeout=3600))
            old,warnings=read_ledger(ledger)
            if warnings:raise ValueError('重评清单损坏，停止后续阶段')
            for record in old.values():
                if verified_record(record):history[record['fingerprint']]=record
            reused=0
            for job in jobs:
                fp=object_digest(input_context(job,plans,ATTACHMENT)[0]);r=history.get(fp)
                if r and verified_record(r):
                    reused+=1
                    if old.get(job_key(job),{}).get('record_id')!=r['record_id']:append_run_row(ledger,r)
            write_json(out/'serial_status.json',dict(stage=stage,status='运行中',reused=reused,expected=len(jobs)))
            cmd=[sys.executable,'-X','utf8',str(root/'solver/evaluate_official.py'),'--cases','1-100','--cores','2,3,4,5',
                 '--problems',','.join(k[-1] for k in kinds),'--workers',str(resolution.workers),'--max-tasks',str(len(jobs)),
                 '--plans-dir',str(plans),'--output-dir',str(folder)]
            if three_way:cmd.append('--three-way')
            print(f'{stage}（问题{stage[-1]}）已复用{reused}/{len(jobs)}；只运行缺项',flush=True)
            subprocess.run(cmd,cwd=root,check=True)
            summary=json.loads((folder/'summary.json').read_text(encoding='utf8'))
            if summary['official_success']!=len(jobs) or summary['failed_or_invalid'] or summary['pending']:
                raise ValueError('本阶段未完整，停止后续阶段')
            rows,_=read_ledger(ledger);differences=[]
            for key,record in rows.items():
                history[record['fingerprint']]=record
                reference=expected.get(key)
                if reference and (record['real']!=reference['real'] or record['plan_sha256']!=reference['plan_sha256']):
                    differences.append(dict(job_id=key,expected=reference,actual=record['real']))
            write_json(folder/'comparison.json',dict(compared=sum(key in expected for key in rows),differences=differences))
            if differences:raise ValueError('本机官方结果与原档案不同；已保存差异，停止后续阶段')
            if three_way and len(summary['three_way'])!=400:raise ValueError('同来源三组未完整')
            print(json.dumps(summary['groups'],ensure_ascii=False),flush=True)
            write_json(out/'serial_status.json',dict(stage=stage,status='已完成',official_success=summary['official_success']))


if __name__=='__main__':main()
