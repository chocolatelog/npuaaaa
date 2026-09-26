"""固定已精评结构候选池，检验共享资源重放；绝不重复调用原官方。"""
import argparse
import json
from pathlib import Path
import statistics
import time

from official_protocol import ATTACHMENT, digest, object_digest, verified_record, input_context
from run_all import ensure_run_manifest, append_run_row
from run_saved_refine import write_json
from scene_a_candidate_score import SceneACandidateScorer
from runtime_resources import initialize_worker_threads


def assert_consistent(row):
    if not row['consistent'] or row['replay'] != row['official']:
        raise ValueError('重放与原官方不一致，停止候选排序接入')


def run(args):
    initialize_worker_threads()
    out=Path(args.output_dir).resolve();out.mkdir(parents=True,exist_ok=True)
    cases=args.cases.split(',');jobs=[];files=[]
    for directory in args.source_dir:
        for p in sorted((Path(directory)/'checkpoints').glob('*.json')):
            checkpoint=json.loads(p.read_text(encoding='utf-8'))
            if checkpoint['case'] not in cases:continue
            files.append(p)
            paths={c['plan_id']:Path(c['plan_path']) for m in checkpoint['models'] for c in m['candidates']}
            for c in checkpoint['candidates']:
                record=c['official']
                if not verified_record(record):raise ValueError('原官方凭据失效，不能作参照')
                plan=paths[c['plan_id']]
                if digest(plan)!=record['plan_sha256']:raise ValueError('候选方案与原官方不匹配')
                job=dict(case=checkpoint['case'],N=checkpoint['N'],kind='problem_1',plan_scene='A')
                if object_digest(input_context(job,plan.parent,ATTACHMENT)[0])!=record['fingerprint']:
                    raise ValueError('固定池官方上下文与当前输入不一致')
                jobs.append(dict(case=checkpoint['case'],N=checkpoint['N'],plan_id=c['plan_id'],
                    plan=str(plan),record=record,coarse=c['feature']['coarse_score']))
                files.append(plan)
    if not jobs:raise ValueError('没有可核验候选')
    jobs=sorted({(j['case'],j['N'],j['plan_id']):j for j in jobs}.values(),key=lambda j:(j['case'],j['plan_id']))
    config=ATTACHMENT/'data/config.txt'
    files+=list(Path(__file__).parent.glob('*.py'))+list((ATTACHMENT/'code').glob('*.py'))+[config]
    files+=[ATTACHMENT/'data'/f'{case}.json' for case in sorted({j['case'] for j in jobs})]
    fp=ensure_run_manifest(out/'results.jsonl',files,dict(cases=cases,jobs=[(j['case'],j['N'],j['plan_id']) for j in jobs]))
    rows=[];scorer=None;current=None
    for i,j in enumerate(jobs,1):
        path=out/'checkpoints'/f"{j['case']}_N{j['N']}_{j['plan_id']}.json"
        if path.exists():
            old=json.loads(path.read_text(encoding='utf-8'))
            if old['fingerprint']==fp and old['plan_sha256']==digest(j['plan']):
                assert_consistent(old)
                rows.append(old);print(f'[{i}/{len(jobs)}] 恢复已核验候选',flush=True);continue
        if current!=j['case']:
            current=j['case']
            graph=json.loads((ATTACHMENT/'data'/f'{current}.json').read_text(encoding='utf-8'))
            scorer=SceneACandidateScorer(graph,config)
        plan=json.loads(Path(j['plan']).read_text(encoding='utf-8'));start=time.perf_counter()
        result=scorer.evaluate(plan);truth=j['record']['real']
        row=dict(case=j['case'],N=j['N'],plan_id=j['plan_id'],fingerprint=fp,
            plan_sha256=digest(j['plan']),record_id=j['record']['record_id'],
            coarse=j['coarse'],official=truth,replay=result['real'],consistent=result['real']==truth,
            elapsed=time.perf_counter()-start,diagnostics=result,
            absolute_coarse_error=abs(j['coarse']/truth['makespan']-1))
        write_json(path,row);append_run_row(out/'results.jsonl',row);rows.append(row)
        print(f"[{i}/{len(jobs)}] {j['case']} 重放一致={row['consistent']}；{row['elapsed']:.2f}秒",flush=True)
        assert_consistent(row)
    write_json(out/'summary.json',dict(count=len(rows),consistent=sum(r['consistent'] for r in rows),
        coarse_mean_absolute_error=statistics.mean(r['absolute_coarse_error'] for r in rows),
        replay_mean_seconds=statistics.mean(r['elapsed'] for r in rows),
        official_new_calls=0,scope='仅已送原官方候选的固定池，不代表未评完整池召回'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-dir',action='append',required=True)
    p.add_argument('--cases',default='case_003,case_048,case_087')
    p.add_argument('--output-dir',required=True)
    args=p.parse_args()
    from online_shared import process_lock
    with process_lock(Path(args.output_dir)/'run.lock'):
        run(args)
