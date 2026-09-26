"""断点恢复必须核验输入、运行身份、输出及原官方凭据。"""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'solver'))
import run_saved_refine as runner


def test_resume_invalidated_by_input_output_code_and_receipt(tmp_path, monkeypatch):
    source, output, checkpoint = [tmp_path/name for name in ('input.json', 'output.json', 'checkpoint.json')]
    source.write_text('input'); output.write_text('output')
    row = {'run_fingerprint': 'version1', 'input_plan_sha256': runner.digest(source),
           'output_plan_sha256': runner.digest(output), 'official_record': {'ok': True}}
    runner.write_json(checkpoint, row)
    monkeypatch.setattr(runner, 'verified_record', lambda r: r['ok'])
    assert runner.completed(checkpoint, source, output, 'version1') == row
    assert runner.completed(checkpoint, source, output, 'version2') is None
    source.write_text('changed')
    assert runner.completed(checkpoint, source, output, 'version1') is None
    source.write_text('input'); output.write_text('changed')
    assert runner.completed(checkpoint, source, output, 'version1') is None
    output.write_text('output'); row['official_record']['ok'] = False
    runner.write_json(checkpoint, row)
    assert runner.completed(checkpoint, source, output, 'version1') is None


def test_interrupted_checkpoint_is_not_completed(tmp_path):
    checkpoint = tmp_path/'checkpoint.json'
    checkpoint.write_text('{"unfinished":')
    assert runner.completed(checkpoint, tmp_path/'in', tmp_path/'out', 'version1') is None


def test_large_graph_skips_search_but_officially_compares_seed(tmp_path, monkeypatch):
    attachment = tmp_path/'attachment'
    (attachment/'data').mkdir(parents=True)
    (attachment/'code').mkdir()
    graph = {'ops': [{'id': i, 'op': 'ADD', 'pipe': 'PIPE_V', 'cycles': 10} for i in (1, 2)],
             'edges': [], 'tensors': []}
    runner.write_json(attachment/'data/case_001.json', graph)
    (attachment/'data/config.txt').write_text('configuration fingerprint fixture')
    monkeypatch.setattr(runner, 'ATTACHMENT', attachment)
    source, seed_dir = tmp_path/'source', tmp_path/'seed'
    parent = {'node_to_subgraph': {'1': 0, '2': 1}, 'core_schedules': [[0, 1], []]}
    seed = {**parent, 'core_schedules': [[0], [1]]}
    runner.write_json(source/'case_001_B_N2.json', parent)
    runner.write_json(seed_dir/'case_001_B_N2.json', seed)
    evaluated = []
    def official(job, out, plans, **kw):
        path = plans/'case_001_B_N2.json'
        plan = json.loads(path.read_text(encoding='utf-8'))
        evaluated.append(plan)
        return {'status': 'official_success', 'record_id': str(len(evaluated)),
                'plan_sha256': runner.digest(path),
                'real': {'makespan': 80 if plan == seed else 100, 'added_copy_bytes': 0,
                         'scheduled_copy_bytes': 0, 'spill_added': 0, 'partition_added': 0}}
    monkeypatch.setattr(runner, 'evaluate_job', official)
    settings = dict(input_dir=str(source), seed_dir=str(seed_dir),
                    output_dir=str(tmp_path/'output'), timeout=10, refine_op_limit=1)
    row, reused = runner.run_task(dict(case='case_001', N=2, kind='problem_2', plan_scene='B'),
                                 settings, 'fixed', {})
    assert not reused and len(evaluated) == 2
    assert row['real']['makespan'] == 80 and row['baseline']['makespan'] == 100
    assert row['audit']['status'] == 'search_skipped_large_graph'
    assert json.loads((tmp_path/'output/plans/case_001_B_N2.json').read_text()) == seed
def test_reuse_history_accepts_final_and_full_candidate_ledgers(tmp_path):
    import json
    import run_saved_refine as runner
    records=[dict(status='official_success',fingerprint=f'fp{i}',record_id=str(i)) for i in range(3)]
    p=tmp_path/'history.jsonl'
    rows=[records[0],dict(official_record=records[1],candidate_receipts=[records[1],records[2],dict(status='failed',fingerprint='bad')]),dict(status='proxy',fingerprint='proxy')]
    p.write_text('\n'.join(json.dumps(r) for r in rows),encoding='utf8')
    history=runner.load_reuse_history([p])
    assert set(history)=={'fp0','fp1','fp2'}
    assert history['fp2']==records[2]


def test_cache_event_improvement_saves_checkpoint_and_resumes(tmp_path,monkeypatch):
    from scenario_contract import CODE
    from evaluation_validation import read_evaluation_config
    from multicore_cut_evaluate_problem_3 import evaluate_problem_3
    from official_protocol import compact_result
    config=read_evaluation_config(str(CODE.parent/'data/config.txt'))
    graph=dict(ops=[dict(id=i,op='ADD',pipe='PIPE_V',cycles=1) for i in range(3)],
        tensors=[dict(id=10,pos='UB',size=600),dict(id=20,pos='UB',size=1200)],
        edges=[dict(source=10,target=0),dict(source=10,target=1),dict(source=20,target=2)])
    parent=dict(node_to_subgraph={str(i):i for i in range(3)},core_schedules=[[0],[1,2]])
    attachment=tmp_path/'attachment';source=tmp_path/'source';output=tmp_path/'output'
    runner.write_json(attachment/'data/case_001.json',graph)
    (attachment/'data/config.txt').write_text('test configuration fingerprint')
    runner.write_json(source/'case_001_C_N2.json',parent)
    monkeypatch.setattr(runner,'ATTACHMENT',attachment)
    calls=[]
    def official(job,out,plans,**kwargs):
        path=plans/'case_001_C_N2.json';plan=json.loads(path.read_text(encoding='utf8'))
        raw=evaluate_problem_3(graph,plan,60,config['capacity'],500,10000,600)
        result=tmp_path/f'raw_{len(calls)}.json';runner.write_json(result,raw);calls.append(plan)
        return dict(status='official_success',record_id=str(len(calls)),plan_sha256=runner.digest(path),
            result_path=str(result),real=compact_result(raw))
    monkeypatch.setattr(runner,'evaluate_job',official)
    monkeypatch.setattr(runner,'verified_record',lambda record:True)
    settings=dict(input_dir=str(source),output_dir=str(output),timeout=10,
        cache_event_window=True,screen_candidates=6,stage_cache_mib=1,official_limit=3)
    job=dict(case='case_001',N=2,kind='problem_3',plan_scene='C')
    row,reused=runner.run_task(job,settings,'test-run',{})
    assert not reused and row['real']['makespan']<row['baseline']['makespan']
    assert row['input_plan_sha256']==runner.digest(source/'case_001_C_N2.json')
    count=len(calls)
    resumed,reused=runner.run_task(job,settings,'test-run',{})
    assert reused and resumed==row and len(calls)==count


def test_boundary_tail_entry_saves_real_official_selection(tmp_path,monkeypatch):
    from test_boundary_release import example
    from multicore_cut_evaluate_problem_1 import evaluate_scene_a
    from official_protocol import compact_result
    graph,parent,raw=example()
    attachment=tmp_path/'attachment';source=tmp_path/'source';output=tmp_path/'output'
    runner.write_json(attachment/'data/case_001.json',graph)
    (attachment/'data/config.txt').write_text('test configuration fingerprint')
    runner.write_json(source/'case_001_A_N2.json',parent)
    monkeypatch.setattr(runner,'ATTACHMENT',attachment)
    calls=[]
    def official(job,out,plans,**kwargs):
        path=plans/'case_001_A_N2.json';plan=json.loads(path.read_text(encoding='utf8'))
        result=evaluate_scene_a(graph,plan,60,raw['capacity_bytes'],1000,100)
        result_path=tmp_path/f'raw_{len(calls)}.json';runner.write_json(result_path,result);calls.append(plan)
        return dict(status='official_success',record_id=str(len(calls)),plan_sha256=runner.digest(path),
            result_path=str(result_path),real=compact_result(result))
    monkeypatch.setattr(runner,'evaluate_job',official)
    monkeypatch.setattr(runner,'verified_record',lambda record:True)
    settings=dict(input_dir=str(source),output_dir=str(output),timeout=10,
        boundary_tail=True,screen_candidates=12,stage_cache_mib=1,official_limit=6)
    job=dict(case='case_001',N=2,kind='problem_1',plan_scene='A')
    row,reused=runner.run_task(job,settings,'test-run',{})
    assert row['real']['makespan']<row['baseline']['makespan'] and not reused
    assert row['input_plan_sha256']==runner.digest(source/'case_001_A_N2.json')
    count=len(calls)
    resumed,reused=runner.run_task(job,settings,'test-run',{})
    assert reused and resumed==row and len(calls)==count
