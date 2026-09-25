"""官方结果必须绑定输入；三组对照不得混淆方案与硬件。"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'solver'))
import evaluate_official as entry


def test_three_way_jobs_keep_b_plan_for_hardware_comparison():
    assert hasattr(entry, 'build_jobs'), '缺少显式三组对照任务生成'
    jobs = entry.build_jobs(['case_001'], [2], ['problem_2', 'problem_3'], True, False)
    assert {(j['kind'], j['plan_scene']) for j in jobs} == {
        ('problem_2', 'B'), ('problem_3', 'B'), ('problem_3', 'C')}


def fixture_inputs(tmp_path):
    attach = tmp_path/'attachment'
    (attach/'data').mkdir(parents=True)
    (attach/'code').mkdir()
    (attach/'data/case_001.json').write_text('{}')
    (attach/'data/config.txt').write_text('config-v1')
    (attach/'code/evaluator.py').write_text('# version 1')
    plans = tmp_path/'plans'
    plans.mkdir()
    (plans/'case_001_C_N2.json').write_text(json.dumps({
        'node_to_subgraph': {'1': 0}, 'core_schedules': [[0], []]}))
    return attach, plans


def test_cache_tracks_content_and_preserves_previous_attempts(tmp_path):
    assert hasattr(entry, 'evaluate_job'), '缺少内容绑定的官方评估'
    attach, plans = fixture_inputs(tmp_path)
    calls = []
    def runner(cmd, **kwargs):
        from types import SimpleNamespace
        calls.append(cmd)
        for flag, content in [('-o', json.dumps({'makespan': 10,
            'data_movement_bytes': {'added_copy_bytes': 0, 'scheduled_copy_bytes': 60},
            'cache_stats': {'hit_bytes': 30, 'miss_bytes': 30}})),
            ('--trace-output', '{}'), ('--log-output', 'ok')]:
            Path(cmd[cmd.index(flag)+1]).write_text(content)
        return SimpleNamespace(returncode=0, stdout='', stderr='')
    job = {'case': 'case_001', 'kind': 'problem_3', 'N': 2, 'plan_scene': 'C'}
    out = tmp_path/'out'
    out.mkdir()
    stale = out/'case_001_problem_3_N2_res.json'
    stale.write_text('{"makespan": 1}')
    kwargs = dict(output_dir=out, plans_dir=plans, attachment=attach, runner=runner)
    first = entry.evaluate_job(job, **kwargs)
    assert first['status'] == 'official_success' and len(calls) == 1
    second = entry.evaluate_job(job, **kwargs)
    assert second == first and len(calls) == 1
    (plans/'case_001_C_N2.json').touch()
    assert entry.evaluate_job(job, **kwargs) == first and len(calls) == 1
    (attach/'data/config.txt').write_text('config-v2')
    third = entry.evaluate_job(job, **kwargs)
    assert third['fingerprint'] != first['fingerprint'] and len(calls) == 2
    assert Path(first['result_path']).exists() and stale.read_text() == '{"makespan": 1}'
    Path(third['result_path']).write_text('{"makespan": 1}')
    fourth = entry.evaluate_job(job, **kwargs)
    assert len(calls) == 3 and fourth['record_id'] != third['record_id']
    assert fourth['real']['cache_hit_rate'] == .5
    from run_all import append_run_row
    from aggregate import write_reports
    ledger = out/'results.jsonl'
    append_run_row(ledger, fourth)
    summary = write_reports(ledger, out)
    assert summary['official_success'] == 1
    import csv
    csv_rows = list(csv.DictReader((out/'summary.csv').open(encoding='utf-8-sig')))
    assert float(csv_rows[0]['makespan']) == summary['rows'][0]['makespan']
    reports = [out/p for p in ('summary.csv','summary.md','summary.json','experiment_report.md')]
    before = [p.read_bytes() for p in reports]
    write_reports(ledger, out)
    assert [p.read_bytes() for p in reports] == before
    # 最新失败不得偷偷回退到较早成功再报告为当前已完成。
    append_run_row(ledger, {**fourth, 'record_id': 'failed-attempt', 'status': 'failed'})
    summary = write_reports(ledger, out)
    assert summary['official_success'] == 0 and summary['failed_or_invalid'] == 1


def test_missing_c_plan_cannot_reuse_old_result(tmp_path):
    assert hasattr(entry, 'evaluate_job'), '缺少内容绑定的官方评估'
    attach, plans = fixture_inputs(tmp_path)
    job = {'case': 'case_002', 'kind': 'problem_3', 'N': 2, 'plan_scene': 'C'}
    result = entry.evaluate_job(job, tmp_path/'out', plans, attachment=attach)
    assert result['status'] == 'failed'


def test_report_uses_same_b_plan_to_separate_hardware_and_algorithm():
    import aggregate
    assert hasattr(aggregate, 'three_way'), '缺少同方案硬件收益归因'
    def row(kind, scene, time, digest):
        return {'case': 'case_001', 'N': 2, 'kind': kind, 'plan_scene': scene,
                'plan_sha256': digest, 'input_sha256': 'graph', 'config_sha256': 'config',
                'evaluator_sha256': 'code', 'real': {'makespan': time}}
    rows = [row('problem_2', 'B', 100, 'b'), row('problem_3', 'B', 80, 'b'),
            row('problem_3', 'C', 60, 'c')]
    result = aggregate.three_way(rows)
    assert result[0]['hardware_speedup'] == 1.25
    assert result[0]['algorithm_speedup'] == pytest.approx(4/3)
    rows[1]['plan_sha256'] = 'other-b'
    assert aggregate.three_way(rows) == []


def test_legacy_entry_cannot_start_deleting_or_overwriting():
    import improve_runs
    assert hasattr(improve_runs, 'legacy_disabled'), '旧的破坏性实验入口必须显式停用'
    with pytest.raises(SystemExit, match='run_all'):
        improve_runs.legacy_disabled()
