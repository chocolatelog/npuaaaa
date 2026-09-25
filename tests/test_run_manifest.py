"""续跑必须匹配代码、输入和实验参数。"""
import sys
import json
import hashlib
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'solver'))
import run_all


def test_seed_zero_preserves_legacy_and_variants_are_distinct():
    legacy = int.from_bytes(hashlib.sha256(b'case_065|A|4').digest()[:4], 'big')
    assert run_all.stable_seed('case_065', 'A', 4, seed_base=0) == legacy
    assert len({run_all.stable_seed('case_065', 'A', 4, seed_base=i)
                for i in range(3)}) == 3


def test_manifest_creates_nested_parent_and_rejects_environment_change(tmp_path):
    log = tmp_path / 'nested' / 'run.jsonl'
    source = tmp_path / 'source.py'
    source.write_text('version = 1')
    run_all.ensure_run_manifest(log, [source], {'seed_base': 0})
    assert log.with_suffix('.manifest.json').exists()
    with pytest.raises(ValueError, match='指纹'):
        run_all.ensure_run_manifest(log, [source], {'seed_base': 1})


def test_resume_requires_saved_plan_and_bound_result(tmp_path):
    assert hasattr(run_all, 'resume_status'), '缺少方案与结果绑定的续跑判定'
    assert hasattr(run_all, 'bind_entry'), '缺少统一结果绑定'
    plan = {'node_to_subgraph': {'1': 0}, 'core_schedules': [[0], []]}
    path = tmp_path / 'case_001_A_N2.json'
    path.write_text(json.dumps(plan), encoding='utf-8')
    entry = {'case': 'case_001', 'scene': 'A', 'N': 2,
             'real': {'makespan': 100, 'added_copy_bytes': 0,
                      'scheduled_copy_bytes': 64, 'partition_added': 0, 'spill_added': 0}}
    run_all.bind_entry(entry, path)
    assert run_all.resume_status(entry, tmp_path) == 'official_success'
    entry['real']['makespan'] = 1
    assert run_all.resume_status(entry, tmp_path) == 'retry'
    entry['real']['makespan'] = 100
    path.write_text('{}', encoding='utf-8')
    assert run_all.resume_status(entry, tmp_path) == 'retry'


def test_proxy_pending_is_not_official_success_and_errors_retry(tmp_path):
    assert hasattr(run_all, 'resume_status'), '缺少区分代理与官方的续跑判定'
    path = tmp_path / 'case_001_A_N2.json'
    path.write_text(json.dumps({'node_to_subgraph': {'1': 0},
                                'core_schedules': [[0], []]}), encoding='utf-8')
    row = {'case': 'case_001', 'scene': 'A', 'N': 2, 'real': None, 'n_ops': 13000}
    run_all.bind_entry(row, path)
    assert run_all.resume_status(row, tmp_path) == 'official_pending'
    row['real'] = {'error': 'timeout'}
    run_all.bind_entry(row, path)
    assert run_all.resume_status(row, tmp_path) == 'retry'


def test_missing_official_on_small_graph_must_retry(tmp_path):
    path = tmp_path / 'case_001_A_N2.json'
    path.write_text(json.dumps({'node_to_subgraph': {'1': 0},
                                'core_schedules': [[0], []]}), encoding='utf-8')
    row = {'case': 'case_001', 'scene': 'A', 'N': 2, 'real': None, 'n_ops': 10}
    run_all.bind_entry(row, path)
    assert run_all.resume_status(row, tmp_path) == 'retry'


def test_retry_log_keeps_latest_attempt_and_reports_partial_line(tmp_path):
    assert hasattr(run_all, 'read_run_rows'), '缺少续跑日志恢复'
    path = tmp_path / 'runs.jsonl'
    row = {'case': 'case_001', 'scene': 'A', 'N': 2, 'real': None}
    path.write_text(json.dumps(row) + '\n' + json.dumps({**row, 'attempt': 2}) + '\n{"case":', encoding='utf-8')
    rows, warnings = run_all.read_run_rows(path)
    assert rows[('case_001', 'A', 2)]['attempt'] == 2
    assert warnings and warnings[0]['line'] == 3


def test_resume_rejects_changed_code_and_unlabelled_log(tmp_path):
    log = tmp_path / 'run.jsonl'
    source = tmp_path / 'source.py'
    source.write_text('version = 1')
    run_all.ensure_run_manifest(log, [source], {'event_rerank': True})
    log.write_text('{}\n')
    run_all.ensure_run_manifest(log, [source], {'event_rerank': True})
    source.write_text('version = 2')
    with pytest.raises(ValueError, match='指纹'):
        run_all.ensure_run_manifest(log, [source], {'event_rerank': True})
    legacy = tmp_path / 'legacy.jsonl'
    legacy.write_text('{}\n')
    with pytest.raises(ValueError, match='缺少'):
        run_all.ensure_run_manifest(legacy, [source], {})
