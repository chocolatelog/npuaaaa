"""新数学实验必须重试失败候选，且不能把不一致的重放记为通过。"""
import json
import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'solver'))
import audit_priority_candidates as priority
import audit_candidate_replay as replay


def test_failed_candidate_checkpoint_is_retryable_and_success_binds_plan(tmp_path, monkeypatch):
    plan = tmp_path / 'case_001_A_N5.json'
    plan.write_text('{}', encoding='utf8')
    saved = tmp_path / 'candidate.json'
    job = dict(case='case_001', N=5, kind='problem_1', plan_scene='A')
    saved.write_text(json.dumps(dict(fingerprint='run', official=dict(status='failed'))))
    assert priority.load_candidate_checkpoint(saved, 'run', plan, job) is None
    truth = dict(status='official_success', plan_sha256=priority.digest(plan), fingerprint='context')
    saved.write_text(json.dumps(dict(fingerprint='run', official=truth)))
    monkeypatch.setattr(priority, 'verified_record', lambda r: True)
    monkeypatch.setattr(priority, 'input_context', lambda *args: ({'fixture': 1},))
    monkeypatch.setattr(priority, 'object_digest', lambda value: 'context')
    assert priority.load_candidate_checkpoint(saved, 'run', plan, job)['official'] == truth
    plan.write_text('{"changed":true}', encoding='utf8')
    with pytest.raises(ValueError, match='方案'):
        priority.load_candidate_checkpoint(saved, 'run', plan, job)


def test_incomplete_priority_task_cannot_be_skipped_on_resume(tmp_path, monkeypatch):
    plan = tmp_path / 'plan.json'; plan.write_text('{}', encoding='utf8')
    row = dict(fingerprint='run', selected_official=dict(plan_sha256=priority.digest(plan)),
               candidates=[dict(official=dict(status='failed'))])
    checkpoint = tmp_path / 'checkpoint.json'
    checkpoint.write_text(json.dumps(row))
    monkeypatch.setattr(priority, 'verified_record', lambda r: r.get('status') != 'failed')
    assert priority.completed_priority(checkpoint, 'run', plan) is None


def test_inconsistent_replay_remains_blocked_after_restart():
    with pytest.raises(ValueError, match='不一致'):
        replay.assert_consistent(dict(consistent=False, replay={'makespan': 11}, official={'makespan': 10}))
    with pytest.raises(ValueError, match='不一致'):
        replay.assert_consistent(dict(consistent=True, replay={'makespan': 11}, official={'makespan': 10}))
    replay.assert_consistent(dict(consistent=True, replay={'makespan': 10}, official={'makespan': 10}))


def test_priority_granularity_respects_cap_work_budget_and_coverage():
    graph=dict(ops=[dict(id=i,op='ADD',pipe='PIPE_V',cycles=10) for i in range(100)],tensors=[],edges=[])
    fine=priority.make_priority_model(graph,5,1)
    coarse=priority.make_priority_model(graph,5,480)
    assert len(fine.blocks)==100 and len(coarse.blocks)==20
    for model,cap in ((fine,1),(coarse,480)):
        assert sorted(i for block in model.blocks for i in block)==list(range(100))
        assert all(len(block)<=cap and len(block)*10<=50 for block in model.blocks)
    with pytest.raises(ValueError):priority.make_priority_model(graph,5,0)
