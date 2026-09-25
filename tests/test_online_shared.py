"""在线接入的保底、缓存、证据复用和候选级恢复契约。"""
import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'solver'))


def fixture():
    from model import Model
    from run_all import plan_digest
    graph = {'ops': [{'id': i, 'op': 'MATMUL', 'pipe': 'PIPE_M', 'cycles': 100}
                     for i in (1, 2)], 'edges': [], 'tensors': []}
    model = Model(graph, block_ops_cap=1)
    plan = model.plan_from([0, 1], [[0, 1], []])
    result = {'plan': plan, 'real': metric(300), 'est': [300, 0], 'elapsed': 1,
              'log': {'selected_plan_id': plan_digest(plan), 'official_candidates': []}}
    return graph, model, result


def metric(makespan):
    return dict(makespan=makespan, added_copy_bytes=0, scheduled_copy_bytes=0,
                partition_added=0, spill_added=0)


def test_shared_local_cache_replays_each_order():
    import online_shared as online
    graph, model, baseline = fixture()
    local = online.SharedLocalModel(graph)
    a = local.evaluate(baseline['plan'])
    b = local.evaluate(model.plan_from([0, 1], [[0], [1]]))
    assert a['makespan'] == 300 and b['makespan'] == 100
    assert local.stats['models'] == 1 and local.stats['hits'] == 1
    assert a['tasks'] == b['tasks']


@pytest.mark.parametrize('mode', ['accept', 'mismatch', 'worse', 'error'])
def test_official_guard_and_same_plan_metrics(tmp_path, monkeypatch, mode):
    import online_shared as online
    from shared_candidate_pool import make_candidate
    from run_all import plan_digest
    graph, model, baseline = fixture()
    proposal = model.plan_from([0, 1], [[0], [1]])
    row = make_candidate(model, proposal, 100, 'test')
    row.update(features={'boundary_bytes': 0}, allocated_family='schedule')
    monkeypatch.setattr(online, 'generate_online', lambda *a, **k: ([row], {'test': True}))
    calls = []
    def resource(*args):
        calls.append('resource')
        return metric(100)
    def official(*args):
        calls.append('official')
        if mode == 'error':
            raise RuntimeError('评测器异常')
        return metric({'accept': 100, 'mismatch': 101, 'worse': 301}[mode])
    checkpoint = online.Checkpoint(tmp_path, 'frozen')
    original = copy.deepcopy(baseline)
    out = online.refine_shared(graph, model, baseline, checkpoint, None,
                               resource=resource, official=official)
    assert baseline == original
    assert out['real'] == metric(100 if mode == 'accept' else 300)
    assert out['plan'] == (proposal if mode == 'accept' else baseline['plan'])
    assert out['log']['selected_plan_id'] == plan_digest(out['plan'])
    assert out['log']['final_est'] == list(out['est'])
    assert calls == ['resource', 'official']
    resumed = online.refine_shared(graph, model, baseline, checkpoint, None,
                                   resource=resource, official=official)
    assert resumed == out and calls == ['resource', 'official']


def test_resume_after_candidate_does_not_repeat_resource(tmp_path, monkeypatch):
    import online_shared as online
    from shared_candidate_pool import make_candidate
    graph, model, baseline = fixture()
    row = make_candidate(model, model.plan_from([0, 1], [[0], [1]]), 100, 'test')
    row.update(features={'boundary_bytes': 0}, allocated_family='schedule')
    monkeypatch.setattr(online, 'generate_online', lambda *a, **k: ([row], {}))
    checkpoint = online.Checkpoint(tmp_path, 'frozen')
    calls = []
    def resource(*a):
        calls.append(1)
        return metric(100)
    def interrupted(*a):
        raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        online.refine_shared(graph, model, baseline, checkpoint, None,
                             resource=resource, official=interrupted)
    online.refine_shared(graph, model, baseline, checkpoint, None,
                         resource=resource, official=lambda *a: metric(100))
    assert calls == [1]
    with pytest.raises(ValueError):
        online.Checkpoint(tmp_path, 'changed').read('generation')


def test_online_region_has_no_history_input_and_preserves_coverage():
    import online_shared as online
    from model import Model
    from scene_a_event import derive_multicore_plan
    from evaluation_validation import validate_task_order
    graph = {'ops': [{'id': i, 'op': 'MATMUL', 'pipe': 'PIPE_M', 'cycles': 100}
                     for i in range(4)], 'edges': [], 'tensors': []}
    model = Model(graph, block_ops_cap=1)
    for n in (2, 3, 4, 5):
        plan = model.plan_from([0] * 4, [[0]] + [[] for _ in range(n-1)])
        menu, orders, audit = online.generate_region(graph, model, plan,
                                                    online.SharedLocalModel(graph))
        assert audit['background_model'] == 'local_approximation'
        assert audit['parents'] <= 12 and audit['resource_calls'] == 0
        for row in menu['candidate_features']:
            proposal = model.plan_from(row['assignment'], row['orders'])
            validate_task_order(derive_multicore_plan(graph, proposal))
            assert set(proposal['node_to_subgraph']) == {'0', '1', '2', '3'}
        assert isinstance(orders['rows'], list)


def test_real_entry_uses_new_stage_and_resumes_saved_baseline(tmp_path, monkeypatch):
    import json
    import types
    import online_shared as online
    import run_all
    import pipeline
    from shared_candidate_pool import make_candidate
    graph, model, baseline = fixture()
    (tmp_path/'case_001.json').write_text(json.dumps(graph), encoding='utf-8')
    out = tmp_path/'plans'
    out.mkdir()
    monkeypatch.setattr(run_all, 'DATA', str(tmp_path))
    monkeypatch.setattr(run_all, 'block_cap_for', lambda n: 1)
    calls = []
    def old_solver(*a, **kw):
        calls.append(1)
        return copy.deepcopy(baseline)
    monkeypatch.setattr(pipeline, 'solve_case', old_solver)
    monkeypatch.setattr(online, 'historical_index', lambda: None)
    monkeypatch.setitem(sys.modules, 'torch', types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: True), set_num_threads=lambda n: None))
    row = make_candidate(model, model.plan_from([0, 1], [[0], [1]]), 100, 'test')
    row.update(features={'boundary_bytes': 0}, allocated_family='schedule')
    monkeypatch.setattr(online, 'generate_online', lambda *a, **k: ([row], {}))
    task = dict(case='case_001', scene='A', n=2, out_plans=str(out), shared_budget=True,
                fingerprint='entry-test')
    first = run_all.solve_task(task)
    assert first['real'] == metric(100)
    assert first['log']['shared_budget']['accepted']
    assert run_all.resume_status(first, out) == 'official_success'
    assert first['log']['selected_plan_id'] == first['plan_id']
    second = run_all.solve_task(task)
    assert {k: v for k, v in second.items() if k != 'active_task_wall_seconds'} == {
        k: v for k, v in first.items() if k != 'active_task_wall_seconds'}
    assert calls == [1]


def test_process_lock_rejects_duplicate_without_removing_file(tmp_path):
    from online_shared import process_lock
    path = tmp_path/'run.lock'
    with process_lock(path):
        with pytest.raises(RuntimeError):
            with process_lock(path):
                pass
    assert path.exists()
    with process_lock(path):
        pass


def test_history_inventory_does_not_require_nonexistent_legacy_archive(tmp_path, monkeypatch):
    import online_shared as online
    path = tmp_path/'branch_experiments.jsonl'
    path.write_text('{}', encoding='utf-8')
    path.with_suffix('.manifest.json').write_text('{}', encoding='utf-8')
    monkeypatch.setattr(online, 'ROOT', tmp_path)
    monkeypatch.setattr(online, 'history_paths', lambda: [path])
    assert all(p.exists() for p in online.history_files())
