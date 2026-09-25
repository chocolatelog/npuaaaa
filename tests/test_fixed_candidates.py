"""固定候选实验的预算与输入身份不能随续跑漂移。"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'solver'))


def test_fixed_candidate_limit_and_duplicate_ids(tmp_path):
    path = Path(__file__).resolve().parents[1] / 'solver/run_fixed_candidates.py'
    assert path.exists(), '缺少固定候选重放入口'
    from run_fixed_candidates import load_candidates
    manifest = tmp_path / 'candidates.json'
    rows = [{'id': f'p{i}', 'case': 'case_065', 'N': 4,
             'plan_path': str(tmp_path / f'p{i}.json')} for i in range(3)]
    for row in rows:
        Path(row['plan_path']).write_text('{}')
    manifest.write_text(json.dumps(rows))
    assert [r['id'] for r in load_candidates(manifest, 2)] == ['p0', 'p1']
    manifest.write_text(json.dumps([rows[0], rows[0]]))
    with pytest.raises(ValueError, match='重复'):
        load_candidates(manifest, 0)


def test_compare_protocol_attempts_uses_latest_and_ignores_failure(tmp_path):
    from compare_runs_audit import load_rows, compare
    path = tmp_path / 'run.jsonl'
    base = {'case': 'case_001', 'scene': 'A', 'N': 2, 'protocol_version': 2,
            'attempt': 1, 'real': {'makespan': 100}}
    path.write_text(json.dumps(base) + '\n' + json.dumps({**base, 'attempt': 2,
                    'real': {'error': 'evaluation failed'}}) + '\n')
    rows = load_rows(path)
    assert rows[('case_001', 'A', 2)]['attempt'] == 2
    assert compare(path, path)['summary']['count'] == 0
