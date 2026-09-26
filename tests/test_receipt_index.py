import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'solver'))
from receipt_index import build_receipt_index


def receipt(root, folder, fingerprint, valid=True):
    path = root / folder / 'receipt.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    row = dict(status='official_success', fingerprint=fingerprint,
               record_id=folder, valid=valid, elapsed=7, receipt_path=str(path))
    path.write_text(json.dumps(row), encoding='utf8')
    return row


def test_recovers_unlogged_attempt_and_rejects_invalid_duplicate(tmp_path):
    good = receipt(tmp_path, 'old/official/attempts/job/known/a', 'known')
    receipt(tmp_path, 'new/official/attempts/job/known/b', 'known', False)
    receipt(tmp_path, 'other/official/attempts/job/unwanted/c', 'unwanted')
    history, audit = build_receipt_index(tmp_path, {'known'}, lambda r: r['valid'])
    assert history == {'known': good}
    assert audit['invalid_receipts'] == 1
    assert audit['matched_receipts'] == 2


def test_records_repeated_cost_and_ignores_broken_receipt(tmp_path):
    receipt(tmp_path, 'old/official/attempts/job/key/a', 'key')
    duplicate = receipt(tmp_path, 'interrupted/official/attempts/job/key/b', 'key')
    bad = tmp_path / 'broken' / 'receipt.json'
    bad.parent.mkdir(); bad.write_text('{', encoding='utf8')
    history, audit = build_receipt_index(tmp_path, {'key'}, lambda r: True,
                                        duplicate_roots=[tmp_path / 'interrupted'])
    assert history['key']['record_id'].startswith('old/')
    assert audit['duplicate_attempts'] == [duplicate]
    assert audit['duplicate_seconds'] == 7
    assert len(audit['malformed_receipts']) == 1
