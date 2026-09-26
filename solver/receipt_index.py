"""从不可变官方尝试恢复内容凭据；不依赖最终胜出清单或任务检查点。"""
import json
import os
from pathlib import Path


def build_receipt_index(root, fingerprints, verify, duplicate_roots=(), progress=None):
    wanted = set(fingerprints)
    roots = tuple(Path(p).resolve() for p in duplicate_roots)
    buckets = {}
    audit = dict(scanned_receipts=0, matched_receipts=0, invalid_receipts=0,
                 malformed_receipts=[], duplicate_attempts=[], duplicate_seconds=0.0)
    for directory, dirs, files in os.walk(root):
        # 冻结目录只含源码/输入；凭据位于原实验输出目录，避免扫描复制工作区。
        dirs[:] = [d for d in dirs if not d.startswith('night_')
                   and d not in ('.git', '__pycache__', '.pytest_cache')]
        if 'receipt.json' not in files:
            continue
        path = Path(directory) / 'receipt.json'
        audit['scanned_receipts'] += 1
        try:
            row = json.loads(path.read_text(encoding='utf8'))
            if not isinstance(row, dict):
                raise ValueError('凭据必须是对象')
        except (OSError, ValueError) as exc:
            audit['malformed_receipts'].append(dict(path=str(path), error=str(exc)))
            continue
        fp = row.get('fingerprint')
        if fp not in wanted or row.get('status') != 'official_success':
            continue
        audit['matched_receipts'] += 1
        if not verify(row):
            audit['invalid_receipts'] += 1
            continue
        buckets.setdefault(fp, {})[row['record_id']] = row
        if progress and audit['matched_receipts'] % 100 == 0:
            progress(audit['scanned_receipts'], len(buckets), len(wanted))

    def in_interrupted(row):
        return any(Path(row['receipt_path']).resolve().is_relative_to(p) for p in roots)

    history = {}
    for fp, records in buckets.items():
        outside = [r for r in records.values() if not in_interrupted(r)]
        ordered = sorted(outside or records.values(), key=lambda r: (r.get('updated_at', ''), r['record_id']))
        history[fp] = ordered[0]
        # 外部历史凭据本轮启动之前已存在。记录本轮重复的成功尝试，不把复用计入耗时。
        if outside:
            audit['duplicate_attempts'].extend(r for r in records.values() if in_interrupted(r))
    audit['duplicate_seconds'] = sum(r.get('elapsed', 0) for r in audit['duplicate_attempts'])
    audit.update(wanted=len(wanted), recovered=len(history), missing=sorted(wanted-history.keys()))
    return history, audit
