"""官方结果逐任务配对；同时报告等权相对变化和总时间比，避免混用。"""
import argparse
import hashlib
import json
import statistics
from pathlib import Path


def load_rows(path):
    rows = {}
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        row = json.loads(line)
        key = (row['case'], row['scene'], row['N'])
        if key in rows:
            old = rows[key]
            if not (old.get('protocol_version') == row.get('protocol_version') == 2
                    and row.get('attempt', 0) > old.get('attempt', 0)
                    and row.get('fingerprint') == old.get('fingerprint')):
                raise ValueError(f'重复任务: {key}')
        rows[key] = row
    return rows


def compare(base_path, new_path):
    base, new = load_rows(base_path), load_rows(new_path)
    rows = []
    for key in sorted(base.keys() & new.keys()):
        a, b = base[key], new[key]
        ra, rb = a.get('real') or {}, b.get('real') or {}
        if ra.get('error') or rb.get('error') or not ra.get('makespan') or not rb.get('makespan'):
            continue
        delta = rb['makespan'] / ra['makespan'] - 1
        rows.append({'case': key[0], 'scene': key[1], 'N': key[2],
                     'base': ra['makespan'], 'new': rb['makespan'],
                     'relative_change': delta,
                     'seed_equal': a.get('seed') is not None and a.get('seed') == b.get('seed')})
    def summary(sub):
        if not sub:
            return {'count': 0}
        return {'count': len(sub),
                'mean_relative_change': statistics.mean(r['relative_change'] for r in sub),
                'total_time_ratio_change': sum(r['new'] for r in sub) / sum(r['base'] for r in sub) - 1,
                'wins': sum(r['new'] < r['base'] for r in sub),
                'losses': sum(r['new'] > r['base'] for r in sub),
                'ties': sum(r['new'] == r['base'] for r in sub),
                'regressions_over_5pct': sum(r['relative_change'] > .05 for r in sub),
                'all_seeds_match': all(r['seed_equal'] for r in sub)}
    return {'base_path': str(base_path), 'new_path': str(new_path),
            'base_sha256': hashlib.sha256(Path(base_path).read_bytes()).hexdigest(),
            'new_sha256': hashlib.sha256(Path(new_path).read_bytes()).hexdigest(),
            'summary': summary(rows),
            'by_core': {str(n): summary([r for r in rows if r['N'] == n]) for n in (2, 3, 4, 5)},
            'rows': rows,
            'limits': '固定种子不等于固定搜索轨迹：墙钟预算会改变评估次数。只比较官方值；旧日志代理值可能错配方案。'}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base', required=True)
    p.add_argument('--new', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    report = compare(args.base, args.new)
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: report[k] for k in ('summary', 'by_core')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
