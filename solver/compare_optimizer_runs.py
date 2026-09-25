"""Compare paired official results from two solver.run_all JSONL logs."""

import argparse
import csv
import json
import statistics
from pathlib import Path


def read_log(path):
    rows = {}
    with open(path, encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            key = row['case'], row['scene'], row['N']
            if key in rows:
                raise ValueError(f'duplicate task in {path}: {key}')
            rows[key] = row
    return rows


def read_singlecore(path):
    with open(path, newline='', encoding='utf-8') as stream:
        return {row['case']: float(row['singlecore_makespan'])
                for row in csv.DictReader(stream)}


def compare(baseline, candidate, singlecore, allow_seed_mismatch=False,
            skip_unofficial=False):
    if baseline.keys() != candidate.keys():
        missing = sorted(baseline.keys() - candidate.keys())
        extra = sorted(candidate.keys() - baseline.keys())
        raise ValueError(f'task sets differ: missing={missing}, extra={extra}')
    pairs = []
    unofficial = 0
    for key in sorted(baseline):
        old, new = baseline[key], candidate[key]
        if not allow_seed_mismatch and old.get('seed') != new.get('seed'):
            raise ValueError(f'search seeds differ for {key}')
        if not old.get('real') or not new.get('real'):
            if not skip_unofficial:
                raise ValueError(f'official result missing for {key}')
            unofficial += 1
            continue
        old_mk = float(old['real']['makespan'])
        new_mk = float(new['real']['makespan'])
        pairs.append((key, old_mk, new_mk))
    improved = sum(new < old for _, old, new in pairs)
    regressed = sum(new > old for _, old, new in pairs)
    summary = {
        'tasks': len(pairs), 'unofficial_tasks': unofficial,
        'improved': improved,
        'equal': len(pairs) - improved - regressed,
        'regressed': regressed,
        'mean_speedup_gain_percent': round(
            statistics.mean((old / new - 1) * 100
                            for _, old, new in pairs), 5),
        'mean_runtime_baseline_seconds': round(statistics.mean(
            baseline[key]['elapsed'] for key, _, _ in pairs), 2),
        'mean_runtime_candidate_seconds': round(statistics.mean(
            candidate[key]['elapsed'] for key, _, _ in pairs), 2),
        'by_scene_and_cores': {},
    }
    for scene in sorted({key[1] for key, _, _ in pairs}):
        for cores in sorted({key[2] for key, _, _ in pairs}):
            group = [(key, old, new) for key, old, new in pairs
                     if key[1] == scene and key[2] == cores]
            if not group:
                continue
            sc = [singlecore[key[0]] for key, _, _ in group]
            summary['by_scene_and_cores'][f'{scene}_N{cores}'] = {
                'tasks': len(group),
                'baseline_mean_speedup': round(statistics.mean(
                    one / old for one, (_, old, _) in zip(sc, group)), 4),
                'candidate_mean_speedup': round(statistics.mean(
                    one / new for one, (_, _, new) in zip(sc, group)), 4),
                'mean_speedup_gain_percent': round(statistics.mean(
                    (old / new - 1) * 100 for _, old, new in group), 5),
            }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('baseline_log', type=Path)
    parser.add_argument('candidate_log', type=Path)
    parser.add_argument('--singlecore', type=Path,
                        default=Path('results/summary.csv'))
    parser.add_argument('--allow-seed-mismatch', action='store_true')
    parser.add_argument('--skip-unofficial', action='store_true')
    args = parser.parse_args()
    summary = compare(read_log(args.baseline_log),
                      read_log(args.candidate_log),
                      read_singlecore(args.singlecore),
                      allow_seed_mismatch=args.allow_seed_mismatch,
                      skip_unofficial=args.skip_unofficial)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
