"""Summarize 800 plans after separately verifying the large cases."""

import argparse
import csv
import json
import statistics
from pathlib import Path


def read_rows(path):
    rows = {}
    with open(path, encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            key = row['case'], row['scene'], row['N']
            if key in rows:
                raise ValueError(f'duplicate task in {path}: {key}')
            rows[key] = row
    return rows


def summarize(baseline, solved, large, singlecore):
    if len(solved) != 800 or baseline.keys() != solved.keys():
        raise ValueError('both solver logs must contain the same 800 tasks')
    merged = {}
    for key, row in solved.items():
        result = row['real'] if row['real'] is not None else large.get(
            key, {}).get('real')
        if not result or 'makespan' not in result:
            raise ValueError(f'official result missing for {key}: {result}')
        merged[key] = result
    if len(large) != 152:
        raise ValueError(f'expected 152 large-case results, found {len(large)}')
    paired = [key for key, row in baseline.items() if row['real'] is not None]
    result = {
        'plans': len(solved),
        'official_results': len(merged),
        'large_results': len(large),
        'paired_official_tasks': len(paired),
        'improved': sum(merged[k]['makespan'] < baseline[k]['real']['makespan']
                        for k in paired),
        'equal': sum(merged[k]['makespan'] == baseline[k]['real']['makespan']
                     for k in paired),
        'regressed': sum(merged[k]['makespan'] > baseline[k]['real']['makespan']
                         for k in paired),
        'mean_paired_speedup_gain_percent': round(statistics.mean(
            (baseline[k]['real']['makespan'] / merged[k]['makespan'] - 1)
            * 100 for k in paired), 5),
        'by_scene_and_cores': {},
    }
    for scene in ('A', 'B'):
        for cores in (2, 3, 4, 5):
            keys = [key for key in merged
                    if key[1] == scene and key[2] == cores]
            old_keys = [key for key in keys if key in paired]
            if len(keys) != 100 or len(old_keys) != 81:
                raise ValueError(f'incomplete configuration: {scene}, {cores}')
            base_speedups = [singlecore[k[0]] /
                             baseline[k]['real']['makespan']
                             for k in old_keys]
            new_paired_speedups = [singlecore[k[0]] /
                                   merged[k]['makespan'] for k in old_keys]
            new_full_speedups = [singlecore[k[0]] /
                                 merged[k]['makespan'] for k in keys]
            result['by_scene_and_cores'][f'{scene}_N{cores}'] = {
                'paired_81_baseline_mean_speedup': round(
                    statistics.mean(base_speedups), 4),
                'paired_81_new_mean_speedup': round(
                    statistics.mean(new_paired_speedups), 4),
                'full_100_new_mean_speedup': round(
                    statistics.mean(new_full_speedups), 4),
                'paired_81_gain_percent': round(statistics.mean(
                    (baseline[k]['real']['makespan'] /
                     merged[k]['makespan'] - 1) * 100
                    for k in old_keys), 5),
            }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('baseline_log', type=Path)
    parser.add_argument('solver_log', type=Path)
    parser.add_argument('large_official_log', type=Path)
    parser.add_argument('--singlecore', type=Path,
                        default=Path('results/summary.csv'))
    args = parser.parse_args()
    with open(args.singlecore, newline='', encoding='utf-8') as stream:
        singlecore = {row['case']: float(row['singlecore_makespan'])
                      for row in csv.DictReader(stream)}
    summary = summarize(read_rows(args.baseline_log),
                        read_rows(args.solver_log),
                        read_rows(args.large_official_log), singlecore)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
