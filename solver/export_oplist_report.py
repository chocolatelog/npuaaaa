"""Recompute the completed 800-task experiment and export auditable JSON."""

import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results'


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def key(row):
    return row['case'], row['scene'], row['N']


def read_log(path):
    with path.open(encoding='utf-8-sig') as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if len({key(row) for row in rows}) != len(rows):
        raise ValueError(f'Duplicate task keys: {path}')
    return rows


def valid_real(result):
    if not isinstance(result, dict) or result.get('error'):
        return False
    values = result.get('makespan'), result.get('added_copy_bytes')
    return all(isinstance(v, (int, float)) and not isinstance(v, bool)
               and math.isfinite(v) for v in values) and values[0] > 0


def comparison(rows, baseline_field):
    pairs = [(row[baseline_field], row['makespan']) for row in rows
             if row[baseline_field] is not None]
    return dict(tasks=len(pairs),
                wins=sum(new < old for old, new in pairs),
                ties=sum(new == old for old, new in pairs),
                losses=sum(new > old for old, new in pairs),
                mean_paired_gain_percent=statistics.mean(
                    (old / new - 1) * 100 for old, new in pairs) if pairs else None)


def metrics(rows):
    values = [row['speedup'] for row in rows]
    return dict(tasks=len(rows), mean_speedup=statistics.mean(values),
                median_speedup=statistics.median(values),
                min_speedup=min(values), max_speedup=max(values),
                population_stddev=statistics.pstdev(values),
                input_mean_speedup=statistics.mean(row['input_speedup'] for row in rows),
                mean_makespan=statistics.mean(row['makespan'] for row in rows),
                versus_input=comparison(rows, 'input_makespan'),
                versus_historical=comparison(rows, 'historical_makespan'))


def main():
    log_path = RESULTS / 'solve_log_oplist_full800.jsonl'
    manifest_path = log_path.with_suffix('.manifest.json')
    summary_path = log_path.with_suffix('.summary.json')
    reference_path = RESULTS / 'solve_log_guided_full.jsonl'
    singlecore_path = RESULTS / 'summary.csv'
    manifest, summary = read_json(manifest_path), read_json(summary_path)
    rows = read_log(log_path)
    reference = {key(row): row for row in read_log(reference_path)}
    with singlecore_path.open(newline='', encoding='utf-8-sig') as stream:
        singlecore = {row['case']: float(row['singlecore_makespan'])
                      for row in csv.DictReader(stream)}
    expected = set(itertools.product(
        (f'case_{n:03d}' for n in range(1, 101)), ('A', 'B'), (2, 3, 4, 5)))
    if set(map(tuple, manifest['expected_keys'])) != expected or {key(row) for row in rows} != expected:
        raise ValueError('Expected exactly the complete 800-task grid')
    plan_dir = RESULTS / 'plans_oplist_full800'
    expected_files = {f'{case}_{scene}_N{cores}.json' for case, scene, cores in expected}
    if {path.name for path in plan_dir.glob('*.json')} != expected_files:
        raise ValueError('Output plans do not match the expected 800 tasks')
    tasks = []
    for row in sorted(rows, key=key):
        k = key(row)
        real, baseline = row.get('real'), row.get('baseline')
        if row['status'] != 'official' or not valid_real(real) or not valid_real(baseline):
            raise ValueError(f'Missing valid official result: {k}')
        if (real['makespan'], real['added_copy_bytes']) > (baseline['makespan'], baseline['added_copy_bytes']):
            raise ValueError(f'Official objective regressed: {k}')
        plan_path = plan_dir / f'{k[0]}_{k[1]}_N{k[2]}.json'
        payload = json.dumps(read_json(plan_path), sort_keys=True, separators=(',', ':'))
        if hashlib.sha256(payload.encode()).hexdigest() != row['plan_sha256']:
            raise ValueError(f'Plan signature mismatch: {k}')
        speedup = singlecore[k[0]] / real['makespan']
        input_speedup = singlecore[k[0]] / baseline['makespan']
        for field, value in (('speedup', speedup), ('baseline_speedup', input_speedup)):
            if not math.isclose(row[field], value, rel_tol=1e-12):
                raise ValueError(f'Speedup mismatch: {k}/{field}')
        historical = reference.get(k, {}).get('real')
        paired = valid_real(historical)
        tasks.append(dict(
            case=k[0], scene=k[1], N=k[2], n_ops=row['n_ops'], large=row['large'],
            historically_comparable=paired, status=row['status'],
            singlecore_makespan=singlecore[k[0]], makespan=real['makespan'],
            input_makespan=baseline['makespan'],
            historical_makespan=historical['makespan'] if paired else None,
            speedup=speedup, input_speedup=input_speedup,
            added_copy_bytes=real['added_copy_bytes'], input_added_copy_bytes=baseline['added_copy_bytes'],
            elapsed_seconds=row['elapsed'], selected_source=row['selected_source'],
            plan_path=plan_path.relative_to(ROOT).as_posix(), plan_sha256=row['plan_sha256']))
    groups = {}
    for scene, cores in itertools.product(('A', 'B'), (2, 3, 4, 5)):
        group = [row for row in tasks if (row['scene'], row['N']) == (scene, cores)]
        paired = [row for row in group if row['historically_comparable']]
        large = [row for row in group if row['large']]
        if (len(group), len(paired), len(large)) != (100, 81, 19):
            raise ValueError(f'Unexpected population: {scene}/{cores}')
        populations = dict(full_100=metrics(group), paired_81=metrics(paired), large_19=metrics(large))
        for current, old in (('full_100', 'all_official'), ('paired_81', 'paired_historical')):
            original = summary['by_scene_and_cores'][f'{scene}_N{cores}'][old]
            if not math.isclose(populations[current]['mean_speedup'], original['mean_speedup'], rel_tol=1e-12):
                raise ValueError('Original summary mean mismatch')
            for field in ('wins', 'ties', 'losses'):
                if populations[current]['versus_input'][field] != original[field]:
                    raise ValueError('Original summary comparison mismatch')
        groups[f'{scene}_N{cores}'] = populations
    report = dict(
        experiment_date='2026-09-25', expected=800, completed=len(tasks), official_tasks=len(tasks),
        metric='arithmetic mean(singlecore_makespan / official_makespan)',
        comparison_note='versus_input: freshly evaluated input portfolio; versus_historical: solve_log_guided_full.jsonl',
        all_800=metrics(tasks), by_scene_and_cores=groups,
        large_case_ids=sorted({row['case'] for row in tasks if row['large']}),
        baseline_official_calls=sum(len(row.get('baseline_trace', [])) for row in rows),
        candidate_official_calls=sum(len(row.get('candidate_trace', [])) for row in rows),
        candidate_errors=sum(bool(t.get('error')) for row in rows for t in row.get('candidate_trace', [])),
        generator_errors=sum(bool(row.get('generator_stats', {}).get('error')) for row in rows),
        candidate_budget_exhausted=sum(bool(row.get('generator_stats', {}).get('candidate_budget_exhausted')) for row in rows),
        wall_seconds=summary['wall_seconds'], run_config=manifest['config'],
        verification=dict(expected_grid_matches=True, output_plan_count=800,
                          output_plan_signatures_verified=800, speedup_equations_verified=800,
                          original_summary_matches=True, fresh_evaluator_replay_performed=False),
        source_files={path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                      for path in (log_path, manifest_path, summary_path, reference_path, singlecore_path)})
    for field in ('baseline_official_calls', 'candidate_official_calls', 'candidate_errors',
                  'generator_errors', 'candidate_budget_exhausted'):
        if report[field] != summary[field]:
            raise ValueError(f'Original summary mismatch: {field}')
    for field in ('tasks', 'wins', 'ties', 'losses'):
        if report['all_800']['versus_historical'][field] != summary['versus_historical'][field]:
            raise ValueError(f'Historical comparison mismatch: {field}')
    for name, content in (('oplist_full800.statistics.json', report), ('oplist_full800.tasks.json', tasks)):
        (RESULTS / name).write_text(json.dumps(content, ensure_ascii=False, indent=2) + chr(10), encoding='utf-8')
    print(json.dumps(dict(tasks=800, plan_hashes_verified=800, all_800=report['all_800'],
                         means={name: {scope: value['mean_speedup'] for scope, value in data.items()}
                                for name, data in groups.items()}), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
