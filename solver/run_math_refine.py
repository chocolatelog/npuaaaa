"""Isolated refinement experiments; only official results replace incumbents."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from compare_optimizer_runs import read_singlecore
from run_all import DATA, parse_cases, stable_seed


def official_key(result):
    if not isinstance(result, dict) or result.get('error'):
        raise ValueError('official evaluation failed: ' + repr(result))
    key = (result.get('makespan'), result.get('added_copy_bytes'))
    if any(isinstance(x, bool) or not isinstance(x, (int, float))
           or not math.isfinite(x) for x in key) or key[0] <= 0:
        raise ValueError('invalid official objective')
    return key


def signature(plan):
    return hashlib.sha256(json.dumps(
        plan, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def rerun_large(task, graph, started):
    """Rerun the existing large-graph fallback without inventing official scores."""
    from pipeline import solve_case
    from run_all import block_cap_for, budget_for
    case, scene, cores = task['case'], task.get('scene', 'B'), task.get('N', 5)
    name = f'{case}_{scene}_N{cores}.json'
    baseline_path = next((Path(d) / name for d in task['baseline_dirs']
                          if (Path(d) / name).is_file()), None)
    if baseline_path is None:
        raise ValueError('no incumbent for ' + name)
    incumbent = json.loads(baseline_path.read_text(encoding='utf-8'))
    if len(incumbent.get('core_schedules', [])) != cores:
        raise ValueError('baseline core count mismatch: ' + str(baseline_path))
    eligible = sum(o['op'] not in ('COPY_IN', 'COPY_OUT') for o in graph['ops'])
    result = solve_case(graph, N=cores, scene=scene,
                        time_budget=budget_for(len(graph['ops'])), seed=task['seed'],
                        block_ops_cap=block_cap_for(eligible))
    candidate_hash, incumbent_hash = signature(result['plan']), signature(incumbent)
    # A different proxy-only proposal cannot replace a historical incumbent.
    same = candidate_hash == incumbent_hash
    (Path(task['output_dir']) / name).write_text(
        json.dumps(incumbent, ensure_ascii=False), encoding='utf-8')
    return dict(case=case, scene=scene, N=cores, seed=task['seed'],
                n_ops=len(graph['ops']), evaluation_status='proxy_only',
                baseline=None, real=None, baseline_source=str(baseline_path),
                baseline_trace=[], candidate_trace=[],
                selected_source='large_fallback_identical' if same else 'large_incumbent_retained',
                fallback_rerun=True, fallback_matches_incumbent=same,
                fallback_plan_sha256=candidate_hash, plan_sha256=incumbent_hash,
                fallback_est_mk=result['est'][0], fallback_est_added=result['est'][1],
                fallback_log=result['log'],
                baseline_speedup=None, speedup=None,
                exclusion_reason='above 12000-op official cap; no new official score',
                elapsed=round(time.monotonic()-started, 3))


def refine_one(task):
    from bound_certificate import compute_certificate
    from model import load_graph
    if task.get('generator', 'cp') == 'op-list':
        from op_list_candidates import generate_op_candidates as generate_local_candidates
    else:
        from pipe_local_cp import generate_local_candidates
    from pipeline import real_evaluate
    started = time.monotonic()
    case = task['case']
    scene, cores = task.get('scene', 'B'), task.get('N', 5)
    if scene not in ('A', 'B') or cores not in (2, 3, 4, 5):
        raise ValueError('supported configurations are A/B and 2--5 cores')
    graph = load_graph(str(Path(DATA) / (case + '.json')))
    if len(graph['ops']) > 12000:
        if task.get('large_policy') == 'rerun-fallback':
            return rerun_large(task, graph, started)
        raise ValueError('official gate excludes graphs above 12000 ops')
    name = f'{case}_{scene}_N{cores}.json'
    seen, baseline_trace = set(), []
    incumbent = baseline_real = baseline_source = None
    for directory in task['baseline_dirs']:
        path = Path(directory) / name
        if not path.is_file():
            continue
        plan = json.loads(path.read_text(encoding='utf-8'))
        if len(plan.get('core_schedules', [])) != cores:
            raise ValueError('baseline core count mismatch: ' + str(path))
        sig = signature(plan)
        if sig in seen:
            continue
        seen.add(sig)
        real = real_evaluate(graph, plan, scene)
        key = official_key(real)
        baseline_trace.append(dict(source=str(path), sha256=sig, real=real))
        if baseline_real is None or key < official_key(baseline_real):
            incumbent, baseline_real, baseline_source = plan, real, str(path)
    if incumbent is None:
        raise ValueError('no incumbent for ' + name)
    initial_certificate = compute_certificate(
        graph, cores, singlecore_makespan=task['singlecore'],
        incumbent_makespan=baseline_real['makespan'])
    best_real, selected_source = baseline_real, 'incumbent'
    traces, round_stats = [], []
    for round_index in range(task['rounds']):
        if len(traces) >= task['max_official']:
            break
        candidates, stats = generate_local_candidates(
            graph, incumbent, num_cores=cores,
            seconds_per_solve=task['seconds_per_solve'],
            max_solves=task['max_solves'], movable_limit=task['movable_limit'],
            seed=task['seed'] + round_index)
        round_stats.append(stats)
        improved = False
        for candidate in candidates:
            if len(traces) >= task['max_official']:
                break
            plan = candidate['plan']
            sig = signature(plan)
            if sig in seen:
                continue
            seen.add(sig)
            real = real_evaluate(graph, plan, scene)
            record = dict(round=round_index, source=candidate['source'],
                          sha256=sig, cp_stats=candidate.get('stats', {}),
                          real=real, accepted=False)
            traces.append(record)
            try:
                key = official_key(real)
            except ValueError as exc:
                record['error'] = str(exc)
                continue
            if key < official_key(best_real):
                incumbent, best_real = plan, real
                selected_source = candidate['source']
                record['accepted'], improved = True, True
        if not improved:
            break
    assert official_key(best_real) <= official_key(baseline_real)
    certificate = compute_certificate(
        graph, cores, singlecore_makespan=task['singlecore'],
        incumbent_makespan=best_real['makespan'])
    (Path(task['output_dir']) / name).write_text(
        json.dumps(incumbent, ensure_ascii=False), encoding='utf-8')
    return dict(case=case, scene=scene, N=cores, seed=task['seed'], n_ops=len(graph['ops']),
                evaluation_status='official',
                baseline=baseline_real, real=best_real, baseline_source=baseline_source,
                baseline_trace=baseline_trace, selected_source=selected_source,
                candidate_trace=traces, round_stats=round_stats,
                initial_certificate=initial_certificate, certificate=certificate,
                plan_sha256=signature(incumbent), elapsed=round(time.monotonic()-started,3),
                baseline_speedup=task['singlecore']/baseline_real['makespan'],
                speedup=task['singlecore']/best_real['makespan'])


def summarize(rows, expected):
    keys = [(r['case'], r['scene'], r['N']) for r in rows]
    if len(set(keys)) != len(keys) or len(rows) != expected:
        raise ValueError('incomplete or duplicate results')
    official = [r for r in rows if r.get('evaluation_status', 'official') == 'official']
    for row in official:
        official_key(row['real'])
        official_key(row['baseline'])
    def metrics(group):
        return dict(tasks=len(group),
                    wins=sum(r['real']['makespan'] < r['baseline']['makespan'] for r in group),
                    ties=sum(r['real']['makespan'] == r['baseline']['makespan'] for r in group),
                    losses=sum(r['real']['makespan'] > r['baseline']['makespan'] for r in group),
                    baseline_mean_speedup=statistics.mean(r['baseline_speedup'] for r in group) if group else None,
                    mean_speedup=statistics.mean(r['speedup'] for r in group) if group else None)
    grouped = {}
    for scene, cores in sorted({(r['scene'], r['N']) for r in rows}):
        group = [r for r in official if (r['scene'], r['N']) == (scene, cores)]
        grouped[f'{scene}_N{cores}'] = metrics(group)
        grouped[f'{scene}_N{cores}']['proxy_only_tasks'] = sum(
            (r['scene'], r['N']) == (scene, cores) and r not in official for r in rows)
    result = dict(metrics(official), tasks=len(rows), official_tasks=len(official),
                proxy_only_tasks=len(rows)-len(official),
                scope='requested task matrix; speedup means use official tasks only',
                by_scene_and_cores=grouped,
                baseline_official_calls=sum(len(r['baseline_trace']) for r in rows),
                candidate_official_calls=sum(len(r['candidate_trace']) for r in rows),
                candidate_errors=sum(bool(t.get('error')) for r in rows for t in r['candidate_trace']),
                task_seconds=sum(r['elapsed'] for r in rows))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cases', default='5,24,48,49,56,80,82,88')
    p.add_argument('--scenes', default='B')
    p.add_argument('--cores', default='5')
    p.add_argument('--large-policy', choices=['reject', 'rerun-fallback'], default='reject')
    p.add_argument('--generator', choices=['cp','op-list'], default='cp')
    p.add_argument('--baseline-dir', action='append', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--log-file', required=True)
    p.add_argument('--singlecore', default='results/summary.csv')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--seconds-per-solve', type=float, default=2.0)
    p.add_argument('--max-solves', type=int, default=6)
    p.add_argument('--movable-limit', type=int, default=8)
    p.add_argument('--max-official', type=int, default=12)
    p.add_argument('--rounds', type=int, default=2)
    p.add_argument('--seed', type=int, default=0)
    a = p.parse_args()
    if a.seconds_per_solve <= 0 or min(a.workers,a.max_solves,a.movable_limit,a.max_official,a.rounds) < 1:
        p.error('budgets and workers must be positive')
    cases = parse_cases(a.cases)
    if len(set(cases)) != len(cases):
        p.error('duplicate cases')
    scenes, cores = a.scenes.split(','), [int(n) for n in a.cores.split(',')]
    if len(set(scenes)) != len(scenes) or not set(scenes) <= {'A', 'B'}:
        p.error('scenes must be unique A/B values')
    if len(set(cores)) != len(cores) or not set(cores) <= {2, 3, 4, 5}:
        p.error('cores must be unique values from 2--5')
    matrix = [(case, scene, n) for case in cases for scene in scenes for n in cores]
    output, logfile = Path(a.output_dir).resolve(), Path(a.log_file).resolve()
    if output.exists() or logfile.exists():
        p.error('use new output paths; existing results are never overwritten')
    baselines = [str(Path(d).resolve()) for d in a.baseline_dir]
    if any(not Path(d).is_dir() for d in baselines):
        p.error('baseline directory missing')
    if any(not any((Path(d)/f'{case}_{scene}_N{n}.json').is_file()
                   for d in baselines) for case, scene, n in matrix):
        p.error('baseline plan missing for requested task matrix')
    import ortools
    from ortools.sat.python import cp_model
    singlecore = read_singlecore(a.singlecore)
    files = list(HERE.glob('*.py'))
    files += list((HERE.parent / '通用神经网络处理器下的多核调度问题附件' / 'code').glob('*.py'))
    files += [Path(DATA)/(case+'.json') for case in cases] + [Path(a.singlecore)]
    files += [Path(d)/f'{case}_{scene}_N{n}.json' for d in baselines
              for case, scene, n in matrix if (Path(d)/f'{case}_{scene}_N{n}.json').is_file()]
    hashes = {str(f.resolve()): hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
    output.mkdir(parents=True)
    logfile.parent.mkdir(parents=True, exist_ok=True)
    logfile.with_suffix('.manifest.json').write_text(json.dumps(dict(
        config=vars(a), ortools_version=ortools.__version__, python=sys.version,
        hashes=hashes, target=3.8, local_cp_bound_scope='restricted surrogate only'),
        ensure_ascii=False, indent=2), encoding='utf-8')
    tasks = [dict(case=case, scene=scene, N=n, large_policy=a.large_policy,
                  baseline_dirs=baselines, output_dir=str(output),
                  singlecore=singlecore[case], seconds_per_solve=a.seconds_per_solve,
                  max_solves=a.max_solves, movable_limit=a.movable_limit,
                  max_official=a.max_official, rounds=a.rounds, generator=a.generator,
                  seed=stable_seed(case,scene,n,a.seed)%2147483647)
             for case, scene, n in matrix]
    rows, failures, started = [], [], time.monotonic()
    with logfile.open('x', encoding='utf-8') as stream:
        with ProcessPoolExecutor(max_workers=a.workers) as pool:
            futures = {pool.submit(refine_one,t): t for t in tasks}
            for future in as_completed(futures):
                try:
                    row = future.result()
                except Exception as exc:
                    task = futures[future]
                    failures.append(dict(case=task['case'], scene=task['scene'],
                                         N=task['N'], error=repr(exc)))
                    print('FAIL', failures[-1], flush=True)
                    continue
                rows.append(row)
                stream.write(json.dumps(row,ensure_ascii=False) + chr(10))
                stream.flush()
                before = row['baseline']['makespan'] if row['baseline'] else 'proxy'
                after = row['real']['makespan'] if row['real'] else 'proxy'
                print(f'[{len(rows)}/{len(tasks)}] {row["case"]} {row["scene"]} N={row["N"]} '
                      f'{before} -> {after} '
                      f'({row["elapsed"]:.1f}s)', flush=True)
    result = dict(failures=failures,completed=len(rows)) if failures else summarize(rows,len(tasks))
    result['wall_seconds'] = round(time.monotonic()-started,3)
    logfile.with_suffix('.summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
