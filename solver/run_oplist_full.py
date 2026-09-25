"""Guarded operation-level experiments across A/B and 2--5 cores.

Large graphs are freshly evaluated in isolated, time-limited processes.
Unverified fallbacks never enter official averages. Historical runs are immutable.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import math
import multiprocessing as mp
from pathlib import Path
import statistics
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from bound_certificate import compute_certificate
from compare_optimizer_runs import read_log, read_singlecore
from model import load_graph
from op_list_candidates import generate_op_candidates
from pipeline import real_evaluate
from run_all import DATA, parse_cases
from run_math_refine import official_key, signature


def _isolated_child(connection, operation, payload):
    try:
        if operation == 'evaluate':
            value = real_evaluate(*payload, verify_cap=10**9)
        elif operation == 'generate':
            graph, plan, cores, count = payload
            value = generate_op_candidates(
                graph, plan, num_cores=cores, max_candidates=count)
        else:
            raise ValueError('unknown isolated operation')
        connection.send(('ok', value))
    except Exception as exc:
        connection.send(('error', repr(exc)))
    finally:
        connection.close()


def isolated_call(operation, payload, timeout):
    context = mp.get_context('spawn')
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(target=_isolated_child, args=(sender, operation, payload))
    process.start()
    sender.close()
    try:
        if not receiver.poll(timeout):
            return {'error': 'timeout', 'operation': operation, 'seconds': timeout}
        try:
            status, value = receiver.recv()
        except EOFError:
            return {'error': 'worker exited without a result', 'operation': operation}
        return value if status == 'ok' else {'error': value, 'operation': operation}
    finally:
        receiver.close()
        process.join(timeout=0.2)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
        if process.is_alive():
            process.kill()
            process.join(timeout=5)
        process.close()


def evaluate(graph, plan, scene, timeout):
    if scene == 'A' or len(graph['ops']) > 12000:
        return isolated_call('evaluate', (graph, plan, scene), timeout)
    return real_evaluate(graph, plan, scene)


def valid_real(value):
    try:
        official_key(value)
    except ValueError:
        return False
    return True


def run_task(task):
    started = time.monotonic()
    case, scene, cores = task['case'], task['scene'], task['N']
    if scene not in ('A', 'B', 'C') or cores not in (2, 3, 4, 5):
        raise ValueError('only A/B/C with 2--5 cores are supported')
    graph = load_graph(str(Path(DATA) / (case + '.json')))
    large = len(graph['ops']) > 12000
    filename = f'{case}_{scene}_N{cores}.json'
    seen, baseline_trace, traces = set(), [], []
    incumbent = initial_real = fallback = source = fallback_source = None
    baseline_names = [filename]
    if scene == 'C':
        baseline_names.append(f'{case}_B_N{cores}.json')
    for directory in task['baseline_dirs']:
        paths = [Path(directory) / name for name in baseline_names]
        path = next((p for p in paths if p.is_file()), None)
        if path is None:
            continue
        plan = json.loads(path.read_text(encoding='utf-8'))
        if len(plan.get('core_schedules', [])) != cores:
            raise ValueError('wrong baseline core count: ' + str(path))
        sig = signature(plan)
        if sig in seen:
            continue
        seen.add(sig)
        if fallback is None:
            fallback, fallback_source = plan, str(path)
        real = evaluate(graph, plan, scene, task['evaluation_seconds'])
        baseline_trace.append(dict(source=str(path), sha256=sig, real=real))
        if valid_real(real) and (initial_real is None or official_key(real) < official_key(initial_real)):
            incumbent, initial_real, source = plan, real, str(path)
    if fallback is None:
        raise ValueError('no input plan for ' + filename)
    best_real, selected_source = initial_real, 'incumbent'
    stats = {'status': 'not_run_unverified_baseline'}
    if incumbent is not None:
        count = task['max_candidates']
        candidate_deadline = time.monotonic() + task.get('candidate_seconds', 300)
        if large:
            generated = isolated_call('generate', (graph, incumbent, cores, count),
                                      min(task['generation_seconds'], task.get('candidate_seconds', 300)))
        else:
            generated = generate_op_candidates(
                graph, incumbent, num_cores=cores, max_candidates=count)
        if isinstance(generated, dict) and generated.get('error'):
            candidates, stats = [], generated
        else:
            candidates, stats = generated
        for candidate in candidates[:count]:
            remaining = candidate_deadline - time.monotonic()
            if remaining <= 0:
                stats['candidate_budget_exhausted'] = True
                break
            plan = candidate['plan']
            sig = signature(plan)
            if sig in seen:
                continue
            seen.add(sig)
            real = evaluate(graph, plan, scene, min(task['evaluation_seconds'], remaining))
            record = dict(source=candidate['source'], sha256=sig, real=real, accepted=False)
            traces.append(record)
            if not valid_real(real):
                record['error'] = repr(real)
                continue
            if official_key(real) < official_key(best_real):
                incumbent, best_real = plan, real
                selected_source, record['accepted'] = candidate['source'], True
        stats['candidate_budget_seconds'] = task.get('candidate_seconds', 300)
        stats['candidate_attempts'] = len(traces)
        assert official_key(best_real) <= official_key(initial_real)
    else:
        incumbent, source = fallback, fallback_source
        selected_source = 'unverified_input_retained'
    certificate = None
    if best_real is not None:
        certificate = compute_certificate(graph, cores, task['singlecore'], best_real['makespan'])
    output = Path(task['output_dir']) / filename
    with output.open('x', encoding='utf-8') as stream:
        json.dump(incumbent, stream, ensure_ascii=False)
    return dict(case=case, scene=scene, N=cores, n_ops=len(graph['ops']), large=large,
                status='official' if best_real is not None else 'unverified_input_retained',
                baseline=initial_real, real=best_real, baseline_source=source,
                baseline_trace=baseline_trace, candidate_trace=traces, generator_stats=stats,
                selected_source=selected_source, plan_sha256=signature(incumbent),
                certificate=certificate, elapsed=round(time.monotonic()-started, 3),
                baseline_speedup=task['singlecore']/initial_real['makespan'] if initial_real else None,
                speedup=task['singlecore']/best_real['makespan'] if best_real else None)


def _metrics(rows):
    if not rows:
        return dict(tasks=0, mean_speedup=None, baseline_mean_speedup=None,
                    wins=0, ties=0, losses=0)
    return dict(tasks=len(rows), mean_speedup=statistics.mean(r['speedup'] for r in rows),
                baseline_mean_speedup=statistics.mean(r['baseline_speedup'] for r in rows),
                wins=sum(r['real']['makespan'] < r['baseline']['makespan'] for r in rows),
                ties=sum(r['real']['makespan'] == r['baseline']['makespan'] for r in rows),
                losses=sum(r['real']['makespan'] > r['baseline']['makespan'] for r in rows))


def summarize(rows, expected_keys, reference):
    keys = [(r['case'], r['scene'], r['N']) for r in rows]
    if len(set(keys)) != len(keys) or set(keys) != set(expected_keys):
        raise ValueError('duplicate, missing, or extra tasks')
    official = [r for r in rows if valid_real(r.get('real'))]
    paired = [r for r in official if valid_real(reference.get(
        (r['case'], r['scene'], r['N']), {}).get('real'))]
    by_group = {}
    for scene, cores in sorted({(r['scene'], r['N']) for r in rows}):
        group = [r for r in official if (r['scene'], r['N']) == (scene, cores)]
        pair = [r for r in paired if (r['scene'], r['N']) == (scene, cores)]
        expected = sum((key[1], key[2]) == (scene, cores) for key in expected_keys)
        expected_pair = sum((key[1], key[2]) == (scene, cores) and
                            valid_real(reference.get(key, {}).get('real')) for key in expected_keys)
        by_group[f'{scene}_N{cores}'] = dict(expected=expected,
            all_official_complete=len(group)==expected, all_official=_metrics(group),
            expected_paired=expected_pair, paired_complete=len(pair)==expected_pair,
            paired_historical=_metrics(pair))
    historic = []
    for row in paired:
        old = reference[(row['case'], row['scene'], row['N'])]['real']['makespan']
        new = row['real']['makespan']
        historic.append((old, new))
    return dict(expected=len(expected_keys), completed=len(rows),
                status_counts=dict(Counter(r['status'] for r in rows)),
                official_tasks=len(official), unverified_tasks=len(rows)-len(official),
                all_official_complete=len(official)==len(expected_keys),
                all_official=_metrics(official), paired_historical=_metrics(paired),
                versus_historical=dict(tasks=len(historic),
                    wins=sum(new<old for old,new in historic),
                    ties=sum(new==old for old,new in historic),
                    losses=sum(new>old for old,new in historic),
                    mean_paired_gain_percent=statistics.mean((old/new-1)*100 for old,new in historic) if historic else None),
                large_official_tasks=sum(r['large'] for r in official),
                baseline_official_calls=sum(len(r.get('baseline_trace', [])) for r in rows),
                candidate_official_calls=sum(len(r.get('candidate_trace', [])) for r in rows),
                candidate_errors=sum(bool(t.get('error')) for r in rows for t in r.get('candidate_trace', [])),
                generator_errors=sum(bool(r.get('generator_stats', {}).get('error')) for r in rows),
                candidate_budget_exhausted=sum(bool(r.get('generator_stats', {}).get('candidate_budget_exhausted')) for r in rows),
                by_scene_and_cores=by_group,
                metric='arithmetic mean(singlecore_makespan / official_makespan)',
                comparison='wins/ties/losses use freshly evaluated input portfolio unless named versus_historical')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', default='1-100')
    parser.add_argument('--scenes', default='A,B')
    parser.add_argument('--cores', default='2,3,4,5')
    parser.add_argument('--baseline-dir', action='append', required=True)
    parser.add_argument('--reference-log', required=True)
    parser.add_argument('--singlecore', default='results/summary.csv')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--log-file', required=True)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--max-candidates', type=int, default=12)
    parser.add_argument('--evaluation-seconds', type=float, default=360)
    parser.add_argument('--generation-seconds', type=float, default=120)
    parser.add_argument('--candidate-seconds', type=float, default=300)
    args = parser.parse_args()
    cases, scenes, cores = parse_cases(args.cases), args.scenes.split(','), [int(n) for n in args.cores.split(',')]
    if len(set(cases)) != len(cases) or len(set(scenes)) != len(scenes) or len(set(cores)) != len(cores):
        parser.error('duplicate task dimensions')
    if not set(scenes) <= {'A','B','C'} or not set(cores) <= {2,3,4,5}:
        parser.error('only A/B/C and 2--5 cores are supported')
    if not 1 <= args.max_candidates <= 12 or args.workers < 1 or any(
            not math.isfinite(v) or v <= 0 for v in (args.evaluation_seconds,args.generation_seconds,args.candidate_seconds)):
        parser.error('invalid budgets')
    output, logfile = Path(args.output_dir).resolve(), Path(args.log_file).resolve()
    if any(p.exists() for p in (output, logfile, logfile.with_suffix('.manifest.json'), logfile.with_suffix('.summary.json'))):
        parser.error('use fresh output paths; existing experiments are immutable')
    baselines = [str(Path(d).resolve()) for d in args.baseline_dir]
    if any(not Path(d).is_dir() for d in baselines):
        parser.error('baseline directory missing')
    reference, singlecore = read_log(args.reference_log), read_singlecore(args.singlecore)
    expected = [(case,scene,n) for case in cases for scene in scenes for n in cores]
    # C has no historical reference rows in the A/B archive. For C runs the
    # input B plan is the warm start; baseline comparison is performed against
    # the freshly evaluated C incumbent inside each task.
    missing_reference = [key for key in expected if key not in reference]
    if missing_reference and any(scene != 'C' for scene in scenes):
        parser.error('reference log is missing requested tasks')
    files = list(HERE.glob('*.py'))
    files += list((HERE.parent/'通用神经网络处理器下的多核调度问题附件'/'code').glob('*.py'))
    files += [Path(DATA)/(case+'.json') for case in cases]
    files += [Path(args.reference_log),Path(args.singlecore)]
    for case,scene,n in expected:
        names = [f'{case}_{scene}_N{n}.json']
        if scene == 'C':
            # No archived C plans exist yet; use the corresponding B plan as
            # a warm start and evaluate it under the official C evaluator.
            names.append(f'{case}_B_N{n}.json')
        paths = [Path(d) / name for d in baselines for name in names]
        if not any(p.is_file() for p in paths):
            parser.error(f'missing baseline: {case}/{scene}/{n}')
        if not math.isfinite(singlecore.get(case,0)) or singlecore.get(case,0) <= 0:
            parser.error('invalid singlecore reference: ' + case)
        files.extend(p for p in paths if p.is_file())
    hashes = {str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    output.mkdir(parents=True)
    logfile.parent.mkdir(parents=True, exist_ok=True)
    manifest = dict(config=vars(args), python=sys.version, hashes=hashes,
        expected_keys=expected, scope='fresh A/B evaluation; official paired and full populations separate',
        hardware_unchanged=True, target_scope='B_N5 paired 81 mean >= 3.8')
    logfile.with_suffix('.manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    tasks = [dict(case=case,scene=scene,N=n,baseline_dirs=baselines,output_dir=str(output),
        singlecore=singlecore[case],max_candidates=args.max_candidates,
        evaluation_seconds=args.evaluation_seconds,generation_seconds=args.generation_seconds,
        candidate_seconds=args.candidate_seconds)
        for case,scene,n in expected]
    started, rows = time.monotonic(), []
    print(f'Starting {len(tasks)} tasks; workers={args.workers}; fresh official large-case evaluation',flush=True)
    with logfile.open('x',encoding='utf-8') as stream:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(run_task,t):t for t in tasks}
            for future in as_completed(futures):
                task = futures[future]
                try:
                    row = future.result()
                except Exception as exc:
                    row = dict(case=task['case'],scene=task['scene'],N=task['N'],
                               status='failed',real=None,error=repr(exc))
                rows.append(row)
                stream.write(json.dumps(row,ensure_ascii=False)+chr(10))
                stream.flush()
                result = row.get('real') or {}
                print(f'[{len(rows)}/{len(tasks)}] {row["case"]}/{row["scene"]}/N{row["N"]} '
                      f'{row["status"]} makespan={result.get("makespan")} elapsed={row.get("elapsed")}',flush=True)
    result = summarize(rows,expected,reference)
    result['wall_seconds'] = round(time.monotonic()-started,3)
    logfile.with_suffix('.summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
    if any(row['status'] != 'official' for row in rows):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
