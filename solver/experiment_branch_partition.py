"""E05 分支、峰值或热点移动试验：固定输入、显卡批特征、12精评/2原官方、可恢复。"""
import argparse
import hashlib
import json
from pathlib import Path
import time

from branch_partition import generate_branch_candidates
from batch_features import BatchFeatures, scalar_features
from experiment_order_pair import binding
from model import Model
from pipeline import real_evaluate
from run_all import (parse_cases, block_cap_for, ensure_run_manifest, run_input_files,
                     append_run_row, valid_official, plan_digest)
from scene_a_event import SceneAEventModel, derive_multicore_plan
from solution import Sol
from task_order_search import search_orders
from partition_identity import select_distinct_partitions

ROOT = Path(__file__).resolve().parents[1]
MODE_LABELS = {'branch':'分支区域', 'peak':'峰值边界', 'hotspot':'累计热点移动','fusion':'四来源融合'}


def seal_result(result, fingerprint):
    # 先正规化为真正落盘的数据模型，避免整数键与字符串键排序不一致。
    result = json.loads(json.dumps(result))
    result['fingerprint'] = fingerprint
    result['binding'] = binding(result)
    return result


def verified(row, fingerprint):
    try:
        content = dict(row)
        expected = content.pop('binding', None)
        if expected != binding(content) or row['fingerprint'] != fingerprint or row['status'] != 'official_success':
            return False
        return (valid_official(row['real']) and all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h
                for p, h in row['artifacts'].items()))
    except (OSError, KeyError, ValueError, TypeError):
        return False


def experiment(row, out, torch, candidate_mode='branch'):
    started = time.perf_counter()
    case, n = row['case'], row['N']
    source = row['variants']['swap']
    input_path = Path(source['plan_path'])
    if hashlib.sha256(input_path.read_bytes()).hexdigest() != source['plan_sha256']:
        raise ValueError('输入方案摘要变化')
    plan = json.loads(input_path.read_text(encoding='utf-8'))
    graph = json.loads((ROOT/'通用神经网络处理器下的多核调度问题附件/data'/f'{case}.json').read_text(encoding='utf-8'))
    baseline = real_evaluate(graph, plan, 'A')
    if not valid_official(baseline) or baseline != source['real']:
        raise ValueError('输入方案原官方结果与冻结结果不一致')
    model = Model(graph, block_ops_cap=block_cap_for(sum(o['op'] not in ('COPY_IN', 'COPY_OUT') for o in graph['ops'])))
    assignment = [plan['node_to_subgraph'][str(block[0])] for block in model.blocks]
    owners = [0] * (max(assignment)+1)
    for c, order in enumerate(plan['core_schedules']):
        for task in order:
            owners[task] = c
    if model.plan_from(assignment, plan['core_schedules']) != plan:
        raise ValueError('输入方案不兼容默认块化')
    parent = Sol(assignment, owners)
    evaluator = SceneAEventModel(graph)
    base_estimate = evaluator.evaluate(plan)
    targets = sorted(base_estimate['tasks'],
                     key=lambda s: (-base_estimate['tasks'][s]['duration'], s))[:3]
    if candidate_mode == 'fusion':
        from partition_fusion import generate_fusion_candidates
        generated,generation=generate_fusion_candidates(model,parent,plan,n,evaluator,base_estimate)
    elif candidate_mode == 'hotspot':
        from hotspot_partition import generate_hotspot_candidates, audit_tensor_aggregation
        before = time.perf_counter()
        tensor_audit = audit_tensor_aggregation(evaluator,plan)
        diagnostic_seconds = time.perf_counter()-before
        generated, generation = generate_hotspot_candidates(model,parent,plan,n,evaluator,base_estimate)
        generation.update(tensor_audit=tensor_audit, diagnostic_seconds=diagnostic_seconds)
        targets = generation['target_tasks']
    elif candidate_mode == 'peak':
        from peak_partition import generate_peak_candidates
        generated, generation = generate_peak_candidates(model, parent, plan, n, evaluator, base_estimate)
        targets = generation['target_tasks']
    else:
        generated, generation = generate_branch_candidates(model, parent, targets, n)
    torch.cuda.reset_peak_memory_stats()
    feature_start = time.perf_counter()
    batch = BatchFeatures(model, n)
    features = batch.evaluate([r['sol'].sg_of_block for r in generated],
                              [r['sol'].core_of_sg for r in generated])
    for r, feature in zip(generated, features):
        if feature != scalar_features(model, r['sol'].sg_of_block, r['sol'].core_of_sg, n):
            raise ValueError('显卡特征与中央处理器不一致')
        r['features'] = feature
    base_features = scalar_features(model,parent.sg_of_block,parent.core_of_sg,n)
    feature_seconds = time.perf_counter()-feature_start
    ranked = sorted(generated, key=lambda r: (r['features']['coarse_score'],
                     tuple(r['sol'].sg_of_block), tuple(r['sol'].core_of_sg)))
    selected = []
    # 每个目标任务至少一个代表，再按粗分数填满；裁掉的候选保留特征和原因。
    for task in targets:
        representative = next((r for r in ranked if r['source']['task'] == task), None)
        if representative is not None:
            selected.append(representative)
    ids = {id(r) for r in selected}
    for r in ranked:
        if len(selected) >= 12:
            break
        if id(r) not in ids:
            selected.append(r); ids.add(id(r))
    if candidate_mode == 'fusion':
        from partition_fusion import select_fusion_candidates
        selected,generation['selection']=select_fusion_candidates(generated)
        ids={id(r) for r in selected}
    evaluated, errors, artifacts = [], [], {}
    for index, r in enumerate(selected):
        before = time.perf_counter()
        try:
            sol = r['sol']
            _, _, info = model.evaluate(sol.sg_of_block, sol.core_of_sg, 'A', n)
            proposal = model.plan_from(sol.sg_of_block, info['orders'])
            estimate = evaluator.evaluate(proposal)
            view = derive_multicore_plan(graph, proposal)
            local, stats = search_orders({s: t['duration'] for s, t in estimate['tasks'].items()},
                view['subgraph_preds'], proposal['core_schedules'],
                bandwidth_floor=estimate['total_copy_bytes']/60, max_evals=400,
                rounds=2, seconds=None, keep=1)
            predicted = estimate['makespan']
            if local and local[0]['makespan'] < predicted:
                proposal['core_schedules'] = local[0]['orders']
                predicted = local[0]['makespan']
            from evaluation_validation import validate_task_order
            validate_task_order(derive_multicore_plan(graph,proposal))
            path = out/f'{row["id"]}_candidate_{index:02d}.json'
            path.write_text(json.dumps(proposal), encoding='utf-8')
            artifacts[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
            evaluated.append({'source': r['source'], 'features': r['features'], 'estimate': predicted,
                'partition_added': estimate['partition_added_bytes'], 'spill_added': estimate['spill_bytes'],
                'total_added': estimate['total_added_bytes'], 'local_search': stats,
                'plan_path': str(path), 'plan_id': plan_digest(proposal), 'seconds': time.perf_counter()-before})
        except Exception as exc:
            errors.append({'source': r['source'], 'error': repr(exc)})
    shortlist = select_distinct_partitions(
        sorted(evaluated, key=lambda r: (r['estimate'], r['total_added'], r['plan_id'])),
        lambda r: json.loads(Path(r['plan_path']).read_text(encoding='utf-8')), 2)
    best_plan, best = plan, baseline
    for r in shortlist:
        proposal = json.loads(Path(r['plan_path']).read_text(encoding='utf-8'))
        truth = real_evaluate(graph, proposal, 'A')
        r['official'] = truth
        if not valid_official(truth):
            errors.append({'stage': 'official', 'result': truth})
            continue
        if (truth['makespan'], truth['added_copy_bytes']) < (best['makespan'], best['added_copy_bytes']):
            best_plan, best = proposal, truth
    final = out/f'{row["id"]}_selected.json'
    final.write_text(json.dumps(best_plan), encoding='utf-8')
    artifacts[str(final)] = hashlib.sha256(final.read_bytes()).hexdigest()
    return {'id': row['id'], 'case': case, 'N': n, 'status': 'official_success',
        'source': str(input_path), 'source_sha256': source['plan_sha256'],
        'baseline': baseline, 'baseline_estimate': base_estimate['makespan'],
        'baseline_spill': base_estimate['spill_bytes'], 'real': best,
        'baseline_features': base_features,
        'generation': generation, 'feature_seconds': feature_seconds, 'candidate_mode': candidate_mode,
        'gpu_peak_allocated': torch.cuda.max_memory_allocated(), 'gpu_cpu_features_equal': True,
        'candidate_features': [{'source': r['source'], 'features': r['features'],
                               'boundary_delta':r['features']['boundary_bytes']-base_features['boundary_bytes'],
                               'compute_load_delta':r['features']['compute_load']-base_features['compute_load'],
                               'admitted': id(r) in ids,
                               'fusion_front':r.get('fusion_front'),
                               'selection_reason':r.get('fusion_selection_reason'),
                               **({'assignment':r['sol'].sg_of_block,'owners':r['sol'].core_of_sg}
                                  if candidate_mode in ('hotspot','fusion') else {})} for r in ranked],
        'evaluated': evaluated, 'precise_count': len(selected), 'official_count': len(shortlist),
        'errors': errors, 'artifacts': artifacts, 'plan_path': str(final),
        'plan_id': plan_digest(best_plan), 'seconds': time.perf_counter()-started,
        'scope': ('固定 E03 完整方案，'+ MODE_LABELS[candidate_mode] +
                  '独立组件，最多12局部精评与2原官方；未启用对照时不代表旧切分器等预算比较')}


def compare_existing(row, result, out):
    """同一原官方输入，旧分位点/合并路径使用相同的计数上限，实际次数另报。"""
    from partition_polish import polish_partition
    started = time.perf_counter()
    graph = json.loads((ROOT/'通用神经网络处理器下的多核调度问题附件/data'/f'{row["case"]}.json').read_text(encoding='utf-8'))
    plan = json.loads(Path(result['source']).read_text(encoding='utf-8'))
    selected, real, audit = polish_partition(graph, plan, result['baseline'], real_evaluate,
        seconds=None, local_search_seconds=None, max_precise=12)
    path = out/f'{row["id"]}_existing_selected.json'
    path.write_text(json.dumps(selected), encoding='utf-8')
    result['artifacts'][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    for i, candidate in enumerate(audit['candidates']):
        candidate_path = out/f'{row["id"]}_existing_candidate_{i}.json'
        candidate_path.write_text(json.dumps(candidate.pop('plan')), encoding='utf-8')
        result['artifacts'][str(candidate_path)] = hashlib.sha256(candidate_path.read_bytes()).hexdigest()
        candidate['plan_path'] = str(candidate_path)
    result['existing'] = {'real': real, 'audit': audit, 'plan_path': str(path),
                          'seconds': time.perf_counter()-started}
    result['scope'] = ('同一 E03 完整输入的'+ MODE_LABELS[result['candidate_mode']] +
                       '与旧分位点/合并独立配对；'
                       '均最多12精评/400次局部重放/2原官方，不是相同实际次数或相同墙钟成本')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cases', default='5,44,49,67,82')
    parser.add_argument('--cores', default='2,4')
    parser.add_argument('--max-tasks', type=int, default=0)
    parser.add_argument('--compare-existing', action='store_true', help='同时运行旧切分器的固定计数对照，结果与成本单列')
    parser.add_argument('--candidate-mode', choices=tuple(MODE_LABELS), default='branch',
                        help='分支区域、溢出事件边界或累计热点移动；单独消融')
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    import torch
    if not torch.cuda.is_available():
        parser.error('本试验需现有 PyTorch 显卡环境')
    cases, cores = parse_cases(args.cases), list(dict.fromkeys(map(int, args.cores.split(','))))
    if args.max_tasks < 0 or any(n not in (2, 3, 4, 5) for n in cores):
        parser.error('核数或分批数非法')
    source = ROOT/'results/p1_e03_r02/order_pairs.jsonl'
    mapping = {(r['case'], r['N']): r for r in map(json.loads, source.read_text(encoding='utf-8').splitlines())}
    selected = [mapping[c, n] for c in cases for n in cores]
    out = Path(args.output_dir).resolve(); out.mkdir(parents=True, exist_ok=True)
    ledger = out/'branch_experiments.jsonl'
    fingerprint = ensure_run_manifest(ledger, run_input_files(cases)+[source]+
        [Path(r['variants']['swap']['plan_path']) for r in selected],
        {'cases': cases, 'cores': cores, 'max_precise': 12, 'max_official': 2,
         'local_replays': 400, 'search_seconds': None, 'gpu': torch.cuda.get_device_name(0),
         'target_tasks': 3, 'levels': 3, 'seed_sets': 6, 'placements': 2,
         'compare_existing': args.compare_existing, 'candidate_mode': args.candidate_mode,
         'hotspot_limits':{'tensors_per_pool':3,'consumer_blocks':8,'target_tasks_per_tensor':4}})
    previous, damaged = {}, []
    if ledger.exists():
        for i, line in enumerate(ledger.read_text(encoding='utf-8').splitlines(), 1):
            try:
                r = json.loads(line); previous[r['id']] = r
            except (ValueError, KeyError):
                damaged.append(i)
    done = {k for k, r in previous.items() if verified(r, fingerprint)}
    pending = [r for r in selected if r['id'] not in done]
    if args.max_tasks:
        pending = pending[:args.max_tasks]
    mode_label = MODE_LABELS[args.candidate_mode]
    print(f'[{len(done)}/{len(selected)}] E05{mode_label}，本批{len(pending)}，损坏行{damaged}', flush=True)
    for r in pending:
        try:
            result = experiment(r, out, torch, candidate_mode=args.candidate_mode)
            if args.compare_existing:
                compare_existing(r, result, out)
            result = seal_result(result,fingerprint)
            done.add(r['id'])
        except Exception as exc:
            result = {'id': r['id'], 'fingerprint': fingerprint, 'status': 'failed', 'error': repr(exc)}
        append_run_row(ledger, result); previous[r['id']] = result
        print(f'[{len(done)}/{len(selected)}] {r["id"]} {result["status"]}，'
              f'官方{result.get("baseline", {}).get("makespan")}→{result.get("real", {}).get("makespan")}，'
              f'合法候选{result.get("generation", {}).get("legal")}，'
              f'候选错误{len(result.get("errors", []))}，'
              f'旧切分器{result.get("existing", {}).get("real", {}).get("makespan")}', flush=True)
    valid = [previous[r['id']] for r in selected if r['id'] in done]
    summary = {'expected': len(selected), 'completed': len(valid),
        'wins': sum(r['real']['makespan'] < r['baseline']['makespan'] for r in valid),
        'losses': sum(r['real']['makespan'] > r['baseline']['makespan'] for r in valid),
        'official_count': sum(r['official_count'] for r in valid),
        'candidate_errors': sum(len(r['errors']) for r in valid)}
    if args.compare_existing:
        summary['wins_vs_existing'] = sum(r['real']['makespan'] < r['existing']['real']['makespan'] for r in valid)
        summary['losses_vs_existing'] = sum(r['real']['makespan'] > r['existing']['real']['makespan'] for r in valid)
    (out/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    if any(r.get('status') == 'failed' for r in previous.values()):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
