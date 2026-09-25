"""共同预算的在线末端组件；现场生成，完整旧方案保底，逐候选恢复。"""
from contextlib import contextmanager
from copy import deepcopy
from functools import lru_cache
import json
import os
from pathlib import Path
import time
import uuid

from evidence_index import digest, file_sha, plan_sha
from scene_a_event import SceneAEventModel, derive_multicore_plan
from evaluation_validation import validate_task_order
from task_order_search import replay_tasks, search_orders

ROOT = Path(__file__).resolve().parents[1]
VERSION = 'online-shared-v1'


class Checkpoint:
    """同一目录只接收同指纹的不可变阶段；异常尾文件保留，不覆盖有效阶段。"""
    def __init__(self, directory, fingerprint):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.fingerprint = fingerprint

    def read(self, name):
        path = self.directory / (name + '.json')
        if not path.exists():
            return None
        record = json.loads(path.read_text(encoding='utf-8'))
        expected = record.pop('binding', None)
        if record.get('fingerprint') != self.fingerprint or digest(record) != expected:
            raise ValueError('检查点指纹或内容绑定失效：' + name)
        return record['value']

    def write(self, name, value):
        value = json.loads(json.dumps(value, ensure_ascii=False))
        prior = self.read(name)
        if prior is not None:
            if prior != value:
                raise ValueError('禁止覆盖已冻结阶段：' + name)
            return prior
        record = {'fingerprint': self.fingerprint, 'value': value}
        record['binding'] = digest(record)
        target = self.directory / (name + '.json')
        temporary = target.with_name(target.name + '.' + uuid.uuid4().hex + '.partial')
        with temporary.open('w', encoding='utf-8') as stream:
            json.dump(record, stream, ensure_ascii=False)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.rename(target)
        return value


@contextmanager
def process_lock(path):
    """锁随进程退出释放；不删除锁文件。"""
    import msvcrt
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as stream:
        if stream.tell() == 0:
            stream.write(b'0')
            stream.flush()
        stream.seek(0)
        try:
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            raise RuntimeError('实验目录已有进程运行，拒绝重复启动') from exc
        try:
            yield
        finally:
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)


class SharedLocalModel(SceneAEventModel):
    """只共享完整成员编号对应的局部统计；核序改变必须重新全图重放。"""
    def __init__(self, graph):
        super().__init__(graph)
        self.cache = {}
        self.stats = {'models': 0, 'hits': 0, 'local_seconds': 0., 'replay_seconds': 0.}

    def evaluate(self, plan):
        key = tuple(sorted(plan['node_to_subgraph'].items()))
        view = derive_multicore_plan(self.graph, plan)
        validate_task_order(view)
        if key not in self.cache:
            start = time.perf_counter()
            estimate = super().evaluate(plan)
            self.stats['local_seconds'] += time.perf_counter() - start
            self.cache[key] = estimate
            self.stats['models'] += 1
        else:
            self.stats['hits'] += 1
        estimate = self.cache[key]
        start = time.perf_counter()
        timing = replay_tasks({t: r['duration'] for t, r in estimate['tasks'].items()},
                              view['subgraph_preds'], plan['core_schedules'],
                              same_wait=100, cross_wait=1000,
                              bandwidth_floor=estimate['total_copy_bytes'] / 60)
        self.stats['replay_seconds'] += time.perf_counter() - start
        if timing is None:
            raise ValueError('局部共享缓存遇到非法完整核序')
        return {**estimate, 'makespan': timing['makespan']}


def generate_region(graph, model, plan, local, progress=None):
    """区域及父方案顺序提案全部现场生成，不读取历史菜单或候选真值。"""
    from shared_candidate_pool import make_candidate
    from solution import Sol
    from region_structure import build_structure_candidates
    from region_group_refine import select_joint_candidates
    from batch_features import scalar_features
    start = time.perf_counter()
    n = len(plan['core_schedules'])
    base = make_candidate(model, plan, 1, 'baseline')
    estimate = local.evaluate(plan)
    menu_policy = 'seed_combinations' if n == 2 else 'diverse'
    rows, structure = build_structure_candidates(
        graph, model, Sol(base['assignment'], base['owners']), plan,
        {'tasks': estimate['tasks']}, core_policy='all', menu_policy=menu_policy)
    for row in rows:
        row['features'] = scalar_features(model, row['assignment'], row['owners'], n)
    parents, screening = select_joint_candidates(rows, limit=12, pool_limit=20)
    proposals = {}
    parent_ids = {plan_sha(model.plan_from(r['assignment'], r['orders'])) for r in parents}
    searches = []
    for i, row in enumerate(parents):
        parent = model.plan_from(row['assignment'], row['orders'])
        prediction = local.evaluate(parent)
        view = derive_multicore_plan(graph, parent)
        found, stats = search_orders(
            {t: r['duration'] for t, r in prediction['tasks'].items()}, view['subgraph_preds'],
            parent['core_schedules'], same_wait=100, cross_wait=1000,
            bandwidth_floor=prediction['total_copy_bytes']/60, beam_width=4,
            rounds=2, max_evals=400, seconds=None, max_sources=12, keep=2, enable_swaps=True)
        searches.append(stats)
        for value in found:
            candidate = {**parent, 'core_schedules': value['orders']}
            pid = plan_sha(candidate)
            if pid in parent_ids:
                continue
            validate_task_order(derive_multicore_plan(graph, candidate))
            origin = {'parent_plan_id': plan_sha(parent), 'neighborhood': value['origin']}
            if pid in proposals:
                proposals[pid]['origins'].append(origin)
            else:
                proposals[pid] = {'plan': candidate, 'coarse': value['makespan'], 'origins': [origin]}
        if progress:
            progress('区域父方案顺序', i+1, len(parents))
    return {'candidate_features': rows}, {'rows': list(proposals.values())}, {
        'seconds': time.perf_counter()-start, 'structure': structure, 'screening': screening,
        'menu_policy': menu_policy, 'parents': len(parents), 'searches': searches,
        'resource_calls': 0, 'background_model': 'local_approximation'}


def generate_online(graph, model, plan, torch, checkpoint=None, progress=None):
    from shared_candidate_pool import generate_pools
    from shared_budget import unify_scores, select_shared
    start = time.perf_counter()
    local = SharedLocalModel(graph)
    regional = checkpoint.read('region') if checkpoint else None
    if regional is None:
        menu, orders, region_audit = generate_region(graph, model, plan, local, progress)
        regional = {'menu': menu, 'orders': orders, 'audit': region_audit}
        if checkpoint:
            regional = checkpoint.write('region', regional)
    generated = checkpoint.read('pools') if checkpoint else None
    if generated is None:
        pools, audit = generate_pools(graph, model, plan, regional['menu'], regional['orders'],
            torch, local=local, progress=(lambda i, n: progress('切分候选', i, n)) if progress else None)
        audit['scope'] = '三类候选现场生成；无历史候选成绩输入'
        generated = {'pools': pools, 'audit': audit}
        if checkpoint:
            generated = checkpoint.write('pools', generated)
    pools, unified = unify_scores(graph, generated['pools'], local=local,
        progress=(lambda i, n, k: progress('统一评分', i, n)) if progress else None)
    before = time.perf_counter()
    selected, selection = select_shared(pools, tie_break='total_copy')
    selection_seconds = time.perf_counter()-before
    audit = {'version': VERSION, 'region': regional['audit'], 'generation': generated['audit'],
             'unified': unified, 'selection': selection, 'local_cache': dict(local.stats),
             'cumulative_generation_seconds': regional['audit']['seconds'] +
                 generated['audit']['seconds'] + unified['seconds'] + selection_seconds,
             'active_seconds': time.perf_counter()-start,
             'scope': '在线局部近似背景，与历史真实背景菜单不是相同输入实验'}
    return selected, audit


def history_paths():
    from audit_evidence_index import evidence_paths
    paths = evidence_paths(ROOT) + [ROOT/'results'/d/name for d, name in (
        ('p1_e08_menu_order_r01', 'menu_order_pairs.jsonl'),
        ('p1_e09_shared_r01', 'shared_pairs.jsonl'),
        ('p1_e09_unified_rank_r01', 'shared_pairs.jsonl'),
        ('p1_e09_total_tie_r01', 'shared_pairs.jsonl'))]
    return [p for p in paths if p.exists()]


def history_files():
    paths = history_paths()
    files = []
    for path in paths:
        files += [path, path.with_suffix('.manifest.json')]
        archive = path.parent/'frozen_source/source.zip'
        # 某些旧分支清单只有绑定指标；load_evidence 会按证据种类审查必须的快照。
        if archive.exists():
            files.append(archive)
    for directory, name in [('p1_e03_r02', 'order_pairs.jsonl'),
                            ('p1_e06_ddr_r01', 'resource_replay.jsonl')]:
        path = ROOT/'results'/directory/name
        if path.exists():
            files += [path, path.with_suffix('.manifest.json')]
            archive = path.parent/'frozen_source/source.zip'
            if archive.exists():
                files.append(archive)
    return files


@lru_cache(maxsize=1)
def historical_index():
    from evidence_index import load_evidence
    from baseline_evidence import add_baselines
    paths = history_paths()
    if not paths:
        return None
    index = load_evidence(ROOT, paths)
    baseline_path = ROOT/'results/p1_e06_ddr_r01/resource_replay.jsonl'
    if baseline_path.exists():
        ids = [json.loads(line)['id'] for line in baseline_path.read_text(encoding='utf-8').splitlines() if line.strip()]
        add_baselines(index, ROOT, ids)
    index.require_runtime()
    return index


def final_metrics(model, result):
    from pipeline import _proxy_for_plan
    out = deepcopy(result)
    mk, added, info = _proxy_for_plan(model, out['plan'], 'A', len(out['plan']['core_schedules']))
    out['est'] = [mk, added]
    log = out['log']
    log['selected_plan_id'] = plan_sha(out['plan'])
    log['final_est'] = list(out['est'])
    for name, key in [('partition_raw_bytes', 'partition_raw_bytes'),
                      ('partition_added_bytes', 'partition_added_bytes'),
                      ('spill_added_bytes', 'spill_bytes'), ('total_added_bytes', 'total_added_bytes')]:
        log['proxy_'+name] = float(info.get(key, added if key == 'total_added_bytes' else 0))
    log['proxy_memory_overage'] = info.get('memory_overage', {})
    log['proxy_peak_memory_bytes'] = info.get('peak_memory_bytes', {})
    return out


def refine_shared(graph, model, baseline, checkpoint, torch, *, case=None, graph_sha=None,
                  index=None, resource=None, official=None, progress=None):
    from run_all import valid_official
    from pipeline import real_evaluate
    checkpoint.write('input', {'plan_id': plan_sha(baseline['plan']), 'real': baseline['real'],
                               'graph': digest(graph), 'version': VERSION})
    completed = checkpoint.read('complete')
    if completed is not None:
        return completed
    start = time.perf_counter()
    out = deepcopy(baseline)
    audit = {'version': VERSION, 'baseline_plan_id': plan_sha(baseline['plan']),
             'baseline_real': baseline['real'], 'baseline_elapsed': baseline['elapsed'],
             'resource_limit': 6, 'official_limit': 1, 'evaluated': [], 'checks': [],
             'accepted': False, 'errors': []}
    if not valid_official(baseline['real']):
        raise ValueError('在线共同预算必须有原官方保底')
    n = len(baseline['plan']['core_schedules'])
    official = official or real_evaluate
    evaluator = None
    def lookup(pid, kind):
        if index is None:
            return None
        return index.lookup_for_evaluation(case, 'A', n, graph_sha, pid, kind)
    try:
        frozen = checkpoint.read('generation')
        if frozen is None:
            selected, details = generate_online(graph, model, baseline['plan'], torch,
                                               checkpoint=checkpoint, progress=progress)
            frozen = checkpoint.write('generation', {'selected': selected, 'audit': details})
        audit['generation'] = frozen['audit']
        selected = frozen['selected']
        if len(selected) > 6 or len({r['plan_id'] for r in selected}) != len(selected):
            raise ValueError('共同预算超限或完整计划重复')
        # 冻结名单后才能查询历史真值。
        for i, candidate in enumerate(selected):
            pid, plan = candidate['plan_id'], candidate['plan']
            if plan_sha(plan) != pid:
                raise ValueError('候选计划摘要失效')
            validate_task_order(derive_multicore_plan(graph, plan))
            row = checkpoint.read('resource_'+pid)
            if row is None:
                known = lookup(pid, 'resource') or lookup(pid, 'official')
                before = time.perf_counter()
                if known:
                    metrics = known['metrics']
                elif resource:
                    metrics = resource(graph, plan)
                else:
                    from scene_a_replay import build_resource_evaluator
                    from local_template_cache import TemplateCache
                    from lookahead_place import compact_result
                    if evaluator is None:
                        evaluator = build_resource_evaluator(unified=True, template_cache=TemplateCache())
                    raw = evaluator(graph, plan, 60, {'L1': 524288, 'UB': 131072}, 1000, 100)
                    metrics = compact_result(raw, plan['core_schedules'])['real']
                if not valid_official(metrics):
                    raise ValueError('资源筛选指标不完整')
                if metrics['partition_added'] != max(0, candidate['features']['boundary_bytes']-model.original_copy_bytes):
                    raise ValueError('资源边界与显卡特征不一致')
                row = checkpoint.write('resource_'+pid, {'plan_id': pid, 'resource': metrics,
                    'kind': known['kind'] if known else 'resource', 'reused': known is not None,
                    'sources': known['sources'] if known else [], 'seconds': time.perf_counter()-before,
                    'coarse': candidate['coarse'], 'family': candidate['allocated_family']})
            audit['evaluated'].append(row)
            if progress:
                progress('全图资源复核', i+1, len(selected))
        rank = lambda r: (r['makespan'], r['added_copy_bytes'])
        ranked = sorted(audit['evaluated'], key=lambda r: (*rank(r['resource']), r['plan_id']))
        improving = [r for r in ranked if rank(r['resource']) < rank(baseline['real'])][:1]
        for row in improving:
            pid = row['plan_id']
            plan = next(r['plan'] for r in selected if r['plan_id'] == pid)
            check = checkpoint.read('official_'+pid)
            if check is None:
                known = lookup(pid, 'official')
                before = time.perf_counter()
                try:
                    truth = known['metrics'] if known else official(graph, plan, 'A')
                except Exception as exc:
                    truth = {'error': repr(exc)}
                check = checkpoint.write('official_'+pid, {'plan_id': pid, 'official': truth,
                    'matches': valid_official(truth) and truth == row['resource'],
                    'reused': known is not None, 'seconds': time.perf_counter()-before,
                    'sources': known['sources'] if known else []})
            audit['checks'].append(check)
            if check['matches'] and rank(check['official']) < rank(baseline['real']):
                out['plan'], out['real'] = plan, check['official']
                audit['accepted'] = True
            else:
                audit['errors'].append('新增原官方确认失败或与资源结果不一致，保留原方案')
        out = final_metrics(model, out)
    except Exception as exc:
        out = deepcopy(baseline)
        audit['accepted'] = False
        audit['errors'].append(repr(exc))
        out['log']['final_est'] = list(out['est'])
        out['log']['selected_plan_id'] = plan_sha(out['plan'])
    audit['selected_plan_id'] = plan_sha(out['plan'])
    audit['active_seconds'] = time.perf_counter()-start
    audit['resource_new'] = sum(not r['reused'] for r in audit['evaluated'])
    audit['official_new'] = sum(not r['reused'] for r in audit['checks'])
    audit['status'] = 'fallback_error' if audit['errors'] else 'completed'
    audit['accounted_stage_seconds'] = audit.get('generation', {}).get('cumulative_generation_seconds', 0.) + sum(
        r['seconds'] for r in audit['evaluated'] + audit['checks'])
    out['log']['shared_budget'] = audit
    for check in audit['checks']:
        out['log'].setdefault('official_candidates', []).append({
            'plan_id': check['plan_id'], 'source': 'online_shared', 'official': check['official'],
            'evaluation_source': 'original_official', 'reused': check['reused']})
    # 恢复时已完成阶段的成本不归零；当前进程实际墙钟另记。
    out['elapsed'] = baseline['elapsed'] + max(audit['active_seconds'], audit['accounted_stage_seconds'])
    return checkpoint.write('complete', out)
