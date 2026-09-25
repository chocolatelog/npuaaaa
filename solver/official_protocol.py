"""官方评估内容指纹、不可覆盖的尝试及统一结果清单校验。"""
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ATTACHMENT = ROOT / '通用神经网络处理器下的多核调度问题附件'
SCENES = {'problem_1': 'A', 'problem_2': 'B', 'problem_3': 'C', 'singlecore': None}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def object_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def job_key(job):
    return f"{job['case']}_{job['kind']}_{job['plan_scene'] or 'single'}_N{job['N']}"


def compact_result(result):
    dm = result.get('data_movement_bytes', {})
    value = {'makespan': result.get('makespan'),
             'added_copy_bytes': dm.get('added_copy_bytes'),
             'scheduled_copy_bytes': dm.get('scheduled_copy_bytes'),
             'partition_added': dm.get('partition_added_copy_bytes'),
             'spill_added': dm.get('spill_added_copy_bytes')}
    for key in ('makespan', 'added_copy_bytes', 'scheduled_copy_bytes'):
        number = value[key]
        if (isinstance(number, bool) or not isinstance(number, (int, float))
                or not math.isfinite(number) or number < 0 or (key == 'makespan' and number == 0)):
            raise ValueError(f'官方结果数值非法: {key}={number}')
    if result.get('error'):
        raise ValueError('官方结果含错误')
    cache = result.get('cache_stats')
    if cache is not None:
        hit, miss = cache.get('hit_bytes', 0), cache.get('miss_bytes', 0)
        if any(not isinstance(x, (int, float)) or not math.isfinite(x) or x < 0 for x in (hit, miss)):
            raise ValueError('缓存字节非法')
        value.update(cache_hit_bytes=hit, cache_miss_bytes=miss,
                     cache_hit_rate=hit/(hit+miss) if hit+miss else 0.0)
    return value


def input_context(job, plans_dir, attachment):
    attachment = Path(attachment).resolve()
    graph = attachment/'data'/f"{job['case']}.json"
    config = attachment/'data/config.txt'
    scene = job['plan_scene']
    if job['kind'] not in SCENES or (job['kind'] != 'singlecore' and
            scene not in ({'B', 'C'} if job['kind'] == 'problem_3' else {SCENES[job['kind']]})):
        raise ValueError('问题与方案场景不匹配')
    plan = None if job['kind'] == 'singlecore' else Path(plans_dir).resolve()/f"{job['case']}_{scene}_N{job['N']}.json"
    if plan:
        value = json.loads(plan.read_text(encoding='utf-8'))
        if len(value['core_schedules']) != job['N']:
            raise ValueError('方案核数与任务不一致')
    code = {p.name: digest(p) for p in sorted((attachment/'code').glob('*.py'))}
    evaluator_hash = object_digest(code)
    context = {'protocol_version': 3, 'job': job, 'input_sha256': digest(graph),
               'config_sha256': digest(config), 'evaluator_sha256': evaluator_hash,
               'plan_sha256': digest(plan) if plan else None,
               'python': sys.version, 'platform': platform.platform(),
               'wrapper_sha256': digest(__file__), 'official_sources': code}
    return context, graph, config, plan


def verified_record(record):
    """验证历史记录与完整结果；源码后续变动不影响历史证据可读性。"""
    try:
        if record.get('status') != 'official_success':
            return False
        receipt = Path(record['receipt_path'])
        if json.loads(receipt.read_text(encoding='utf-8')) != record:
            return False
        if any(digest(path) != sha for path, sha in record['artifacts'].items()):
            return False
        result = json.loads(Path(record['result_path']).read_text(encoding='utf-8'))
        return compact_result(result) == record['real'] and object_digest(record['context']) == record['fingerprint']
    except (OSError, ValueError, KeyError, TypeError):
        return False


def evaluate_job(job, output_dir, plans_dir, attachment=ATTACHMENT, timeout=3600, runner=None):
    """不读取旧无指纹结果；成功复用，失败重试，每次尝试写入新目录。"""
    started = time.perf_counter()
    folder = Path(output_dir).resolve()
    record = {**job, 'job_id': job_key(job), 'record_id': uuid.uuid4().hex,
              'protocol_version': 3, 'updated_at': datetime.now().astimezone().isoformat(timespec='minutes')}
    try:
        context, graph, config, plan = input_context(job, plans_dir, attachment)
        fingerprint = object_digest(context)
        record.update(fingerprint=fingerprint, context=context,
                      **{k: context[k] for k in ('input_sha256', 'config_sha256', 'evaluator_sha256', 'plan_sha256')})
        cache_dir = folder/'attempts'/record['job_id']/fingerprint
        for receipt in sorted(cache_dir.glob('*/receipt.json'), reverse=True):
            try:
                old = json.loads(receipt.read_text(encoding='utf-8'))
                if old.get('fingerprint') == fingerprint and verified_record(old):
                    return old
            except (OSError, ValueError):
                continue
        attempt = cache_dir/record['record_id']
        attempt.mkdir(parents=True, exist_ok=False)
        result_path, trace_path, log_path = [attempt/name for name in ('result.json', 'trace.json', 'official.log')]
        script = 'singlecore_evaluate.py' if job['kind'] == 'singlecore' else f"multicore_cut_evaluate_{job['kind']}.py"
        cmd = [sys.executable, '-X', 'utf8', str(Path(attachment).resolve()/'code'/script), str(graph)]
        if plan:
            cmd.append(str(plan))
        cmd += ['--config', str(config), '-o', str(result_path), '--trace-output', str(trace_path), '--log-output', str(log_path)]
        execution = (runner or subprocess.run)(cmd, cwd=str(attachment), capture_output=True,
                                              text=True, encoding='utf-8', errors='replace', timeout=timeout)
        (attempt/'console.txt').write_text(execution.stdout+'\n'+execution.stderr, encoding='utf-8')
        if execution.returncode:
            raise ValueError(f'官方评估返回 {execution.returncode}: {execution.stderr[-1000:]}')
        after, _, _, _ = input_context(job, plans_dir, attachment)
        if after != context:
            raise ValueError('评估期间输入发生变更，结果不采纳')
        result = json.loads(result_path.read_text(encoding='utf-8'))
        record.update(status='official_success', real=compact_result(result), result_path=str(result_path),
                      artifacts={str(p): digest(p) for p in (result_path, trace_path, log_path)},
                      receipt_path=str(attempt/'receipt.json'), elapsed=time.perf_counter()-started)
        (attempt/'receipt.json').write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
    except Exception as exc:
        record.update(status='failed', error=repr(exc), elapsed=time.perf_counter()-started)
    return record


def read_ledger(path):
    latest, warnings = {}, []
    if Path(path).exists():
        for index, line in enumerate(Path(path).read_text(encoding='utf-8').splitlines(), 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                latest[record['job_id']] = record
            except (KeyError, ValueError, TypeError):
                warnings.append(index)
    return latest, warnings
