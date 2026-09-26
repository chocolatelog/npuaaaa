"""完整方案候选组合：保留来源，补空核，官方评估后才允许替换。"""
import copy
import json
import math
from pathlib import Path

from official_protocol import object_digest
from scenario_contract import validate_plan


def lift_plan(plan, cores):
    old = len(plan['core_schedules'])
    if not 2 <= old <= cores <= 5:
        raise ValueError('只允许2～5核之间补空核，不允许截断核心')
    result = copy.deepcopy(plan)
    result['core_schedules'].extend([] for _ in range(cores-old))
    return result


def candidate_paths(case, scene, cores, directories, include_lower=False):
    seen = set()
    for directory in directories:
        for n in (range(2, cores+1) if include_lower else [cores]):
            path = Path(directory).resolve()/f'{case}_{scene}_N{n}.json'
            if path.exists() and path not in seen:
                seen.add(path)
                yield path, n


def load_candidates(case, scene, cores, directories, include_lower=False):
    for path, n in candidate_paths(case, scene, cores, directories, include_lower):
        source = dict(path=str(path), source_cores=n, target_cores=cores,
                      transformation='append_empty_cores' if n<cores else 'identity')
        try:
            plan = json.loads(path.read_text(encoding='utf-8'))
            if len(plan['core_schedules']) != n:
                raise ValueError('来源核数与方案不一致')
            yield dict(plan=lift_plan(plan, cores), source=source)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            yield dict(source=source, load_error=repr(exc))


def official_key(value):
    key = (value['makespan'], value['added_copy_bytes'])
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in key):
        raise ValueError('官方结果必须为有限数值')
    if key[0] <= 0 or key[1] < 0:
        raise ValueError('官方结果范围非法')
    return key


def select_official_portfolio(graph, parent, baseline, scene, candidates, evaluate):
    validate_plan(graph, parent, scene)
    official_key(baseline)
    chosen, best = parent, baseline
    seen = {object_digest(parent)}
    audit = dict(status='official_portfolio_only', baseline_official=baseline,
                 errors=[], candidates=[], duplicate_count=0, selected_source='parent')
    for item in candidates:
        source = item['source']
        try:
            if item.get('load_error'):
                raise ValueError(item['load_error'])
            plan = item['plan']
            if len(plan['core_schedules']) != len(parent['core_schedules']):
                raise ValueError('目标核数与父方案不一致')
            validate_plan(graph, plan, scene)
            key = object_digest(plan)
            if key in seen:
                audit['duplicate_count'] += 1
                audit['candidates'].append(dict(source=source,plan_id=key,status='duplicate'))
                continue
            seen.add(key)
            truth = evaluate(graph, plan, scene)
            accepted = official_key(truth) < official_key(best)
            audit['candidates'].append(dict(source=source, plan_id=key,
                status='official_success', official=truth, accepted=accepted))
            if accepted:
                chosen, best = plan, truth
                audit['selected_source'] = source
        except (ValueError, RuntimeError, OSError, KeyError, TypeError) as exc:
            audit['errors'].append(dict(source=source, error=repr(exc)))
    audit['final_official'] = best
    return chosen, best, audit
