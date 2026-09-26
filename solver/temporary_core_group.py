"""临时核心协作组的独立合法性与释放时刻模拟。"""
from __future__ import annotations


def simulate_temporary_core_group(tasks, core_set, orders, external=None):
    cores = sorted({int(core) for core in core_set})
    if not cores:
        return {'legal': False, 'reason': 'empty_core_set'}
    assignments = {}
    for core, row in orders.items():
        core = int(core)
        if core not in cores:
            return {'legal': False, 'reason': 'core_outside_group'}
        for task_id in row:
            if task_id not in tasks:
                return {'legal': False, 'reason': 'unknown_task'}
            if task_id in assignments:
                return {'legal': False, 'reason': 'duplicate_task_assignment'}
            assignments[task_id] = core
    group_task_ids = set(assignments)
    task_ids = set(tasks)
    for task_id in group_task_ids:
        task = tasks[task_id]
        if float(task.get('duration', 0)) < 0:
            return {'legal': False, 'reason': 'negative_duration'}
        for pred in task.get('preds', ()):
            if pred not in task_ids:
                return {'legal': False, 'reason': 'unknown_predecessor'}
    positions = {task_id: index for core, row in orders.items()
                 for index, task_id in enumerate(row)}
    remaining = set(group_task_ids)
    finish = {}
    core_ready = {core: 0.0 for core in cores}
    schedule = []
    while remaining:
        choices = []
        for core in cores:
            row = list(orders.get(core, ()))
            for task_id in row:
                if task_id not in remaining:
                    continue
                preds = tasks[task_id].get('preds', ())
                if any(pred in group_task_ids and pred not in finish for pred in preds):
                    continue
                start = max([core_ready[core]] +
                            [finish[pred] for pred in preds if pred in group_task_ids])
                choices.append((start, core, positions[task_id], task_id))
                break
        if not choices:
            return {'legal': False, 'reason': 'dependency_cycle_or_blocked_order'}
        start, core, _position, task_id = min(choices)
        end = start + float(tasks[task_id].get('duration', 0))
        finish[task_id] = end
        core_ready[core] = end
        remaining.remove(task_id)
        schedule.append({'task_id': task_id, 'core': core,
                         'start': start, 'completion': end})
    external_delays = {}
    for row in external or ():
        task_id = str(row['task_id'])
        core = int(row['core'])
        ready = float(row.get('ready_time', 0))
        duration = float(row.get('duration', 0))
        group_release = core_ready.get(core, 0.0)
        external_delays[task_id] = max(0.0, max(ready, group_release) - ready)
    return {
        'legal': True,
        'reason': None,
        'schedule': schedule,
        'release_times': {core: core_ready[core] for core in cores},
        'external_delays': external_delays,
        'group_finish': max(finish.values(), default=0.0),
        'group_width': len(cores),
    }
