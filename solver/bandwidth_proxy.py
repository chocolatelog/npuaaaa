"""确定性的共享带宽事件代理。

该模块只用于候选排序和诊断。它不改变官方评测器的搬运或硬件语义。
请求按流体公平共享带宽，完成事件优先于同刻新请求，便于复现边界时序。
"""
from __future__ import annotations

import heapq


def _sweep_approx(normalized, bandwidth):
    """用理想服务区间估计大批请求的共享带宽竞争。

    该路径只负责候选排序，显式标记为近似；小批请求仍走逐事件精确
    重放。理想区间的活动面积给出平均并发度，避免对每个完成事件反复
    扫描活动集合。
    """
    intervals = []
    endpoints = []
    for row in normalized:
        ideal = row['size'] / bandwidth if row['size'] else 0.0
        end = row['start'] + ideal
        intervals.append((row, ideal, end))
        endpoints.append((row['start'], 1))
        endpoints.append((end, -1))
    endpoints.sort(key=lambda item: (item[0], item[1]))
    active = 0
    last = endpoints[0][0] if endpoints else 0.0
    area = 0.0
    max_concurrency = 0
    for point, delta in endpoints:
        if point > last:
            area += active * (point - last)
            last = point
        active += delta
        max_concurrency = max(max_concurrency, active)
    ideal_makespan = max((end for _row, _ideal, end in intervals), default=0.0)
    average_active = area / ideal_makespan if ideal_makespan > 0 else 1.0
    factor = max(1.0, average_active)
    rows = []
    per_core = {}
    for row, ideal, _end in intervals:
        contention = ideal * (factor - 1.0)
        completion = row['start'] + ideal * factor
        item = {
            'request_id': row['request_id'],
            'core': row['core'],
            'critical': row['critical'],
            'start': row['start'],
            'completion': completion,
            'service_cycles': completion - row['start'],
            'contention_cycles': contention,
            'size': row['size'],
        }
        rows.append(item)
        if row['core'] is not None:
            key = str(row['core'])
            per_core[key] = per_core.get(key, 0.0) + contention
    return {
        'events': rows,
        'makespan': max((row['completion'] for row in rows), default=0.0),
        'total_contention_cycles': sum(row['contention_cycles'] for row in rows),
        'critical_path_wait': sum(row['contention_cycles'] for row in rows
                                  if row['critical']),
        'per_core_contention_cycles': per_core,
        'max_concurrency': max_concurrency,
        'request_count': len(rows),
        'mode': 'sweep_approx',
    }


def simulate_bandwidth(events, bandwidth=60.0, exact_limit=512):
    if bandwidth <= 0:
        raise ValueError('带宽必须为正数')
    if exact_limit is not None and int(exact_limit) < 1:
        raise ValueError('精确请求上限必须为正数或 None')
    normalized = []
    seen_request_ids = set()
    for event in events or ():
        row = dict(event)
        request_id = str(row.get('request_id'))
        size = float(row.get('size', 0))
        start = float(row.get('start', 0))
        if not request_id or size < 0 or start < 0:
            raise ValueError('带宽请求字段非法')
        if request_id in seen_request_ids:
            raise ValueError('带宽请求编号重复')
        seen_request_ids.add(request_id)
        normalized.append({
            'request_id': request_id,
            'start': start,
            'size': size,
            'core': row.get('core'),
            'critical': bool(row.get('critical', False)),
        })
    normalized.sort(key=lambda row: (row['start'], row['request_id']))
    if exact_limit is not None and len(normalized) > int(exact_limit):
        return _sweep_approx(normalized, bandwidth)
    active = {}
    completion_heap = []
    completed = {}
    index = 0
    now = 0.0
    # All active requests receive the same fluid service rate.  Store each
    # request's initial work level and advance one shared service cursor;
    # relative ordering therefore stays in a heap without rescanning active
    # requests at every event.
    service_level = 0.0
    max_concurrency = 0
    while index < len(normalized) or active:
        if not active:
            now = max(now, normalized[index]['start'])
            while index < len(normalized) and normalized[index]['start'] <= now:
                row = normalized[index]
                active[row['request_id']] = row
                heapq.heappush(completion_heap,
                               (service_level + row['size'],
                                row['request_id']))
                index += 1
            max_concurrency = max(max_concurrency, len(active))
            continue
        rate = bandwidth / len(active)
        next_key = completion_heap[0][0]
        next_completion = now + max(0.0, next_key - service_level) / rate
        next_arrival = normalized[index]['start'] if index < len(normalized) else None
        target = next_completion if next_arrival is None else min(next_completion, next_arrival)
        elapsed = max(0.0, target - now)
        service_level += rate * elapsed
        now = target
        # 完成事件先退役，保证同刻新请求不分走已经释放的带宽。
        while completion_heap and completion_heap[0][0] <= service_level + 1e-9:
            _key, rid = heapq.heappop(completion_heap)
            row = active.pop(rid, None)
            if row is None:
                continue
            ideal = row['size'] / bandwidth if row['size'] else 0.0
            completed[rid] = {
                'request_id': rid,
                'core': row['core'],
                'critical': row['critical'],
                'start': row['start'],
                'completion': now,
                'service_cycles': now - row['start'],
                'contention_cycles': max(0.0, now - row['start'] - ideal),
                'size': row['size'],
            }
        while index < len(normalized) and normalized[index]['start'] <= now + 1e-9:
            row = normalized[index]
            active[row['request_id']] = row
            heapq.heappush(completion_heap,
                           (service_level + row['size'],
                            row['request_id']))
            index += 1
        max_concurrency = max(max_concurrency, len(active))
    rows = [completed[row['request_id']] for row in normalized]
    per_core = {}
    for row in rows:
        if row['core'] is not None:
            key = str(row['core'])
            per_core[key] = per_core.get(key, 0.0) + row['contention_cycles']
    return {
        'events': rows,
        'makespan': max((row['completion'] for row in rows), default=0.0),
        'total_contention_cycles': sum(row['contention_cycles'] for row in rows),
        'critical_path_wait': sum(row['contention_cycles'] for row in rows
                                  if row['critical']),
        'per_core_contention_cycles': per_core,
        'max_concurrency': max_concurrency,
        'request_count': len(rows),
        'mode': 'exact',
    }
