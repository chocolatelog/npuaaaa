"""问题三的只读 FIFO 缓存事件重放。

该模块服务于候选排序和语义审计，不替代附件中的官方评测器。输入可以是
``issue_time`` 形式的访问事件，也可以是官方观测器导出的 ``cache_events``。
所有访问都在完成后才插入 FIFO；在途的同一张量不会被错误地当作命中。
"""
from __future__ import annotations

import heapq
from collections import OrderedDict


def _number(value, name):
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise ValueError(f'{name}必须是有限数')
    if value < 0 or value != value or value in (float('inf'), float('-inf')):
        raise ValueError(f'{name}必须是非负有限数')
    return value


def replay_fifo_cache(events, capacity, ddr_bandwidth=60.0,
                      cache_bandwidth=250.0):
    """重放访问事件并返回逐事件命中、完成和淘汰信息。

    ``events`` 每项至少包含 ``tensor_id``、``size``（或 ``size_bytes``）和
    ``issue_time``（或 ``time``）；可选 ``core_id``、``op_id``、``critical``。
    事件同刻按核心、操作编号和输入顺序稳定排序。完成事件先于同刻新发射。
    """
    capacity = int(_number(capacity, '缓存容量'))
    if ddr_bandwidth <= 0 or cache_bandwidth <= 0:
        raise ValueError('带宽必须为正数')
    normalized = []
    for index, raw in enumerate(events or ()):
        row = dict(raw)
        tensor_id = row.get('tensor_id', row.get('logical_tid'))
        if tensor_id is None:
            raise ValueError('缓存事件缺少 tensor_id')
        size = row.get('size', row.get('size_bytes'))
        size = int(_number(size, '张量大小'))
        issue = row.get('issue_time', row.get('time', row.get('start', 0)))
        issue = _number(issue, '发射时间')
        normalized.append({
            'index': index,
            'tensor_id': int(tensor_id),
            'size': size,
            'issue_time': issue,
            'core_id': row.get('core_id', row.get('core')),
            'op_id': row.get('op_id', row.get('op')),
            'critical': bool(row.get('critical', False)),
        })
    normalized.sort(key=lambda r: (r['issue_time'],
                                  -1 if r['core_id'] is None else int(r['core_id']),
                                  -1 if r['op_id'] is None else int(r['op_id']),
                                  r['index']))

    entries = OrderedDict()  # tensor_id -> size
    inflight = {}             # tensor_id -> set(access index)
    completions = []          # (completion time, access index)
    by_index = {row['index']: row for row in normalized}
    result = {}
    used = 0
    evictions = 0

    def complete_until(now):
        nonlocal used, evictions
        while completions and completions[0][0] <= now + 1e-9:
            completion, access_index = heapq.heappop(completions)
            access = by_index[access_index]
            tensor_id = access['tensor_id']
            inflight.get(tensor_id, set()).discard(access_index)
            item = result[access_index]
            item['completion_time'] = completion
            if tensor_id in entries:
                item['insert_time'] = None
                continue
            item['insert_time'] = completion
            if access['size'] <= capacity and access['size'] > 0:
                evicted = []
                while entries and used + access['size'] > capacity:
                    old_id, old_size = entries.popitem(last=False)
                    used -= old_size
                    evicted.append(old_id)
                    evictions += 1
                if access['size'] <= capacity:
                    entries[tensor_id] = access['size']
                    used += access['size']
                item['evicted_ids'] = evicted
            else:
                item['evicted_ids'] = []

    for access in normalized:
        complete_until(access['issue_time'])
        tensor_id = access['tensor_id']
        hit = tensor_id in entries and access['size'] <= capacity
        path = 'cache' if hit else 'ddr'
        bandwidth = cache_bandwidth if hit else ddr_bandwidth
        service = access['size'] / bandwidth if access['size'] else 0.0
        completion = access['issue_time'] + service
        item = {
            'event_index': access['index'],
            'tensor_id': tensor_id,
            'size': access['size'],
            'core_id': access['core_id'],
            'op_id': access['op_id'],
            'issue_time': access['issue_time'],
            'path': path,
            'hit': hit,
            'service_cycles': service,
            'completion_time': None,
            'insert_time': None,
            'evicted_ids': [],
            'critical': access['critical'],
        }
        result[access['index']] = item
        # 同一张量的在途读取也必须是 miss，但各请求仍各自完成并尝试插入。
        inflight.setdefault(tensor_id, set()).add(access['index'])
        heapq.heappush(completions, (completion, access['index']))
    complete_until(float('inf'))

    ordered = [result[row['index']] for row in normalized]
    hits = sum(row['size'] for row in ordered if row['hit'])
    misses = sum(row['size'] for row in ordered if not row['hit'])
    return {
        'events': ordered,
        'capacity_bytes': capacity,
        'hit_bytes': hits,
        'miss_bytes': misses,
        'hit_count': sum(1 for row in ordered if row['hit']),
        'miss_count': sum(1 for row in ordered if not row['hit']),
        'hit_rate': hits / max(1, hits + misses),
        'evictions': evictions,
        'final_entries': [{'tensor_id': tid, 'size': size}
                          for tid, size in entries.items()],
        'used_bytes_final': used,
        'mode': 'deterministic_proxy',
        'timing_semantics': 'independent_service_approximation',
    }


def replay_official_events(cache_events, capacity, ddr_bandwidth=60.0,
                           cache_bandwidth=250.0):
    """把官方 ``miss/hit`` 访问记录转换为代理重放输入。

    官方的 ``insert`` 记录不是新的访问，因此会被忽略；它只用于语义审计。
    """
    accesses = []
    for row in cache_events or ():
        if row.get('event') not in {'miss', 'hit'}:
            continue
        accesses.append({
            'tensor_id': row.get('tensor_id'),
            'size_bytes': row.get('size_bytes', row.get('size', 0)),
            'issue_time': row.get('time', 0),
            'core_id': row.get('core_id'),
            'op_id': row.get('op_id'),
        })
    return replay_fifo_cache(accesses, capacity, ddr_bandwidth, cache_bandwidth)


def audit_official_cache(result):
    """按官方已观测完成时刻审计 FIFO；不是预测新的执行时间。"""
    events = result['cache_events']
    timeline = {(c['core_id'], op['op_id']): op
                for c in result['per_core_timeline'] for op in c['ops']}
    capacity = result['cache_capacity_bytes']
    queue = []
    for index, e in enumerate(events):
        if e['event'] not in ('hit', 'miss'):
            continue
        op = timeline[e['core_id'], e['op_id']]
        if op['start'] != e['time'] or op['end'] < op['start']:
            raise ValueError('缓存事件和官方操作时间线不一致')
        queue.append((e['time'], 1, index, e))
        queue.append((op['end'], 0, e['core_id'], e))
    # 完成先于发射；同刻完成按官方核心遍历顺序；访问保留观测顺序。
    queue.sort(key=lambda x: x[:3])
    entries = OrderedDict(); used = hit = miss = 0
    inserts = []; mismatches = []
    for now, phase, _, e in queue:
        tid, size = e['tensor_id'], e['size_bytes']
        if phase:
            predicted = 'hit' if tid in entries else 'miss'
            if predicted != e['event']:
                mismatches.append({'time':now,'tensor_id':tid,'expected':e['event'],'actual':predicted})
            if predicted == 'hit':hit += size
            else:miss += size
        elif tid not in entries and size <= capacity:
            evicted = []
            while entries and used + size > capacity:
                old, old_size = entries.popitem(last=False); used -= old_size; evicted.append(old)
            entries[tid] = size; used += size
            inserts.append({'time':now,'event':'insert','tensor_id':tid,'size_bytes':size,
                            'used_bytes':used,'evicted_tensor_ids':evicted,
                            'core_id':e['core_id'],'op_id':e['op_id']})
    final = [{'tensor_id':k,'size_bytes':v} for k,v in entries.items()]
    checks = {'accesses':not mismatches,
              'inserts':inserts == [e for e in events if e['event']=='insert'],
              'hit_bytes':hit == result['cache_stats']['hit_bytes'],
              'miss_bytes':miss == result['cache_stats']['miss_bytes'],
              'final_entries':final == result['cache_final_entries'],
              'capacity':used == result['cache_used_bytes_final'] and used <= capacity}
    return {'consistent':all(checks.values()),'checks':checks,'mismatches':mismatches,
            'mode':'observed_official_timeline_audit','access_count':sum(e['event'] in ('hit','miss') for e in events)}
