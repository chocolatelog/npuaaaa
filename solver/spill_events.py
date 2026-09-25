"""给定核内图和算子顺序，按下一次使用最晚优先规则计数溢出搬运。

只计数，不构造张量重命名和扩展调度图；每次调用的容量池独立。
需要输入已补齐边界搬运的局部图，不能用全图包围区间代替子图成员。
"""
from collections import defaultdict


def count_spill_events(graph, sequence, capacity, event_limit=0, tensor_limit=0):
    """event_limit 为每个存储池保留的最大诊断事件数；默认不保存轨迹。"""
    if isinstance(event_limit, bool) or not isinstance(event_limit, int) or event_limit < 0:
        raise ValueError('诊断事件数必须为非负整数')
    if isinstance(tensor_limit, bool) or not isinstance(tensor_limit, int) or tensor_limit < 0:
        raise ValueError('诊断张量数必须为非负整数')
    ops = {op['id']: op for op in graph['ops']}
    if len(sequence) != len(ops) or set(sequence) != set(ops):
        raise ValueError('算子顺序必须完整且不重复')
    tensors = {t['id']: t for t in graph['tensors']}
    steps = {op: i for i, op in enumerate(sequence)}
    uses, inputs, outputs = defaultdict(set), defaultdict(set), defaultdict(set)
    for edge in graph['edges']:
        src, dst = edge['source'], edge['target']
        if src in ops and dst in tensors:
            uses[dst].add(steps[src])
            outputs[src].add(dst)
        elif src in tensors and dst in ops:
            uses[src].add(steps[dst])
            inputs[dst].add(src)
    backing_candidates = defaultdict(set)
    for oid, op in ops.items():
        if op['op'] != 'COPY_IN':
            continue
        external = [t for t in inputs[oid] if tensors[t]['pos'] == 'DDR']
        if len(external) == 1:
            for tid in outputs[oid]:
                if tensors[tid]['pos'] != 'DDR':
                    backing_candidates[tid].add(external[0])
    backed = {t for t, candidates in backing_candidates.items() if len(candidates) == 1}
    events = [[] for _ in sequence]
    for tid, tensor in tensors.items():
        if tensor['pos'] == 'DDR' or tid not in uses:
            continue
        if tensor['pos'] not in capacity:
            raise ValueError(f'未知片上存储池: {tensor["pos"]}')
        ordered = sorted(uses[tid])
        for i, step in enumerate(ordered):
            events[step].append((tid, ordered[i + 1] if i + 1 < len(ordered) else None))
    active = {pool: {} for pool in capacity}
    resident = dict.fromkeys(capacity, 0)
    peak_alloc, peak_resident = dict(resident), dict(resident)
    read_bytes = write_bytes = evictions = 0
    by_pool = dict.fromkeys(capacity, 0)
    retained = {pool: [] for pool in capacity} if event_limit else None
    event_counts = dict.fromkeys(capacity, 0) if event_limit else None
    tensor_stats = {} if tensor_limit else None
    for step, accesses in enumerate(events):
        current = {tid for tid, _ in accesses}
        for tid, next_use in accesses:
            t = tensors[tid]
            pool = t['pos']
            if tid not in active[pool]:
                resident[pool] += t['size']
            active[pool][tid] = next_use
        for pool, cap in capacity.items():
            peak_alloc[pool] = max(peak_alloc[pool], resident[pool])
            while resident[pool] > cap:
                eligible = [tid for tid, nxt in active[pool].items()
                            if nxt is not None and tid not in current]
                if not eligible:
                    raise ValueError(f'当前算子不可腾挪: step={step}, pool={pool}')
                # 字典保持驻留插入次序，同下一次使用时按此顺序打破平局。
                victim = max(eligible, key=lambda tid: active[pool][tid])
                size = tensors[victim]['size']
                if tensor_limit:
                    summary = tensor_stats.setdefault(victim, {'tensor_id': victim, 'pool': pool,
                        'size': size, 'read_bytes': 0, 'write_bytes': 0, 'evictions': 0,
                        'reload_by_op': {}})
                    summary['read_bytes'] += size
                    summary['write_bytes'] += 0 if victim in backed else size
                    summary['evictions'] += 1
                    reload_op = sequence[active[pool][victim]]
                    summary['reload_by_op'][reload_op] = summary['reload_by_op'].get(reload_op, 0) + 1
                if event_limit:
                    event = {'pool': pool, 'step': step, 'op_id': sequence[step],
                        'tensor_id': victim, 'size': size,
                        'next_use_step': active[pool][victim],
                        'next_use_distance': active[pool][victim] - step,
                        'backed_before': victim in backed, 'read_bytes': size,
                        'write_bytes': 0 if victim in backed else size,
                        'resident_before_bytes': resident[pool],
                        'capacity_overage_bytes': resident[pool] - cap}
                    retained[pool].append(event)
                    retained[pool].sort(key=lambda e: (
                        -(e['read_bytes'] + e['write_bytes']), -e['next_use_distance'],
                        e['step'], e['tensor_id']))
                    if len(retained[pool]) > event_limit:
                        retained[pool].pop()
                    event_counts[pool] += 1
                read_bytes += size
                by_pool[pool] += size
                if victim not in backed:
                    write_bytes += size
                    by_pool[pool] += size
                    backed.add(victim)
                evictions += 1
                resident[pool] -= size
                del active[pool][victim]
            peak_resident[pool] = max(peak_resident[pool], resident[pool])
        # 输出分配和当前输入同时参与容量检查，之后才允许末次释放。
        for tid, next_use in accesses:
            if next_use is None:
                pool = tensors[tid]['pos']
                del active[pool][tid]
                resident[pool] -= tensors[tid]['size']
    result = {'spill_bytes': read_bytes + write_bytes, 'spill_read_bytes': read_bytes,
            'spill_write_bytes': write_bytes, 'eviction_count': evictions,
            'spill_by_pool': by_pool, 'peak_alloc_bytes': peak_alloc,
            'peak_resident_bytes': peak_resident}
    if event_limit:
        result['spill_events'] = retained
        result['spill_events_omitted'] = {pool: event_counts[pool] - len(retained[pool])
                                         for pool in capacity}
    if tensor_limit:
        consumer_steps = defaultdict(list)
        for oid, tids in inputs.items():
            for tid in tids:
                if tid in tensor_stats:
                    consumer_steps[tid].append(steps[oid])
        per_pool = {pool: [] for pool in capacity}
        for tid, row in tensor_stats.items():
            consumers = sorted(consumer_steps[tid])
            row.update(total_bytes=row['read_bytes']+row['write_bytes'],
                       consumer_steps=consumers,
                       first_consumer_step=consumers[0] if consumers else None,
                       last_consumer_step=consumers[-1] if consumers else None)
            per_pool[row['pool']].append(row)
        for rows in per_pool.values():
            rows.sort(key=lambda r: (-r['total_bytes'], -r['evictions'], r['tensor_id']))
        result['spill_tensors'] = {p: rows[:tensor_limit] for p,rows in per_pool.items()}
        result['spill_tensors_omitted'] = {p: max(0,len(rows)-tensor_limit) for p,rows in per_pool.items()}
        result['spill_tensor_omitted_bytes'] = {
            p: sum(r['total_bytes'] for r in rows[tensor_limit:]) for p,rows in per_pool.items()}
    return result
