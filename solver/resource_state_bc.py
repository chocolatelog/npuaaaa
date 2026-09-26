"""B/C候选的轻量资源表。

资源表只用于候选生成和诊断。官方展开器仍是唯一真值来源；这里显式区分
片上池的占用、外存搬运事件和 C 的 FIFO 逻辑缓存状态，避免把命中率当作
总耗时的替代指标。
"""
from __future__ import annotations

from collections import OrderedDict, defaultdict
from copy import deepcopy
import heapq
import math


class FifoCache:
    """按逻辑张量编号管理的先进先出缓存，命中不刷新队列。"""

    def __init__(self, capacity=1048576):
        if isinstance(capacity, bool) or int(capacity) < 0:
            raise ValueError('缓存容量必须为非负整数')
        self.capacity = int(capacity)
        self.entries = OrderedDict()
        self.pending = defaultdict(set)
        self._completions = []
        self._request_id = 0
        self._now = 0.0
        self.used = 0
        self.stats = {'hit_bytes': 0, 'miss_bytes': 0, 'evictions': 0,
                      'oversize_bytes': 0}

    def contains(self, tensor_id):
        return tensor_id in self.entries

    def access(self, tensor_id, size, now=0.0, duration=0.0):
        now, duration = float(now), float(duration)
        if not math.isfinite(duration) or duration < 0:
            raise ValueError('读取时长必须为非负有限数')
        self.complete_until(now)
        size = max(0, int(size))
        if size == 0:
            return 'miss'
        if tensor_id in self.entries:
            self.stats['hit_bytes'] += size
            outcome = 'hit'
        else:
            self.stats['miss_bytes'] += size
            if size > self.capacity:
                self.stats['oversize_bytes'] += size
                return 'oversize'
            outcome = 'inflight_miss' if self.pending.get(tensor_id) else 'miss'
        # 每次请求独立完成，命中读取也可能在完成前被淘汰，因此也须登记。
        request_id = self._request_id
        self._request_id += 1
        self.pending[tensor_id].add(request_id)
        heapq.heappush(self._completions, (now + duration, request_id, tensor_id, size))
        if duration == 0:
            self.complete_until(now)
        return outcome

    def complete_until(self, now):
        """按完成时间处理所有张量；同刻按请求登记顺序稳定处理。

        时长由调用方给出，未模拟共享带宽；仅保证此输入时间线上的缓存状态。
        调用方必须按时间递增发射，完成先于同刻新发射。
        """
        now = float(now)
        if not math.isfinite(now) or now < self._now:
            raise ValueError('缓存事件时间必须为递增的非负有限数')
        while self._completions and self._completions[0][0] <= now:
            _, request_id, tensor_id, size = heapq.heappop(self._completions)
            self.pending[tensor_id].discard(request_id)
            if not self.pending[tensor_id]:
                self.pending.pop(tensor_id)
            self._insert(tensor_id, size)
        self._now = now

    def _insert(self, tensor_id, size):
        if tensor_id in self.entries or size > self.capacity or size <= 0:
            return
        while self.entries and self.used + size > self.capacity:
            _old_id, (old_size, _sequence) = self.entries.popitem(last=False)
            self.used -= old_size
            self.stats['evictions'] += 1
        self.entries[tensor_id] = (size, len(self.entries))
        self.used += size


class VersionedResourceTable:
    """按核心维护资源快照，并用反向索引做局部失效。

    这个对象借鉴路由表的 ``前缀到下一跳`` 反向索引思想，但更新仍在
    单轮快照内确定性执行：候选方案改变某些张量后，只提升真正接触这些
    张量的核心版本，其余核心继续复用旧快照。它只服务于候选审计和代理
    重算，官方评测器不读取这里的结果。
    """

    def __init__(self, cores):
        if isinstance(cores, bool) or int(cores) < 1:
            raise ValueError('核心数必须为正整数')
        self.cores = int(cores)
        self._versions = {core: 0 for core in range(self.cores)}
        self._events = {core: [] for core in range(self.cores)}
        self._tensor_cores = defaultdict(set)
        self._invalidations = []

    def update(self, core, events):
        """替换一个核心的局部资源条目，并提升该核心版本。"""
        core = int(core)
        if core not in self._events:
            raise IndexError(f'核心编号越界: {core}')
        old_events = self._events[core]
        for event in old_events:
            tensor_id = event.get('tensor_id')
            if tensor_id is not None:
                self._tensor_cores[int(tensor_id)].discard(core)
        normalized = []
        for event in events or ():
            row = dict(event)
            if 'tensor_id' in row:
                row['tensor_id'] = int(row['tensor_id'])
                self._tensor_cores[row['tensor_id']].add(core)
            normalized.append(row)
        self._events[core] = normalized
        self._versions[core] += 1
        return self._versions[core]

    def invalidate_tensors(self, tensor_ids, reason='tensor_changed'):
        """让接触指定张量的核心失效，返回稳定排序后的核心编号。"""
        ids = sorted({int(tensor_id) for tensor_id in tensor_ids})
        affected = sorted({core for tensor_id in ids
                           for core in self._tensor_cores.get(tensor_id, ())})
        for core in affected:
            self._versions[core] += 1
        self._invalidations.append({
            'tensor_ids': ids,
            'cores': affected,
            'reason': str(reason),
        })
        return affected

    def snapshot(self):
        """返回可序列化的同轮快照，不暴露内部可变容器。"""
        return {
            'versions': {str(core): version
                         for core, version in self._versions.items()},
            'tensor_cores': {
                str(tensor_id): sorted(cores)
                for tensor_id, cores in sorted(self._tensor_cores.items())
                if cores
            },
            'events': {
                str(core): deepcopy(events)
                for core, events in self._events.items()
            },
            'invalidations': deepcopy(self._invalidations),
        }


def _order_positions(graph, plan):
    ops = {int(op['id']): op for op in graph.get('ops', ())
           if op.get('op') not in {'COPY_IN', 'COPY_OUT'}}
    mapping = {int(k): int(v) for k, v in plan['node_to_subgraph'].items()}
    core_of = {sg: core for core, row in enumerate(plan['core_schedules'])
               for sg in row}
    # 完整原图拓扑包含COPY/张量中介；不假设编号递增就是依赖顺序。
    from dag_priority import _topology
    nodes={int(o['id']) for o in graph.get('ops',())}|{int(t['id']) for t in graph.get('tensors',())}
    predecessors={node:set() for node in nodes}
    for edge in graph.get('edges',()):
        u,v=int(edge['source']),int(edge['target'])
        if u not in nodes or v not in nodes:raise ValueError('资源表原图边端点不存在')
        predecessors[v].add(u)
    topo,_=_topology(predecessors)
    members=defaultdict(list)
    for op in topo:
        if op in mapping:members[mapping[op]].append(op)
    positions = {}
    for core, row in enumerate(plan['core_schedules']):
        cursor = 0
        clock = 0.0
        for sg in row:
            for op in members[int(sg)]:
                positions[op] = (core, cursor, clock)
                cursor += 1
                clock += max(1, int(ops[op].get('cycles', 1)))
    return ops, mapping, core_of, positions


def build_resource_table(graph, plan, scene, l2_capacity=1048576):
    """构造确定性的 B/C 资源表和逐张量读取事件。"""
    if scene not in {'B', 'C'}:
        raise ValueError('资源表只支持场景 B/C')
    ops, mapping, core_of, positions = _order_positions(graph, plan)
    tensors = {int(t['id']): dict(t) for t in graph.get('tensors', ())}
    producers = defaultdict(set)
    consumers = defaultdict(set)
    for edge in graph.get('edges', ()):
        src, dst = int(edge['source']), int(edge['target'])
        if src in ops and dst in tensors:
            producers[dst].add(src)
        if src in tensors and dst in ops:
            consumers[src].add(dst)
    events = []
    per_core = defaultdict(list)
    for tid, cons in consumers.items():
        tensor = tensors.get(tid, {})
        size = max(0, int(tensor.get('size', 0)))
        # 原图DDR输入在官方每核展开时转换为本地UB后COPY_IN；不能提前排除。
        # 同一生产核上的消费不产生跨核COPY_IN。溢出重载仍须官方展开诊断。
        producer_cores = {positions[p][0] for p in producers.get(tid, ())
                          if p in positions}
        by_core = {}
        for op in sorted(cons, key=lambda x: positions.get(x, (0, 0))):
            core = positions.get(op, (0, 0))[0]
            if producer_cores and core in producer_cores:
                continue
            by_core.setdefault(core, op)
        for op in sorted(by_core.values(), key=lambda x: positions.get(x, (0, 0))):
            core, slot, _start = positions.get(op, (0, 0, 0.0))
            _start = positions.get(op, (0, 0, 0.0))[2]
            event = {'tensor_id': tid, 'size': size, 'op': op,
                     'core': core, 'slot': slot, 'time': _start,
                     'producer_cores': sorted({positions[p][0] for p in producers.get(tid, ())
                                               if p in positions})}
            events.append(event)
            per_core[core].append(event)
    events.sort(key=lambda e: (e['time'], e['core'], e['op'], e['tensor_id']))

    # 私有池峰值：按本核首次使用到最后一次使用的区间估计，只作风险信号。
    pool_peaks = {'L1': 0, 'UB': 0}
    for core, rows in per_core.items():
        active = {}
        last = {}
        for event in rows:
            last[event['tensor_id']] = event['slot']
        for event in rows:
            tid, size = event['tensor_id'], event['size']
            pool = tensors.get(tid, {}).get('pos', 'L1')
            if pool == 'DDR':
                pool = 'UB'
            if pool not in ('L1', 'UB'):
                pool = 'L1'
            active.setdefault(tid, (size, event['slot']))
            current = sum(value[0] for value in active.values())
            pool_peaks[pool] = max(pool_peaks[pool],
                                   sum(value[0] for key, value in active.items()
                                       if ('UB' if tensors.get(key, {}).get('pos') == 'DDR'
                                           else tensors.get(key, {}).get('pos', 'L1')) == pool))
            for key, value in list(active.items()):
                if last.get(key, value[1]) <= event['slot']:
                    del active[key]

    cache = FifoCache(l2_capacity)
    if scene == 'C':
        for event in events:
            event['cache_result'] = cache.access(
                event['tensor_id'], event['size'], event['time'],
                event['size'] / 60.0)
    return {
        'scene': scene,
        'events': events,
        'pool_peaks': pool_peaks,
        'cache': dict(cache.stats),
        'cache_hit_rate': (cache.stats['hit_bytes'] /
                           max(1, cache.stats['hit_bytes'] + cache.stats['miss_bytes']))
        if scene == 'C' else 0.0,
        'core_loads': {str(core): sum(max(1, int(ops[e['op']].get('cycles', 1)))
                                    for e in rows if e['op'] in ops)
                       for core, rows in per_core.items()},
    }
