"""B/C独立闭环事件模型：官方只读展开，自主推导发射/缓存/共享服务。

尚不代替原官方评估；通过历史轨迹校准后才能用于候选排序。
"""
from collections import OrderedDict,defaultdict
import math
import time

from bc_observed_graph import expand
from dynamic_service_pool import SharedServicePool
from schedule_step3 import PIPES,PIPE_SLOTS,_op_duration,_uses_ddr_bandwidth,op_pipe


class FifoResidency:
    """只负责完成后的驻留状态；发射、在途请求和退休由事件控制器管理。"""
    def __init__(self,capacity):
        if type(capacity) is not int or capacity<0:raise ValueError('缓存容量必须为非负整数')
        self.capacity=capacity;self.used=0;self.entries=OrderedDict()

    def contains(self,key):return key in self.entries

    def insert(self,key,size):
        if type(size) is not int or size<0:raise ValueError('张量大小必须为非负整数')
        if key is None or size>self.capacity or key in self.entries:return None
        evicted=[]
        while self.entries and self.used+size>self.capacity:
            old,old_size=self.entries.popitem(last=False);self.used-=old_size;evicted.append(old)
        self.entries[key]=size;self.used+=size
        return evicted


def _copy_info(task,op_id):
    op=task['op_by_id'][op_id]
    tids=(task['out_tids'][op_id] if op['op']=='COPY_IN' else task['in_tids'][op_id] if op['op']=='COPY_OUT' else ())
    tids=[tid for tid in tids if task['tensor_by_id'][tid].get('pos')!='DDR']
    if not tids:return None,0
    tensor=task['tensor_by_id'][tids[0]]
    return tensor.get('logical_tid',tids[0]),tensor['size']


def simulate_bc_plan(graph,plan,hardware,scene,*,max_events=1000000,expander=None):
    if scene not in ('B','C'):raise ValueError('事件模型仅支持B/C')
    if PIPE_SLOTS!=1:raise ValueError('当前模型仅核验了官方单在飞槽语义')
    if type(max_events) is not int or max_events<1:raise ValueError('事件上限必须为正整数')
    started=time.perf_counter()
    bandwidth=hardware['bandwidth_bytes_per_cycle'];cross=hardware['cross_core_copy_delay_cycles']
    if isinstance(bandwidth,bool) or not math.isfinite(bandwidth) or bandwidth<=0:raise ValueError('带宽非法')
    if type(cross) is not int or cross<0:raise ValueError('跨核等待必须非负整数')
    cache_bandwidth=hardware['cache_bandwidth_bytes_per_cycle'] if scene=='C' else bandwidth
    if isinstance(cache_bandwidth,bool) or not math.isfinite(cache_bandwidth) or cache_bandwidth<=0:raise ValueError('缓存带宽非法')
    tasks,links=(expand if expander is None else expander)(graph,plan,hardware)
    expanded_at=time.perf_counter()
    pools={name:SharedServicePool() for name in ('DDR','CACHE_READ')}
    cache=FifoResidency(hardware['cache_capacity_bytes']) if scene=='C' else None
    pipeline_keys=[(core,pipe) for core in range(len(plan['core_schedules'])) for pipe in PIPES]
    cursors={key:0 for key in pipeline_keys};running={};done=set();operations={};paths={};cache_keys={}
    external=defaultdict(list)
    for link in links:
        external[link['target_core'],link['target_copy_in_id']].append((link['source_core'],link['source_copy_out_id']))
    total=sum(len(task['op_by_id']) for task in tasks.values())
    cache_events=[];stats=dict(copy_in_hits=0,copy_in_misses=0,hit_bytes=0,miss_bytes=0)
    peak={name:0 for name in pools};now=0;events=0

    def refresh(path):
        for item,finish in pools[path].finish_times().items():operations[item]['end']=finish

    while len(done)<total:
        events+=1
        if events>max_events:raise ValueError('闭环事件模型超过事件预算')
        for pool in pools.values():pool.advance(now)
        changed=set()
        # 同刻退休顺序严格遵循官方核心/流水线次序；退休结束后再发射。
        for executor in pipeline_keys:
            item=running.get(executor)
            if item is None or operations[item]['end']>now:continue
            core,op_id=item;op=tasks[core]['op_by_id'][op_id]
            path=paths.get(item)
            if path in pools:
                pools[path].retire([item],now);changed.add(path)
            done.add(item);running.pop(executor);cursors[executor]+=1
            if cache is not None and op['op']=='COPY_IN':
                key,size=_copy_info(tasks[core],op_id);evicted=cache.insert(key,size)
                if evicted is not None:
                    cache_events.append(dict(time=now,event='insert',tensor_id=key,size_bytes=size,
                        used_bytes=cache.used,evicted_tensor_ids=evicted,core_id=core,op_id=op_id))
        for path in changed:refresh(path)

        future_releases=[]
        for executor in pipeline_keys:
            if executor in running:continue
            core,pipe=executor;task=tasks[core];order=task['pipe_ops'].get(pipe,());cursor=cursors[executor]
            if cursor>=len(order):continue
            op_id=order[cursor];item=(core,op_id);op=task['op_by_id'][op_id]
            if any((core,p) not in done for p in task['op_preds'][op_id]):continue
            if any(p not in done for p in external[item]):continue
            release=max((operations[p]['end']+cross for p in external[item]),default=0)
            if release>now:future_releases.append(release);continue
            key,size=_copy_info(task,op_id)
            eligible=cache is not None and op['op']=='COPY_IN' and key is not None and size>0
            hit=eligible and cache.contains(key)
            service=_op_duration(op,task['in_tids'],task['out_tids'],task['tensor_by_id'],cache_bandwidth if hit else bandwidth)
            operations[item]=dict(start=now,end=now+service,op=op['op'],pipe=op_pipe(op),subgraph_id=task['op_subgraph'].get(op_id))
            running[executor]=item
            if eligible:
                stats['copy_in_hits' if hit else 'copy_in_misses']+=1
                stats['hit_bytes' if hit else 'miss_bytes']+=size
                cache_keys[item]=key
                cache_events.append(dict(time=now,event='hit' if hit else 'miss',tensor_id=key,size_bytes=size,
                                         core_id=core,op_id=op_id,op=op['op']))
            path=('CACHE_READ' if hit else 'DDR' if _uses_ddr_bandwidth(op,task['in_tids'],task['out_tids'],task['tensor_by_id']) else None)
            if path is not None:
                paths[item]=path;pools[path].issue(item,service,now);refresh(path)
                peak[path]=max(peak[path],len(pools[path].remaining))
        if len(done)==total:break
        deadlines=[operations[item]['end'] for item in running.values()]+future_releases
        if not deadlines:raise ValueError(f'闭环调度死锁，t={now}，剩余{total-len(done)}')
        next_now=min(deadlines)
        if next_now<=now:raise ValueError('闭环事件无时间推进')
        now=next_now

    for item,entry in operations.items():
        entry['duration']=entry['end']-entry['start']
        if entry['op'] in ('COPY_IN','COPY_OUT'):entry['memory_path']=paths.get(item,'ON_CHIP')
        if item in cache_keys:entry.update(cache_hit=paths.get(item)=='CACHE_READ',cache_tensor_id=cache_keys[item])
    stats.update(hits=stats['copy_in_hits'],accesses=stats['copy_in_hits']+stats['copy_in_misses'])
    total_bytes=stats['hit_bytes']+stats['miss_bytes'];stats['hit_rate']=stats['hit_bytes']/total_bytes if total_bytes else 0.0
    return dict(makespan=max((entry['end'] for entry in operations.values()),default=0),operations=operations,
        cache_stats=stats,cache_events=cache_events,
        cache_final_entries=[dict(tensor_id=key,size_bytes=size) for key,size in cache.entries.items()] if cache else [],
        cache_used_bytes_final=cache.used if cache else 0,events=events,peak_requests=peak,
        expansion_seconds=expanded_at-started,event_seconds=time.perf_counter()-expanded_at,
        scope='独立闭环事件候选模型；仅原图/计划/配置作为输入，原官方仍为裁决来源')
