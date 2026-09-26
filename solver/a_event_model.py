"""A独立事件模型：整任务屏障、同核等待、跨核释放与共享DDR闭环。"""
import hashlib
import math
import time
import types

from scenario_contract import CODE
from multicore_cut_evaluate_problem_1 import _build_scene_a_tasks
from schedule_step3 import PIPES,PIPE_SLOTS,_op_duration,_uses_ddr_bandwidth,op_pipe
from dynamic_service_pool import SharedServicePool
from stage_cache import StageMemo


class CachedAExpansion:
    """仅复制官方A展开函数的命名空间；不修改原函数/模块/输入。"""
    def __init__(self,max_bytes=128*1024*1024):
        sha=hashlib.sha256()
        for path in sorted(CODE.glob('*.py')):
            sha.update(path.name.encode('utf8'));sha.update(b'\0');sha.update(path.read_bytes())
        self.memo=StageMemo(max_bytes,sha.hexdigest())
        namespace=dict(_build_scene_a_tasks.__globals__)
        def wrap(name,function):
            def cached(*args,**kwargs):return self.memo.call(name,function,*args,**kwargs)
            return cached
        for name in ('step1_schedule','step2_spill_insertion','prepare_step3_execution'):
            namespace[name]=wrap(name,namespace[name])
        self._build=types.FunctionType(_build_scene_a_tasks.__code__,namespace,
            _build_scene_a_tasks.__name__,_build_scene_a_tasks.__defaults__,_build_scene_a_tasks.__closure__)
        self._build.__kwdefaults__=_build_scene_a_tasks.__kwdefaults__

    def __call__(self,graph,plan,hardware):
        return self._build(graph,plan,hardware['bandwidth_bytes_per_cycle'],hardware['capacity_bytes'])


def simulate_a_plan(graph,plan,hardware,*,expander=None,max_events=1000000):
    """硬件参数来自原配置；观测时间线、命中和官方makespan不参与预测。"""
    if PIPE_SLOTS!=1:raise ValueError('当前模型仅核验了官方单在飞槽')
    if type(max_events) is not int or max_events<1:raise ValueError('事件预算必须为正整数')
    bandwidth=hardware['bandwidth_bytes_per_cycle']
    if isinstance(bandwidth,bool) or not math.isfinite(bandwidth) or bandwidth<=0:raise ValueError('带宽非法')
    cross=hardware['task_cross_core_wait_cycles'];same=hardware['task_same_core_wait_cycles']
    if any(type(v) is not int or v<0 for v in (cross,same)):raise ValueError('任务等待必须为非负整数')
    started=time.perf_counter()
    if expander is None:
        tasks,traffic,movement,view=_build_scene_a_tasks(graph,plan,bandwidth,hardware['capacity_bytes'])
    else:tasks,traffic,movement,view=expander(graph,plan,hardware)
    expanded_at=time.perf_counter();cores=view['num_cores'];orders=view['core_orders']
    if any(not t['seq'] for t in tasks.values()):raise ValueError('不支持空任务')
    pool=SharedServicePool();now=0;events=0;running={};operations={};done=set();task_end={};task_start={}
    active={c:None for c in range(cores)};index={c:0 for c in range(cores)};previous={c:None for c in range(cores)}
    cursors={(task_id,pipe):0 for task_id in tasks for pipe in PIPES}
    remaining={task_id:len(t['seq']) for task_id,t in tasks.items()};ddr=set();peak=0
    executors=[(c,p) for c in range(cores) for p in PIPES]

    def refresh():
        for item,end in pool.finish_times().items():operations[item]['end']=end

    def release(task_id):
        task=tasks[task_id];core=task['core_id']
        if any(p not in task_end for p in task['pred_tasks']):return None
        ready=0 if previous[core] is None else previous[core]+same
        return max([ready]+[task_end[p]+cross for p in task['pred_tasks'] if tasks[p]['core_id']!=core])

    while len(task_end)<len(tasks):
        events+=1
        if events>max_events:raise ValueError('A闭环事件超过预算')
        pool.advance(now);retired=[]
        for executor in executors:
            item=running.get(executor)
            if item is None or operations[item]['end']>now:continue
            task_id,op_id=item;done.add(item);running.pop(executor)
            cursors[task_id,executor[1]]+=1;remaining[task_id]-=1
            if item in ddr:retired.append(item)
        if retired:pool.retire(retired,now);refresh()
        # 所有同刻操作退休后统一结束任务，再逐核心激活新任务。
        for core,task_id in list(active.items()):
            if task_id is not None and remaining[task_id]==0:
                task_end[task_id]=now;previous[core]=now;active[core]=None;index[core]+=1
        future=[]
        for core in range(cores):
            if active[core] is not None or index[core]>=len(orders.get(core,[])):continue
            task_id=orders[core][index[core]];ready=release(task_id)
            if ready is None:continue
            if ready>now:future.append(ready);continue
            active[core]=task_id;task_start[task_id]=now
        for executor in executors:
            core,pipe=executor;task_id=active[core]
            if executor in running or task_id is None:continue
            task=tasks[task_id];seq=task['pipe_ops'].get(pipe,());cursor=cursors[task_id,pipe]
            if cursor>=len(seq):continue
            op_id=seq[cursor];item=(task_id,op_id)
            if any((task_id,p) not in done for p in task['op_preds'][op_id]):continue
            op=task['op_by_id'][op_id]
            work=_op_duration(op,task['in_tids'],task['out_tids'],task['tensor_by_id'],bandwidth)
            operations[item]=dict(start=now,end=now+work,core_id=core,pipe=op_pipe(op),op=op['op'])
            running[executor]=item
            if _uses_ddr_bandwidth(op,task['in_tids'],task['out_tids'],task['tensor_by_id']):
                pool.issue(item,work,now);ddr.add(item);refresh();peak=max(peak,len(pool.remaining))
        if len(task_end)==len(tasks):break
        deadlines=[operations[item]['end'] for item in running.values()]+future
        if not deadlines:raise ValueError(f'A闭环事件死锁，时间{now}')
        following=min(deadlines)
        if following<=now:raise ValueError('A闭环事件时间不推进')
        now=following
    return dict(makespan=max(task_end.values(),default=0),operations=operations,
        task_times={task_id:(task_start[task_id],end) for task_id,end in task_end.items()},
        data_movement_bytes=movement,cross_task_traffic=traffic,events=events,peak_requests=peak,
        expansion_seconds=expanded_at-started,event_seconds=time.perf_counter()-expanded_at,
        scope='A独立任务/操作闭环预测，原官方只读展开；尚不替代正式评估')
