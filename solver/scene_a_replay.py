"""E06：核心/任务/共享外存表与受控从头重放；尚无区域表和完整快照恢复。"""
from dataclasses import dataclass
import hashlib
import inspect
import math
from pathlib import Path

from scene_a_fast import official, EXPECTED_SHA256


class TaskCoreState:
    """完成事件更新全部后继；只激活固定核队首，不跳过阻塞任务。"""
    def __init__(self,tasks,orders,num_cores,cross_wait,same_wait):
        self.tasks=tasks
        self.orders={c:list(orders.get(c,[])) for c in range(num_cores)}
        self.cross_wait,self.same_wait=cross_wait,same_wait
        self.status={t:'waiting' for t in tasks}
        self.start,self.end={},{}
        self.core_index=dict.fromkeys(range(num_cores),0)
        self.core_active=dict.fromkeys(range(num_cores),None)
        self.core_previous_end=dict.fromkeys(range(num_cores),None)
        self.pending_predecessors={t:len(task['pred_tasks']) for t,task in tasks.items()}
        self.successors={t:[] for t in tasks}
        for t,task in tasks.items():
            for p in task['pred_tasks']:self.successors[p].append(t)
        self.cross_release=dict.fromkeys(tasks,0)
        self.remaining_ops={t:len(task['seq']) for t,task in tasks.items()}
        self.unfinished_ops={t:set(task['seq']) for t,task in tasks.items()}
        self.task_version=dict.fromkeys(tasks,0)
        self.core_version=dict.fromkeys(range(num_cores),0)
        self.position={t:i for i,t in enumerate(tasks)}
        self.completed_count=0

    def release_time(self,task):
        if self.pending_predecessors[task]:return None
        core=self.tasks[task]['core_id'];previous=self.core_previous_end[core]
        return max(0 if previous is None else previous+self.same_wait,self.cross_release[task])

    def activate(self,task,now):
        core=self.tasks[task]['core_id'];index=self.core_index[core]
        release=self.release_time(task)
        if (self.status[task]!='waiting' or self.core_active[core] is not None or
            index>=len(self.orders[core]) or self.orders[core][index]!=task or release is None or release>now):
            raise ValueError('任务不是已就绪的空闲核心队首')
        self.status[task]='active';self.start[task]=now;self.core_active[core]=task
        self.task_version[task]+=1;self.core_version[core]+=1

    def complete_op(self,task,op):
        if self.status[task]!='active' or op not in self.unfinished_ops[task]:
            raise ValueError('重复完成或非活动任务操作')
        self.unfinished_ops[task].remove(op);self.remaining_ops[task]-=1
        self.task_version[task]+=1;self.core_version[self.tasks[task]['core_id']]+=1

    def complete(self,task,now):
        if self.status[task]!='active' or self.remaining_ops[task] or now<self.start[task]:
            raise ValueError('任务未全部完成或重复退役')
        core=self.tasks[task]['core_id']
        self.status[task]='done';self.end[task]=now
        self.core_active[core]=None;self.core_previous_end[core]=now;self.core_index[core]+=1
        self.task_version[task]+=1;self.core_version[core]+=1;self.completed_count+=1
        for succ in self.successors[task]:
            self.pending_predecessors[succ]-=1
            if self.tasks[succ]['core_id']!=core:
                self.cross_release[succ]=max(self.cross_release[succ],now+self.cross_wait)
            self.task_version[succ]+=1

    def active_tasks_in_original_order(self):
        return sorted((t for t in self.core_active.values() if t is not None),key=self.position.get)


@dataclass(frozen=True)
class DDRSnapshot:
    last_update: float
    version: int
    remaining: tuple


class DDRResource:
    """工作量单位为独占服务周期；严格保留官方容差及整数完成时刻。"""
    def __init__(self):
        self.remaining={}
        self.last_update=0
        self.version=0
        self._identity=object()

    def advance(self,now):
        if now < self.last_update:
            raise ValueError('资源时间不可倒退')
        elapsed=now-self.last_update
        if elapsed:
            self.version+=1
        while elapsed > 1e-9:
            active=[item for item,work in self.remaining.items() if work > 1e-9]
            if not active:break
            minimum=min(self.remaining[item] for item in active)
            finish_delta=minimum*len(active)
            if finish_delta >= elapsed-1e-9:
                share=elapsed/len(active)
                for item in active:self.remaining[item]=max(0.0,self.remaining[item]-share)
                break
            for item in active:self.remaining[item]=max(0.0,self.remaining[item]-minimum)
            elapsed-=finish_delta
        self.last_update=now

    def add(self,item,duration,now):
        if item in self.remaining or not math.isfinite(duration) or duration < 0:
            raise ValueError('外存请求重复或服务量非法')
        self.advance(now)
        self.remaining[item]=float(duration)
        self.version+=1

    def remove(self,item,now):
        if item not in self.remaining:raise KeyError(item)
        self.advance(now)
        self.remaining.pop(item)
        self.version+=1

    def forecast(self):
        ordered=sorted((max(0.0,work),item) for item,work in self.remaining.items())
        projected,cursor,previous,active_count={},float(self.last_update),0.0,len(ordered)
        i=0
        while i<len(ordered):
            work=ordered[i][0]
            cursor+=(work-previous)*active_count
            j=i
            while j<len(ordered) and abs(ordered[j][0]-work)<=1e-9:
                projected[ordered[j][1]]=int(math.ceil(cursor-1e-9));j+=1
            active_count-=j-i;previous=work;i=j
        return (self._identity,self.version),projected

    def is_current(self,token):
        return token==(self._identity,self.version)

    def snapshot(self):
        # 保持插入顺序，避免浮点服务更新或平局行为因快照重排变化。
        return DDRSnapshot(self.last_update,self.version,tuple(self.remaining.items()))

    @classmethod
    def from_snapshot(cls,snapshot):
        result=cls();result.last_update=snapshot.last_update;result.version=snapshot.version
        result.remaining=dict(snapshot.remaining)
        return result


def build_resource_evaluator(unified=False,template_cache=None):
    """隔离命名空间，仅提取外存状态所有权与已验证剩余计数；原官方不变。"""
    if hashlib.sha256(Path(official.__file__).read_bytes()).hexdigest()!=EXPECTED_SHA256:
        raise RuntimeError('官方源码已变化，拒绝自动迁移外存语义')
    source=inspect.getsource(official.evaluate_scene_a)
    def replace(before,after):
        nonlocal source
        if source.count(before)!=1:raise RuntimeError('预期受控替换位置不唯一')
        source=source.replace(before,after,1)
    replace('    ddr_last_update = 0\n',
            '    ddr_resource = DDRResource()\n    ddr_remaining_work = ddr_resource.remaining\n')
    start=source.index('    def advance_ddr_work(now):\n')
    end=source.index('    def queue_if_ready(task_id, op_id):\n',start)
    source=source[:start]+'''    def advance_ddr_work(now):
        ddr_resource.advance(now)

    def reschedule_ddr(now):
        if not ddr_remaining_work:
            return {}
        if ddr_resource.last_update != now:
            raise ValueError('预测前必须结算到当前事件时刻')
        _, projected = ddr_resource.forecast()
        for item, end in projected.items():
            op_end[item] = end
        for executor_key, running in executors.items():
            executors[executor_key] = [
                (item, projected.get(item, end)) for item, end in running]
        return projected

'''+source[end:]
    replace('                    ddr_remaining_work.pop(item)\n',
            '                    ddr_resource.remove(item, now)\n')
    replace('                            ddr_remaining_work[item] = float(duration)\n',
            '                            ddr_resource.add(item, duration, now)\n')
    replace('    op_status, pred_remaining, op_start, op_end = {}, {}, {}, {}\n',
            '    op_status, pred_remaining, op_start, op_end = {}, {}, {}, {}\n'
            "    task_remaining = {task_id: len(task['seq']) for task_id, task in tasks.items()}\n")
    replace("                op_status[item] = 'done'\n",
            "                op_status[item] = 'done'\n                task_remaining[task_id] -= 1\n")
    replace("            task_items = [key(task_id, op_id) for op_id in tasks[task_id]['seq']]\n"
            "            if all(op_status[item] == 'done' for item in task_items):\n",
            '            if task_remaining[task_id] == 0:\n')
    if unified:
        start=source.index("    task_status = {task_id: 'waiting' for task_id in tasks}\n")
        end=source.index('    executors = {\n',start)
        source=source[:start]+'''    resource = TaskCoreState(tasks, core_orders, num_cores, cross_core_wait, same_core_wait)
    task_status, task_start, task_end = resource.status, resource.start, resource.end
    core_index = resource.core_index
    core_active_task = resource.core_active
    core_previous_end = resource.core_previous_end
'''+source[end:]
        replace("    task_remaining = {task_id: len(task['seq']) for task_id, task in tasks.items()}\n",
                '    task_remaining = resource.remaining_ops\n')
        replace("        task_status[task_id] = 'active'\n        task_start[task_id] = now\n"
                "        core_active_task[task['core_id']] = task_id\n",'        resource.activate(task_id, now)\n')
        replace('                task_remaining[task_id] -= 1\n','                resource.complete_op(task_id, op_id)\n')
        start=source.index('    def task_release_time(task_id):\n')
        end=source.index('    def activate_ready_tasks(now):\n',start)
        source=source[:start]+'''    def task_release_time(task_id):
        return resource.release_time(task_id)

'''+source[end:]
        replace("        for task_id, status in list(task_status.items()):\n"
                "            if status != 'active':\n                continue\n",
                '        for task_id in resource.active_tasks_in_original_order():\n')
        replace("                core_id = tasks[task_id]['core_id']\n"
                "                task_status[task_id] = 'done'\n"
                '                task_end[task_id] = now\n'
                '                core_active_task[core_id] = None\n'
                '                core_previous_end[core_id] = now\n'
                '                core_index[core_id] += 1\n',
                '                resource.complete(task_id, now)\n')
        replace("        if all(status == 'done' for status in task_status.values()):\n",
                '        if resource.completed_count == len(tasks):\n')
    namespace=dict(official.__dict__)
    namespace.update(DDRResource=DDRResource,TaskCoreState=TaskCoreState,__name__=__name__)
    if template_cache is not None:
        from scene_a_fast import build_counter_evaluator
        # 复用E04已验证的完整局部任务模板接口；每次命中解码独立副本。
        cached_helper=build_counter_evaluator(template_cache=template_cache)
        namespace['_build_scene_a_tasks']=cached_helper.__globals__['_build_scene_a_tasks']
    exec(compile(source,'<scene_a_resource_replay>','exec'),namespace)
    return namespace['evaluate_scene_a']
