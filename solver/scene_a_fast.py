"""场景 A 官方函数的受控计数式副本；不修改官方文件或模块全局状态。"""
import hashlib
import inspect
from pathlib import Path
import sys

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / '通用神经网络处理器下的多核调度问题附件/code').is_dir())
CODE = ROOT / '通用神经网络处理器下的多核调度问题附件/code'
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))
import multicore_cut_evaluate_problem_1 as official

EXPECTED_SHA256 = '2095f188a6c24ce3899f156bef21d50dcd87cbd9368488046b1e77e2bf91af3f'


def build_counter_evaluator(template_cache=None):
    path = Path(official.__file__)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != EXPECTED_SHA256:
        raise RuntimeError('官方场景 A 源码指纹变化，计数式副本需要重新审查验证')
    source = inspect.getsource(official.evaluate_scene_a)
    replacements = [
        ("    op_status, pred_remaining, op_start, op_end = {}, {}, {}, {}\n",
         "    op_status, pred_remaining, op_start, op_end = {}, {}, {}, {}\n"
         "    task_remaining = {task_id: len(task['seq']) for task_id, task in tasks.items()}\n"),
        ("                op_status[item] = 'done'\n",
         "                op_status[item] = 'done'\n"
         "                task_remaining[task_id] -= 1\n"),
        ("            task_items = [key(task_id, op_id) for op_id in tasks[task_id]['seq']]\n"
         "            if all(op_status[item] == 'done' for item in task_items):\n",
         "            if task_remaining[task_id] == 0:\n")]
    for before, after in replacements:
        if source.count(before) != 1:
            raise RuntimeError('官方场景 A 预期代码位置不唯一，拒绝生成副本')
        source = source.replace(before, after, 1)
    # 单独复制全局名字绑定，原官方函数及辅助模块不被替换。
    namespace = dict(official.__dict__)
    namespace['__name__'] = __name__
    if template_cache is not None:
        # 只替换副本命名空间里的局部阶段函数；全局官方模块不变。
        # 完整局部图、顺序、容量、带宽均在各调用参数中参与缓存键。
        sources = {p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in sorted(CODE.glob('*.py'))}
        import json
        template_cache.source_fingerprint=hashlib.sha256(
            json.dumps(sources,sort_keys=True).encode()).hexdigest()
        def prepare_local(graph,capacity,bandwidth):
            seq=official.step1_schedule(graph)
            result2=official.step2_spill_insertion(graph,seq,capacity=capacity)
            extended=official._build_extended_graph(graph,result2)
            prepared=official.prepare_step3_execution(extended,capacity=capacity,bandwidth=bandwidth)
            return result2,prepared
        namespace['_cached_prepare_local']=template_cache.wrap('whole_local_task_v2',prepare_local)
        helper=inspect.getsource(official._build_scene_a_tasks)
        first=("        seq = step1_schedule(graph)\n"
               "        result2 = step2_spill_insertion(graph, seq, capacity=capacity)\n")
        second=("        ext_graph = _build_extended_graph(graph, result2)\n"
                "        prepared = prepare_step3_execution(\n"
                "            ext_graph, capacity=capacity, bandwidth=bandwidth)\n")
        if helper.count(first)!=1 or helper.count(second)!=1:
            raise RuntimeError('官方局部准备结构变化，拒绝启用模板缓存')
        helper=helper.replace(first,"        result2, prepared = _cached_prepare_local(graph, capacity, bandwidth)\n",1)
        helper=helper.replace(second,'',1)
        exec(compile(helper,'<scene_a_cached_local_tasks>','exec'),namespace)
    exec(compile(source, '<scene_a_counter_replica>', 'exec'), namespace)
    replica = namespace['evaluate_scene_a']
    replica.__name__ = 'evaluate_scene_a_fast'
    return replica


evaluate_scene_a_fast = build_counter_evaluator()
