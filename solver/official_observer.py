"""第二、三问原官方函数的只读观测；不替换共享模块，不改变官方指令代码。

每次调用创建独立的函数全局字典，阶段输入和返回值仅深复制记录。
第三问只保留官方已经导出的缓存事件，不虚构嵌套缓存队列观测。
"""
from copy import deepcopy
import hashlib
from pathlib import Path
import sys
from types import FunctionType

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / '通用神经网络处理器下的多核调度问题附件/code'
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))
import multicore_cut_evaluate_problem_2 as official_b
import multicore_cut_evaluate_problem_3 as official_c
from evaluation_validation import read_evaluation_config, validate_execution


def _source_digests():
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(CODE.glob('*.py'))}


def _isolated(function, namespace):
    """直接复用代码对象，保留默认参数与闭包，仅隔离名字绑定。"""
    copied = FunctionType(function.__code__, namespace, function.__name__,
                          function.__defaults__, function.__closure__)
    copied.__kwdefaults__ = function.__kwdefaults__
    return copied


class OfficialObserver:
    """真实配置驱动的官方适配器；显式配置路径可用于合成诊断。

默认读取附件配置，任何诊断配置都随结果返回并参与缓存标识。
输出中的 ``result`` 始终是原官方完整返回值，不增加或修改其字段。
"""

    def __init__(self, config_path=None):
        self.config_path = Path(config_path) if config_path is not None else CODE.parent / 'data/config.txt'
        self._sources = _source_digests()

    def context(self, scene):
        """读取当前配置与完整源码指纹；已装载源码变化时拒绝混用。"""
        if scene not in ('B', 'C'):
            raise ValueError('官方观测仅支持场景 B/C')
        sources = _source_digests()
        if sources != self._sources:
            raise RuntimeError('官方源码在观测器生命周期内发生变化，请重启进程后重新验证')
        module = official_b if scene == 'B' else official_c
        before = hashlib.sha256(self.config_path.read_bytes()).hexdigest()
        settings = read_evaluation_config(str(self.config_path))
        settings['cross_core_copy_delay'] = module.read_scene_b_config(str(self.config_path))['cross_core_copy_delay_cycles']
        if scene == 'C':
            settings.update(module.read_cache_config(str(self.config_path)))
        after = hashlib.sha256(self.config_path.read_bytes()).hexdigest()
        if before != after:
            raise RuntimeError('读取期间评估配置发生变化')
        return {'scene': scene, 'settings': settings, 'config_digest': before,
                'source_digest': sources, 'observer_version': 1}

    def _run(self, graph, plan, scene, observe, expansion):
        context = self.context(scene)
        module = official_b if scene == 'B' else official_c
        namespace = dict(module.__dict__)
        originals = {name: namespace[name] for name in
                     ('step1_schedule', '_prioritize_task_seq', 'step2_spill_insertion',
                      '_build_extended_graph', 'prepare_step3_execution', '_build_scene_b_tasks')}
        observations = []
        pending_step1 = {}

        def step1(graph, *args, **kwargs):
            snapshot = deepcopy(graph)
            result = originals['step1_schedule'](graph, *args, **kwargs)
            pending_step1.update(graph=snapshot, raw_seq=deepcopy(result))
            return result

        def prioritize(graph, raw_seq, op_subgraph, subgraph_order):
            item = {'core_id': len(observations), 'graph': deepcopy(graph),
                    'raw_seq': deepcopy(raw_seq), 'op_subgraph': deepcopy(op_subgraph),
                    'subgraph_order': deepcopy(subgraph_order),
                    'step1': deepcopy(pending_step1) if graph['ops'] else None}
            result = originals['_prioritize_task_seq'](graph, raw_seq, op_subgraph, subgraph_order)
            item['seq'] = deepcopy(result)
            observations.append(item)
            pending_step1.clear()
            return result

        def spill(graph, seq, *args, **kwargs):
            snapshot = {'graph': deepcopy(graph), 'seq': deepcopy(seq),
                        'args': deepcopy(args), 'kwargs': deepcopy(kwargs)}
            result = originals['step2_spill_insertion'](graph, seq, *args, **kwargs)
            observations[-1]['step2_input'] = snapshot
            observations[-1]['result2'] = deepcopy(result)
            return result

        def extended(graph, result2):
            # 空核跳过第一、二步；这里直接读取官方实际创建的空结果。
            observations[-1]['result2'] = deepcopy(result2)
            result = originals['_build_extended_graph'](graph, result2)
            observations[-1]['extended_graph'] = deepcopy(result)
            return result

        def prepare(graph, *args, **kwargs):
            observations[-1]['prepare_input'] = {'graph': deepcopy(graph),
                                                 'args': deepcopy(args), 'kwargs': deepcopy(kwargs)}
            result = originals['prepare_step3_execution'](graph, *args, **kwargs)
            observations[-1]['prepared'] = deepcopy(result)
            return result

        if observe:
            namespace.update(step1_schedule=step1, _prioritize_task_seq=prioritize,
                             step2_spill_insertion=spill, _build_extended_graph=extended,
                             prepare_step3_execution=prepare)
        builder = _isolated(module._build_scene_b_tasks, namespace)
        namespace['_build_scene_b_tasks'] = builder
        settings = context['settings']
        # 输入独立副本也隔离官方返回图中的可变引用。
        graph_copy, plan_copy = deepcopy(graph), deepcopy(plan)
        try:
            if expansion:
                value = builder(graph_copy, plan_copy, settings['bandwidth'], settings['capacity'])
                validate_execution(value[0], value[1])
                field = 'built'
            else:
                function = module.evaluate_scene_b if scene == 'B' else module.evaluate_problem_3
                # evaluate_scene_* 的代码对象通过本次命名空间解析其构图入口；
                # 其余参数仍完全来自 config.txt，不使用观测器内置数值。
                value = _isolated(function, namespace)(graph_copy, plan_copy, **settings)
                field = 'result'
            if self.context(scene) != context:
                raise RuntimeError('评估期间配置或官方源码发生变化，结果不采纳')
            return {field: value, 'observations': observations, **context}
        finally:
            # 恢复的仅是本次调用的独立字典，共享官方模块从未被修改。
            namespace.update(originals)

    def evaluate(self, graph, plan, scene, observe=True):
        return self._run(graph, plan, scene, observe, expansion=False)

    def expand(self, graph, plan, scene):
        return self._run(graph, plan, scene, True, expansion=True)
