"""可信进程内纯函数内容缓存；预算统计序列化负载，不代表进程内存。"""
from collections import OrderedDict
from functools import wraps
import hashlib
import pickle
import types


class _HashSink:
    def __init__(self):
        self.digest = hashlib.sha256()

    def write(self, data):
        self.digest.update(data)
        return len(data)


class StageMemo:
    """单进程、单线程有界缓存。函数必须是确定的、无副作用纯函数。"""
    def __init__(self, max_bytes, source_version):
        if type(max_bytes) is not int or max_bytes < 0:
            raise ValueError('缓存字节预算必须为非负整数')
        if not isinstance(source_version, str) or not source_version:
            raise ValueError('必须给出源码版本')
        self.max_bytes = max_bytes
        self.source_version = source_version
        self.resident_bytes = 0
        self.stats = dict(hits=0, misses=0, evictions=0, oversized=0)
        self._entries = OrderedDict()
        # 保持函数引用，防止对象销毁后身份地址被另一个函数复用。
        self._functions = {}

    def call(self, stage_tag, function, *args, **kwargs):
        identity = id(function)
        self._functions[identity] = function
        sink = _HashSink()
        pickle.Pickler(sink, protocol=pickle.HIGHEST_PROTOCOL).dump(
            (self.source_version, stage_tag, identity, args, kwargs))
        key = sink.digest.digest()
        if key in self._entries:
            payload = self._entries.pop(key)
            self._entries[key] = payload
            self.stats['hits'] += 1
            return pickle.loads(payload)
        self.stats['misses'] += 1
        value = function(*args, **kwargs)
        payload = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
        if len(payload) > self.max_bytes:
            self.stats['oversized'] += 1
            return value
        while self.resident_bytes + len(payload) > self.max_bytes:
            _, old = self._entries.popitem(last=False)
            self.resident_bytes -= len(old)
            self.stats['evictions'] += 1
        self._entries[key] = payload
        self.resident_bytes += len(payload)
        return value


class CachedSceneExpansion:
    """仅复制展开器命名空间注入缓存，不修改官方函数、文件或模块全局。"""
    def __init__(self, max_bytes=128 * 1024 * 1024):
        from scenario_contract import CODE
        from multicore_cut_evaluate_problem_2 import _build_scene_b_tasks

        digest = hashlib.sha256()
        for path in sorted(CODE.glob('*.py')):
            digest.update(path.name.encode('utf8'))
            digest.update(b'\0')
            digest.update(path.read_bytes())
        self.memo = StageMemo(max_bytes, digest.hexdigest())
        namespace = dict(_build_scene_b_tasks.__globals__)
        for name in ('step1_schedule', 'step2_spill_insertion', 'prepare_step3_execution'):
            original = namespace[name]

            def wrap(tag, function):
                @wraps(function)
                def cached(*args, **kwargs):
                    return self.memo.call(tag, function, *args, **kwargs)
                return cached

            namespace[name] = wrap(name, original)
        self._build = types.FunctionType(
            _build_scene_b_tasks.__code__, namespace,
            _build_scene_b_tasks.__name__, _build_scene_b_tasks.__defaults__,
            _build_scene_b_tasks.__closure__)
        self._build.__kwdefaults__ = _build_scene_b_tasks.__kwdefaults__

    def __call__(self, graph, plan, hardware):
        from scenario_contract import validate_plan
        from evaluation_validation import validate_execution

        validate_plan(graph, plan, 'B')
        tasks, links, _, _, _ = self._build(
            graph, plan, hardware['bandwidth_bytes_per_cycle'], hardware['capacity_bytes'])
        validate_execution(tasks, links)
        return tasks, links
