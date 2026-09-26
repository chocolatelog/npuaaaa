"""按完整编号方案缓存第二、三问官方展开；每次交付独立可变副本。

仅复用整份展开，不是资源执行副本，不共享可变执行状态。
缓存仅在本进程内存在；持久化证据与恢复由运行器负责。
"""
from collections import OrderedDict
from copy import deepcopy
import hashlib
import json

from official_observer import OfficialObserver


class BCExpansionCache:
    """条目数有界的整份展开缓存，默认最多保留八个完整状态。"""

    def __init__(self, observer=None, max_entries=8):
        if isinstance(max_entries, bool) or not isinstance(max_entries, int) or max_entries < 0:
            raise ValueError('展开缓存上限必须是非负整数')
        self.observer = observer if observer is not None else OfficialObserver()
        self.max_entries = max_entries
        self._entries = OrderedDict()
        self.hits = 0
        self.misses = 0

    def __len__(self):
        return len(self._entries)

    def expand(self, graph, plan, scene):
        context = self.observer.context(scene)
        # 保留所有子图编号和每核完整列表，不使用分区规范化摘要。
        payload = {'graph': graph, 'plan': plan, 'context': context, 'cache_version': 1}
        key = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False,
                                       separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()
        if key in self._entries:
            self.hits += 1
            self._entries.move_to_end(key)
            return deepcopy(self._entries[key])
        self.misses += 1
        result = self.observer.expand(graph, plan, scene)
        if any(result[name] != value for name, value in context.items()):
            raise RuntimeError('展开期间配置发生变化，缓存结果不采纳')
        if self.max_entries:
            self._entries[key] = deepcopy(result)
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)
        return result
