"""独立运行资源管理；只约束并发，不改变搜索预算或算法结果。

字节参数均使用整数。自动并发是保守起点，并非实测最优并发。
设备预算应包含每进程张量缓存和设备上下文，不能只填临时张量大小。
"""
from __future__ import annotations

import ctypes
import importlib
import importlib.metadata
import os
import platform
import sys
import warnings
from dataclasses import asdict, dataclass

GIB = 1024 ** 3
THREAD_ENV = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
              'NUMEXPR_NUM_THREADS')


@dataclass(frozen=True)
class WorkerResolution:
    requested: str | int
    workers: int
    recommended_workers: int
    cpu_limit: int
    memory_limit: int | None
    gpu_limit: int | None
    available_memory: int | None
    reserve: int
    per_worker_estimate: int
    gpu_available_memory: int | None
    gpu_reserve: int
    gpu_per_worker_estimate: int
    advice: tuple[str, ...]

    def to_dict(self):
        return asdict(self)


def _integer(name, value, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f'{name} 必须是大于等于 {minimum} 的整数')
    return value


def memory_snapshot():
    """优先调用操作系统获取物理内存；探测失败明确记空，不伪造容量。"""
    try:
        if os.name == 'nt':
            class MemoryStatus(ctypes.Structure):
                _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong),
                            ('total_phys', ctypes.c_ulonglong),
                            ('avail_phys', ctypes.c_ulonglong),
                            ('total_page', ctypes.c_ulonglong),
                            ('avail_page', ctypes.c_ulonglong),
                            ('total_virtual', ctypes.c_ulonglong),
                            ('avail_virtual', ctypes.c_ulonglong),
                            ('avail_extended', ctypes.c_ulonglong)]
            status = MemoryStatus()
            status.length = ctypes.sizeof(status)
            if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                raise OSError('系统内存查询失败')
            return {'total_bytes': status.total_phys, 'available_bytes': status.avail_phys,
                    'source': 'GlobalMemoryStatusEx', 'error': None}
        # 其他操作系统仅在安装了探测库时使用，不自动安装。
        psutil = importlib.import_module('psutil')
        memory = psutil.virtual_memory()
        return {'total_bytes': int(memory.total), 'available_bytes': int(memory.available),
                'source': 'psutil', 'error': None}
    except (ImportError, OSError, AttributeError) as exc:
        return {'total_bytes': None, 'available_bytes': None,
                'source': None, 'error': str(exc)}


def resolve_workers(requested='auto', *, cpu_count=None, available_memory=None,
                    reserve=2 * GIB, per_worker_estimate=512 * 1024 ** 2,
                    gpu_available_memory=None, gpu_reserve=GIB,
                    gpu_per_worker_estimate=0):
    """根据处理器、空闲内存和可选显存预算返回并发与可记录的决策。

    12 个逻辑处理器以 8 并发开始；其他机器取逻辑数的三分之二，至少 1。
    手动正整数保持原值，超过建议容量时发出警告。自动模式无法容纳一个
    进程时拒绝启动。未提供内存值会现场探测；显存仅由调用方显式提供。
    """
    if requested != 'auto':
        if isinstance(requested, str) and requested.isascii() and requested.isdecimal():
            requested = int(requested)
        _integer('并发数', requested, 1)
    logical_count = _integer('逻辑处理器数',
                             (os.cpu_count() or 1) if cpu_count is None else cpu_count, 1)
    for name, value in (('内存保留量', reserve), ('显存保留量', gpu_reserve),
                        ('每进程显存估计', gpu_per_worker_estimate)):
        _integer(name, value)
    _integer('每进程内存估计', per_worker_estimate, 1)
    if available_memory is None:
        available_memory = memory_snapshot()['available_bytes']
    if available_memory is not None:
        _integer('可用内存', available_memory)
    if gpu_available_memory is not None:
        _integer('可用显存', gpu_available_memory)
    if gpu_per_worker_estimate and gpu_available_memory is None:
        raise ValueError('启用每进程显存预算时必须提供可用显存')
    cpu_limit = max(1, logical_count * 2 // 3)
    memory_limit = (None if available_memory is None else
                    max(0, available_memory - reserve) // per_worker_estimate)
    gpu_limit = (None if not gpu_per_worker_estimate else
                 max(0, gpu_available_memory - gpu_reserve) // gpu_per_worker_estimate)
    recommended = min(value for value in (cpu_limit, memory_limit, gpu_limit)
                      if value is not None)
    advice = []
    if available_memory is None:
        advice.append('无法读取可用内存，请显式提供内存预算；当前建议仅考虑已知资源。')
    if requested == 'auto' and recommended < 1:
        raise RuntimeError('保留资源后不足以容纳单个工作进程，请释放资源或修正实测内存/显存估计。')
    workers = recommended if requested == 'auto' else requested
    if requested != 'auto' and workers > recommended:
        advice.append(f'手动并发 {workers} 保持不变；当前资源建议至多 {recommended}，'
                      f'处理器/内存/显存上限为 {cpu_limit}/{memory_limit}/{gpu_limit}。'
                      '请降低并发或用实测峰值修正预算。')
    for message in advice:
        warnings.warn(message, RuntimeWarning, stacklevel=2)
    return WorkerResolution(requested, workers, recommended, cpu_limit, memory_limit,
                            gpu_limit, available_memory, reserve, per_worker_estimate,
                            gpu_available_memory, gpu_reserve, gpu_per_worker_estimate,
                            tuple(advice))


def initialize_worker_threads():
    """在父进程创建池之前及子进程初始化时调用，避免底层线程乘并发数。

    父进程先调用可使新进程在导入数值库之前继承限制；对于已加载的张量库
    同时设置其运行时线程数。本函数不会主动导入张量库或安装依赖。
    """
    for name in THREAD_ENV:
        os.environ[name] = '1'
    torch = sys.modules.get('torch')
    if torch is not None:
        torch.set_num_threads(1)
    # 已加载的数值库可能早于初始化器建立线程池；可用时同步限制现有池。
    if any(name in sys.modules for name in ('numpy', 'scipy')):
        try:
            threadpoolctl = importlib.import_module('threadpoolctl')
        except ImportError:
            warnings.warn('数值库已加载且缺少线程池控制库，请在创建子进程前设置线程环境变量。',
                          RuntimeWarning, stacklevel=2)
        else:
            threadpoolctl.threadpool_limits(limits=1)
    return {name: os.environ[name] for name in THREAD_ENV}


def collect_runtime_environment(*, include_torch=False):
    """采集可直接写入运行报告的环境；默认只读包版本而不导入张量库。

    空闲内存/显存是采样时刻的动态值，应单独记录，不能纳入断点续跑的
    稳定身份摘要。显存采样也不等于整个运行期间的全设备峰值。
    """
    packages = {}
    for name in ('numpy', 'scipy', 'torch', 'tqdm', 'pytest', 'psutil', 'threadpoolctl'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    report = {
        'python': {'executable': sys.executable, 'version': sys.version,
                   'implementation': platform.python_implementation()},
        'platform': platform.platform(),
        'cpu': {'logical_count': os.cpu_count() or 1, 'processor': platform.processor()},
        'memory': memory_snapshot(), 'packages': packages,
        'thread_environment': {name: os.environ.get(name) for name in THREAD_ENV},
        'torch': {'inspected': bool(include_torch)},
    }
    if not include_torch:
        return report
    info = report['torch']
    info.update(cuda_available=False, devices=[], error=None)
    try:
        torch = importlib.import_module('torch')
        info.update(version=str(torch.__version__), cuda_runtime=torch.version.cuda,
                    cuda_available=bool(torch.cuda.is_available()),
                    cpu_threads=int(torch.get_num_threads()))
        if info['cuda_available']:
            info['cudnn_version'] = torch.backends.cudnn.version()
            for index in range(torch.cuda.device_count()):
                properties = torch.cuda.get_device_properties(index)
                device = {'index': index, 'name': properties.name,
                          'total_memory_bytes': int(properties.total_memory),
                          'compute_capability': [properties.major, properties.minor]}
                try:
                    free, total = torch.cuda.mem_get_info(index)
                    device.update(available_memory_bytes=int(free),
                                  reported_total_memory_bytes=int(total))
                except (RuntimeError, OSError) as exc:
                    device.update(available_memory_bytes=None, memory_error=str(exc))
                info['devices'].append(device)
    except (ImportError, OSError, RuntimeError, AttributeError) as exc:
        info['error'] = str(exc)
    return report
