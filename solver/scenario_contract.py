"""只读官方方案契约适配；B/C不使用A整任务串行边判死锁。"""
from pathlib import Path
import sys

CODE = Path(__file__).resolve().parents[1] / '通用神经网络处理器下的多核调度问题附件/code'
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))
from stub_multicore_cut_and_schedule import (
    derive_multicore_plan, _build_op_adjacency, _contract_excluded_copy_nodes)
from evaluation_validation import validate_task_order


def validate_plan(graph, plan, scenario='B'):
    if scenario not in ('A', 'B', 'C'):
        raise ValueError('未知场景')
    try:
        view = derive_multicore_plan(graph, plan)
        if scenario == 'A':
            validate_task_order(view)
        return view
    except (ValueError, RuntimeError) as exc:
        raise ValueError(str(exc)) from exc


def operation_dependencies(graph):
    _, successors = _build_op_adjacency(graph)
    eligible = sorted(o['id'] for o in graph['ops'] if o['op'] not in ('COPY_IN', 'COPY_OUT'))
    return _contract_excluded_copy_nodes(eligible, successors)

