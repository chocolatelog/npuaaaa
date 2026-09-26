"""入口接受自动并发及一核，解析不触发求解或覆盖已有结果。"""
import sys
from pathlib import Path
from unittest.mock import patch
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'solver'))
import run_all
import runtime_resources


def test_auto_workers_and_single_core_reach_runner(tmp_path):
    seen=[]
    argv=['run_all','--cases','1','--cores','1','--scenes','B',
          '--workers','auto','--output-dir',str(tmp_path/'plans'),
          '--log-file',str(tmp_path/'run.jsonl')]
    with (patch.object(sys,'argv',argv),
          patch.object(runtime_resources.os,'cpu_count',return_value=12),
          patch.object(runtime_resources,'memory_snapshot',return_value={'available_bytes':16*runtime_resources.GIB}),
          patch.object(run_all,'run_jobs',side_effect=lambda a,c,n,s:seen.append((a.workers,n,s)))):
        run_all.main()
    assert seen == [(8,[1],['B'])]


def test_auto_workers_low_memory_refuses_before_runner(tmp_path, capsys):
    argv=['run_all','--cases','1','--cores','1','--scenes','B',
          '--workers','auto','--output-dir',str(tmp_path/'plans'),
          '--log-file',str(tmp_path/'run.jsonl')]
    with (patch.object(sys,'argv',argv),
          patch.object(runtime_resources.os,'cpu_count',return_value=12),
          patch.object(runtime_resources,'memory_snapshot',return_value={'available_bytes':2*runtime_resources.GIB}),
          patch.object(run_all,'run_jobs') as runner,
          pytest.raises(SystemExit) as error):
        run_all.main()
    assert error.value.code == 2
    runner.assert_not_called()
    assert '不足以容纳单个工作进程' in capsys.readouterr().err


def test_worker_environment_set_before_solver_import():
    import inspect
    body=inspect.getsource(run_all.solve_task)
    assert body.index('initialize_worker_threads()') < body.index('from pipeline import solve_case')
