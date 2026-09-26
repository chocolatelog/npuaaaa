"""并发预算、子进程线程和按需环境采集的机制验收。"""
import importlib
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'solver'))
GIB = 1024 ** 3


class RuntimeResourceTests(unittest.TestCase):
    def runtime(self):
        self.assertIsNotNone(importlib.util.find_spec('runtime_resources'),
                             '缺少独立资源管理模块')
        return importlib.import_module('runtime_resources')

    def resolve(self, requested='auto', **kwargs):
        values = dict(cpu_count=12, available_memory=32 * GIB,
                      reserve=2 * GIB, per_worker_estimate=GIB)
        values.update(kwargs)
        return self.runtime().resolve_workers(requested, **values)

    def test_twelve_logical_processors_start_at_eight(self):
        result = self.resolve()
        self.assertEqual(result.workers, 8)
        self.assertEqual(result.cpu_limit, 8)
        self.assertEqual(result.requested, 'auto')
        json.dumps(result.to_dict())

    def test_memory_reserve_limits_concurrency(self):
        result = self.resolve(available_memory=5 * GIB)
        self.assertEqual(result.workers, 3)
        self.assertEqual(result.memory_limit, 3)

    def test_device_budget_limits_concurrency(self):
        result = self.resolve(gpu_available_memory=7 * GIB,
                              gpu_reserve=GIB, gpu_per_worker_estimate=2 * GIB)
        self.assertEqual(result.workers, 3)
        self.assertEqual(result.gpu_limit, 3)

    def test_auto_refuses_when_even_one_worker_exceeds_budget(self):
        with self.assertRaisesRegex(RuntimeError, '单个'):
            self.resolve(available_memory=2 * GIB)
        with self.assertRaisesRegex(RuntimeError, '单个'):
            self.resolve(gpu_available_memory=GIB, gpu_reserve=GIB,
                         gpu_per_worker_estimate=GIB)

    def test_manual_count_is_preserved_with_actionable_advice(self):
        with self.assertWarnsRegex(RuntimeWarning, '建议.*3'):
            result = self.resolve('12', available_memory=5 * GIB)
        self.assertEqual(result.workers, 12)
        self.assertEqual(result.recommended_workers, 3)
        self.assertTrue(result.advice)

    def test_invalid_worker_and_budget_inputs_are_rejected(self):
        for value in (0, -1, 1.5, True, '0', '-2', '1.5', 'all'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.resolve(value)
        for kwargs in ({'cpu_count': 0}, {'reserve': -1},
                       {'per_worker_estimate': 0}, {'available_memory': -1},
                       {'gpu_per_worker_estimate': GIB},
                       {'gpu_reserve': -1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.resolve(**kwargs)

    def test_single_cpu_remains_usable_and_manual_small_count_is_quiet(self):
        self.assertEqual(self.resolve(cpu_count=1).workers, 1)
        self.assertFalse(self.resolve(2).advice)

    def test_thread_initializer_overrides_existing_values(self):
        runtime = self.runtime()
        names = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                 'NUMEXPR_NUM_THREADS')
        with patch.dict(os.environ, {name: '12' for name in names}):
            runtime.initialize_worker_threads()
            self.assertEqual({os.environ[name] for name in names}, {'1'})

    def test_default_report_is_json_serializable_and_does_not_import_torch(self):
        self.runtime()
        program = ("import json,sys; sys.path.insert(0, 'solver'); "
                   "import runtime_resources as r; "
                   "a=r.collect_runtime_environment(); "
                   "assert 'torch' not in sys.modules; print(json.dumps(a))")
        completed = subprocess.run([sys.executable, '-c', program], cwd=ROOT,
                                   text=True, capture_output=True, check=True)
        report = json.loads(completed.stdout)
        self.assertEqual(report['python']['executable'], sys.executable)
        self.assertIn('torch', report['packages'])
        self.assertGreaterEqual(report['cpu']['logical_count'], 1)
        self.assertGreater(report['memory']['available_bytes'], 0)
        self.assertFalse(report['torch']['inspected'])

    def test_explicit_torch_inspection_reports_missing_dependency(self):
        runtime = self.runtime()
        with patch.object(runtime.importlib, 'import_module',
                          side_effect=ImportError('测试用缺失依赖')):
            report = runtime.collect_runtime_environment(include_torch=True)
        self.assertTrue(report['torch']['inspected'])
        self.assertFalse(report['torch']['cuda_available'])
        self.assertIn('测试用缺失依赖', report['torch']['error'])


if __name__ == '__main__':
    unittest.main()
