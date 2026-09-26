"""第二、三问官方观测的逐字段等价与独立状态测试。"""
import copy
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'solver'))
sys.path.insert(0, str(ROOT / '通用神经网络处理器下的多核调度问题附件/code'))
import multicore_cut_evaluate_problem_2 as official_b
import multicore_cut_evaluate_problem_3 as official_c
from stub_multicore_cut_and_schedule import MulticoreCutError


def api():
    assert importlib.util.find_spec('official_observer') is not None, '官方只读观测接口尚未实现'
    from official_observer import OfficialObserver
    return OfficialObserver


def cold_graph():
    return {'ops': [
        {'id': 1, 'op': 'COPY_IN', 'pipe': 'PIPE_MTE2', 'cycles': 1},
        {'id': 2, 'op': 'MATMUL', 'pipe': 'PIPE_M', 'cycles': 100},
        {'id': 3, 'op': 'MATMUL', 'pipe': 'PIPE_M', 'cycles': 100}],
        'tensors': [{'id': 10001, 'pos': 'DDR', 'size': 600},
                    {'id': 10002, 'pos': 'L1', 'size': 600},
                    {'id': 10003, 'pos': 'UB', 'size': 60}],
        'edges': [{'source': a, 'target': b} for a, b in
                  [(10001, 1), (1, 10002), (10002, 2), (10002, 3), (2, 10003), (10003, 3)]]}


def configuration(tmp_path, l1=524288, cache=1048576, bandwidth=60):
    path = tmp_path / 'config.txt'
    path.write_text(f'[capacity]\nL1 {l1}\nUB 131072\n[bandwidth]\nbandwidth {bandwidth}\n'
                    '[multicore_scene_b]\ncross_core_copy_delay_cycles 500\n'
                    f'[problem_3]\ncache_capacity_bytes {cache}\ncache_bandwidth_bytes_per_cycle 250\n',
                    encoding='utf-8')
    return path


@pytest.mark.parametrize('scene,mod', [('B', official_b), ('C', official_c)])
def test_observation_matches_complete_official_result_and_empty_core(scene, mod):
    observer = api()()
    graph = cold_graph()
    plan = {'node_to_subgraph': {'2': 0, '3': 1}, 'core_schedules': [[0], [], [1]]}
    before = copy.deepcopy((graph, plan))
    bindings = dict(mod.__dict__)
    args = {'bandwidth': 60, 'capacity': {'L1': 524288, 'UB': 131072}, 'cross_core_copy_delay': 500}
    function = mod.evaluate_scene_b if scene == 'B' else mod.evaluate_problem_3
    if scene == 'C':
        args.update(cache_capacity_bytes=1048576, cache_bandwidth_bytes_per_cycle=250)
    expected = function(graph, plan, **args)
    observed = observer.evaluate(graph, plan, scene)
    assert observed['result'] == expected
    assert observer.evaluate(graph, plan, scene, observe=False)['observations'] == []
    assert (graph, plan) == before
    assert set(bindings) == set(mod.__dict__)
    assert all(mod.__dict__[name] is value for name, value in bindings.items())
    cores = observed['observations']
    assert [item['core_id'] for item in cores] == [0, 1, 2]
    assert cores[1]['seq'] == cores[1]['result2']['seq_ext'] == []
    assert cores[1]['graph']['ops'] == []
    assert all('memory_dependencies' in item['prepared']['step3'] for item in cores)
    expanded = observer.expand(graph, plan, scene)
    expected_build = mod._build_scene_b_tasks(graph, plan, args['bandwidth'], args['capacity'])
    assert expanded['built'] == expected_build
    assert expanded['observations'] == cores
    assert len(observed['source_digest']) == len(list((ROOT / '通用神经网络处理器下的多核调度问题附件/code').glob('*.py')))


def test_cold_read_events_and_explicit_zero_cache_config(tmp_path):
    graph = cold_graph()
    plan = {'node_to_subgraph': {'2': 0, '3': 1}, 'core_schedules': [[0], [1]]}
    observer = api()()
    result = observer.evaluate(graph, plan, 'C')['result']
    misses = [e for e in result['cache_events'] if e['event'] == 'miss' and e['tensor_id'] == 10002]
    assert len(misses) == 2 and all(e['time'] == 0 for e in misses)
    zero = api()(configuration(tmp_path, cache=0)).evaluate(graph, plan, 'C')['result']
    baseline = observer.evaluate(graph, plan, 'B')['result']
    assert zero['makespan'] == baseline['makespan'] == 722
    assert zero['data_movement_bytes'] == baseline['data_movement_bytes']
    assert zero['cache_capacity_bytes'] == 0


def test_spill_records_are_actual_intermediates(tmp_path):
    graph = {'ops': [{'id': i, 'op': 'MATMUL', 'pipe': 'PIPE_M', 'cycles': 1} for i in range(1, 10)],
             'tensors': [], 'edges': []}
    for tid, uses in [(1001, [1, 5, 9]), (1002, [2, 3]), (1003, [6, 7])]:
        graph['tensors'].append({'id': tid, 'pos': 'L1', 'size': 60})
        graph['edges'].append({'source': uses[0], 'target': tid})
        graph['edges'].extend({'source': tid, 'target': op} for op in uses[1:])
    plan = {'node_to_subgraph': {str(i): i for i in range(1, 10)}, 'core_schedules': [list(range(1, 10))]}
    observer = api()(configuration(tmp_path, l1=100))
    obs = observer.expand(graph, plan, 'B')['observations'][0]
    expected = official_b.step2_spill_insertion(obs['graph'], obs['seq'], capacity={'L1': 100, 'UB': 131072})
    assert obs['result2'] == expected
    assert obs['result2']['spill_records']
    assert obs['extended_graph'] == official_b._build_extended_graph(obs['graph'], expected)


def test_bounded_cache_distinguishes_core_order_scene_and_returns_copies(tmp_path):
    observer = api()(configuration(tmp_path))
    from scene_bc_expand import BCExpansionCache
    cache = BCExpansionCache(observer, max_entries=2)
    graph = {'ops': [{'id': i, 'op': 'MATMUL', 'pipe': 'PIPE_M', 'cycles': 1} for i in (1, 2)],
             'tensors': [], 'edges': []}
    plan = {'node_to_subgraph': {'1': 7, '2': 9}, 'core_schedules': [[7, 9], []]}
    original = copy.deepcopy(plan)
    first = cache.expand(graph, plan, 'B')
    first['built'][0][0]['seq'].append(-1)
    assert -1 not in cache.expand(graph, plan, 'B')['built'][0][0]['seq']
    assert cache.hits == 1 and cache.misses == 1
    reverse = copy.deepcopy(plan)
    reverse['core_schedules'][0].reverse()
    second = cache.expand(graph, reverse, 'B')
    assert second['observations'][0]['seq'] == [2, 1]
    cache.expand(graph, reverse, 'C')
    assert cache.misses == 3 and len(cache) == 2
    assert plan == original
    configuration(tmp_path, bandwidth=30)
    cache.expand(graph, reverse, 'C')
    assert cache.misses == 4


def test_failed_observation_does_not_change_official_bindings():
    observer = api()()
    original = official_b._build_scene_b_tasks
    with pytest.raises(MulticoreCutError):
        observer.expand(cold_graph(), {'node_to_subgraph': {}, 'core_schedules': [[]]}, 'B')
    assert official_b._build_scene_b_tasks is original
    with pytest.raises(ValueError, match='场景'):
        observer.evaluate(cold_graph(), {}, 'A')


def test_shared_observer_keeps_concurrent_calls_isolated():
    observer = api()()
    graph = cold_graph()
    plans = [
        {'node_to_subgraph': {'2': 0, '3': 1}, 'core_schedules': [[0], [1]]},
        {'node_to_subgraph': {'2': 0, '3': 1}, 'core_schedules': [[0, 1], []]},
    ]
    jobs = [(plan, scene) for plan in plans for scene in ('B', 'C')]
    expected = [observer.evaluate(graph, plan, scene) for plan, scene in jobs]
    with ThreadPoolExecutor(max_workers=4) as pool:
        actual = list(pool.map(lambda job: observer.evaluate(graph, *job), jobs))
    assert actual == expected


def test_cache_retains_numbered_plan_graph_and_source_binding(tmp_path):
    observer = api()(configuration(tmp_path))
    from scene_bc_expand import BCExpansionCache
    cache = BCExpansionCache(observer, max_entries=2)
    graph = cold_graph()
    plan = {'node_to_subgraph': {'2': 0, '3': 1}, 'core_schedules': [[0], [1]]}
    cache.expand(graph, plan, 'B')
    renamed = {'node_to_subgraph': {'2': 8, '3': 9}, 'core_schedules': [[8], [9]]}
    cache.expand(graph, renamed, 'B')
    changed_graph = copy.deepcopy(graph)
    changed_graph['ops'][1]['cycles'] += 1
    cache.expand(changed_graph, renamed, 'B')
    assert cache.hits == 0 and cache.misses == 3
    disabled = BCExpansionCache(observer, max_entries=0)
    disabled.expand(graph, plan, 'B')
    assert len(disabled) == 0
    # 只改观测器自己的期待值，不改官方文件，验证源码变化不可命中旧条目。
    observer._sources = {}
    with pytest.raises(RuntimeError, match='源码'):
        cache.expand(changed_graph, renamed, 'B')
