import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'solver'))


def test_fifo_cache_keeps_inflight_reads_as_misses_and_does_not_refresh_hits():
    from resource_state_bc import FifoCache

    cache = FifoCache(capacity=100)
    assert cache.access('a', 60) == 'miss'
    assert cache.access('a', 60) == 'hit'
    assert cache.access('b', 40) == 'miss'
    # 命中不刷新先进先出顺序，a 仍是最早条目。
    assert cache.access('c', 40) == 'miss'
    assert cache.stats['evictions'] == 1
    assert cache.contains('a') is False


def test_resource_table_reports_c_cache_bytes_and_b_pool_peaks():
    from resource_state_bc import build_resource_table

    graph = {
        'ops': [
            {'id': 1, 'op': 'A', 'pipe': 'PIPE_M', 'cycles': 2},
            {'id': 2, 'op': 'B', 'pipe': 'PIPE_V', 'cycles': 3},
        ],
        'tensors': [{'id': 10, 'pos': 'L1', 'size': 40}],
        'edges': [{'source': 1, 'target': 10}, {'source': 10, 'target': 2}],
    }
    plan = {'node_to_subgraph': {'1': 0, '2': 1},
            'core_schedules': [[0], [1]]}
    b = build_resource_table(graph, plan, 'B')
    c = build_resource_table(graph, plan, 'C')
    assert b['pool_peaks']['L1'] >= 40
    assert c['cache']['miss_bytes'] >= 0
    assert 'events' in c


def test_versioned_table_invalidates_only_cores_touching_changed_tensors():
    from resource_state_bc import VersionedResourceTable

    table = VersionedResourceTable(cores=3)
    table.update(0, [{'tensor_id': 7, 'size': 8}])
    table.update(1, [{'tensor_id': 9, 'size': 8}])
    before = table.snapshot()
    affected = table.invalidate_tensors([7])
    assert affected == [0]
    assert table.snapshot()['versions']['0'] > before['versions']['0']
    assert table.snapshot()['versions']['1'] == before['versions']['1']


def test_fifo_multiple_inflight_reads_do_not_delay_first_completion():
    from resource_state_bc import FifoCache
    cache = FifoCache(100)
    assert cache.access('a', 60, 0, 10) == 'miss'
    assert cache.access('a', 60, 1, 20) == 'inflight_miss'
    assert cache.access('a', 60, 10, 2) == 'hit'
    assert cache.used == 60


def test_fifo_completes_other_tensors_before_new_access():
    from resource_state_bc import FifoCache
    cache = FifoCache(100)
    cache.access('a', 60, 0, 2)
    cache.access('b', 60, 3, 1)
    # b 在4完成并淘汰a，无需再次访问b才能触发入缓存。
    assert cache.access('a', 60, 4, 1) == 'miss'
    assert list(cache.entries) == ['b']


def test_fifo_hit_inflight_can_reinsert_after_eviction_without_refreshing():
    from resource_state_bc import FifoCache
    cache = FifoCache(100)
    cache.access('a', 60, 0, 0)
    assert cache.access('a', 60, 1, 5) == 'hit'
    cache.access('b', 60, 2, 0)
    assert not cache.contains('a')
    assert cache.access('a', 60, 6, 1) == 'hit'
    assert list(cache.entries) == ['a']
    assert cache.used == 60
    assert cache.stats['evictions'] == 2


def test_fifo_request_timeline_matches_independent_cache_replay():
    from resource_state_bc import FifoCache
    from cache_replay import replay_fifo_cache
    events = [dict(tensor_id=t, size=s, issue_time=n)
              for t,s,n in [(1,60,0),(1,60,1),(2,60,7),(1,60,8),(1,60,13),(3,40,14),(1,60,18)]]
    expected = replay_fifo_cache(events,100,10,20)
    cache = FifoCache(100)
    for event, row in zip(events,expected['events']):
        outcome = cache.access(event['tensor_id'],event['size'],event['issue_time'],row['service_cycles'])
        assert (outcome == 'hit') == row['hit']
    cache.complete_until(100)
    assert cache.stats['hit_bytes'] == expected['hit_bytes']
    assert cache.stats['miss_bytes'] == expected['miss_bytes']
    assert cache.stats['evictions'] == expected['evictions']
    assert list(cache.entries) == [e['tensor_id'] for e in expected['final_entries']]


def test_resource_positions_follow_dependency_not_numeric_operator_id():
    from resource_state_bc import build_resource_table
    graph=dict(ops=[dict(id=100,op='ADD',cycles=7),dict(id=2,op='ADD',cycles=3)],
               tensors=[dict(id=i,size=40,pos='UB') for i in (10,88,99)],
               edges=[dict(source=a,target=b) for a,b in [(99,100),(100,10),(10,2),(88,2)]])
    plan=dict(node_to_subgraph={'100':0,'2':0},core_schedules=[[0]])
    table=build_resource_table(graph,plan,'C')
    times={e['tensor_id']:e['time'] for e in table['events']}
    assert times[99]==0 and times[88]==7


def test_original_ddr_input_is_still_read_into_local_buffer_and_cache_eligible():
    from resource_state_bc import build_resource_table
    graph=dict(ops=[dict(id=1,op='ADD',cycles=3)],tensors=[dict(id=10,size=40,pos='DDR')],
               edges=[dict(source=10,target=1)])
    plan=dict(node_to_subgraph={'1':0},core_schedules=[[0]])
    table=build_resource_table(graph,plan,'C')
    assert len(table['events'])==1 and table['events'][0]['tensor_id']==10
    assert table['pool_peaks']['UB']==40
