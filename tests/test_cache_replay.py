import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'solver'))


def test_fifo_replay_inserts_only_after_completion_and_hits_next_access():
    from cache_replay import replay_fifo_cache

    result = replay_fifo_cache([
        {'tensor_id': 1, 'size': 60, 'issue_time': 0},
        {'tensor_id': 1, 'size': 60, 'issue_time': 0.5},
        {'tensor_id': 1, 'size': 60, 'issue_time': 2},
    ], capacity=100, ddr_bandwidth=60, cache_bandwidth=250)
    assert [row['hit'] for row in result['events']] == [False, False, True]
    assert result['miss_count'] == 2
    assert result['hit_count'] == 1


def test_fifo_replay_evicts_oldest_and_rejects_oversize():
    from cache_replay import replay_fifo_cache

    result = replay_fifo_cache([
        {'tensor_id': 1, 'size': 60, 'issue_time': 0},
        {'tensor_id': 2, 'size': 60, 'issue_time': 2},
        {'tensor_id': 3, 'size': 120, 'issue_time': 4},
    ], capacity=100, ddr_bandwidth=60, cache_bandwidth=250)
    assert result['evictions'] == 1
    assert result['events'][-1]['hit'] is False
    assert result['used_bytes_final'] == 60


def test_official_insert_events_are_not_counted_as_accesses():
    from cache_replay import replay_official_events

    result = replay_official_events([
        {'event': 'miss', 'tensor_id': 1, 'size_bytes': 10, 'time': 0},
        {'event': 'insert', 'tensor_id': 1, 'size_bytes': 10, 'time': 1},
        {'event': 'hit', 'tensor_id': 1, 'size_bytes': 10, 'time': 2},
    ], capacity=100, ddr_bandwidth=10, cache_bandwidth=100)
    assert len(result['events']) == 2


def test_duplicate_inflight_completion_preserves_fifo_and_capacity():
    from cache_replay import replay_fifo_cache
    r = replay_fifo_cache([
        {'tensor_id': 1, 'size': 40, 'time': 0},
        {'tensor_id': 1, 'size': 40, 'time': 0},
        {'tensor_id': 2, 'size': 40, 'time': 2},
        {'tensor_id': 3, 'size': 40, 'time': 4},
    ], 100)
    assert r['used_bytes_final'] == sum(x['size'] for x in r['final_entries'])
    assert r['evictions'] == 1
    assert [x['tensor_id'] for x in r['final_entries']] == [2, 3]


def test_official_timeline_drives_fill_not_ideal_bandwidth():
    from cache_replay import audit_official_cache
    result = {'cache_capacity_bytes': 100, 'cache_events': [
        {'event':'miss','time':0,'tensor_id':1,'size_bytes':60,'core_id':0,'op_id':1},
        {'event':'miss','time':1,'tensor_id':1,'size_bytes':60,'core_id':1,'op_id':2},
        {'event':'insert','time':3,'tensor_id':1,'size_bytes':60,'used_bytes':60,
         'evicted_tensor_ids':[],'core_id':0,'op_id':1}],
        'per_core_timeline': [
            {'core_id':0,'ops':[{'op_id':1,'op':'COPY_IN','start':0,'end':3}]},
            {'core_id':1,'ops':[{'op_id':2,'op':'COPY_IN','start':1,'end':4}]}],
        'cache_stats':{'hit_bytes':0,'miss_bytes':120},
        'cache_final_entries':[{'tensor_id':1,'size_bytes':60}],
        'cache_used_bytes_final':60}
    assert audit_official_cache(result)['consistent']
