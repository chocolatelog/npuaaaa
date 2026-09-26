import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'solver'))


def test_shared_bandwidth_recomputes_completion_when_requests_overlap():
    from bandwidth_proxy import simulate_bandwidth

    result = simulate_bandwidth([
        {'request_id': 'a', 'start': 0, 'size': 60, 'core': 0, 'critical': True},
        {'request_id': 'b', 'start': 0, 'size': 60, 'core': 1, 'critical': False},
    ], bandwidth=60)
    rows = {row['request_id']: row for row in result['events']}
    assert result['max_concurrency'] == 2
    assert rows['a']['completion'] == 2.0
    assert rows['a']['contention_cycles'] == 1.0
    assert result['critical_path_wait'] == 1.0


def test_bandwidth_proxy_processes_completion_before_same_time_arrival():
    from bandwidth_proxy import simulate_bandwidth

    result = simulate_bandwidth([
        {'request_id': 'a', 'start': 0, 'size': 60, 'core': 0},
        {'request_id': 'b', 'start': 1, 'size': 60, 'core': 1},
    ], bandwidth=60)
    rows = {row['request_id']: row for row in result['events']}
    assert rows['a']['completion'] == 1.0
    assert rows['b']['completion'] == 2.0
    assert result['max_concurrency'] == 1


def test_bandwidth_proxy_is_diagnostic_only_and_opt_in():
    from model import Model

    graph = {
        'ops': [
            {'id': 0, 'op': 'MATMUL', 'pipe': 'PIPE_M', 'cycles': 10},
            {'id': 1, 'op': 'MATMUL', 'pipe': 'PIPE_M', 'cycles': 20},
        ],
        'tensors': [],
        'edges': [],
    }
    model = Model(graph, block_ops_cap=1)
    baseline = model.evaluate([0, 0], [0], 'A', 1, use_cache=False)
    assert 'bandwidth_proxy' not in baseline[2]

    model.use_bandwidth_proxy = True
    with_proxy = model.evaluate([0, 0], [0], 'A', 1, use_cache=False)
    assert with_proxy[:2] == baseline[:2]
    assert with_proxy[2]['bandwidth_proxy']['request_count'] == 0


def test_bandwidth_proxy_rejects_duplicate_request_ids():
    from bandwidth_proxy import simulate_bandwidth

    try:
        simulate_bandwidth([
            {'request_id': 'same', 'start': 0, 'size': 1},
            {'request_id': 'same', 'start': 1, 'size': 1},
        ])
    except ValueError as exc:
        assert '重复' in str(exc)
    else:
        raise AssertionError('duplicate request ids must be rejected')


def test_bandwidth_proxy_scales_to_large_same_time_batch():
    from bandwidth_proxy import simulate_bandwidth

    events = [
        {'request_id': str(index), 'start': 0, 'size': 1, 'core': index % 4}
        for index in range(5000)
    ]
    started = time.perf_counter()
    result = simulate_bandwidth(events, bandwidth=60)
    elapsed = time.perf_counter() - started
    assert len(result['events']) == len(events)
    assert result['max_concurrency'] == len(events)
    assert elapsed < 0.8


def test_bandwidth_proxy_marks_bounded_large_batch_as_approximate():
    from bandwidth_proxy import simulate_bandwidth

    events = [
        {'request_id': str(index), 'start': 0, 'size': 1, 'core': index % 4}
        for index in range(300)
    ]
    result = simulate_bandwidth(events, bandwidth=60, exact_limit=128)
    assert result['mode'] == 'sweep_approx'
    assert result['request_count'] == len(events)
    assert result['max_concurrency'] == len(events)
