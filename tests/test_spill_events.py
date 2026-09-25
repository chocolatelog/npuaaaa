"""用可手算的多次逐出反例核验逐事件搬运计数。"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'solver'))
sys.path.insert(0, str(ROOT / '通用神经网络处理器下的多核调度问题附件' / 'code'))
from spill_events import count_spill_events
from schedule_step2 import step2_spill_insertion


def graph_for_uses(uses, backing=False):
    ops = [{'id': i + 1, 'op': 'MATMUL', 'pipe': 'PIPE_M', 'cycles': 1}
           for i in range(9)]
    tensors, edges = [], []
    for i, steps in enumerate(uses):
        tid = 1001 + i
        tensors.append({'id': tid, 'pos': 'L1', 'size': 60})
        edges.append({'source': steps[0] + 1, 'target': tid})
        edges.extend({'source': tid, 'target': s + 1} for s in steps[1:])
    if backing:
        ops[0].update(op='COPY_IN', pipe='PIPE_MTE2')
        tensors.append({'id': 2001, 'pos': 'DDR', 'size': 60})
        edges.append({'source': 2001, 'target': 1})
    return {'ops': ops, 'tensors': tensors, 'edges': edges}


@pytest.mark.parametrize('backing,expected', [(False, 180), (True, 120)])
def test_repeated_eviction_writes_once_and_reloads_twice(backing, expected):
    graph = graph_for_uses([[0, 4, 8], [1, 2], [5, 6]], backing)
    seq, cap = list(range(1, 10)), {'L1': 100, 'UB': 100}
    actual = count_spill_events(graph, seq, cap)
    official = step2_spill_insertion(graph, seq, cap)
    total = sum(s['size'] * (1 + int(s['spill_out_copies_data']))
                for s in official['spill_records'])
    assert actual['spill_bytes'] == total == expected
    assert actual['eviction_count'] == 2


def test_dead_tensor_frees_before_later_allocation():
    graph = graph_for_uses([[0, 1], [2, 3], [4, 5]])
    actual = count_spill_events(graph, list(range(1, 10)), {'L1': 100, 'UB': 100})
    assert actual['spill_bytes'] == 0
    assert actual['peak_alloc_bytes']['L1'] == 60


def test_current_operands_cannot_be_evicted_to_hide_infeasibility():
    graph = graph_for_uses([[0, 2], [1, 2]])
    with pytest.raises(ValueError, match='当前算子'):
        count_spill_events(graph, list(range(1, 10)), {'L1': 100, 'UB': 100})


def test_bounded_event_diagnostics_preserve_all_existing_counts():
    graph = graph_for_uses([[0, 4, 8], [1, 2], [5, 6]])
    seq, cap = list(range(1, 10)), {'L1': 100, 'UB': 100}
    base = count_spill_events(graph, seq, cap)
    detailed = count_spill_events(graph, seq, cap, event_limit=3)
    assert {k: detailed[k] for k in base} == base
    events = detailed['spill_events']['L1']
    assert len(events) == 2
    assert [(e['step'], e['next_use_step'], e['read_bytes'], e['write_bytes']) for e in events] == [
        (1, 4, 60, 60), (5, 8, 60, 0)]
    assert events[0]['op_id'] == seq[events[0]['step']]
    assert events[0]['tensor_id'] == 1001 and events[0]['resident_before_bytes'] == 120
    assert events[0]['capacity_overage_bytes'] == 20
    limited = count_spill_events(graph, seq, cap, event_limit=1)
    assert limited['spill_events']['L1'] == events[:1]
    assert limited['spill_events_omitted']['L1'] == 1
    assert 'spill_events' not in base


def test_backed_tensor_diagnostic_has_no_extra_writeback():
    graph = graph_for_uses([[0, 4, 8], [1, 2], [5, 6]], backing=True)
    result = count_spill_events(graph, list(range(1, 10)), {'L1': 100, 'UB': 100}, event_limit=3)
    assert all(e['write_bytes'] == 0 and e['backed_before'] for e in result['spill_events']['L1'])


def test_tensor_aggregate_matches_official_and_keeps_repeat_cost():
    graph = graph_for_uses([[0, 4, 8], [1, 2], [5, 6]], backing=True)
    seq, cap = list(range(1, 10)), {'L1': 100, 'UB': 100}
    plain = count_spill_events(graph, seq, cap)
    detailed = count_spill_events(graph, seq, cap, tensor_limit=3)
    assert {k: detailed[k] for k in plain} == plain
    hot = detailed['spill_tensors']['L1'][0]
    assert hot['tensor_id'] == 1001
    assert (hot['read_bytes'], hot['write_bytes'], hot['evictions']) == (120, 0, 2)
    assert hot['consumer_steps'] == [4, 8]
    assert hot['first_consumer_step'] == 4 and hot['last_consumer_step'] == 8
    assert hot['reload_by_op'] == {5: 1, 9: 1}
    assert sum(r['size'] for r in step2_spill_insertion(graph, seq, cap)['spill_records']) == hot['read_bytes']
    assert 'spill_tensors' not in plain


def test_cumulative_small_tensor_outranks_one_large_eviction():
    graph = graph_for_uses([[0, 4, 8], [1, 2], [5, 6]])
    # 在原序列后加入一次较大张量逐出；60字节两次搬运=180超过80字节一次=160。
    graph['ops'].extend({'id': i, 'op': 'MATMUL', 'pipe': 'PIPE_M', 'cycles': 1} for i in range(10, 14))
    graph['tensors'].extend([{'id': 1004, 'pos': 'L1', 'size': 80}, {'id': 1005, 'pos': 'L1', 'size': 30}])
    graph['edges'].extend({'source': a, 'target': b} for a,b in [(10,1004),(1004,13),(11,1005),(1005,12)])
    result = count_spill_events(graph, list(range(1,14)), {'L1':100,'UB':100}, tensor_limit=1)
    assert result['spill_tensors']['L1'][0]['tensor_id'] == 1001
    assert result['spill_tensors_omitted']['L1'] == 1
    assert result['spill_tensor_omitted_bytes']['L1'] == 160
