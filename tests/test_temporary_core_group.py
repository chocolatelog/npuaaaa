import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'solver'))


def test_temporary_core_group_releases_members_independently():
    from temporary_core_group import simulate_temporary_core_group

    result = simulate_temporary_core_group(
        tasks={
            'heavy_a': {'duration': 5, 'preds': []},
            'heavy_b': {'duration': 2, 'preds': []},
            'tail': {'duration': 1, 'preds': ['heavy_a']},
        },
        core_set=[0, 1],
        orders={0: ['heavy_a'], 1: ['heavy_b']},
        external=[{'task_id': 'tail', 'core': 1, 'ready_time': 0, 'duration': 1}],
    )
    assert result['legal'] is True
    assert result['release_times'][1] == 2.0
    assert result['release_times'][0] == 5.0
    assert result['external_delays']['tail'] == 2.0


def test_temporary_core_group_rejects_one_task_on_multiple_cores():
    from temporary_core_group import simulate_temporary_core_group

    result = simulate_temporary_core_group(
        tasks={'x': {'duration': 1, 'preds': []}},
        core_set=[0, 1],
        orders={0: ['x'], 1: ['x']},
    )
    assert result['legal'] is False
    assert result['reason'] == 'duplicate_task_assignment'


def test_temporary_core_group_rejects_unknown_predecessor():
    from temporary_core_group import simulate_temporary_core_group

    result = simulate_temporary_core_group(
        tasks={'x': {'duration': 1, 'preds': ['missing']}},
        core_set=[0],
        orders={0: ['x']},
    )
    assert result['legal'] is False
    assert result['reason'] == 'unknown_predecessor'
