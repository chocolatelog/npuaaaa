"""参数扫描不污染官方固定配置，覆盖完整轴并隔离断点身份。"""
from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'solver'))


def test_axes_share_default_only_and_have_required_endpoints():
    from cache_sensitivity import sweep_points
    points = sweep_points()
    assert len(points) == len(set(points)) == 12
    assert (0, 250) in points and (4 * 1048576, 250) in points
    assert (1048576, 60) in points and (1048576, 500) in points
    assert points[0] == (1048576, 250)
    assert all(c == 1048576 or b == 250 for c, b in points)


def test_config_changes_only_two_requested_values():
    from cache_sensitivity import sensitivity_config
    source = '# 固定配置\n[bandwidth]\nbandwidth 60\n[problem_3]\ncache_capacity_bytes 1048576\ncache_bandwidth_bytes_per_cycle 250\n'
    assert sensitivity_config(source, 1048576, 250) == source
    result = sensitivity_config(source, 0, 500)
    assert result == source.replace('cache_capacity_bytes 1048576', 'cache_capacity_bytes 0').replace('cache_bandwidth_bytes_per_cycle 250', 'cache_bandwidth_bytes_per_cycle 500')
    with pytest.raises(ValueError):
        sensitivity_config(source, -1, 250)
    with pytest.raises(ValueError):
        sensitivity_config(source.replace('cache_capacity_bytes', 'wrong'), 0, 500)


def test_reporting_keeps_missing_cases_out_of_completed_curve():
    from cache_sensitivity import summarize_rows
    rows = [dict(case='case_001', capacity_bytes=0, cache_bandwidth=250,
                 real=dict(makespan=100, added_copy_bytes=4, scheduled_copy_bytes=10,
                           spill_added=0, cache_hit_rate=0), baseline_makespan=80)]
    result = summarize_rows(rows, expected_cases=['case_001', 'case_002'])
    group = next(g for g in result['groups'] if g['capacity_bytes'] == 0)
    assert not group['complete'] and group['count'] == 1
    assert group['mean_makespan'] is None
    assert result['expected'] == 24 and result['completed'] == 1
