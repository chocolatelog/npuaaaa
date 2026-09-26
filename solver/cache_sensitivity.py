"""固定C方案的L2容量/带宽敏感性；与默认配置正式成绩分开统计。"""
import re
import statistics

MIB = 1024 ** 2


def sweep_points():
    # 默认点共享一次；两条单因素曲线，不冒称完整笛卡尔网格。
    return [(MIB, 250)] + [(c, 250) for c in (0, MIB // 2, 2*MIB, 3*MIB, 4*MIB)] + [
        (MIB, b) for b in (60, 100, 200, 300, 400, 500)]


def sensitivity_config(source, capacity, bandwidth):
    if type(capacity) is not int or not 0 <= capacity <= 4*MIB:
        raise ValueError('缓存容量超出0～4MiB')
    if type(bandwidth) is not int or not 60 <= bandwidth <= 500:
        raise ValueError('缓存带宽超出60～500字节/周期')
    for key, value in (('cache_capacity_bytes', capacity), ('cache_bandwidth_bytes_per_cycle', bandwidth)):
        source, count = re.subn(r'(?m)^(' + key + r'[ \t]+)\d+([ \t]*\r?)$',
                               lambda m: m[1] + str(value) + m[2], source)
        if count != 1:
            raise ValueError('配置项必须恰好一次：' + key)
    return source


def summarize_rows(rows, expected_cases):
    expected_cases = set(expected_cases)
    seen = set()
    for r in rows:
        key = (r['case'], r['capacity_bytes'], r['cache_bandwidth'])
        if key in seen or r['case'] not in expected_cases or key[1:] not in sweep_points():
            raise ValueError('扫描结果重复或不属于预期范围')
        seen.add(key)
    groups = []
    for capacity, bandwidth in sweep_points():
        subset = [r for r in rows if (r['capacity_bytes'], r['cache_bandwidth']) == (capacity, bandwidth)]
        complete = len(subset) == len(expected_cases)
        group = dict(capacity_bytes=capacity, capacity_mib=capacity/MIB, cache_bandwidth=bandwidth,
                     count=len(subset), complete=complete,
                     mean_makespan=statistics.mean(r['real']['makespan'] for r in subset) if complete else None)
        if complete:
            for field in ('added_copy_bytes', 'scheduled_copy_bytes', 'spill_added', 'cache_hit_rate'):
                group['mean_' + field] = statistics.mean(r['real'][field] for r in subset)
            group['mean_ratio_vs_default'] = statistics.mean(r['baseline_makespan']/r['real']['makespan'] for r in subset)
            group['slower_than_default'] = sum(r['real']['makespan'] > r['baseline_makespan'] for r in subset)
        groups.append(group)
    return dict(expected=len(expected_cases)*len(sweep_points()), completed=len(rows), groups=groups,
                scope='原官方评测器的非默认硬件参数敏感性；固定方案；不并入默认竞赛成绩')
