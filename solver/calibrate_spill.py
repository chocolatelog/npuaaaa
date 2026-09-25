"""用官方评估真值离线校准 spill 估计器。

数据：results/official/ 中 1300 项评估的 spill_added_copy_bytes 真值
     + 对应方案解码回 (sg_of_block, core_of_sg) 重新计算四路信号。
拟合：非负最小二乘 (NNLS)，80/20 按用例划分留出验证。
输出：solver/spill_coefs.json + 拟合报告。
"""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
RESULTS = os.path.normpath(os.path.join(HERE, '..', 'results'))
ATTACH = os.path.normpath(os.path.join(
    HERE, '..', '通用神经网络处理器下的多核调度问题附件'))
SEGMENTED_OUT = os.path.join(RESULTS, 'spill_coefs_segmented_v1.json')
L1_CAP = 524288.0


def feature_vector(row):
    """返回可在代理阶段获得的分段拟合特征。"""
    sig = row['sig']
    peak = row['peak']
    return [sig['L1p'], sig['L1w'], sig['UBp'], sig['UBw'],
            peak['L1'], peak['UB'], row['partition_added'] / 1e6,
            row['n_ops'] / 1000.0, 1.0]


def segment_key(row):
    """按规模、L1 压力、核数和场景划分，保持段数量可控。"""
    n_ops = row['n_ops']
    size = 'small' if n_ops <= 2000 else ('medium' if n_ops <= 6000 else 'large')
    pressure = row['peak']['L1'] / L1_CAP
    pressure_bin = 'low' if pressure <= 0.85 else ('mid' if pressure <= 1.25 else 'high')
    core_bin = 'lowcore' if row['N'] <= 2 else 'highcore'
    return f"{row['scene']}|{size}|{pressure_bin}|{core_bin}"


def split_by_case(data):
    """按算例切分，强制关键失败样本留出，防止同一算例泄漏。"""
    cases = sorted({row['case'] for row in data})
    holdout = {'case_044', 'case_067'}
    holdout.update(cases[::5])
    train = [row for row in data if row['case'] not in holdout]
    test = [row for row in data if row['case'] in holdout]
    return train, test, sorted(holdout)


def scene_from_parts(parts):
    """把官方 problem 编号映射到题目场景。"""
    if len(parts) > 3 and parts[2] == 'problem':
        return {1: 'A', 2: 'B', 3: 'C'}.get(int(parts[3]))
    return None


def block_cap_for(n_eligible):
    return int(max(24, min(240, n_eligible // 400)))


def decode_plan(model, plan):
    sg_of_block = []
    for blk in model.blocks:
        sg_of_block.append(plan['node_to_subgraph'][str(blk[0])])
    K = max(sg_of_block) + 1
    core_of_sg = [0] * K
    for core_id, order in enumerate(plan['core_schedules']):
        for sg in order:
            if sg < K:
                core_of_sg[sg] = core_id
    return sg_of_block, core_of_sg


def collect():
    from model import load_graph, Model
    data = []
    stats = {'files': 0, 'valid': 0, 'skipped_missing_plan': 0,
             'skipped_invalid_plan': 0, 'by_scene': {s: 0 for s in 'ABC'}}
    graphs = {}
    files = sorted(glob.glob(os.path.join(RESULTS, 'official', '*_res.json')))
    for path in files:
        stats['files'] += 1
        name = os.path.basename(path)[:-len('_res.json')]
        parts = name.split('_')
        case = '_'.join(parts[:2])
        r = json.load(open(path, encoding='utf-8'))
        dm = r.get('data_movement_bytes', {})
        real_spill = dm.get('spill_added_copy_bytes')
        if real_spill is None:
            continue
        if 'singlecore' in name:
            scene, n = 'A', 1
            plan = None
        else:
            scene = scene_from_parts(parts)
            if scene is None:
                stats['skipped_invalid_plan'] += 1
                continue
            n = int(parts[4][1:])
            pp = os.path.join(RESULTS, 'plans', f'{case}_{scene}_N{n}.json')
            if not os.path.exists(pp):
                stats['skipped_missing_plan'] += 1
                continue
            try:
                plan = json.load(open(pp, encoding='utf-8'))
            except (OSError, ValueError):
                stats['skipped_invalid_plan'] += 1
                continue
        if case not in graphs:
            g = load_graph(os.path.join(ATTACH, 'data', f'{case}.json'))
            n_el = sum(1 for o in g['ops']
                       if o['op'] not in ('COPY_IN', 'COPY_OUT'))
            graphs[case] = Model(g, block_ops_cap=block_cap_for(n_el))
        model = graphs[case]
        if plan is None:
            nb = len(model.blocks)
            sg_of_block, core_of_sg = [0] * nb, [0]
        else:
            try:
                sg_of_block, core_of_sg = decode_plan(model, plan)
            except (KeyError, TypeError, ValueError, IndexError):
                stats['skipped_invalid_plan'] += 1
                continue
        _, _, info = model.evaluate(sg_of_block, core_of_sg, scene,
                                    max(n, 1), use_cache=False)
        sig = info['spill_sig']
        data.append({
            'case': case, 'scene': scene, 'N': n,
            'real': float(real_spill),
            'sig': {key: float(sig.get(key, 0.0))
                    for key in ('L1p', 'L1w', 'UBp', 'UBw')},
            'peak': {key: float(info.get('peak_memory_bytes', {}).get(key, 0.0))
                     for key in ('L1', 'UB')},
            'partition_added': float(info.get('partition_added_bytes', 0.0)),
            'n_ops': len(model.graph_json['ops']),
        })
        stats['valid'] += 1
        stats['by_scene'][scene] += 1
    return data, stats


def nnls(X, y, iters=20000, lr=0.01, l2=0.0):
    """非负最小二乘：优先 scipy，退化时用坐标下降。"""
    try:
        from scipy.optimize import nnls as scipy_nnls
        import numpy as np
        w, _ = scipy_nnls(np.array(X, dtype=float), np.array(y, dtype=float))
        return list(w)
    except ImportError:
        pass
    # 坐标下降（每维解析解 + 非负截断）
    import math
    m, d = len(X), len(X[0])
    w = [0.0] * d
    col = [[row[j] for row in X] for j in range(d)]
    norms = [sum(v * v for v in c) or 1.0 for c in col]
    for _ in range(200):
        for j in range(d):
            resid = y[:]
            for i in range(m):
                pred = sum(w[k] * X[i][k] for k in range(d) if k != j)
                resid[i] = y[i] - pred
            dot = sum(col[j][i] * resid[i] for i in range(m))
            w[j] = max(0.0, dot / norms[j])
    return w


def evaluate_fit(X, y, w):
    import math
    errs, abs_rel = [], []
    for row, t in zip(X, y):
        pred = max(0.0, sum(w[j] * row[j] for j in range(len(w))))
        errs.append(pred - t)
        abs_rel.append(abs(pred - t) / max(1.0, t))
    n = len(y)
    my = sum(y) / n
    ss_res = sum(e * e for e in errs)
    ss_tot = sum((t - my) ** 2 for t in y)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float('nan')
    return r2, sum(abs_rel) / n


def spearman(a, b):
    ra = {v: i for i, v in enumerate(sorted(a))}
    order = sorted(range(len(a)), key=lambda i: a[i])
    rank_a = [0] * len(a)
    for i, idx in enumerate(order):
        rank_a[idx] = i
    order = sorted(range(len(b)), key=lambda i: b[i])
    rank_b = [0] * len(b)
    for i, idx in enumerate(order):
        rank_b[idx] = i
    import statistics as st
    ma, mb = st.mean(rank_a), st.mean(rank_b)
    cov = sum((x - ma) * (y - mb) for x, y in zip(rank_a, rank_b)) / len(a)
    sd = st.pstdev(rank_a) * st.pstdev(rank_b)
    return cov / sd if sd > 0 else float('nan')


def main():
    data, collect_stats = collect()
    print(f'样本: {len(data)}')
    print('样本有效性:', collect_stats)
    train, test, holdout_cases = split_by_case(data)
    print('留出算例:', ','.join(holdout_cases))

    # 全局父模型只用于分段收缩和小样本回退；不覆盖当前 active 系数文件。
    parent_x = [feature_vector(row) for row in train]
    parent_y = [row['real'] for row in train]
    parent = nnls(parent_x, parent_y)
    grouped = {}
    for row in train:
        grouped.setdefault(segment_key(row), []).append(row)
    models = {}
    for key, rows in grouped.items():
        if len(rows) < 30:
            continue
        local = nnls([feature_vector(row) for row in rows],
                     [row['real'] for row in rows])
        weight = len(rows) / (len(rows) + 50.0)
        models[key] = [weight * a + (1.0 - weight) * b
                       for a, b in zip(local, parent)]

    def predict(row, coefs):
        return max(0.0, sum(a * b for a, b in
                            zip(coefs, feature_vector(row))))

    def segmented_predict(row):
        return predict(row, models.get(segment_key(row), parent))

    print('全局特征数:', len(parent), '有效分段:', len(models))
    print('全局父模型系数:', [round(float(v), 6) for v in parent])

    report = {
        'schema_version': 'spill-segmented-v1',
        'feature_names': ['L1p', 'L1w', 'UBp', 'UBw', 'peak_L1',
                          'peak_UB', 'partition_added_MB', 'n_ops_k', 'bias'],
        'split': {'holdout_cases': holdout_cases,
                  'train_rows': len(train), 'test_rows': len(test)},
        'collection': collect_stats,
        'parent_coefs': [float(v) for v in parent],
        'segments': {key: [float(v) for v in value]
                     for key, value in models.items()},
    }
    metrics = {}
    for label, rows, fn in (
            ('训练集', train, segmented_predict),
            ('留出集', test, segmented_predict)):
        pred = [fn(row) for row in rows]
        real = [row['real'] for row in rows]
        positive = [(p, t) for p, t in zip(pred, real) if t > 0]
        mape = (sum(abs(p - t) / t for p, t in positive) /
                max(1, len(positive)))
        mae = sum(abs(p - t) for p, t in zip(pred, real)) / max(1, len(real))
        metrics[label] = {'mae': mae, 'positive_mape': mape,
                          'spearman': spearman(pred, real),
                          'rows': len(rows)}
        print(f'{label}: MAE={mae:.1f} 正spill MAPE={mape:.1%} '
              f'排序相关={spearman(pred, real):.3f} (n={len(rows)})')

    report['metrics'] = metrics
    with open(SEGMENTED_OUT, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print('saved ->', SEGMENTED_OUT)

    # 仅在留出集上报告关键失败算例，不能把它们用于拟合。
    for case in ('case_044', 'case_067'):
        rows = [row for row in test if row['case'] == case]
        if rows:
            vals = [segmented_predict(row) for row in rows]
            print(f'{case}: proxy spill [{min(vals):.1f}, {max(vals):.1f}] '
                  f'real [{min(row["real"] for row in rows):.1f}, '
                  f'{max(row["real"] for row in rows):.1f}]')


if __name__ == '__main__':
    main()
