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
    graphs = {}
    files = sorted(glob.glob(os.path.join(RESULTS, 'official', '*_res.json')))
    for path in files:
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
            scene = 'A' if parts[2] == 'problem' and parts[3] == '1' else 'B'
            n = int(parts[-1][1:])
            pp = os.path.join(RESULTS, 'plans', f'{case}_{scene}_N{n}.json')
            if not os.path.exists(pp):
                continue
            plan = json.load(open(pp, encoding='utf-8'))
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
            sg_of_block, core_of_sg = decode_plan(model, plan)
        _, _, info = model.evaluate(sg_of_block, core_of_sg, scene,
                                    max(n, 1), use_cache=False)
        sig = info['spill_sig']
        data.append({
            'case': case, 'scene': scene, 'N': n,
            'real': float(real_spill),
            'x': [sig['L1p'], sig['L1w'], sig['UBp'], sig['UBw'], 1.0],
        })
    return data


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
    data = collect()
    print(f'样本: {len(data)}')
    # 80/20 按用例划分
    cases = sorted({d['case'] for d in data})
    test_cases = set(cases[::5])
    tr = [d for d in data if d['case'] not in test_cases]
    te = [d for d in data if d['case'] in test_cases]
    X = [d['x'] for d in tr]
    y = [d['real'] for d in tr]
    w = nnls(X, y)
    print('NNLS 系数 [a·L1p, b·L1w, c·UBp, d·UBw, e] =',
          [round(v, 4) for v in w])
    r2_tr, mape_tr = evaluate_fit(X, y, w)
    r2_te, mape_te = evaluate_fit([d['x'] for d in te], [d['real'] for d in te], w)
    print(f'训练集 R2={r2_tr:.3f} MAPE={mape_tr:.1%} | 留出集 R2={r2_te:.3f} MAPE={mape_te:.1%}')
    # 排序能力对比（全局，按场景）
    for scene in ('A', 'B'):
        sub = [d for d in data if d['scene'] == scene]
        reals = [d['real'] for d in sub]
        pred_h = [2 * (max(d['x'][0], d['x'][1]) + max(d['x'][2], d['x'][3]))
                  for d in sub]
        pred_c = [max(0.0, sum(w[j] * d['x'][j] for j in range(5)))
                  for d in sub]
        print(f'场景{scene}: 真值排序相关 启发式={spearman(pred_h, reals):.3f} '
              f'校准后={spearman(pred_c, reals):.3f} (n={len(sub)})')
    with open(os.path.join(HERE, 'spill_coefs.json'), 'w') as f:
        json.dump({'coefs': w,
                   'note': 'spill_bytes = a*L1p+b*L1w+c*UBp+d*UBw+e, NNLS on official results'}, f, indent=1)
    print('saved -> solver/spill_coefs.json')


if __name__ == '__main__':
    main()
