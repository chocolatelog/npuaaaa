"""按题目 1.4 节三个官方指标 + 代理模型精度评估求解结果。"""
import json
import glob
import os
import statistics as st
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.normpath(os.path.join(HERE, '..', 'results'))

rows = defaultdict(dict)
for p in glob.glob(os.path.join(RESULTS, 'official', '*_res.json')):
    name = os.path.basename(p)[:-len('_res.json')]
    parts = name.split('_')
    case = '_'.join(parts[:2])
    r = json.load(open(p, encoding='utf-8'))
    dm = r.get('data_movement_bytes', {})
    if 'singlecore' in name:
        rows[case]['sc'] = (r['makespan'],
                            dm.get('original_graph_copy_bytes'),
                            dm.get('scheduled_copy_bytes'))
    else:
        prob, n = parts[2] + '_' + parts[3], int(parts[-1][1:])
        cs = r.get('cache_stats', {})
        hit = (cs.get('hit_bytes', 0) /
               max(1, cs.get('hit_bytes', 0) + cs.get('miss_bytes', 0)))
        rows[case][(prob, n)] = {
            'mk': r['makespan'], 'added': dm.get('added_copy_bytes', 0),
            'orig': dm.get('original_graph_copy_bytes', 0),
            'part': dm.get('partition_added_copy_bytes', 0) or 0,
            'spill': dm.get('spill_added_copy_bytes', 0) or 0,
            'hit': hit,
        }

print('=== 指标1: Makespan -> 加速比 / 并行效率 ===')
for prob in ('problem_1', 'problem_2', 'problem_3'):
    line = []
    for n in (2, 3, 4, 5):
        sps = [rows[c]['sc'][0] / rows[c][(prob, n)]['mk']
               for c in rows if (prob, n) in rows[c] and 'sc' in rows[c]]
        line.append(f'N={n}: {st.mean(sps):.3f}(中位{st.median(sps):.2f}) '
                    f'效率{st.mean(sps) / n * 100:.0f}%')
    print(f'{prob}: ' + ' | '.join(line))

print()
print('=== 指标2: 总额外数据搬运量 ===')
for prob in ('problem_1', 'problem_2', 'problem_3'):
    for n in (2, 4, 5):
        added = [rows[c][(prob, n)]['added'] / 1e6 for c in rows
                 if (prob, n) in rows[c]]
        ratio = [rows[c][(prob, n)]['added'] / max(1, rows[c][(prob, n)]['orig'])
                 for c in rows if (prob, n) in rows[c]]
        part = [rows[c][(prob, n)]['part'] / 1e6 for c in rows
                if (prob, n) in rows[c]]
        spill = [rows[c][(prob, n)]['spill'] / 1e6 for c in rows
                 if (prob, n) in rows[c]]
        zero = sum(1 for a in added if a < 1e4) / max(1, len(added))
        print(f'{prob} N={n}: 均值{st.mean(added):6.2f}MB 中位{st.median(added):6.2f}MB '
              f'| added/原始流量={st.mean(ratio):5.2f}x '
              f'| 切分贡献{st.mean(part):5.2f} + spill贡献{st.mean(spill):5.2f}MB '
              f'| 零额外搬运用例占比{zero:.0%}')

print()
print('=== 指标3: 问题3 只读 Cache 命中率 ===')
for n in (2, 3, 4, 5):
    hits = [rows[c][('problem_3', n)]['hit'] for c in rows
            if ('problem_3', n) in rows[c]]
    l2sp = [rows[c][('problem_2', n)]['mk'] / rows[c][('problem_3', n)]['mk']
            for c in rows if ('problem_3', n) in rows[c]
            and ('problem_2', n) in rows[c]]
    pairs = sorted(zip(hits, l2sp), reverse=True)[:5]
    print(f'N={n}: 命中率 均值{st.mean(hits):.2%} 中位{st.median(hits):.2%} '
          f'最大{max(hits):.1%} | L2加速比均值{st.mean(l2sp):.4f} '
          f'| 高命中用例(命中率->L2加速比): '
          + ', '.join(f'{h:.0%}->{s:.3f}' for h, s in pairs))

# 命中率与 L2 加速比的相关性
hs, ls = [], []
for c in rows:
    if ('problem_3', 4) in rows[c] and ('problem_2', 4) in rows[c]:
        hs.append(rows[c][('problem_3', 4)]['hit'])
        ls.append(rows[c][('problem_2', 4)]['mk'] / rows[c][('problem_3', 4)]['mk'])
mh, ml = st.mean(hs), st.mean(ls)
cov = sum((h - mh) * (l - ml) for h, l in zip(hs, ls)) / len(hs)
corr = cov / (st.pstdev(hs) * st.pstdev(ls))
print(f'\n命中率与L2加速比的Pearson相关(N=4): r={corr:.3f}')

print()
print('=== 代理模型精度 (est vs 官方真值, 已校验用例) ===')
est_mk, real_mk, est_ad, real_ad, ops = [], [], [], [], []
for line in open(os.path.join(RESULTS, 'solve_log.jsonl'), encoding='utf-8'):
    e = json.loads(line)
    if e['real'] and e['real'].get('makespan'):
        est_mk.append(e['est_mk'])
        real_mk.append(e['real']['makespan'])
        est_ad.append(e['est_added'])
        real_ad.append(e['real']['added_copy_bytes'])
        ops.append(e['n_ops'])
err_mk = [(e - r) / r * 100 for e, r in zip(est_mk, real_mk)]
err_ad = [(e - r) / max(1, r) * 100 for e, r in zip(est_ad, real_ad)]
import math
def mape(xs):
    return sum(abs(x) for x in xs) / len(xs)
def smape(xs):
    return sum(min(abs(x), 200) for x in xs) / len(xs)
# 秩相关（排序一致性——搜索引导质量的关键）
def rank_corr(a, b):
    ra = {v: i for i, v in enumerate(sorted(range(len(a)), key=lambda i: a[i]))}
    idx_a = sorted(range(len(a)), key=lambda i: a[i])
    idx_b = sorted(range(len(b)), key=lambda i: b[i])
    rank_a = [0] * len(a)
    rank_b = [0] * len(b)
    for i, idx in enumerate(idx_a):
        rank_a[idx] = i
    for i, idx in enumerate(idx_b):
        rank_b[idx] = i
    mh_, mb_ = st.mean(rank_a), st.mean(rank_b)
    cov = sum((x - mh_) * (y - mb_) for x, y in zip(rank_a, rank_b)) / len(a)
    return cov / (st.pstdev(rank_a) * st.pstdev(rank_b))
print(f'样本数: {len(est_mk)} (<=1.2万op用例)')
print(f'Makespan: MAPE {mape(err_mk):.1f}%, 偏差中位 {st.median(err_mk):.1f}%, '
      f'80%分位 {sorted(err_mk)[int(len(err_mk)*0.8)]:.1f}%, 秩相关 {rank_corr(est_mk, real_mk):.3f}')
print(f'新增搬运: MAPE {mape(err_ad):.1f}%, 秩相关 {rank_corr(est_ad, real_ad):.3f}')
big = [(abs(e - r) / r) for e, r, o in zip(est_mk, real_mk, ops) if o > 4000]
small = [(abs(e - r) / r) for e, r, o in zip(est_mk, real_mk, ops) if o <= 4000]
print(f'Makespan 误差 | <=4kop: {st.mean(small)*100:.1f}%, 4k-12kop: {st.mean(big)*100:.1f}%')
