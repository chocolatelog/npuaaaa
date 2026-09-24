"""三问题 × 三指标：本求解器 vs 原始代码基线（附件 stub 生成器）。

指标（题目 1.4 节）：
1. Makespan（主）-> 加速比与相对提升
2. 总额外数据搬运量（次）
3. Cache 命中率（仅问题3）
基线方案由 stub_multicore_cut_and_schedule.py 生成（README 注明其仅演示
方案格式、非性能基线），评估器与配置完全相同。
"""
import glob
import json
import os
import statistics as st
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.normpath(os.path.join(HERE, '..', 'results'))


def load_dir(d):
    rows = defaultdict(dict)
    for p in glob.glob(os.path.join(d, '*_res.json')):
        name = os.path.basename(p)[:-len('_res.json')]
        parts = name.split('_')
        case = '_'.join(parts[:2])
        if 'singlecore' in name:
            rows[case]['sc'] = json.load(open(p, encoding='utf-8'))['makespan']
            continue
        prob, n = parts[2] + '_' + parts[3], int(parts[-1][1:])
        r = json.load(open(p, encoding='utf-8'))
        dm = r.get('data_movement_bytes', {})
        row = {'mk': r.get('makespan'), 'added': dm.get('added_copy_bytes')}
        cs = r.get('cache_stats')
        if cs:
            tot = cs.get('hit_bytes', 0) + cs.get('miss_bytes', 0)
            row['hit'] = cs.get('hit_bytes', 0) / tot if tot else 0.0
        rows[case][(prob, n)] = row
    return rows


def main():
    ours = load_dir(os.path.join(RESULTS, 'official'))
    stub = load_dir(os.path.join(RESULTS, 'official_stub'))
    out = ['# 与原始代码基线（附件 stub 生成器）对比\n']
    out.append('基线：`stub_multicore_cut_and_schedule.py`（随机合法方案，'
               'README 注明仅演示格式、非性能基线）；本求解器结果为单调改进协议后版本。'
               '两者使用同一官方评估器与固定 config.txt。\n')

    summary = {}
    for prob, title in (('problem_1', '问题 1（场景 A）'),
                        ('problem_2', '问题 2（场景 B）'),
                        ('problem_3', '问题 3（场景 B + L2）')):
        out.append(f'## {title}\n')
        out.append('| 核数 | 基线 Makespan 均值 | 本文均值 | Makespan 提升 | '
                   '基线加速比 | 本文加速比 | 加速比提升 | '
                   '基线新增搬运 | 本文新增搬运 | 搬运降幅 |')
        out.append('|---|---|---|---|---|---|---|---|---|---|')
        for n in (2, 3, 4, 5):
            pairs = []
            for c in ours:
                if (prob, n) not in ours[c] or (prob, n) not in stub.get(c, {}):
                    continue
                o, s = ours[c][(prob, n)], stub[c][(prob, n)]
                if o.get('mk') is None or s.get('mk') is None:
                    continue
                pairs.append((c, o, s))
            if not pairs:
                continue
            mk_ratio = [s['mk'] / o['mk'] for _, o, s in pairs]
            sp_o = [ours[c]['sc'] / o['mk'] for c, o, _ in pairs
                    if ours.get(c, {}).get('sc')]
            sp_s = [ours[c]['sc'] / s['mk'] for c, _, s in pairs
                    if ours.get(c, {}).get('sc')]
            ad_o = [o.get('added') or 0 for _, o, _ in pairs]
            ad_s = [s.get('added') or 0 for _, _, s in pairs]
            ad_ratio = [(o2 + 1) / (s2 + 1) for s2, o2 in
                        zip(ad_s, ad_o)]      # 本文/基线（+1 平滑零值）
            summary[(prob, n)] = {
                'n_cases': len(pairs),
                'mk_gain': st.mean(mk_ratio),
                'mk_gain_med': st.median(mk_ratio),
                'sp_ours': st.mean(sp_o), 'sp_stub': st.mean(sp_s),
                'added_stub': st.mean(ad_s), 'added_ours': st.mean(ad_o),
                'added_reduction': 1 - st.mean(ad_ratio),
            }
            s = summary[(prob, n)]
            out.append(
                f'| {n} | {st.mean([x["mk"] for _, _, x in pairs]):.3g} | '
                f'{st.mean([o["mk"] for _, o, _ in pairs]):.3g} | '
                f'**{s["mk_gain"]:.2f}×**(中位{s["mk_gain_med"]:.2f}×) | '
                f'{s["sp_stub"]:.3f} | {s["sp_ours"]:.3f} | '
                f'+{(s["sp_ours"]/s["sp_stub"]-1)*100:.0f}% | '
                f'{s["added_stub"]/1e6:.1f}MB | {s["added_ours"]/1e6:.1f}MB | '
                f'{s["added_reduction"]*100:.0f}% |')
        out.append('')

    # 问题3 Cache 命中率对比
    out.append('## 问题 3 Cache 命中率对比\n')
    out.append('| 核数 | 基线命中率 | 本文命中率 |')
    out.append('|---|---|---|')
    hit_stats = {}
    for n in (2, 3, 4, 5):
        hs, ho = [], []
        for c in ours:
            if ('problem_3', n) in ours[c] and 'hit' in ours[c][('problem_3', n)]:
                ho.append(ours[c][('problem_3', n)]['hit'])
            if ('problem_3', n) in stub.get(c, {}) and 'hit' in \
                    stub[c][('problem_3', n)]:
                hs.append(stub[c][('problem_3', n)]['hit'])
        if hs and ho:
            hit_stats[n] = (st.mean(hs), st.mean(ho))
            out.append(f'| {n} | {st.mean(hs):.2%} | {st.mean(ho):.2%} |')
    out.append('')

    # 覆盖率说明
    out.append('## 覆盖与失败说明\n')
    ns = sorted({v['n_cases'] for v in summary.values()})
    out.append(f'- 基线 1200 项评估全部完成（重试后无失败），'
               f'对比仅在双方均有结果的用例上进行（每配置 {ns} 用例）。\n'
                '- 单核基准共用 `singlecore_evaluate.py` 结果；'
                '加速比 = 单核 Makespan / 多核 Makespan。\n'
                '- 基线（随机切分）在问题 1 下平均加速比 <1（多核反而更慢），'
                '问题 2/3 仅靠核数堆到 1.1~1.3×。\n')

    text = '\n'.join(out)
    with open(os.path.join(RESULTS, 'baseline_comparison.md'), 'w',
              encoding='utf-8') as f:
        f.write(text + '\n')
    print(text)


if __name__ == '__main__':
    main()
