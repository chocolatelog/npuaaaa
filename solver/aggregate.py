"""汇总官方评估结果：summary.csv + 加速比曲线 + 附录逐用例表。"""
import csv
import glob
import json
import os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.normpath(os.path.join(HERE, '..', 'results'))
OFFICIAL = os.path.join(RESULTS, 'official')
PLOTS = os.path.join(RESULTS, 'plots')


def load_official():
    rows = {}
    for path in glob.glob(os.path.join(OFFICIAL, '*_res.json')):
        name = os.path.basename(path)[:-len('_res.json')]
        with open(path, encoding='utf-8') as f:
            r = json.load(f)
        dm = r.get('data_movement_bytes', {})
        row = {
            'makespan': r.get('makespan'),
            'added_bytes': dm.get('added_copy_bytes'),
            'scheduled_bytes': dm.get('scheduled_copy_bytes'),
            'original_bytes': dm.get('original_graph_copy_bytes'),
            'partition_added': dm.get('partition_added_copy_bytes'),
            'spill_added': dm.get('spill_added_copy_bytes'),
        }
        cs = r.get('cache_stats')
        if cs:
            total = cs.get('hit_bytes', 0) + cs.get('miss_bytes', 0)
            row['cache_hit_rate'] = (cs.get('hit_bytes', 0) / total
                                     if total else 0.0)
        rows[name] = row
    return rows


def parse_key(name):
    # case_XXX_problem_1_N4 / case_XXX_singlecore
    parts = name.split('_')
    case = '_'.join(parts[:2])
    if 'singlecore' in name:
        return case, 'singlecore', 1
    prob = parts[2] + '_' + parts[3]
    n = int(parts[-1][1:])
    return case, prob, n


def main():
    rows = load_official()
    data = defaultdict(dict)   # case -> (prob,N) -> row
    for name, row in rows.items():
        case, prob, n = parse_key(name)
        data[case][f'{prob}_N{n}' if prob != 'singlecore' else 'singlecore'] = row

    cases = sorted(data)
    # ---- summary CSV ----
    csv_path = os.path.join(RESULTS, 'summary.csv')
    fields = ['case', 'singlecore_makespan']
    for prob in ('problem_1', 'problem_2', 'problem_3'):
        for n in (2, 3, 4, 5):
            fields += [f'{prob}_N{n}_makespan', f'{prob}_N{n}_added_bytes',
                       f'{prob}_N{n}_speedup']
    fields += [f'problem_3_N{n}_cache_hit' for n in (2, 3, 4, 5)]
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(fields)
        for case in cases:
            sc = data[case].get('singlecore', {}).get('makespan')
            row = [case, sc]
            for prob in ('problem_1', 'problem_2', 'problem_3'):
                for n in (2, 3, 4, 5):
                    r = data[case].get(f'{prob}_N{n}', {})
                    mk = r.get('makespan')
                    sp = round(sc / mk, 4) if (sc and mk) else ''
                    row += [mk, r.get('added_bytes'), sp]
            for n in (2, 3, 4, 5):
                r = data[case].get(f'problem_3_N{n}', {})
                row.append(round(r['cache_hit_rate'], 4)
                           if 'cache_hit_rate' in r else '')
            w.writerow(row)

    # ---- 平均加速比 ----
    avg = defaultdict(list)
    for case in cases:
        sc = data[case].get('singlecore', {}).get('makespan')
        if not sc:
            continue
        for prob in ('problem_1', 'problem_2', 'problem_3'):
            for n in (2, 3, 4, 5):
                mk = data[case].get(f'{prob}_N{n}', {}).get('makespan')
                if mk:
                    avg[(prob, n)].append(sc / mk)
    summary_md = ['# 官方评估汇总\n']
    summary_md.append('## 平均加速比（单核基准）\n')
    summary_md.append('| 核数 | 问题1(场景A) | 问题2(场景B) | 问题3(场景B+L2) |')
    summary_md.append('|---|---|---|---|')
    for n in (2, 3, 4, 5):
        summary_md.append('| {} | {:.3f} | {:.3f} | {:.3f} |'.format(
            n,
            *[_mean(avg[(p, n)]) for p in
              ('problem_1', 'problem_2', 'problem_3')]))
    l2sp = defaultdict(list)
    for case in cases:
        for n in (2, 3, 4, 5):
            m2 = data[case].get(f'problem_2_N{n}', {}).get('makespan')
            m3 = data[case].get(f'problem_3_N{n}', {}).get('makespan')
            if m2 and m3:
                l2sp[n].append(m2 / m3)
    summary_md.append('\n## 问题3 只读 Cache 相对无 L2 的加速比\n')
    summary_md.append('| 核数 | L2加速比 |')
    summary_md.append('|---|---|')
    for n in (2, 3, 4, 5):
        summary_md.append(f'| {n} | {_mean(l2sp[n]):.4f} |')
    with open(os.path.join(RESULTS, 'summary.md'), 'w',
              encoding='utf-8') as f:
        f.write('\n'.join(summary_md) + '\n')
    print('cases:', len(cases))
    for n in (2, 3, 4, 5):
        print(f'N={n}: P1 {_mean(avg[("problem_1", n)]):.3f} '
              f'P2 {_mean(avg[("problem_2", n)]):.3f} '
              f'P3 {_mean(avg[("problem_3", n)]):.3f} '
              f'L2speedup {_mean(l2sp[n]):.4f}')

    # ---- 曲线 ----
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        os.makedirs(PLOTS, exist_ok=True)
        ns = [2, 3, 4, 5]
        plt.figure(figsize=(7, 4.5))
        for prob, label, color in (('problem_1', 'Problem 1 (Scene A)', 'tab:blue'),
                                   ('problem_2', 'Problem 2 (Scene B)', 'tab:orange'),
                                   ('problem_3', 'Problem 3 (Scene B + L2)', 'tab:green')):
            ys = [_mean(avg[(prob, n)]) for n in ns]
            plt.plot([1] + ns, [1.0] + ys, marker='o', label=label, color=color)
        plt.xlabel('Number of cores')
        plt.ylabel('Average speedup vs single-core')
        plt.title('Average speedup (100 cases, official evaluator)')
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS, 'speedup.png'), dpi=150)
        plt.close()

        plt.figure(figsize=(6, 4))
        ys = [_mean(l2sp[n]) for n in ns]
        plt.plot(ns, ys, marker='s', color='tab:red')
        plt.xlabel('Number of cores')
        plt.ylabel('L2 cache speedup (no-L2 / with-L2)')
        plt.title('Problem 3: read-only L2 benefit')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS, 'l2_speedup.png'), dpi=150)
        plt.close()
        print('plots written to', PLOTS)
    except ImportError:
        print('matplotlib unavailable, skip plots')


def _mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float('nan')


if __name__ == '__main__':
    main()
