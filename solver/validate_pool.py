"""构造池分层验证实验（方案 §17）。

每场景×每核数：10 小图 + 10 中图 + 10 大图，只跑构造池（无元启发式），
记录 R0 12 表、逐层保留/淘汰、最终种子血统、新旧对照与多样性指标。
输出：results/pool_validation.json + pool_validation_report.md（10 项交付）。
"""
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ATTACH = os.path.normpath(os.path.join(
    HERE, '..', '通用神经网络处理器下的多核调度问题附件'))
RESULTS = os.path.normpath(os.path.join(HERE, '..', 'results'))

# 按规模分层抽 30 用例（小 <=2k, 中 2k-8k, 大 >8k op）
SMALL = ['case_001', 'case_006', 'case_010', 'case_019', 'case_026',
         'case_029', 'case_032', 'case_057', 'case_065', 'case_071']
MEDIUM = ['case_002', 'case_004', 'case_011', 'case_013', 'case_018',
          'case_023', 'case_033', 'case_034', 'case_055', 'case_100']
LARGE = ['case_003', 'case_014', 'case_016', 'case_025', 'case_030',
         'case_041', 'case_047', 'case_053', 'case_076', 'case_091']
CASES = [('small', c) for c in SMALL] + [('medium', c) for c in MEDIUM] + \
    [('large', c) for c in LARGE]


def run_one(task):
    from model import load_graph, Model
    from solution import Context, Sol
    from construct_refine import (build_construct_candidates,
                                  candidate_metrics)
    from construct import legacy_pool
    from run_all import budget_for
    case, tier, scene, n = task
    g = load_graph(os.path.join(ATTACH, 'data', f'{case}.json'))
    n_el = sum(1 for o in g['ops'] if o['op'] not in ('COPY_IN', 'COPY_OUT'))
    cap = int(max(24, min(240, n_el // 400)))
    m = Model(g, block_ops_cap=cap)
    ctx = Context(m, n, scene)
    t0 = time.time()
    trace = []
    try:
        finals, layers = build_construct_candidates(
            m, n, scene, ctx, deadline=time.time() + 0.2 * budget_for(len(g['ops'])),
            trace=trace)
    except Exception as e:
        return {'case': case, 'tier': tier, 'scene': scene, 'N': n,
                'error': repr(e)}
    dt = time.time() - t0
    # 旧池对照
    legacy_best = None
    try:
        for _name, _p, sg, core in legacy_pool(m, n, scene):
            mk, ad, _ = ctx.evaluate(Sol(list(sg), list(core)))
            if legacy_best is None or mk < legacy_best:
                legacy_best = mk
    except Exception:
        pass
    seeds = [{'paradigm': c.paradigm, 'granularity': c.granularity,
              'refine': c.refine_level,
              'mk': c.metrics['makespan'], 'added': c.metrics['added'],
              'spill': c.metrics['spill'], 'K': c.metrics['K']}
             for c in finals]
    r0_kept = [l for l in trace if isinstance(l, dict) and l.get('layer') == 'R0']
    return {
        'case': case, 'tier': tier, 'scene': scene, 'N': n,
        'elapsed': round(dt, 1),
        'legacy_best': legacy_best,
        'R0_best': layers.get('R0_best'),
        'R1_best': layers.get('R1_best'),
        'R2_best': layers.get('R2_best'),
        'R3_best': layers.get('R3_best'),
        'final_best': min((s['mk'] for s in seeds), default=None),
        'seeds': seeds,
        'dedup_eliminated': sum(1 for l in trace if isinstance(l, dict)
                                and l.get('reason') in ('exact_dup',
                                                        'struct_dup')),
        'paradigms_in_final': sorted({s['paradigm'] for s in seeds}),
        'granularities_in_final': sorted({s['granularity'] for s in seeds}),
    }


def main():
    scenes = ['A', 'B', 'C']
    cores = [2, 4]
    tasks = [(c, tier, s, n) for (tier, c) in CASES
             for s in scenes for n in cores]
    print(f'{len(tasks)} pool-only validations')
    results = []
    with ProcessPoolExecutor(max_workers=10) as ex:
        futs = {ex.submit(run_one, t): t for t in tasks}
        nd = 0
        for fut in as_completed(futs):
            results.append(fut.result())
            nd += 1
            if nd % 30 == 0:
                print(f'  [{nd}/{len(tasks)}]', flush=True)
    with open(os.path.join(RESULTS, 'pool_validation.json'), 'w',
              encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=1)

    # ---- 汇总报告 ----
    ok = [r for r in results if 'error' not in r]
    import statistics as st
    lines = ['# 构造池重构验证报告（范式×粒度×场景×精化）\n']

    def ratio_best(r, key):
        lb = r['legacy_best']
        v = r.get(key)
        if lb is None or v is None:
            return None
        return v / lb

    lines.append('## 1. 新旧对照与逐层单调性（代理 Makespan / 旧池最优）\n')
    lines.append('| 层 | 均值 | 中位 | 劣化>2% 占比 |')
    lines.append('|---|---|---|---|')
    for key in ('R0_best', 'R1_best', 'R2_best', 'R3_best', 'final_best'):
        rs = [ratio_best(r, key) for r in ok]
        rs = [x for x in rs if x is not None]
        if not rs:
            continue
        bad = sum(1 for x in rs if x > 1.02) / len(rs)
        lines.append(f'| {key} | {st.mean(rs):.3f} | {st.median(rs):.3f} '
                     f'| {bad:.1%} |')
    # 层间单调性
    mono_ok = sum(1 for r in ok
                  if r.get('R0_best') and r.get('final_best')
                  and r['final_best'] <= r['R0_best'] * 1.001)
    lines.append(f'\n逐层单调（final ≤ R0×1.001）：{mono_ok}/{len(ok)}')

    lines.append('\n## 2. 范式/粒度存活（最终种子中出现率）\n')
    from collections import Counter
    par = Counter(s['paradigm'] for r in ok for s in r['seeds'])
    gran = Counter(s['granularity'] for r in ok for s in r['seeds'])
    lines.append(f"- 范式: {dict(par)}")
    lines.append(f"- 粒度: {dict(gran)}")

    lines.append('\n## 3. 多样性\n')
    multi_par = sum(1 for r in ok if len(r['paradigms_in_final']) >= 2)
    lines.append(f'- 最终种子含 >=2 范式: {multi_par}/{len(ok)}')
    dup_ratio = st.mean([r['dedup_eliminated'] for r in ok])
    lines.append(f'- 平均每任务去重淘汰数: {dup_ratio:.1f}')

    lines.append('\n## 4. 耗时\n')
    for tier in ('small', 'medium', 'large'):
        ts = [r['elapsed'] for r in ok if r['tier'] == tier]
        lines.append(f'- {tier}: 均值 {st.mean(ts):.1f}s 中位 '
                     f'{st.median(ts):.1f}s')
    errors = [r for r in results if 'error' in r]
    if errors:
        lines.append(f"\n## 异常（{len(errors)}）\n")
        for r in errors[:5]:
            lines.append(f"- {r['case']}/{r['scene']}/N{r['N']}: "
                         f"{r['error'][:100]}")
    text = '\n'.join(lines)
    with open(os.path.join(RESULTS, 'pool_validation_report.md'), 'w',
              encoding='utf-8') as f:
        f.write(text + '\n')
    print(text)


if __name__ == '__main__':
    main()
