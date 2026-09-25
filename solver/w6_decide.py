"""w6 大图官方评估 + 逐案裁决（w6 vs 现有官方，词典序保留更优）。"""
import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
ATTACH = os.path.normpath(os.path.join(HERE, '..', '通用神经网络处理器下的多核调度问题附件'))
RESULTS = os.path.normpath(os.path.join(HERE, '..', 'results'))
PY = sys.executable


def lexi_better(mk_a, ad_a, mk_b, ad_b, tol=0.003):
    if mk_a < mk_b * (1 - tol):
        return True
    if mk_a > mk_b * (1 + tol):
        return False
    return (ad_a or 0) < (ad_b or 0)


def official_eval(case, n, plan_path, stem):
    cmd = [PY, '-X', 'utf8', 'code/multicore_cut_evaluate_problem_1.py',
           os.path.join(ATTACH, 'data', f'{case}.json'), plan_path,
           '--config', 'data/config.txt', '-o', stem + '_res.json',
           '--trace-output', 'NUL', '--log-output', 'NUL']
    r = subprocess.run(cmd, cwd=ATTACH, capture_output=True, text=True,
                       encoding='utf-8', errors='replace', timeout=7200)
    if r.returncode != 0 or not os.path.exists(stem + '_res.json'):
        return None
    res = json.load(open(stem + '_res.json', encoding='utf-8'))
    dm = res.get('data_movement_bytes', {})
    return {'makespan': res['makespan'],
            'added': dm.get('added_copy_bytes')}


def run_one(job):
    case, n = job
    plan = os.path.join(RESULTS, 'plans_w6', f'{case}_A_N{n}.json')
    stem = os.path.join(RESULTS, 'official_w6', f'{case}_problem_1_N{n}')
    os.makedirs(os.path.dirname(stem), exist_ok=True)
    r = official_eval(case, n, plan, stem)
    return case, n, r


def main():
    # 1) 找 w6 中缺真值的任务
    needs = []
    for line in open(os.path.join(RESULTS, 'solve_log_w6.jsonl'),
                     encoding='utf-8'):
        e = json.loads(line)
        if e.get('real') is None:
            needs.append((e['case'], e['N']))
    print(f'{len(needs)} 大图需官方评估')
    # 2) 评估
    w6_res = {}
    with ProcessPoolExecutor(max_workers=10) as ex:
        futs = {ex.submit(run_one, j): j for j in needs}
        for fut in as_completed(futs):
            case, n, r = fut.result()
            if r:
                w6_res[(case, n)] = r
    print(f'完成 {len(w6_res)}/{len(needs)}')
    # 3) 逐案裁决
    wins = kept = 0
    deltas = []
    for (case, n), r6 in sorted(w6_res.items()):
        pp = os.path.join(RESULTS, 'official', f'{case}_problem_1_N{n}_res.json')
        if not os.path.exists(pp):
            continue
        cur = json.load(open(pp, encoding='utf-8'))
        cur_mk = cur['makespan']
        cur_ad = cur['data_movement_bytes'].get('added_copy_bytes')
        d = (r6['makespan'] - cur_mk) / cur_mk * 100
        deltas.append(d)
        if lexi_better(r6['makespan'], r6['added'], cur_mk, cur_ad):
            # w6 胜：方案与官方结果都替换
            shutil_src = os.path.join(RESULTS, 'plans_w6', f'{case}_A_N{n}.json')
            os.replace(shutil_src, os.path.join(RESULTS, 'plans', f'{case}_A_N{n}.json'))
            os.replace(os.path.join(RESULTS, 'official_w6',
                                    f'{case}_problem_1_N{n}_res.json'), pp)
            wins += 1
        else:
            kept += 1
    import statistics as st
    print(f'裁决: w6 胜 {wins} / 保留旧 {kept}')
    if deltas:
        print(f'w6 相对现有: 均值 {st.mean(deltas):+.2f}% 中位 {st.median(deltas):+.2f}%')


if __name__ == '__main__':
    main()
