"""历史改进实现，仅保留用于阅读；命令行入口已停用。
替代流程：run_all.py 独立求解，evaluate_official.py 内容绑定评估。
旧修改时间清理和问题三跟随 B 的流程已取消，不删除历史文件。
"""
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
RESULTS = os.path.normpath(os.path.join(HERE, '..', 'results'))
ATTACH = os.path.normpath(os.path.join(
    HERE, '..', '通用神经网络处理器下的多核调度问题附件'))
PY = sys.executable
VERIFY_CAP = 12000
DECISIONS = os.path.join(RESULTS, 'improve_decisions.json')


def old_official_mk(case, scene, n):
    prob = 'problem_1' if scene == 'A' else 'problem_2'
    p = os.path.join(RESULTS, 'official', f'{case}_{prob}_N{n}_res.json')
    if not os.path.exists(p):
        return None, None
    r = json.load(open(p, encoding='utf-8'))
    dm = r.get('data_movement_bytes', {})
    return r.get('makespan'), dm.get('added_copy_bytes')


def key_of(rmk, raded):
    from model import BW
    return rmk + 0.2 * (raded or 0) / BW


def lexi_better(mk_a, ad_a, mk_b, ad_b, tol=0.003):
    """词典序：Makespan 严格优先（tol 内并列比新增搬运）。"""
    if mk_a < mk_b * (1 - tol):
        return True
    if mk_a > mk_b * (1 + tol):
        return False
    return (ad_a or 0) < (ad_b or 0)


def solve_one(task):
    from model import load_graph, BW
    from pipeline import solve_case
    from run_all import budget_for, block_cap_for, stable_seed
    case, scene, n, phase = task
    graph = load_graph(os.path.join(ATTACH, 'data', f'{case}.json'))
    n_ops = len(graph['ops'])
    n_el = sum(1 for o in graph['ops']
               if o['op'] not in ('COPY_IN', 'COPY_OUT'))
    t0 = time.time()
    r = solve_case(graph, N=n, scene=scene,
                   time_budget=budget_for(n_ops),
                   seed=stable_seed(case, scene + '|' + phase, n),
                   block_ops_cap=block_cap_for(n_el),
                   spill_calibrated=True)
    elapsed = time.time() - t0
    cand = os.path.join(RESULTS, 'plans_cand', f'{case}_{scene}_N{n}.json')
    os.makedirs(os.path.dirname(cand), exist_ok=True)
    with open(cand, 'w', encoding='utf-8') as f:
        json.dump(r['plan'], f)
    old_mk, old_ad = old_official_mk(case, scene, n)
    rec = {'case': case, 'scene': scene, 'N': n, 'elapsed': round(elapsed, 1),
           'old_mk': old_mk, 'old_added': old_ad, 'phase': phase}
    new_real = r['real']
    if n_ops <= VERIFY_CAP and new_real and old_mk is not None:
        new_mk, new_ad = new_real['makespan'], new_real['added_copy_bytes']
        rec['new_mk'], rec['new_added'] = new_mk, new_ad
        if lexi_better(new_mk, new_ad, old_mk, old_ad):
            shutil.copyfile(cand, os.path.join(
                RESULTS, 'plans', f'{case}_{scene}_N{n}.json'))
            rec['action'] = 'replaced_small'
        else:
            rec['action'] = 'kept_old'
    else:
        rec['action'] = 'needs_official'
    return rec


def eval_big(rec):
    """官方评估大用例候选并择优。"""
    case, scene, n = rec['case'], rec['scene'], rec['N']
    prob = 'problem_1' if scene == 'A' else 'problem_2'
    stem = os.path.join(RESULTS, 'official_cand', f'{case}_{prob}_N{n}')
    os.makedirs(os.path.dirname(stem), exist_ok=True)
    plan = os.path.join(RESULTS, 'plans_cand', f'{case}_{scene}_N{n}.json')
    cmd = [PY, '-X', 'utf8', f'code/multicore_cut_evaluate_{prob}.py',
           os.path.join(ATTACH, 'data', f'{case}.json'), plan,
           '--config', 'data/config.txt', '-o', stem + '_res.json',
           '--trace-output', stem + '_trace.json',
           '--log-output', stem + '_log.txt']
    r = subprocess.run(cmd, cwd=ATTACH, capture_output=True, text=True,
                       encoding='utf-8', errors='replace', timeout=7200)
    if r.returncode != 0:
        rec['action'] = 'eval_fail'
        return rec
    res = json.load(open(stem + '_res.json', encoding='utf-8'))
    new_mk = res['makespan']
    new_ad = res['data_movement_bytes'].get('added_copy_bytes')
    rec['new_mk'], rec['new_added'] = new_mk, new_ad
    if rec['old_mk'] is None or lexi_better(new_mk, new_ad,
                                            rec['old_mk'], rec['old_added']):
        shutil.copyfile(plan, os.path.join(
            RESULTS, 'plans', f'{case}_{scene}_N{n}.json'))
        rec['action'] = 'replaced_big'
    else:
        rec['action'] = 'kept_old'
    return rec


def legacy_disabled():
    raise SystemExit('旧实验入口已停用：请使用 run_all.py 的新日志/方案目录及 evaluate_official.py；历史结果保留。')


def main():
    legacy_disabled()
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cores', default='2,3,4,5')
    parser.add_argument('--workers', type=int, default=10)
    parser.add_argument('--phase', default='v4')
    args = parser.parse_args()
    cores = [int(x) for x in args.cores.split(',')]

    # 备份当前方案（一次性）
    backup = os.path.join(RESULTS, 'plans_v1_backup')
    if not os.path.exists(backup):
        shutil.copytree(os.path.join(RESULTS, 'plans'), backup)
        print('backed up plans -> plans_v1_backup')

    cases = sorted(f'case_{i:03d}' for i in range(1, 101))
    done = set()
    if os.path.exists(DECISIONS):
        with open(DECISIONS, encoding='utf-8') as f:
            for line in f:
                try:
                    e = json.loads(line)
                    if e.get('phase') == args.phase:
                        done.add((e['case'], e['scene'], e['N']))
                except Exception:
                    pass
    tasks = [(c, s, n, args.phase) for c in cases for s in ('A', 'B')
             for n in cores if (c, s, n) not in done]
    print(f'Phase A: {len(tasks)} solves')
    recs = []
    t0 = time.time()
    with open(DECISIONS, 'a', encoding='utf-8') as logf:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(solve_one, t): t for t in tasks}
            nd = 0
            for fut in as_completed(futs):
                rec = fut.result()
                recs.append(rec)
                logf.write(json.dumps(rec, ensure_ascii=False) + '\n')
                logf.flush()
                nd += 1
                if nd % 40 == 0:
                    print(f'  [{nd}/{len(tasks)}] {time.time()-t0:.0f}s',
                          flush=True)
    bigs = [r for r in recs if r['action'] == 'needs_official']
    print(f'Phase B: {len(bigs)} big-case official evaluations')
    replaced = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(eval_big, r): r for r in bigs}
        for i, fut in enumerate(as_completed(futs)):
            rec = fut.result()
            if rec['action'] == 'replaced_big':
                replaced += 1
            if (i + 1) % 10 == 0:
                print(f'  [{i+1}/{len(bigs)}]', flush=True)
    n_small = sum(1 for r in recs if r['action'] == 'replaced_small')
    print(f'Phase A+B done: 小图替换 {n_small}, 大图替换 {replaced}, '
          f'耗时 {time.time()-t0:.0f}s')
    print('Phase C: 重新生成被替换条目的官方结果')
    n_del = regenerate_changed()
    if n_del:
        subprocess.run([PY, '-X', 'utf8', os.path.join(HERE, 'evaluate_official.py'),
                        '--cores', args.cores, '--workers', str(args.workers)],
                       cwd=HERE)
    print(f'all done: removed {n_del} stale official results')


def regenerate_changed():
    """旧修改时间失效入口已停用；不删除任何历史文件。"""
    legacy_disabled()


if __name__ == '__main__':
    main()
