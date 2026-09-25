"""问题三逐案裁决：B 方案 vs C 方案（各自 P3 官方评估），逐案保留更优。

流程：
1. 快照当前 P3 结果（C 方案版）-> official_p3_Cplan/
2. 移走 C 方案 -> P3 重评（回退读 B 方案）-> 快照 -> official_p3_Bplan/
3. 逐案词典序比较：C 胜 -> 恢复 C 方案与其结果；B 胜 -> 保持 B 结果（C 方案留在池目录）
4. 追加 eval_manifest 记录使状态一致
"""
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from repro import (EVAL_MANIFEST, append_manifest, eval_key, file_sha256,
                   load_manifest)

RESULTS = os.path.normpath(os.path.join(HERE, '..', 'results'))
OFFICIAL = os.path.join(RESULTS, 'official')
PLANS = os.path.join(RESULTS, 'plans')
C_POOL = os.path.join(RESULTS, 'plans_C_pool')
SNAP_C = os.path.join(RESULTS, 'official_p3_Cplan')
SNAP_B = os.path.join(RESULTS, 'official_p3_Bplan')
PY = sys.executable


def lexi_better(mk_a, ad_a, mk_b, ad_b, tol=0.003):
    if mk_a < mk_b * (1 - tol):
        return True
    if mk_a > mk_b * (1 + tol):
        return False
    return (ad_a or 0) < (ad_b or 0)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--step', required=True,
                        choices=['snapshot_c', 'eval_b', 'decide'])
    args = parser.parse_args()

    if args.step == 'snapshot_c':
        os.makedirs(SNAP_C, exist_ok=True)
        n = 0
        for f in os.listdir(OFFICIAL):
            if '_problem_3_' in f and f.endswith('_res.json'):
                shutil.copyfile(os.path.join(OFFICIAL, f),
                                os.path.join(SNAP_C, f))
                n += 1
        print(f'snapshotted {n} C-plan P3 results -> {SNAP_C}')
        # 移走 C 方案（P3 将回退读 B）
        os.makedirs(C_POOL, exist_ok=True)
        m = 0
        for f in os.listdir(PLANS):
            if f.endswith('.json') and '_C_N' in f:
                shutil.move(os.path.join(PLANS, f), os.path.join(C_POOL, f))
                m += 1
        print(f'moved {m} C plans -> {C_POOL}')
        return

    if args.step == 'eval_b':
        os.makedirs(SNAP_B, exist_ok=True)
        r = subprocess.run([PY, '-X', 'utf8',
                            os.path.join(HERE, 'evaluate_official.py'),
                            '--cases', '1-100', '--workers', '10'],
                           cwd=HERE)
        if r.returncode != 0:
            sys.exit('eval failed')
        n = 0
        for f in os.listdir(OFFICIAL):
            if '_problem_3_' in f and f.endswith('_res.json'):
                shutil.copyfile(os.path.join(OFFICIAL, f),
                                os.path.join(SNAP_B, f))
                n += 1
        print(f'snapshotted {n} B-plan P3 results -> {SNAP_B}')
        return

    if args.step == 'decide':
        wins_c = wins_b = ties = 0
        deltas = []
        report = []
        for f in sorted(os.listdir(SNAP_C)):
            if not f.endswith('_res.json'):
                continue
            rc = json.load(open(os.path.join(SNAP_C, f), encoding='utf-8'))
            rb = json.load(open(os.path.join(SNAP_B, f), encoding='utf-8'))
            mkc = rc['makespan']
            adc = rc['data_movement_bytes'].get('added_copy_bytes')
            mkb = rb['makespan']
            adb = rb['data_movement_bytes'].get('added_copy_bytes')
            case_prob = f[:-len('_res.json')]
            parts = case_prob.split('_')
            case = '_'.join(parts[:2])
            n = int(parts[-1][1:])
            plan_c = os.path.join(C_POOL, f'{case}_C_N{n}.json')
            plan_b = os.path.join(PLANS, f'{case}_B_N{n}.json')
            if lexi_better(mkc, adc, mkb, adb):
                # C 胜：恢复 C 方案与 C 结果
                shutil.copyfile(plan_c, os.path.join(
                    PLANS, f'{case}_C_N{n}.json'))
                shutil.copyfile(os.path.join(SNAP_C, f),
                                os.path.join(OFFICIAL, f))
                ph = file_sha256(plan_c)
                scene = 'C'
                wins_c += 1
                d = (mkc - mkb) / mkb * 100
            else:
                # B 胜：official 已是 B 结果（eval_b 步骤写入）
                ph = file_sha256(plan_b)
                scene = 'B'
                if abs(mkc - mkb) <= mkb * 0.003:
                    ties += 1
                else:
                    wins_b += 1
                d = (mkc - mkb) / mkb * 100
            deltas.append(d)
            report.append({'case': case, 'N': n, 'mk_Cplan': mkc,
                           'mk_Bplan': mkb, 'delta_pct': round(d, 2),
                           'winner': scene})
            append_manifest(EVAL_MANIFEST, {
                '_key': eval_key(case, 'problem_3', n), 'case': case,
                'problem': 'problem_3', 'N': n,
                'graph': load_manifest(EVAL_MANIFEST).get(
                    eval_key(case, 'problem_3', n), {}).get('graph', ''),
                'config': load_manifest(EVAL_MANIFEST).get(
                    eval_key(case, 'problem_3', n), {}).get('config', ''),
                'plan': ph, 'plan_scene': scene, 'status': 'ok'})
        import statistics as st
        with open(os.path.join(RESULTS, 'p3_decision_report.json'),
                  'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=1)
        print(f'C 胜 {wins_c} / B 胜 {wins_b} / 并列 {ties}')
        print(f'C 方案相对 B 方案的 P3 Makespan: 均值 {st.mean(deltas):+.2f}% '
              f'中位 {st.median(deltas):+.2f}%')


if __name__ == '__main__':
    main()
