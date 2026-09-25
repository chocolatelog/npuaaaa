"""重放固定方案，比较旧代理与场景 A 逐事件精评，不重新搜索。"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time

from scene_a_event import SceneAEventModel
from calibrate_spill import decode_plan, block_cap_for
from model import Model

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--truth', default='results/spill_events_audit_control48_v1.jsonl')
    p.add_argument('--plans', default='results/plans_control_base_v1')
    p.add_argument('--output', default='results/scene_a_event_model_v1.jsonl')
    args = p.parse_args()
    truth = [json.loads(s) for s in Path(args.truth).read_text(encoding='utf-8').splitlines()]
    out = Path(args.output)
    fingerprint = hashlib.sha256(''.join(digest(Path(__file__).with_name(name)) for name in
        ('scene_a_event.py', 'spill_events.py', 'model.py', 'audit_scene_a_model.py')).encode()).hexdigest()
    rows = [json.loads(s) for s in out.read_text(encoding='utf-8').splitlines()] if out.exists() else []
    if any(r['source_hash'] != fingerprint for r in rows):
        raise ValueError('代码发生变化，请使用新输出文件')
    done = {(r['case'], r['N']) for r in rows}
    graphs = {}
    for i, t in enumerate(truth, 1):
        case, n = t['case'], t['N']
        gp = ROOT / '通用神经网络处理器下的多核调度问题附件/data' / f'{case}.json'
        pp = Path(args.plans) / f'{case}_A_N{n}.json'
        if digest(pp) != t['plan_sha256'] or digest(gp) != t['graph_sha256']:
            raise ValueError('官方真值与输入图或方案指纹不匹配')
        if (case, n) in done:
            print(f'[{i}/{len(truth)}] 已完成，跳过', flush=True)
            continue
        if case not in graphs:
            g = json.loads(gp.read_text(encoding='utf-8'))
            cap = block_cap_for(sum(o['op'] not in ('COPY_IN', 'COPY_OUT') for o in g['ops']))
            graphs[case] = (Model(g, block_ops_cap=cap), SceneAEventModel(g))
        model, precise = graphs[case]
        plan = json.loads(pp.read_text(encoding='utf-8'))
        sg, cores = decode_plan(model, plan)
        if model.plan_from(sg, plan['core_schedules']) != plan:
            raise ValueError('保存方案不能无损转换到当前块结构')
        before = time.perf_counter()
        old_mk, old_added, old_info = model.evaluate(sg, cores, 'A', n,
            use_cache=False, orders_override=plan['core_schedules'])
        old_seconds = time.perf_counter() - before
        before = time.perf_counter()
        new = precise.evaluate(plan)
        elapsed = time.perf_counter() - before
        mismatch = sum(a['spill_bytes'] != b['official_spill_bytes']
                       for a, b in zip(new['tasks'].values(), t['task_rows']))
        if len(new['tasks']) != len(t['task_rows']):
            raise ValueError('局部图数不同')
        row = {'case': case, 'N': n, 'scene': 'A', 'source_hash': fingerprint,
               'plan_sha256': t['plan_sha256'], 'official_makespan': t['official_makespan'],
               'official_spill_bytes': t['official_spill_bytes'],
               'old_makespan': old_mk, 'new_makespan': new['makespan'],
               'old_spill_bytes': old_info['spill_bytes'], 'new_spill_bytes': new['spill_bytes'],
               'old_relative_error': old_mk/t['official_makespan']-1,
               'new_relative_error': new['makespan']/t['official_makespan']-1,
               'mismatched_tasks': mismatch, 'old_seconds': old_seconds,
               'new_seconds': elapsed, 'new_detail': new}
        with out.open('a', encoding='utf-8') as f:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
            f.flush()
        rows.append(row)
        print(f'[{i}/{len(truth)}] {case}/N{n}: 字节错配={mismatch}, '
              f'耗时误差={row["old_relative_error"]:.1%}->{row["new_relative_error"]:.1%}', flush=True)
    summary = {'count': len(rows), 'mismatched_tasks': sum(r['mismatched_tasks'] for r in rows),
               'old_mape_ratio': statistics.mean(abs(r['old_relative_error']) for r in rows),
               'new_mape_ratio': statistics.mean(abs(r['new_relative_error']) for r in rows),
               'old_seconds': sum(r['old_seconds'] for r in rows),
               'new_seconds': sum(r['new_seconds'] for r in rows),
               'scope': '同一固定方案上的代理误差，未验证候选排序及搜索结果'}
    out.with_suffix('.summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
