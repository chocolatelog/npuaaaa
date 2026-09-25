"""只读比较超限启发式与 op 级 LRU spill 估计，不改变求解流程。"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from calibrate_spill import decode_plan, block_cap_for  # noqa: E402
from model import Model, load_graph  # noqa: E402

ROOT = os.path.normpath(os.path.join(HERE, '..'))
ATTACH = os.path.join(ROOT, '通用神经网络处理器下的多核调度问题附件')
PLANS = os.path.join(ROOT, 'results', 'plans_case044_067_instrumented_v2')
LOG = os.path.join(ROOT, 'results',
                   'solve_log_case044_067_instrumented_v2.jsonl')


def main():
    official = {}
    with open(LOG, encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)
            official[(row['case'], row['N'])] = row['real']['spill_added']
    rows = []
    for case in ('case_044', 'case_067'):
        graph = load_graph(os.path.join(ATTACH, 'data', f'{case}.json'))
        n_eligible = sum(1 for op in graph['ops']
                         if op['op'] not in ('COPY_IN', 'COPY_OUT'))
        for n in (2, 3, 4, 5):
            plan_path = os.path.join(PLANS, f'{case}_A_N{n}.json')
            plan = json.load(open(plan_path, encoding='utf-8'))
            model = Model(graph, block_ops_cap=block_cap_for(n_eligible))
            sg, core = decode_plan(model, plan)
            values = []
            for use_lru in (False, True):
                model.use_lru_spill = use_lru
                _mk, _added, info = model.evaluate(
                    sg, core, 'A', n, use_cache=False)
                values.append(info['spill_bytes'])
            real = official[(case, n)]
            rows.append({'case': case, 'N': n,
                         'heuristic': values[0], 'lru': values[1],
                         'official': real,
                         'lru_ratio': values[1] / real if real else None})
            print(case, f'N={n}',
                  f'heuristic={values[0]:.1f}',
                  f'lru={values[1]:.1f}',
                  f'official={real:.1f}',
                  f'lru_rel={values[1] / real:.3f}')
    errors = [abs(row['lru'] - row['official'])
              for row in rows if row['official'] > 0]
    rel = [err / row['official'] for err, row in zip(errors, rows)
           if row['official'] > 0]
    report = {
        'schema_version': 'lru-spill-audit-v1',
        'rows': rows,
        'metrics': {
            'rows': len(rows),
            'mae': sum(errors) / max(1, len(errors)),
            'mape': sum(rel) / max(1, len(rel)),
            'within_half_to_double': sum(0.5 <= row['lru_ratio'] <= 2.0
                                         for row in rows),
        },
    }
    out = os.path.join(ROOT, 'results', 'lru_spill_audit_v1.json')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print('saved ->', out)


if __name__ == '__main__':
    main()
