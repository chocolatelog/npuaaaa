"""从逐任务检查点分析官方配对结果、搬运、溢出、排序局限及成本。"""
import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

from official_protocol import verified_record, digest


def analyze(folder):
    folder = Path(folder)
    rows = [json.loads(p.read_text(encoding='utf-8')) for p in sorted((folder/'checkpoints').glob('*.json'))]
    for row in rows:
        if not verified_record(row['official_record']):
            raise ValueError('失效的官方凭据，禁止生成性能结论')
        name = f"{row['case']}_{row['plan_scene']}_N{row['N']}.json"
        if digest(folder/'plans'/name) != row['output_plan_sha256']:
            raise ValueError('输出方案被更改')
    paired = []
    proxy_pairs = []
    for row in rows:
        baseline, final = row['baseline'], row['real']
        audit = row['audit']
        item = {k: row[k] for k in ('case', 'N', 'plan_scene', 'elapsed')}
        item.update(baseline=baseline['makespan'], final=final['makespan'],
                    reduction_fraction=1-final['makespan']/baseline['makespan'],
                    errors=len(audit.get('errors', [])))
        for metric in ('added_copy_bytes', 'scheduled_copy_bytes', 'spill_added', 'partition_added'):
            item[metric+'_delta'] = final[metric]-baseline[metric]
        if 'seed_best_official' in audit:
            seed = audit['seed_best_official']['makespan']
            item.update(seed_best=seed, search_reduction_fraction=1-final['makespan']/seed)
        paired.append(item)
        if row['plan_scene'] == 'A':
            for candidate in audit.get('candidates', []):
                truth = candidate.get('official') or {}
                if truth.get('makespan', 0) > 0 and candidate.get('estimate') is not None:
                    proxy_pairs.append({'case': row['case'], 'N': row['N'],
                        'proxy': candidate['estimate'], 'official': truth['makespan'],
                        'relative_error': (candidate['estimate']-truth['makespan'])/truth['makespan']})
    groups = []
    for n in sorted({r['N'] for r in paired}):
        sample = [r for r in paired if r['N'] == n]
        groups.append({'N': n, 'count': len(sample),
            'wins': sum(r['final'] < r['baseline'] for r in sample),
            'losses': sum(r['final'] > r['baseline'] for r in sample),
            'mean_reduction_fraction': statistics.mean(r['reduction_fraction'] for r in sample),
            'mean_seconds': statistics.mean(r['elapsed'] for r in sample),
            'max_seconds': max(r['elapsed'] for r in sample),
            **{k: statistics.mean(r[k] for r in sample) for k in
               ('scheduled_copy_bytes_delta', 'spill_added_delta', 'partition_added_delta')}})
    diagnostics = []
    for row in rows:
        search = row['audit'].get('search', {})
        candidates = search.get('candidates', [])[1:]
        if candidates:
            scores = [c['metrics'].get('proxy_score') for c in candidates]
            diagnostics.append({'case': row['case'], 'N': row['N'],
                'official_candidate_actions': dict(Counter(c['action'] for c in candidates)),
                'distinct_proxy_scores': len(set(s for s in scores if s is not None)),
                'selected_count': len(candidates),
                'scope': '仅官方入选候选；未评完整池，不能据此声称Top-k召回率'})
    value = {'completed': len(rows), 'groups': groups, 'paired': paired,
             'proxy_official_pairs': proxy_pairs, 'candidate_diagnostics': diagnostics,
             'gpu_execution': '本轮固定方案后处理未使用显卡；输入最佳方案曾使用共享预算模块',
             'memory_peak': '本轮未采集，不能用瞬时占用代替全程峰值',
             'conclusion_scope': '官方保底后处理效果；不等同于端到端重新搜索增益'}
    (folder/'paired_analysis.json').write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    return value


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True)
    value = analyze(parser.parse_args().output_dir)
    print(json.dumps({'completed': value['completed'], 'groups': value['groups']}, ensure_ascii=False))
