"""从唯一、经过绑定验证的清单生成表格、汇总和实验报告。"""
import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

from official_protocol import read_ledger, verified_record, job_key, digest


def three_way(rows):
    groups = defaultdict(dict)
    for r in rows:
        groups[(r['case'], r['N'])][(r['kind'], r['plan_scene'])] = r
    result = []
    for (case, n), group in sorted(groups.items()):
        a, b, c = [group.get(key) for key in [('problem_2','B'),('problem_3','B'),('problem_3','C')]]
        if not all((a,b,c)):
            continue
        if a['plan_sha256'] != b['plan_sha256']:
            continue
        if any(len({r.get(field) for r in (a,b,c)}) != 1 or a.get(field) is None
               for field in ('input_sha256','config_sha256','evaluator_sha256')):
            continue
        ma, mb, mc = [r['real']['makespan'] for r in (a,b,c)]
        result.append({'case': case, 'N': n, 'b_problem2': ma, 'b_problem3': mb,
                       'c_problem3': mc, 'hardware_speedup': ma/mb, 'algorithm_speedup': mb/mc,
                       'combined_speedup': ma/mc})
    return result


def build_summary(ledger):
    ledger = Path(ledger)
    latest, malformed = read_ledger(ledger)
    manifest = ledger.with_suffix('.manifest.json')
    expected = None
    if manifest.exists():
        settings = json.loads(manifest.read_text(encoding='utf-8'))['settings']
        if 'jobs' in settings:
            expected = {job_key(j) for j in settings['jobs']}
            latest = {k:v for k,v in latest.items() if k in expected}
    rows = [r for _,r in sorted(latest.items()) if verified_record(r)]
    baseline = {r['case']:r for r in rows if r['kind']=='singlecore'}
    flat = []
    for r in rows:
        item = {k:r.get(k) for k in ('case','N','kind','plan_scene','plan_sha256','fingerprint','record_id')}
        item.update(r['real'])
        single = baseline.get(r['case'])
        item['speedup'] = None
        if single and all(single.get(k)==r.get(k) for k in ('input_sha256','config_sha256','evaluator_sha256')):
            item['speedup'] = single['real']['makespan']/r['real']['makespan']
        flat.append(item)
    groups = defaultdict(list)
    for r in flat:
        if r['kind'] != 'singlecore':
            groups[(r['kind'],r['plan_scene'],r['N'])].append(r)
    means=[]
    for (kind,scene,n), group in sorted(groups.items()):
        values=[r['speedup'] for r in group if r['speedup'] is not None]
        means.append({'kind':kind,'plan_scene':scene,'N':n,'official_count':len(group),
                      'comparable_count':len(values),'mean_speedup':statistics.mean(values) if values else None,
                      'mean_makespan':statistics.mean(r['makespan'] for r in group)})
    return {'schema_version':3,'source':str(ledger.resolve()),'source_sha256':digest(ledger),
            'updated_at':max((r.get('updated_at','') for r in latest.values()),default='未运行'),
            'expected':len(expected) if expected is not None else len(latest),
            'official_success':len(rows),'pending':len(expected-set(latest)) if expected else 0,
            'failed_or_invalid':len(latest)-len(rows),'malformed_lines':malformed,
            'rows':flat,'groups':means,'three_way':three_way(rows)}


def write_reports(ledger, output_dir):
    summary=build_summary(ledger)
    folder=Path(output_dir); folder.mkdir(parents=True,exist_ok=True)
    (folder/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    fields=['case','N','kind','plan_scene','makespan','added_copy_bytes','scheduled_copy_bytes',
            'partition_added','spill_added','cache_hit_bytes','cache_miss_bytes','cache_hit_rate','speedup',
            'plan_sha256','fingerprint','record_id']
    with (folder/'summary.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(summary['rows'])
    lines=['# 官方实验统一汇总','', '所属分支：main（主分支本地工作区）  ',
           '结果协议版本：3；代码版本见各条记录的源码内容指纹  ',
           '更新时间：'+summary['updated_at'],'',
           f"清单内容摘要：{summary['source_sha256']}", '',
           f"预期 {summary['expected']} 项；官方有效 {summary['official_success']} 项；未运行 {summary['pending']} 项；失败或失效 {summary['failed_or_invalid']} 项。",
           '', '表格、本文和实验报告来自同一份已验证清单。没有有效单核对照时不填加速比，不将代理结果混入官方统计。', '',
           '| 问题 | 方案场景 | 核数 | 官方数 | 可比数 | 平均加速比 | 平均耗时 |',
           '|---|---|---:|---:|---:|---:|---:|']
    for r in summary['groups']:
        speed='未评单核' if r['mean_speedup'] is None else f"{r['mean_speedup']:.6f}"
        lines.append(f"| {r['kind'].replace('problem_', '问题')} | {r['plan_scene']} | {r['N']} | {r['official_count']} | {r['comparable_count']} | {speed} | {r['mean_makespan']:.6f} |")
    lines += ['', '## 问题三归因', '',
              '硬件收益使用同一 B 方案比较问题二与问题三；算法收益比较问题三内 B 与 C 方案。只有三份记录的图、配置和评估代码一致，且前两份方案摘要相同，才计算。', '',
              '| 案例 | 核数 | 硬件加速比 | 算法加速比 | 综合加速比 |', '|---|---:|---:|---:|---:|']
    for r in summary['three_way']:
        lines.append(f"| {r['case']} | {r['N']} | {r['hardware_speedup']:.6f} | {r['algorithm_speedup']:.6f} | {r['combined_speedup']:.6f} |")
    if not summary['three_way']:
        lines += ['', '暂无完整且可比的三组结果，不能进行硬件/算法归因。']
    text='\n'.join(lines)+'\n'
    (folder/'summary.md').write_text(text,encoding='utf-8')
    (folder/'experiment_report.md').write_text(text+'\n## 验证范围\n\n本报告只描述清单内的方案评估，不证明完整求解器重复搜索确定性，也不代表全量算法提升。\n',encoding='utf-8')
    print(f"统一汇总：官方有效 {len(summary['rows'])} 项，三组可比 {len(summary['three_way'])} 组",flush=True)
    return summary


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--ledger',required=True,help='唯一官方结果清单；旧无指纹目录不自动混入')
    parser.add_argument('--output-dir',required=True)
    args=parser.parse_args()
    write_reports(args.ledger,args.output_dir)


if __name__ == '__main__':
    main()
