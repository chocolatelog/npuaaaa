"""验证导入历史证据，不调用资源或原官方评估器。"""
import argparse,json,time
from pathlib import Path
from evidence_index import load_evidence,file_sha

ROOT=Path(__file__).resolve().parents[1]
MENU_DIRS=('p1_e08_menu_r01','p1_e08_menu_memory_r01','p1_e08_decoupled_r01','p1_e08_decoupled_memory_r01',
           'p1_e08_diverse_r01','p1_e08_diverse_memory_r01','p1_e08_seed_combinations_r01','p1_e08_seed_combinations_r02')

def evidence_paths(root=ROOT):
    return [root/'results'/d/'menu_pairs.jsonl' for d in MENU_DIRS]+[
        root/'results/p1_e08_diverse_potential_r01/potential.jsonl',root/'results/p1_e05_branch_r03/branch_experiments.jsonl']

def main():
    p=argparse.ArgumentParser();p.add_argument('--cases',default='5,44,49,67,82');p.add_argument('--cores',default='2,4')
    p.add_argument('--output-dir',required=True);p.add_argument('--verify-runtime',action='store_true');a=p.parse_args()
    ids=[f'case_{int(c):03d}_N{int(n)}' for c in a.cases.split(',') for n in a.cores.split(',')]
    before=time.perf_counter();index=load_evidence(ROOT,evidence_paths(),ids,
        progress=lambda i,n,count:print(f'[{i}/{n}] 验证导入{count}条证据',flush=True))
    if a.verify_runtime:index.require_runtime()
    summary={**index.summary(),'build_seconds':time.perf_counter()-before,'config_ids':ids}
    inventory=list(index.records.values());out=Path(a.output_dir).resolve();out.mkdir(exist_ok=True,parents=True)
    manifest={'module_sha256':file_sha(Path(__file__).with_name('evidence_index.py')),
              'entry_sha256':file_sha(__file__),'contracts':index.contracts,'sources':index.imports,'config_ids':ids}
    for name,value in [('audit_manifest.json',manifest),('inventory.json',inventory)]:
        target=out/name
        if target.exists():
            if json.loads(target.read_text(encoding='utf-8'))!=value:raise ValueError('索引或源指纹改变，需使用新输出目录')
        else:target.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    target=out/'summary.json'
    if not target.exists():target.write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('imports','config_ids')},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
