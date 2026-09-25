"""以E06同环境原官方全结果衔接E03输入；不跨环境升级旧指标。"""
import ast
import gzip
import hashlib
import json
from pathlib import Path
import zipfile
from evidence_index import digest,file_sha,plan_sha,evaluation_contracts,verified_rows
from experiment_order_pair import verified as verified_order

def read_full_official(row):
    if not row['full_result_equal'] or file_sha(row['source_plan'])!=row['source_sha256']:
        raise ValueError('完整官方证据的输入计划改变')
    if len(row['artifacts'])!=1:raise ValueError('未获得唯一全等结果')
    path=next(iter(row['artifacts']))
    if file_sha(path)!=row['artifacts'][path]:raise ValueError('完整官方结果文件改变')
    full=json.loads(gzip.decompress(Path(path).read_bytes()))
    if digest(full)!=row['result_sha256'].get('official') or any(h!=digest(full) for h in row['result_sha256'].values()):
        raise ValueError('完整结果未绑定原官方或三方不相等')
    dm=full['data_movement_bytes']
    metrics={'makespan':full['makespan'],'added_copy_bytes':dm['added_copy_bytes'],
        'scheduled_copy_bytes':dm['scheduled_copy_bytes'],'partition_added':dm['partition_added_copy_bytes'],
        'spill_added':dm['spill_added_copy_bytes']}
    if metrics!=row['real']:raise ValueError('完整结果与汇总指标不同')
    return metrics,path

def add_baselines(index,root,ids):
    root=Path(root).resolve();order=root/'results/p1_e03_r02/order_pairs.jsonl'
    log=root/'results/p1_e06_ddr_r01/resource_replay.jsonl'
    om=json.loads(order.with_suffix('.manifest.json').read_text(encoding='utf-8'))
    meta=json.loads(log.with_suffix('.manifest.json').read_text(encoding='utf-8'))
    for m in (om,meta):
        if digest({k:v for k,v in m.items() if k!='fingerprint'})!=m['fingerprint']:raise ValueError('原输入证据清单指纹失效')
    if meta['environment']!=index.environment:raise ValueError('E06官方环境不符')
    contracts,deps=evaluation_contracts(root,index.environment)
    if contracts['official']!=index.contracts['official'] or any(meta['files'].get(p)!=h for p,h in deps['official'].items()):
        raise ValueError('E06官方依赖不同')
    archive=log.parent/'frozen_source/source.zip';member='solver/audit_resource_replay.py'
    with zipfile.ZipFile(archive) as z:source=z.read(member)
    if hashlib.sha256(source).hexdigest()!=meta['files'][str(root/member)]:raise ValueError('E06冻结入口不同')
    tree=ast.parse(source.decode('utf-8'))
    params=[n.value for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='params' for t in n.targets)]
    if len(params)!=1 or not isinstance(params[0],ast.Tuple) or len(params[0].elts)!=6 or [ast.literal_eval(x) for x in params[0].elts[2:]]!=[60,{'L1':524288,'UB':131072},1000,100]:
        raise ValueError('E06冻结硬件参数不符')
    orders={r['id']:r for r in map(json.loads,order.read_text(encoding='utf-8').splitlines())}
    refs={r['id']:r for r in verified_rows(log,meta['fingerprint'])};entries=[];baselines={}
    for cid in ids:
        old=orders[cid];ref=refs[cid];variant=old['variants']['swap']
        if not verified_order(old,om['fingerprint']):raise ValueError('E03方案绑定失效')
        if ref['source_sha256']!=variant['plan_sha256'] or ref['source_plan']!=variant['plan_path']:raise ValueError('E03/E06非同一输入')
        metrics,trace=read_full_official(ref)
        if metrics!=variant['real']:raise ValueError('E03/E06输入成绩不同')
        case,n=cid.rsplit('_N',1);n=int(n)
        if (ref['case'],ref['N'])!=(case,n):raise ValueError('配置身份不符')
        gp=root/'通用神经网络处理器下的多核调度问题附件/data'/f'{case}.json';gh=file_sha(gp)
        if meta['files'].get(str(gp))!=gh or om['files'].get(str(gp))!=gh:raise ValueError('原图变更')
        plan=json.loads(Path(ref['source_plan']).read_text(encoding='utf-8'))
        entries.append({'case':case,'scene':'A','N':n,'graph_sha256':gh,'plan_id':plan_sha(plan),
            'kind':'official','contract':contracts['official'],'metrics':metrics,'plan_path':ref['source_plan'],
            'trace':{'path':trace,'sha256':file_sha(trace)},'timeline_level':'full_timeline',
            'provenance':{'ledger':str(log),'ledger_sha256':file_sha(log),'row_id':cid,'binding':ref['binding'],
                          'plan_sha256':ref['source_sha256'],'e03_binding':old['binding']}})
        baselines[cid]={'case':case,'N':n,'plan_path':ref['source_plan'],'real':metrics,'binding':ref['binding']}
    index.add_many(entries);index.imports.append({'ledger':str(log),'sha256':file_sha(log),'accepted_references':len(entries)})
    return baselines
