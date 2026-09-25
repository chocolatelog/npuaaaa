"""按图、完整方案、评估语义隔离的只读历史证据索引。"""
import ast
from copy import deepcopy
import hashlib
import json
import math
import platform
from pathlib import Path
import sys
from importlib.metadata import version,PackageNotFoundError
import zipfile

METRICS=('makespan','added_copy_bytes','scheduled_copy_bytes','partition_added','spill_added')
TIMELINES={'metrics_only':0,'region_summary':1,'full_timeline':2}

def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()
def file_sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def plan_sha(plan):return digest({'n':plan['node_to_subgraph'],'c':plan['core_schedules']})

def runtime_environment():
    packages={}
    for name in ('numpy','scipy','torch'):
        try:packages[name]=version(name)
        except PackageNotFoundError:packages[name]=None
    return {'python':sys.version,'platform':platform.platform(),'packages':packages}

class EvidenceIndex:
    def __init__(self,contracts,environment=None):
        self.contracts=dict(contracts);self.records={};self.imports=[];self.environment=deepcopy(environment)

    def require_runtime(self):
        if self.environment is None or runtime_environment()!=self.environment:
            raise ValueError('当前评测运行环境与证据环境不同；只读索引仍可检查，不得直接复用执行')

    def lookup_for_evaluation(self,*args):
        self.require_runtime()
        return self.get(*args)

    def key(self,case,scene,n,graph_sha,plan_id,kind):
        if kind not in self.contracts:raise ValueError('未知证据级别')
        return case,scene,n,graph_sha,plan_id,kind,self.contracts[kind]

    def add_many(self,entries):
        staged=dict(self.records)
        for entry in entries:
            kind=entry['kind']
            if entry['contract']!=self.contracts.get(kind):raise ValueError('评估语义版本不一致')
            if entry['timeline_level'] not in TIMELINES:raise ValueError('未知时间线级别')
            metrics=entry['metrics']
            if set(metrics)!=set(METRICS) or any(not isinstance(metrics[k],(int,float)) or
                    not math.isfinite(metrics[k]) or metrics[k]<0 for k in METRICS) or metrics['makespan']==0:
                raise ValueError('证据指标不完整或非法')
            key=self.key(entry['case'],entry['scene'],entry['N'],entry['graph_sha256'],entry['plan_id'],kind)
            for other_kind in self.contracts:
                if other_kind==kind:continue
                other=staged.get(self.key(entry['case'],entry['scene'],entry['N'],entry['graph_sha256'],entry['plan_id'],other_kind))
                if other and other['metrics']!=metrics:raise ValueError('原官方与精确资源结果冲突')
            previous=staged.get(key)
            if previous and previous['metrics']!=metrics:raise ValueError('同图同计划同评估语义的指标冲突')
            if previous:
                merged=deepcopy(previous)
                if entry['provenance'] not in merged['sources']:merged['sources'].append(deepcopy(entry['provenance']))
                if TIMELINES[entry['timeline_level']]>TIMELINES[merged['timeline_level']]:
                    merged['timeline_level']=entry['timeline_level'];merged['trace']=deepcopy(entry['trace'])
                staged[key]=merged
            else:
                merged=deepcopy(entry);merged['sources']=[merged.pop('provenance')];staged[key]=merged
        self.records=staged

    def get(self,case,scene,n,graph_sha,plan_id,kind):
        value=self.records.get(self.key(case,scene,n,graph_sha,plan_id,kind))
        return deepcopy(value) if value else None

    def summary(self):
        return {'distinct_entries':len(self.records),
            'by_kind':{kind:sum(r['kind']==kind for r in self.records.values()) for kind in self.contracts},
            'timeline_levels':{level:sum(r['timeline_level']==level for r in self.records.values()) for level in TIMELINES},
            'source_references':sum(len(r['sources']) for r in self.records.values()),
            'contracts':self.contracts,'evaluation_environment':self.environment,'imports':self.imports,'evaluation_calls':0}

def verified_rows(log,fingerprint):
    rows=[]
    for line in Path(log).read_text(encoding='utf-8').splitlines():
        if not line.strip():continue
        row=json.loads(line);body=dict(row);expected=body.pop('binding',None)
        if digest(body)!=expected or row.get('fingerprint')!=fingerprint:raise ValueError('源记录绑定失效')
        for path,sha in row.get('artifacts',{}).items():
            if file_sha(path)!=sha:raise ValueError('证据计划文件发生变化')
        rows.append(row)
    return rows

def evaluation_contracts(root,environment):
    root=Path(root).resolve()
    official=list((root/'通用神经网络处理器下的多核调度问题附件/code').glob('*.py'))
    official += [root/'solver/pipeline.py',root/'solver/official_protocol.py']
    resource=official+[root/'solver'/name for name in ('scene_a_replay.py','scene_a_fast.py','local_template_cache.py','lookahead_place.py')]
    deps={kind:{str(p):file_sha(p) for p in paths} for kind,paths in [('official',official),('resource',resource)]}
    hardware={'scene':'A','bandwidth':60,'capacity':{'L1':524288,'UB':131072},'cross_wait':1000,'same_wait':100}
    return {kind:digest({'files':files,'hardware':hardware,'environment':environment}) for kind,files in deps.items()},deps

def certify_resource_call(root,log,meta):
    archive=log.parent/'frozen_source/source.zip'
    if log.name=='menu_pairs.jsonl':member='solver/experiment_region_menu.py';original=root/member
    elif log.name=='menu_order_pairs.jsonl':member='solver/experiment_menu_order.py';original=root/member
    elif log.name=='shared_pairs.jsonl':member='solver/experiment_shared_budget.py';original=root/member
    elif log.name=='potential.jsonl':member='audit_potential.py';original=root/'results/p1_e08_diverse_r01/audit_potential.py'
    else:raise ValueError('未适配的资源证据入口')
    with zipfile.ZipFile(archive) as z:source=z.read(member)
    if hashlib.sha256(source).hexdigest()!=meta['files'].get(str(original)):raise ValueError('资源入口冻结源码不匹配')
    calls=[node for node in ast.walk(ast.parse(source.decode('utf-8'))) if isinstance(node,ast.Call) and
           isinstance(node.func,ast.Name) and node.func.id=='evaluator']
    if not calls or any(node.keywords or len(node.args)!=6 or [ast.literal_eval(x) for x in node.args[2:]]!=
            [60,{'L1':524288,'UB':131072},1000,100] for node in calls):
        raise ValueError('资源评估硬件参数未通过冻结源码核验')

def load_evidence(root,paths,config_ids=None,progress=None):
    if not paths:raise ValueError('没有证据清单')
    first_meta=json.loads(Path(paths[0]).with_suffix('.manifest.json').read_text(encoding='utf-8'))
    environment=first_meta['environment']
    root=Path(root).resolve();contracts,deps=evaluation_contracts(root,environment);index=EvidenceIndex(contracts,environment)
    wanted=set(config_ids) if config_ids is not None else None
    for number,path in enumerate(paths):
        log=Path(path).resolve();log_sha=file_sha(log);meta_path=log.with_suffix('.manifest.json')
        if log.name not in ('menu_pairs.jsonl','menu_order_pairs.jsonl','shared_pairs.jsonl','potential.jsonl','branch_experiments.jsonl'):
            raise ValueError('未适配的历史清单格式')
        meta=json.loads(meta_path.read_text(encoding='utf-8'));payload={k:v for k,v in meta.items() if k!='fingerprint'}
        if digest(payload)!=meta['fingerprint']:raise ValueError('源清单指纹失效')
        if meta['environment']!=environment:raise ValueError('历史证据运行环境不一致，不能混合复用')
        rows=verified_rows(log,meta['fingerprint']);staged=[];kinds=set();plans={}
        for row in rows:
            cid=row.get('config_id',row['id'])
            if wanted is not None and cid not in wanted:continue
            if row.get('status')=='diagnostic_summary':continue
            if row.get('scene','A')!='A':raise ValueError('当前证据索引只接场景A')
            case,n=(cid.rsplit('_N',1)[0],int(cid.rsplit('_N',1)[1]))
            graph_path=root/'通用神经网络处理器下的多核调度问题附件/data'/f'{case}.json'
            graph_sha=file_sha(graph_path)
            if meta['files'].get(str(graph_path))!=graph_sha:raise ValueError('输入图摘要不一致')
            def add(kind,plan_path,metrics,trace=None):
                if plan_path not in row.get('artifacts',{}):raise ValueError('计划不在源行产物绑定内')
                if plan_path not in plans:plans[plan_path]=json.loads(Path(plan_path).read_text(encoding='utf-8'))
                plan=plans[plan_path]
                if len(plan['core_schedules'])!=n:raise ValueError('计划核心数量与案例键不符')
                kinds.add(kind)
                staged.append({'case':case,'scene':'A','N':n,'graph_sha256':graph_sha,'plan_id':plan_sha(plan),
                    'kind':kind,'contract':contracts[kind],'metrics':{k:metrics[k] for k in METRICS},'plan_path':plan_path,
                    'trace':trace or [],'timeline_level':'region_summary' if trace else 'metrics_only',
                    'provenance':{'ledger':str(log),'ledger_sha256':log_sha,'row_id':row['id'],'binding':row['binding'],
                                  'origin_binding':row.get('origin_binding'),'plan_sha256':row['artifacts'][plan_path]}})
            if row.get('status')=='resource_diagnostic_success':
                add('resource',row['plan_path'],row['resource']);continue
            if row.get('status')!='official_success':raise ValueError('未成功的源记录不能导入')
            add('official',row['plan_path'],row['real'])
            if row.get('existing'):
                existing=row['existing'];add('official',existing['plan_path'],existing['real'])
                for value in existing['audit']['candidates']:
                    if value.get('official') and not value['official'].get('error'):
                        add('official',value['plan_path'],value['official'])
            for value in row.get('evaluated',[]):
                if 'resource' in value:
                    kind=value.get('evaluation_kind','resource')
                    if kind not in ('official','resource'):raise ValueError('未知候选评分证据级别')
                    add(kind,value['plan_path'],value['resource'],value.get('trace'))
                if value.get('official'):add('official',value['plan_path'],value['official'])
            for value in row.get('checks',[]):
                if value.get('matches',True):add('official',value['plan_path'],value['official'])
        for kind in kinds:
            if any(meta['files'].get(name)!=sha for name,sha in deps[kind].items()):raise ValueError(f'{kind}评估依赖发生变化')
        if 'resource' in kinds:certify_resource_call(root,log,meta)
        if file_sha(log)!=log_sha:raise ValueError('读取期间源清单仍在变化，拒绝导入')
        index.add_many(staged)
        index.imports.append({'ledger':str(log),'sha256':log_sha,'accepted_references':len(staged)})
        if progress:progress(number+1,len(paths),len(staged))
    return index
