"""冻结已验证方案的轻量重评包；历史凭据只作引用，异机须重新评测。"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil

from official_protocol import digest, read_ledger, verified_record, input_context, object_digest


def _target(root, relative):
    rel=Path(relative)
    if rel.is_absolute() or '..' in rel.parts:
        raise ValueError('相对路径越界')
    target=(Path(root).resolve()/rel).resolve()
    if not target.is_relative_to(Path(root).resolve()):raise ValueError('路径越界')
    return target


def freeze_files(sources, output, progress=None):
    sources=list(sources); names=[Path(rel).as_posix() for _,rel in sources]
    if len(set(names))!=len(names):raise ValueError('重复输出路径')
    targets=[_target(output,relative) for _,relative in sources]
    records=[]
    for i,((source,relative),target) in enumerate(zip(sources,targets),1):
        source=Path(source); sha=digest(source)
        if target.exists():
            if digest(target)!=sha:raise ValueError('禁止覆盖不一致文件：'+str(target))
        else:
            target.parent.mkdir(parents=True,exist_ok=True)
            pending=target.with_name(target.name+'.pending')
            shutil.copyfile(source,pending)
            if digest(pending)!=sha or digest(source)!=sha:raise ValueError('复制期间源内容变化')
            pending.replace(target)
        records.append(dict(path=Path(relative).as_posix(),sha256=sha,size=target.stat().st_size))
        if progress and (i%100==0 or i==len(sources)):progress(i,len(sources))
    return records


def verify_files(root, records, progress=None):
    seen=set()
    for i,row in enumerate(records,1):
        target=_target(root,row['path'])
        if target in seen:raise ValueError('重复输出路径')
        seen.add(target)
        if not target.is_file() or target.stat().st_size!=row['size'] or digest(target)!=row['sha256']:
            raise ValueError('包内容摘要不一致：'+row['path'])
        if progress and (i%100==0 or i==len(records)):progress(i,len(records))
    return len(records)


def export_bundle(root, directories, output):
    from run_serial_saved import complete_summary
    from run_saved_refine import write_json
    root=Path(root).resolve();output=Path(output).resolve()
    if output==root or not output.is_relative_to(root/'results'):
        raise ValueError('导出目录必须位于项目results子目录')
    sources=[];expected=[];refs=[]
    for scene,folder in zip('ABC',directories):
        folder=Path(folder).resolve()
        summary=complete_summary(folder,500)
        if not summary:raise ValueError('来源不是500项完整有效官方档案：'+str(folder))
        records,warnings=read_ledger(folder/'official/results.jsonl')
        if warnings:raise ValueError('来源清单损坏')
        coverage={(r['case'],r['N']) for r in records.values() if r['N']>1}
        if coverage!={(f'case_{case:03}',n) for case in range(1,101) for n in (2,3,4,5)}:
            raise ValueError('来源缺少完整案例/核数覆盖')
        for record in records.values():
            if not verified_record(record):raise ValueError('来源凭据失效')
            job={k:record[k] for k in ('case','kind','N','plan_scene')}
            if object_digest(input_context(job,folder/'plans',root/'通用神经网络处理器下的多核调度问题附件')[0])!=record['fingerprint']:
                raise ValueError('源数据、环境或官方源码与记录不匹配')
            if record['N']>1:
                if record['plan_scene']!=scene:raise ValueError('来源场景错误')
                name=f"{record['case']}_{scene}_N{record['N']}.json"
                path=folder/'plans'/name
                if digest(path)!=record['plan_sha256']:raise ValueError('方案与官方摘要不一致')
                sources.append((path,Path('plans')/name))
            expected.append({k:record[k] for k in ('case','kind','N','plan_scene','real','plan_sha256','input_sha256','config_sha256','evaluator_sha256')})
            refs.append(dict(record_id=record['record_id'],receipt_path=record['receipt_path'],fingerprint=record['fingerprint']))
        sources.append((folder/'official/summary.json',Path('archive')/f'{scene}_summary.json'))
    for directory in ('solver','tests'):
        sources.extend((p,p.relative_to(root)) for p in sorted((root/directory).glob('*.py')))
    attachment=root/'通用神经网络处理器下的多核调度问题附件'
    sources.extend((p,p.relative_to(root)) for p in sorted((attachment/'code').glob('*.py')))
    sources.extend((attachment/'data'/f'case_{case:03}.json',Path(attachment.name)/'data'/f'case_{case:03}.json') for case in range(1,101))
    sources.append((attachment/'data/config.txt',Path(attachment.name)/'data/config.txt'))
    for name in ('requirements-reproduce.txt','实验计划_问题一.md','实验记录_主分支_main-online-shared-v1.0.md','技术路线_主分支最佳流程_v1.1.md'):
        sources.append((root/name,name))
    spec={str(Path(relative).as_posix()):digest(path) for path,relative in sources}
    output.mkdir(parents=True,exist_ok=True)
    spec_path=output/'export_sources.json'
    if spec_path.exists() and json.loads(spec_path.read_text(encoding='utf8'))!=spec:raise ValueError('导出源改变，使用新输出目录')
    if not spec_path.exists():write_json(spec_path,spec)
    files=freeze_files(sources,output,lambda i,n:print(f'冻结文件[{i}/{n}]',flush=True))
    manifest=dict(schema_version=1,branch='main',version='main-night-r37',
        updated_at=datetime.now().astimezone().isoformat(timespec='minutes'),files=files,
        expected=expected,historical_receipt_references=refs,
        scope='保存方案的跨机器官方重评包；未携带历史10.9GiB轨迹，不宣称默认搜索从零重现该候选档案')
    path=output/'bundle_manifest.json'
    if path.exists():
        old=json.loads(path.read_text(encoding='utf8'))
        manifest['updated_at']=old['updated_at']
        if old!=manifest:raise ValueError('已冻结包清单变化')
    else:write_json(path,manifest)
    verify_files(output,files)
    print(f'重评包内容验证完成：{len(files)}文件，1200多核方案；未重复运行官方实验',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--a',required=True);p.add_argument('--b',required=True);p.add_argument('--c',required=True)
    p.add_argument('--output-dir',required=True);args=p.parse_args()
    export_bundle(Path(__file__).resolve().parents[1],[args.a,args.b,args.c],args.output_dir)


if __name__=='__main__':main()
