"""给定官方发射时刻/内存路径，独立动态服务池预测退休时刻的只读审计。"""
from collections import defaultdict
import time

from bc_observed_graph import expand
from dynamic_service_pool import SharedServicePool
from schedule_step3 import _op_duration,_uses_ddr_bandwidth,PIPES


def audit_service_timeline(graph,plan,raw):
    started=time.perf_counter()
    tasks,_=expand(graph,plan,raw)
    pools={name:SharedServicePool() for name in ('DDR','CACHE_READ')}
    predictions={name:{} for name in pools}
    issues=defaultdict(list);completions=defaultdict(list);times={0};requests={}
    pipe_index={pipe:i for i,pipe in enumerate(PIPES)}
    for core in raw['per_core_timeline']:
        task=tasks[core['core_id']]
        for observed in core['ops']:
            times.update((observed['start'],observed['end']))
            op_id=observed['op_id'];op=task['op_by_id'][op_id]
            path=observed.get('memory_path')
            if path=='CACHE_READ':bandwidth=raw['cache_bandwidth_bytes_per_cycle']
            elif _uses_ddr_bandwidth(op,task['in_tids'],task['out_tids'],task['tensor_by_id']):
                path='DDR';bandwidth=raw['bandwidth_bytes_per_cycle']
            else:continue
            key=(core['core_id'],op_id)
            duration=_op_duration(op,task['in_tids'],task['out_tids'],task['tensor_by_id'],bandwidth)
            record=dict(key=key,path=path,service=duration,start=observed['start'],end=observed['end'],pipe=op['pipe'])
            requests[key]=record;issues[record['start']].append(record);completions[record['end']].append(record)
    mismatches=[];completed=0;peak={name:0 for name in pools}
    for now in sorted(times):
        for record in completions[now]:
            predicted=predictions[record['path']].get(record['key'])
            if predicted!=now:
                mismatches.append(dict(core=record['key'][0],op_id=record['key'][1],path=record['path'],
                    expected_end=now,predicted_end=predicted,issue=record['start'],service=record['service']))
        if mismatches:break
        for pool in pools.values():pool.advance(now)
        for path,pool in pools.items():
            retired=[r['key'] for r in completions[now] if r['path']==path]
            if retired:pool.retire(retired,now);completed+=len(retired)
        for record in sorted(issues[now],key=lambda r:(r['key'][0],pipe_index[r['pipe']],r['key'][1])):
            pools[record['path']].issue(record['key'],record['service'],now)
        for path,pool in pools.items():
            predictions[path]=pool.finish_times();peak[path]=max(peak[path],len(pool.remaining))
    return dict(consistent=not mismatches and completed==len(requests),completed_requests=completed,
                expected_requests=len(requests),mismatches=mismatches,peak_requests=peak,
                elapsed=time.perf_counter()-started,
                scope='原官方发射时刻/缓存路径作为输入；独占服务工作由原算子计算，只预测动态共享后的完成时刻；未预测新发射顺序或缓存命中')
