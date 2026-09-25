"""局部模板复用必须隔离可变状态，并按完整参数和源码版本失效。"""
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'solver'))


def test_cache_separates_results_and_changes_with_arguments():
    from local_template_cache import TemplateCache
    calls=[]
    def original(graph,capacity):
        calls.append(1)
        return {'seq':list(graph['ops']),'capacity':dict(capacity)}
    cache=TemplateCache(max_entries=2,max_bytes=1024*1024,source_fingerprint='v1')
    wrapped=cache.wrap('local',original)
    graph={'ops':[1,2]}
    a=wrapped(graph,{'L1':2});a['seq'].append(9)
    assert wrapped(graph,{'L1':2})['seq']==[1,2] and len(calls)==1
    assert wrapped(graph,{'L1':3})['capacity']['L1']==3 and len(calls)==2
    wrapped({'ops':[1,3]},{'L1':2})
    assert cache.stats['evictions']==1
    cache.source_fingerprint='v2'
    wrapped(graph,{'L1':3})
    assert len(calls)==4


def test_cached_replica_keeps_all_official_fields_and_input_isolation():
    from local_template_cache import TemplateCache
    from scene_a_fast import build_counter_evaluator, official
    graph={'ops':[{'id':i,'op':'MATMUL','pipe':'PIPE_M','cycles':10} for i in (0,1)],
           'tensors':[],'edges':[]}
    cache=TemplateCache(source_fingerprint='test')
    evaluator=build_counter_evaluator(template_cache=cache)
    original=official.step1_schedule
    for orders in ([[0],[1]],[[0,1],[]],[[1],[0]]):
        plan={'node_to_subgraph':{'0':0,'1':1},'core_schedules':orders}
        args=(graph,plan,60,{'L1':524288,'UB':131072},1000,100)
        actual=evaluator(*args)
        assert actual==official.evaluate_scene_a(*args)
        actual['per_core_timeline'].clear()
        assert evaluator(*args)==official.evaluate_scene_a(*args)
    assert cache.stats['hits']>0
    assert official.step1_schedule is original


def test_capacity_counts_local_tasks_instead_of_three_intermediate_stages():
    from local_template_cache import TemplateCache
    from scene_a_fast import build_counter_evaluator
    graph={'ops':[{'id':i,'op':'MATMUL','pipe':'PIPE_M','cycles':10} for i in range(50)],
           'tensors':[],'edges':[]}
    plan={'node_to_subgraph':{str(i):i for i in range(50)},
          'core_schedules':[list(range(0,50,2)),list(range(1,50,2))]}
    cache=TemplateCache(max_entries=128)
    evaluate=build_counter_evaluator(template_cache=cache)
    args=(graph,plan,60,{'L1':524288,'UB':131072},1000,100)
    first=evaluate(*args)
    hits=cache.stats['hits']
    assert evaluate(*args)==first
    assert cache.stats['hits']-hits>=50, '同一分区循环访问不能因三个中间阶段抢占而全失效'
