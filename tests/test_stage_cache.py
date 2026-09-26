"""有界阶段缓存：参数失效、对象隔离、字节预算和原函数命名空间隔离。"""
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'solver'))
from stage_cache import StageMemo


def test_full_arguments_key_and_mutation_isolation():
    calls=[]
    def compute(graph,capacity):
        calls.append(capacity);return {'items':list(graph),'capacity':capacity}
    cache=StageMemo(10000,'source-v1')
    first=cache.call('stage',compute,[1,2],capacity=4)
    first['items'].append(99)
    second=cache.call('stage',compute,[1,2],capacity=4)
    assert second=={'items':[1,2],'capacity':4} and len(calls)==1
    cache.call('stage',compute,[1,2],capacity=5)
    cache.call('stage',compute,[1,3],capacity=4)
    assert len(calls)==3 and cache.stats['hits']==1


def test_serialized_limit_evicts_and_skips_oversized_values():
    calls=[]
    def make(value):calls.append(value);return 'x'*value
    cache=StageMemo(180,'source-v1')
    cache.call('s',make,80);cache.call('s',make,81)
    assert cache.resident_bytes<=180 and cache.stats['evictions']>=1
    cache.call('s',make,1000)
    assert cache.resident_bytes<=180 and cache.stats['oversized']==1


def test_function_identity_does_not_alias():
    cache=StageMemo(1000,'v1')
    assert cache.call('s',lambda x:x+1,2)==3
    assert cache.call('s',lambda x:x+2,2)==4


def test_real_expansion_isolated_and_hardware_change_recomputed():
    import copy
    from stage_cache import CachedSceneExpansion
    from bc_observed_graph import expand
    import multicore_cut_evaluate_problem_2 as official
    from test_bc_copy_window import example
    g,p,r=example()
    names=('step1_schedule','step2_spill_insertion','prepare_step3_execution')
    identities={name:getattr(official,name) for name in names}
    cache=CachedSceneExpansion(1024*1024)
    first=cache(g,p,r)
    expected=copy.deepcopy(first)
    first[0].clear()
    assert cache(g,p,r)==expected==expand(g,p,r)
    assert cache.memo.stats['hits']>=6
    changed=copy.deepcopy(r);changed['bandwidth_bytes_per_cycle']=30
    assert cache(g,p,changed)==expand(g,p,changed)
    assert all(getattr(official,name) is identities[name] for name in names)


def test_cached_event_loop_equal_on_changed_plans_and_scenes():
    import copy
    from stage_cache import CachedSceneExpansion
    from bc_event_model import simulate_bc_plan
    from test_bc_copy_window import example
    from scenario_contract import CODE
    from multicore_cut_evaluate_problem_3 import read_cache_config
    g,p,r=example();r.update(read_cache_config(CODE.parent/'data/config.txt'))
    other=copy.deepcopy(p);other['core_schedules'][0]=[0,3]
    cache=CachedSceneExpansion(1024*1024)
    for scene,plan in [('B',p),('B',other),('C',other),('C',p)]:
        expected=simulate_bc_plan(g,plan,r,scene)
        actual=simulate_bc_plan(g,plan,r,scene,expander=cache)
        for key in expected:
            if key not in ('expansion_seconds','event_seconds'):
                assert actual[key]==expected[key]
