"""动态共享服务池：请求到达、分数服务完成和整数退休的解析反例。"""
from pathlib import Path
import sys
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'solver'))
from dynamic_service_pool import SharedServicePool


def test_arrival_reschedules_all_and_fractional_exhaustion_frees_service():
    pool=SharedServicePool()
    pool.issue('a',1,0);pool.issue('b',2,0)
    assert pool.finish_times()=={'a':2,'b':3}
    pool.issue('c',1,1)
    assert pool.finish_times()=={'a':3,'c':4,'b':4}
    pool.advance(3)
    assert pool.remaining['a']==0
    assert pool.remaining['b']==pytest.approx(.75)
    assert pool.remaining['c']==pytest.approx(.25)
    pool.retire(['a'],3)
    pool.issue('d',1,3)
    assert pool.finish_times()=={'c':4,'b':5,'d':5}


def test_zero_work_stays_until_retirement_and_pools_are_independent():
    ddr=SharedServicePool();cache=SharedServicePool()
    ddr.issue((0,1),2,0);ddr.issue((1,1),2,0)
    old=ddr.finish_times();version=ddr.version
    cache.issue('hit',1,0)
    assert ddr.finish_times()==old and ddr.version==version
    ddr.advance(4)
    assert len(ddr.remaining)==2 and all(x==0 for x in ddr.remaining.values())
    ddr.retire([(0,1),(1,1)],4)
    assert not ddr.remaining


def test_invalid_input_and_early_retire_do_not_mutate():
    pool=SharedServicePool();pool.issue('x',10,2)
    before=(dict(pool.remaining),pool.now,pool.version)
    for action in (lambda:pool.issue('x',1,3),lambda:pool.retire(['x'],5),
                   lambda:pool.advance(1),lambda:pool.issue('y',float('nan'),3),
                   lambda:pool.issue('y',1,float('inf'))):
        with pytest.raises(ValueError):action()
        assert (pool.remaining,pool.now,pool.version)==before


def test_service_replay_matches_real_official_b_and_c():
    from test_bc_copy_window import example
    from service_pool_audit import audit_service_timeline
    from multicore_cut_evaluate_problem_3 import evaluate_problem_3,read_cache_config
    from scenario_contract import CODE
    graph,plan,b=example()
    config=read_cache_config(CODE.parent/'data/config.txt')
    c=evaluate_problem_3(graph,plan,b['bandwidth_bytes_per_cycle'],b['capacity_bytes'],
        b['cross_core_copy_delay_cycles'],config['cache_capacity_bytes'],config['cache_bandwidth_bytes_per_cycle'])
    for raw in (b,c):
        result=audit_service_timeline(graph,plan,raw)
        assert result['consistent'] and result['completed_requests']>0
