"""B/C闭环事件模型不能读取官方发射/命中作为输入。"""
import copy
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'solver'))
from bc_event_model import simulate_bc_plan,FifoResidency


def test_fifo_completion_semantics_including_zero_size():
    cache=FifoResidency(10)
    assert cache.insert(1,0)==[]
    cache.insert(2,6);cache.insert(3,4)
    assert cache.contains(2)
    assert cache.insert(2,6) is None
    assert list(cache.entries)==[1,2,3]
    assert cache.insert(4,5)==[1,2]
    assert list(cache.entries)==[3,4] and cache.used==9
    assert cache.insert(5,11) is None


def test_closed_loop_matches_official_operations_and_cache_events():
    from test_bc_copy_window import example
    from multicore_cut_evaluate_problem_3 import evaluate_problem_3,read_cache_config
    from scenario_contract import CODE
    graph,plan,b=example();config=read_cache_config(CODE.parent/'data/config.txt')
    hardware={k:b[k] for k in ('bandwidth_bytes_per_cycle','capacity_bytes','cross_core_copy_delay_cycles')}
    hardware.update(config)
    c=evaluate_problem_3(graph,plan,hardware['bandwidth_bytes_per_cycle'],hardware['capacity_bytes'],
        hardware['cross_core_copy_delay_cycles'],config['cache_capacity_bytes'],config['cache_bandwidth_bytes_per_cycle'])
    before=copy.deepcopy((graph,plan,hardware))
    for scene,raw in (('B',b),('C',c)):
        predicted=simulate_bc_plan(graph,plan,hardware,scene)
        observed={(core['core_id'],o['op_id']):(o['start'],o['end']) for core in raw['per_core_timeline'] for o in core['ops']}
        assert predicted['makespan']==raw['makespan']
        assert {key:(o['start'],o['end']) for key,o in predicted['operations'].items()}==observed
        if scene=='C':
            assert predicted['cache_events']==raw['cache_events']
            assert predicted['cache_final_entries']==raw['cache_final_entries']
            assert predicted['cache_stats']==raw['cache_stats']
    assert (graph,plan,hardware)==before
