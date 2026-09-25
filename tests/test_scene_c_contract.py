"""官方缓存语义回归：同时首次读取不能虚报命中，零缓存退化为问题二。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'通用神经网络处理器下的多核调度问题附件/code'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'solver'))
from multicore_cut_evaluate_problem_2 import evaluate_scene_b
from multicore_cut_evaluate_problem_3 import evaluate_problem_3
from official_protocol import compact_result


def test_concurrent_cold_reads_are_misses_and_zero_cache_matches_b():
    graph = {'ops': [
        {'id':1,'op':'COPY_IN','pipe':'PIPE_MTE2','cycles':1},
        {'id':2,'op':'MATMUL','pipe':'PIPE_M','cycles':100},
        {'id':3,'op':'MATMUL','pipe':'PIPE_M','cycles':100}],
        'tensors':[{'id':10001,'pos':'DDR','size':600},
                   {'id':10002,'pos':'L1','size':600}, {'id':10003,'pos':'UB','size':60}],
        'edges':[{'source':a,'target':b} for a,b in [(10001,1),(1,10002),
                   (10002,2),(10002,3),(2,10003),(10003,3)]]}
    plan = {'node_to_subgraph':{'2':0,'3':1},'core_schedules':[[0],[1]]}
    cap = {'L1':524288,'UB':131072}
    b = evaluate_scene_b(graph, plan, 60, cap, 500)
    c = evaluate_problem_3(graph, plan, 60, cap, 500, 1048576, 250)
    zero = evaluate_problem_3(graph, plan, 60, cap, 500, 0, 250)
    assert b['makespan'] == zero['makespan'] == 722
    assert b['data_movement_bytes'] == zero['data_movement_bytes']
    assert c['cache_stats']['copy_in_hits'] == 0
    assert c['cache_stats']['miss_bytes'] == 1260
    assert compact_result(c)['cache_hit_rate'] == 0
    misses = [e for e in c['cache_events'] if e['event']=='miss' and e['tensor_id']==10002]
    assert len(misses)==2 and all(e['time']==0 for e in misses)
