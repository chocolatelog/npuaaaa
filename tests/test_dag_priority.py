"""计算松弛的合法范围与加权优先级的依赖安全。"""
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'solver'))
from dag_priority import compute_certificate, priority_features, schedule_ready


def test_copy_tensor_paths_and_independent_pipes_bound():
    g={'ops':[{'id':3,'op':'MATMUL','pipe':'PIPE_M','cycles':10},
              {'id':8,'op':'COPY_OUT','pipe':'PIPE_MTE3','cycles':100},
              {'id':15,'op':'ADD','pipe':'PIPE_V','cycles':6},
              {'id':21,'op':'ADD','pipe':'PIPE_V','cycles':20}],
       'tensors':[{'id':7,'size':8},{'id':12,'size':8}],
       'edges':[{'source':3,'target':7},{'source':7,'target':8},
                {'source':8,'target':12},{'source':12,'target':15}]}
    c=compute_certificate(g,2)
    assert c['predecessors'][15]==[3]
    assert c['dependency_path']==20
    assert c['lower_bound']==20  # 不能把所有流水线工作相加后冒充界
    assert c['scope']=='global_compute_only_relaxation'
    assert compute_certificate(g,1)['lower_bound']==26


def test_invalid_graph_is_rejected_instead_of_partial_topology():
    g={'ops':[{'id':1,'op':'ADD','pipe':'PIPE_V','cycles':1}], 'tensors':[],
       'edges':[{'source':1,'target':1}]}
    with pytest.raises(ValueError):compute_certificate(g,2)
    g['edges']=[{'source':1,'target':99}]
    with pytest.raises(ValueError):compute_certificate(g,2)


def test_rank_never_processes_successor_before_unfinished_predecessor():
    preds={0:set(),1:{0},2:set(),3:{1,2},4:{3}}
    durations={0:0,1:0,2:1,3:2,4:1}
    f=priority_features(preds,durations)
    for alpha in (0,.25,.5):
        r=schedule_ready(preds,durations,{},2,alpha=alpha,same_wait=7,cross_wait=11,bandwidth=60)
        pos={i:k for k,i in enumerate(r['dispatch_order'])}
        for i,ps in preds.items():
            for p in ps:
                assert pos[p]<pos[i]
                assert r['starts'][i]>=r['ends'][p]+(11 if r['core_of'][p]!=r['core_of'][i] else 0)
        assert r==schedule_ready(preds,durations,{},2,alpha=alpha,same_wait=7,cross_wait=11,bandwidth=60)
    assert f['height'][3]==3


def test_same_core_serial_wait_and_bad_numeric_inputs():
    r=schedule_ready({0:set(),1:set()},{0:10,1:9},{},1,alpha=.25,same_wait=7,cross_wait=11,bandwidth=60)
    a,b=r['dispatch_order'];assert r['starts'][b]==r['ends'][a]+7
    with pytest.raises(ValueError):schedule_ready({0:set()},{0:float('nan')},{},2,alpha=0)
    with pytest.raises(ValueError):schedule_ready({0:{8}},{0:1},{},2,alpha=0)
