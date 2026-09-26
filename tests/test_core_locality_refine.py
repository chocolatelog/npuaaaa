"""B/C核亲和迁移的跨核连接计数及负载约束。"""
import copy
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'solver'))


def data():
    graph=dict(ops=[dict(id=i,op='ADD',pipe='PIPE_V',cycles=100 if i==3 else 10) for i in range(4)],
               tensors=[dict(id=10,pos='UB',size=60)],edges=[dict(source=0,target=10),dict(source=10,target=1),dict(source=10,target=2)])
    parent=dict(node_to_subgraph={str(i):i for i in range(4)},core_schedules=[[0,3],[1,2]])
    raw=dict(per_core_timeline=[dict(core_id=0,ops=[dict(op_id=0,start=0),dict(op_id=3,start=10)]),dict(core_id=1,ops=[dict(op_id=1,start=520),dict(op_id=2,start=530)])])
    return graph,parent,raw


def test_traffic_counts_connections_not_consumer_edges():
    from core_locality_refine import CoreTrafficIndex
    g,p,_=data();index=CoreTrafficIndex(g,{0:0,1:1,2:1,3:0},60)
    assert index.total()==dict(bytes=120,service=2,links=1)
    assert index.move_delta(1,0)==dict(bytes=0,service=0,links=0)
    assert index.move_delta(0,1)==dict(bytes=120,service=2,links=1)
    index.move(0,1)
    assert index.total()==dict(bytes=0,service=0,links=0)


def test_original_output_and_multiple_producers():
    from core_locality_refine import CoreTrafficIndex
    g,_,_=data();g['ops'].append(dict(id=4,op='COPY_OUT',pipe='PIPE_MTE3',cycles=1));g['edges'] += [dict(source=3,target=10),dict(source=10,target=4)]
    index=CoreTrafficIndex(g,{0:0,1:1,2:1,3:1},60)
    # 两个输出核各写回一次，另有0->1一对搬运，共4次。
    assert index.total()==dict(bytes=240,service=4,links=1)
    delta=index.move_delta(0,1);index.move(0,1)
    assert delta==dict(bytes=180,service=3,links=1)
    assert index.total()==dict(bytes=60,service=1,links=0)


def test_boundary_bytes_match_official_expansion():
    from core_locality_refine import CoreTrafficIndex
    from scenario_contract import CODE
    from multicore_cut_evaluate_problem_2 import _build_scene_b_tasks
    from evaluation_validation import read_evaluation_config
    config=read_evaluation_config(str(CODE.parent/'data/config.txt'))
    g,p,_=data()
    for core in (0,1):
        plan=copy.deepcopy(p)
        if core==1:plan['core_schedules']=[[3],[0,1,2]]
        mapping={i:c for c,row in enumerate(plan['core_schedules']) for i in row}
        expanded=_build_scene_b_tasks(g,plan,60,config['capacity'])
        # Step2（溢出插入）之前的静态搬运，与本模块计数对齐。
        assert expanded[3]['partition_added_copy_bytes']==CoreTrafficIndex(g,mapping,60).total()['bytes']


def test_locality_candidate_preserves_operations_and_bounds_pipe_work():
    from core_locality_refine import generate_core_locality_candidates
    from scenario_contract import validate_plan
    g,p,r=data();original=copy.deepcopy((g,p,r))
    rows,audit=generate_core_locality_candidates(g,p,r,bandwidth=60)
    assert rows and audit['moves']>=1
    for row in rows:
        plan=row['plan'];validate_plan(g,plan,'B')
        assert plan['node_to_subgraph']==p['node_to_subgraph']
        assert max(sum(g['ops'][i]['cycles'] for i in order) for order in plan['core_schedules'])<=110
        assert row['source']['boundary_bytes_saved']>0
    assert rows==generate_core_locality_candidates(g,p,r,bandwidth=60)[0]
    assert (g,p,r)==original


def test_exchange_joint_delta_does_not_double_count_shared_tensor():
    from core_locality_refine import CoreTrafficIndex
    g,_,_=data();index=CoreTrafficIndex(g,{0:0,1:1,2:0,3:0},60)
    # 交换生产者和其中一个消费者后仍有跨核连接，不能把两个单独收益相加。
    assert index.changes_delta({0:1,1:0})==dict(bytes=0,service=0,links=0)


def test_exchange_unlocks_balanced_cores_without_raising_load():
    from core_locality_refine import generate_core_locality_candidates
    from scenario_contract import validate_plan
    g,p,r=data();g['ops'][3]['cycles']=10
    # 仅0->1传输。两核各20工作，任何单迁移都超上限，交换0与2可消除传输。
    g['edges']=[dict(source=0,target=10),dict(source=10,target=1)]
    single,audit=generate_core_locality_candidates(g,p,r,bandwidth=60)
    assert not single and audit['moves']==0
    rows,audit=generate_core_locality_candidates(g,p,r,bandwidth=60,exchange_only=True)
    assert rows and audit['moves']==2
    for row in rows:
        validate_plan(g,row['plan'],'B')
        assert all(len(order)==2 for order in row['plan']['core_schedules'])
        assert row['source']['boundary_bytes_saved']==120
