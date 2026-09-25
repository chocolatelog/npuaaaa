"""区域切分身份、组外核序及全图列表构造的非平凡约束。"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'solver'))


def test_region_identity_uses_members_not_compressed_labels():
    from region_structure import split_region_identity
    import pytest
    assert split_region_identity([0,1,1,1,2],[2,0,1,1,3],1)==({0,1},{0:2,2:3})
    with pytest.raises(ValueError):split_region_identity([0,1,1,1,2],[0,0,1,1,2],1)


def test_complete_schedule_obeys_external_order_and_all_parents():
    from region_structure import complete_region_orders
    from task_order_search import replay_tasks
    durations={0:10,1:100,2:90,3:10,4:50}
    preds={0:set(),1:{0},2:{0},3:{1,2},4:set()}
    base=[[0,1,3],[4,2]]
    for rule in ('critical','earliest'):
        out=complete_region_orders(durations,preds,base,{1,2,3},(0,1),rule)
        assert out is not None
        assert [[t for t in q if t not in {1,2,3}] for q in out['orders']]==[[0],[4]]
        timing=replay_tasks(durations,preds,out['orders'])
        assert timing is not None and out['coarse']==timing['makespan']
        owners={t:c for c,q in enumerate(out['orders']) for t in q}
        assert out['start'][3]>=max(out['finish'][1]+(1000 if owners[1]!=owners[3] else 0),
                                     out['finish'][2]+(1000 if owners[2]!=owners[3] else 0))
    assert base==[[0,1,3],[4,2]]


def test_external_order_cycle_is_rejected_instead_of_repaired():
    from region_structure import complete_region_orders
    assert complete_region_orders({0:1,1:1,2:1},{0:{1},1:{2},2:set()},[[0,2],[1]],{1},(0,1)) is None


def test_structure_menu_preserves_diamond_join_and_never_splits_single_operator():
    from region_structure import structure_menu
    from solution import Sol
    from types import SimpleNamespace
    model=SimpleNamespace(blocks=[[i] for i in range(6)],block_topo=list(range(6)),block_pos={i:i for i in range(6)},
        block_edges=[((0,1),1),((1,2),1),((1,3),1),((2,4),1),((3,4),1),((4,5),1)],
        block_work_m=[10,1,100,80,1,10],block_work_v=[0]*6)
    parent=Sol([0,1,1,1,1,2],[0,0,1])
    rows,audit=structure_menu(model,parent,[1],4)
    assert len(rows)<=2*4 and audit['max_widths'][1]==2
    branch=next(r for r in rows if r['source']['kind']=='branch' and r['source']['width']==2)
    c=branch['sol'];assert len({c.sg_of_block[i] for i in (1,2,3,4)})==4
    for row in rows:
        child=row['sol'];assert child.validate(model,4)
        assert child.blocks_in_sg[child.sg_of_block[0]]=={0}
        assert child.blocks_in_sg[child.sg_of_block[5]]=={5}
    assert structure_menu(model,parent,[0],4)[1]['max_widths'][0]==1


def test_structure_menu_does_not_invent_multicore_width_on_strict_chain():
    from region_structure import structure_menu
    from solution import Sol
    from types import SimpleNamespace
    model=SimpleNamespace(blocks=[[i] for i in range(4)],block_topo=list(range(4)),block_pos={i:i for i in range(4)},
        block_edges=[((0,1),1),((1,2),1),((2,3),1)],block_work_m=[1,100,100,1],block_work_v=[0]*4)
    rows,audit=structure_menu(model,Sol([0]*4,[0]),[0],4)
    assert len(rows)==1 and rows[0]['source']['width']==1


def test_same_queue_with_different_member_partitions_is_not_duplicate():
    from region_group_refine import select_joint_candidates
    common={'orders':[[0],[1]],'coarse':10,'origins':[{'region':0,'width':2}]}
    rows=[{**common,'assignment':[0,0,1]},{**common,'assignment':[0,1,1]}]
    selected,_=select_joint_candidates(rows)
    assert len(selected)==2


def test_decoupled_core_menu_contains_tied_plans_and_explicit_single_core_splits():
    from model import Model
    from solution import Sol
    from region_structure import build_structure_candidates
    graph={'ops':[{'id':i,'op':'MATMUL','pipe':'PIPE_M','cycles':1000*i}
                  for i in (1,2,3)],'tensors':[],'edges':[]}
    model=Model(graph,block_ops_cap=1)
    parent=Sol([0]*len(model.blocks),[0])
    for n in (2,4):
        plan=model.plan_from(parent.sg_of_block,[[0]]+[[] for _ in range(n-1)])
        baseline={'tasks':{0:{'duration':6000}}}
        tied,ta=build_structure_candidates(graph,model,parent,plan,baseline,core_policy='tied')
        free,fa=build_structure_candidates(graph,model,parent,plan,baseline,core_policy='all')
        key=lambda r:(tuple(r['assignment']),tuple(map(tuple,r['orders'])))
        assert not ta['unfinished_streams'] and not fa['unfinished_streams']
        assert {key(r) for r in tied}<={key(r) for r in free}
        assert any(o['structure']['width']>1 and len(o['requested_cores'])==1
                   for r in free for o in r['origins'])
        assert all(len(o['requested_cores'])==o['structure']['width']
                   for r in tied for o in r['origins'])
        assert parent.sg_of_block==[0]*len(model.blocks)


def test_partition_distance_is_label_free_and_work_weighted():
    from region_structure import partition_distance
    weights={0:1,1:2,2:3,3:4}
    p=[{0,1},{2,3}]
    assert partition_distance(p,list(reversed(p)),weights)==0
    assert partition_distance(p,[{0,2},{1,3}],weights)>0
    assert partition_distance(p,[{0,2},{1,3}],weights)==partition_distance([{0,2},{1,3}],p,weights)


def test_diverse_menu_preserves_old_partitions_and_adds_other_branch_levels():
    from region_structure import structure_menu
    from solution import Sol
    from types import SimpleNamespace
    model=SimpleNamespace(blocks=[[i] for i in range(8)],block_topo=list(range(8)),block_pos={i:i for i in range(8)},
        block_edges=[((0,1),1),((0,2),1),((1,3),1),((2,4),1),((3,5),1),((4,6),1),((5,7),1),((6,7),1)],
        block_work_m=[1,100,90,80,70,60,50,1],block_work_v=[0]*8)
    parent=Sol([0]*8,[0])
    old,_=structure_menu(model,parent,[0],4)
    new,audit=structure_menu(model,parent,[0],4,menu_policy='diverse')
    key=lambda r:tuple(sorted(tuple(sorted(g)) for g in r['sol'].blocks_in_sg if g))
    assert {key(r) for r in old} < {key(r) for r in new}
    assert len(new)<=20 and audit['extra_kept']>0
    assert len({r['source'].get('level') for r in new if r['source']['kind']=='branch'})>1
    assert all(r['sol'].validate(model,4) for r in new)
    again,_=structure_menu(model,parent,[0],4,menu_policy='diverse')
    assert [key(r) for r in new]==[key(r) for r in again]


def test_diverse_menu_global_cap_keeps_both_regions_old_structures():
    from region_structure import structure_menu
    from solution import Sol
    from types import SimpleNamespace
    edges=[((base+i,base+i+5),1) for base in (0,15) for i in range(10)]
    model=SimpleNamespace(blocks=[[i] for i in range(30)],block_topo=list(range(30)),block_pos={i:i for i in range(30)},
        block_edges=edges,block_work_m=[i+1 for i in range(30)],block_work_v=[0]*30)
    parent=Sol([0]*15+[1]*15,[0,1])
    old,_=structure_menu(model,parent,[0,1],5)
    new,audit=structure_menu(model,parent,[0,1],5,menu_policy='diverse')
    key=lambda r:tuple(r['sol'].sg_of_block)
    assert len(new)==20 and len(old)<=18
    assert {key(r) for r in old}<={key(r) for r in new}
    assert audit['extra_unselected']>0 and all(r['sol'].validate(model,5) for r in new)


def test_seed_combinations_fill_unused_slots_without_replacing_old_structures():
    from region_structure import structure_menu
    from solution import Sol
    from types import SimpleNamespace
    model=SimpleNamespace(blocks=[[i] for i in range(6)],block_topo=list(range(6)),block_pos={i:i for i in range(6)},
        block_edges=[((0,i),1) for i in (1,2,3,4)]+[((i,5),1) for i in (1,2,3,4)],
        block_work_m=[1,100,90,80,70,1],block_work_v=[0]*6)
    parent=Sol([0]*6,[0])
    old,_=structure_menu(model,parent,[0],2,menu_policy='diverse')
    new,audit=structure_menu(model,parent,[0],2,menu_policy='seed_combinations')
    key=lambda r:tuple(r['sol'].sg_of_block)
    assert {key(r) for r in old} < {key(r) for r in new}
    assert any(r['source'].get('seeds')==[1,3] for r in new)
    assert len(new)<=20 and audit['seed_extra_kept']>0
    assert all(r['sol'].validate(model,2) for r in new)
