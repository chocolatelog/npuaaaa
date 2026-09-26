"""块化不可通过合并整片环来突破粒度限制。"""
import sys
from pathlib import Path
import random
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'solver'))


def test_frontier_budget_preserves_parallel_groups_for_independent_sources():
    from bounded_coarsening import coarsen
    preds={i:set() for i in range(100)}
    blocks,audit=coarsen(list(preds),preds,{},max_ops=240,work={i:10 for i in preds},max_work=10000,pack_frontiers=True,frontier_width=5)
    assert len(blocks)==5
    assert max(len(b) for b in blocks)==20
    assert audit['frontier_width']==5


def test_frontier_local_budget_keeps_large_singletons_and_cycle_safety():
    from bounded_coarsening import coarsen
    preds={0:set(),1:set(),2:set(),3:{0,1},4:{1,2},5:{3,4}}
    work={0:100,1:1,2:1,3:2,4:2,5:3}
    blocks,audit=coarsen(list(preds),preds,{(p,i):1 for i,ps in preds.items() for p in ps},max_ops=3,work=work,max_work=10,pack_frontiers=True,frontier_width=5)
    assert_acyclic(blocks,preds)
    assert [0] in blocks
    assert audit['oversize_singletons']==[0]


def test_entry_preserving_does_not_pull_join_into_earlier_independent_branch():
    from bounded_coarsening import coarsen
    preds={0:set(),1:set(),2:{0,1},3:{2}}
    blocks,audit=coarsen(list(preds),preds,{(0,2):100,(1,2):1,(2,3):100},max_ops=240,work={i:10 for i in preds},preserve_entries=True)
    assert blocks==[[0],[1],[2,3]]
    assert audit['entry_rejections']==2


def test_entry_preserving_keeps_initial_external_predecessors_for_every_block():
    from bounded_coarsening import coarsen
    rng=random.Random(97)
    for _ in range(20):
        preds={i:{j for j in range(i) if rng.random()<.1} for i in range(40)}
        blocks,_=coarsen(list(preds),preds,{(p,i):rng.randrange(1,10) for i,ps in preds.items() for p in ps},max_ops=6,work={i:1 for i in preds},pack_frontiers=True,preserve_entries=True)
        owner={op:g for g,b in enumerate(blocks) for op in b}
        for g,block in enumerate(blocks):
            initial={owner[p] for p in preds[block[0]] if owner[p]!=g}
            assert all({owner[p] for p in preds[op] if owner[p]!=g}<=initial for op in block)
        assert_acyclic(blocks,preds)


def assert_acyclic(blocks, predecessors):
    owner = {op:i for i,block in enumerate(blocks) for op in block}
    deps = {i:set() for i in range(len(blocks))}
    for op, ps in predecessors.items():
        deps[owner[op]].update(owner[p] for p in ps if owner[p] != owner[op])
    seen = set()
    while True:
        ready = {i for i,ps in deps.items() if i not in seen and ps <= seen}
        if not ready:break
        seen.update(ready)
    assert len(seen) == len(blocks)


def test_rejects_merge_that_would_close_an_alternate_path():
    from bounded_coarsening import coarsen
    preds = {0:set(),1:{0},2:{0},3:{1,2}}
    affinity = {(0,1):100,(0,2):0,(1,3):100,(2,3):1}
    blocks, audit = coarsen([0,1,2,3],preds,affinity,max_ops=3,work={i:1 for i in preds})
    assert_acyclic(blocks,preds)
    assert max(map(len,blocks)) <= 3
    assert audit['cycle_rejections'] >= 1
    assert sorted(op for block in blocks for op in block) == list(preds)


def test_work_cap_and_reachability_budget_are_conservative():
    from bounded_coarsening import coarsen
    preds = {0:set(),1:{0},2:{0},3:{1,2}}
    blocks,audit = coarsen([0,1,2,3],preds,{(0,1):100,(1,3):100,(2,3):1},
                          max_ops=10,work={0:2,1:2,2:20,3:2},max_work=5,reach_budget=1)
    assert_acyclic(blocks,preds)
    assert all(len(b)==1 or sum({0:2,1:2,2:20,3:2}[i] for i in b)<=5 for b in blocks)
    assert audit['oversize_singletons']==[2]


def test_deterministic_bounded_partition_on_random_dags():
    from bounded_coarsening import coarsen
    rng=random.Random(42)
    for _ in range(12):
        preds={i:{j for j in range(i) if rng.random()<.07} for i in range(70)}
        affinity={(p,i):rng.randrange(1,100) for i,ps in preds.items() for p in ps}
        kwargs=dict(max_ops=7,work={i:rng.randrange(1,8) for i in preds},max_work=24,reach_budget=20)
        blocks,audit=coarsen(list(preds),preds,affinity,**kwargs)
        repeated=coarsen(list(preds),preds,affinity,**kwargs)
        assert repeated==(blocks,audit)
        assert_acyclic(blocks,preds)
        assert max(map(len,blocks))<=7
        assert sorted(op for block in blocks for op in block)==list(preds)


def test_model_policy_prevents_legacy_scc_from_creating_oversize_block():
    from model import Model
    graph={'ops':[{'id':i,'op':'ADD','pipe':'PIPE_V','cycles':1} for i in range(4)],
           'tensors':[{'id':100+i,'pos':'UB','size':size} for i,size in enumerate((100,0,100,1))],
           'edges':[{'source':a,'target':b} for a,b in
                    [(0,100),(100,1),(0,101),(101,2),(1,102),(102,3),(2,103),(103,3)]]}
    legacy=Model(graph,block_ops_cap=3)
    assert max(map(len,legacy.blocks))==4
    bounded=Model(graph,block_ops_cap=3,block_policy='bounded')
    assert max(map(len,bounded.blocks))<=3
    assert bounded.coarsening_audit['cycle_rejections']>=1
    assert_acyclic(bounded.blocks,bounded.eligible_preds)


def test_independent_frontier_is_packed_without_inventing_dependencies():
    from bounded_coarsening import coarsen
    preds={i:set() for i in range(100)}
    blocks,audit=coarsen(list(preds),preds,{},max_ops=10,work={i:1 for i in preds},
                         max_work=5,pack_frontiers=True)
    assert len(blocks)==20
    assert max(map(len,blocks))==5
    assert audit['frontier_merges']==80
    assert_acyclic(blocks,preds)


def test_frontier_packing_still_checks_contracted_graph_cycles():
    from bounded_coarsening import coarsen
    # 原拓扑编号允许在处理一条支路后才看到同层的另一条支路。
    preds={0:set(),1:{0},2:set(),3:{1,2},4:{2},5:{3,4}}
    blocks,audit=coarsen(list(preds),preds,{(0,1):9,(1,3):9,(2,4):9,(3,5):9},
                         max_ops=3,work={i:1 for i in preds},pack_frontiers=True)
    assert_acyclic(blocks,preds)
    assert max(map(len,blocks))<=3


def test_layer_windows_keep_five_chains_parallel_and_bound_output_delay():
    from bounded_coarsening import coarsen
    preds={i:({i-5} if i>=5 else set()) for i in range(60)}
    blocks,audit=coarsen(list(preds),preds,{(p,i):1 for i,ps in preds.items() for p in ps},
                         max_ops=240,work={i:1 for i in preds},max_work=100,
                         pack_frontiers=True,frontier_width=5,preserve_entries=True,layer_window=4)
    assert len(blocks)==15
    assert all(len(b)==4 and len({i%5 for i in b})==1 for b in blocks)
    assert all(len({(i//5)//4 for i in b})==1 for b in blocks)
    assert audit['layer_boundary_rejections']>0
    assert_acyclic(blocks,preds)


def test_layer_window_budget_and_coverage_on_random_dags():
    from bounded_coarsening import coarsen
    rng=random.Random(416)
    for _ in range(10):
        preds={i:{j for j in range(i) if rng.random()<.09} for i in range(60)}
        work={i:rng.randint(1,20) for i in preds};levels={}
        for i in preds:levels[i]=max((levels[p] for p in preds[i]),default=-1)+1
        blocks,audit=coarsen(list(preds),preds,{(p,i):1 for i,ps in preds.items() for p in ps},max_ops=8,work=work,pack_frontiers=True,frontier_width=5,preserve_entries=True,layer_window=4)
        assert sorted(i for b in blocks for i in b)==list(preds)
        for b in blocks:
            bands={levels[i]//4 for i in b}
            assert len(bands)==1
            assert sum(work[i] for i in b)<=audit['layer_window_caps'][next(iter(bands))]
        assert_acyclic(blocks,preds)


def test_transitive_entry_accepts_redundant_ancestor_but_not_independent_input():
    from bounded_coarsening import coarsen
    preds={0:set(),1:{0},2:{1},3:{2,0},4:set(),5:{3,4}}
    kw=dict(max_ops=240,work={i:1 for i in preds},preserve_entries=True)
    affinity={(2,3):10,(3,5):10}
    strict,_=coarsen(list(preds),preds,affinity,**kw)
    relaxed,audit=coarsen(list(preds),preds,affinity,entry_ancestors=True,**kw)
    assert strict==[[0],[1],[2],[3],[4],[5]]
    assert relaxed==[[0],[1],[2,3],[4],[5]]
    assert audit['entry_ancestor_relaxations']==1
    assert_acyclic(relaxed,preds)


def test_transitive_entry_never_adds_new_effective_ancestors():
    from bounded_coarsening import coarsen
    rng=random.Random(519)
    for _ in range(20):
        preds={i:{j for j in range(i) if rng.random()<.08} for i in range(50)}
        blocks,_=coarsen(list(preds),preds,{(p,i):rng.randint(1,9) for i,ps in preds.items() for p in ps},max_ops=8,work={i:1 for i in preds},pack_frontiers=True,preserve_entries=True,entry_ancestors=True)
        owner={o:g for g,b in enumerate(blocks) for o in b};ancestors={}
        for g,b in enumerate(blocks):
            initial={owner[p] for p in preds[b[0]] if owner[p]!=g}
            assert all(p<g for p in initial)
            allowed=set(initial)
            for p in initial:allowed.update(ancestors[p])
            actual={owner[p] for o in b for p in preds[o] if owner[p]!=g}
            assert actual<=allowed
            ancestors[g]=allowed
        assert_acyclic(blocks,preds)
