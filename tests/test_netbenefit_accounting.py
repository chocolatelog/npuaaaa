"""合并后工作量必须守恒；旧堆条目不能绕过当前粒度约束。"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'solver'))
from construct_v2 import netbenefit_construct
from model import Model


def graph_for_merge(star=False):
    ops = [{'id': i, 'op': 'MATMUL', 'pipe': 'PIPE_M', 'cycles': work}
           for i, work in enumerate([10, 10, 10] if star else [50, 50, 10000])]
    tensors = [{'id': 10, 'pos': 'UB', 'size': 60}]
    edges = [{'source': 0, 'target': 10}, {'source': 10, 'target': 1}]
    if star:
        tensors.append({'id': 11, 'pos': 'UB', 'size': 60})
        edges += [{'source': 0, 'target': 11}, {'source': 11, 'target': 2}]
    return {'ops': ops, 'tensors': tensors, 'edges': edges}


def test_corrected_cluster_work_conserves_all_member_work():
    model = Model(graph_for_merge(), block_ops_cap=1)
    captured = {}
    def profile(frame, event, result):
        if frame.f_code is netbenefit_construct.__code__ and event == 'return':
            captured.update(frame.f_locals)
    previous = sys.getprofile()
    sys.setprofile(profile)
    try:
        netbenefit_construct(model, 2, 'A', max_sg_ops=2, corrected=True)
    finally:
        sys.setprofile(previous)
    assert len(captured['clusters']) == 2
    assert sum(captured['cl_work']) == sum(model.block_work_m) + sum(model.block_work_v)
    for ci, cid in enumerate(captured['clusters']):
        members = captured['cl_members'][cid]
        assert captured['cl_m'][ci] == sum(model.block_work_m[b] for b in members)
        assert captured['cl_v'][ci] == sum(model.block_work_v[b] for b in members)


def test_corrected_stale_heap_cannot_merge_three_ops_under_cap_two():
    model = Model(graph_for_merge(star=True), block_ops_cap=1)
    groups, cores = netbenefit_construct(model, 2, 'A', max_sg_ops=2, corrected=True)
    sizes = [sum(len(model.blocks[b]) for b, owner in enumerate(groups) if owner == s)
             for s in set(groups)]
    assert sorted(sizes) == [1, 2]


def test_corrected_output_preserves_the_actual_merged_cluster():
    # 第一、二块合并后存活簇编号为 0、2；不能再拿 0、1 下标写回映射。
    graph = graph_for_merge()
    graph['ops'][2]['cycles'] = 10  # 合并簇先分核，使默认编号掩盖不了遗漏成员。
    model = Model(graph, block_ops_cap=1)
    groups, cores = netbenefit_construct(model, 2, 'A', max_sg_ops=2, corrected=True)
    assert groups[0] == groups[1], '已经合并的成员在输出时被错误拆开'
    assert groups[2] != groups[0], '独立重任务被默认编号错误吸入旧簇'
    assert set(groups) == set(range(len(cores))), '不得留下没有成员的调度任务'
