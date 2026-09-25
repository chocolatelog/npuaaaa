"""统一资源状态必须与局部模板缓存协同且保持候选隔离。"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'solver'))


def test_resource_cache_preserves_core_changes_and_result_isolation():
    from scene_a_replay import build_resource_evaluator
    from scene_a_fast import official
    from local_template_cache import TemplateCache
    cache=TemplateCache()
    evaluator=build_resource_evaluator(unified=True,template_cache=cache)
    graph={'ops':[{'id':i,'op':'MATMUL','pipe':'PIPE_M','cycles':10} for i in (0,1)],
           'tensors':[],'edges':[]}
    for orders in ([[0],[1]],[[0,1],[]],[[1],[0]]):
        plan={'node_to_subgraph':{'0':0,'1':1},'core_schedules':orders}
        args=(graph,plan,60,{'L1':524288,'UB':131072},1000,100)
        result=evaluator(*args)
        assert result==official.evaluate_scene_a(*args)
        result['per_core_timeline'].clear()
        assert evaluator(*args)==official.evaluate_scene_a(*args)
    assert cache.stats['hits']>0
