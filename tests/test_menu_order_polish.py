"""固定成员菜单的顺序候选：旧切分不变、原方案隔离、固定调用预算。"""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'solver'))


def test_order_variants_keep_members_and_respect_search_budget(tmp_path):
    from menu_order_polish import generate_order_variants,select_order_variants
    from model import Model
    from scene_a_event import derive_multicore_plan
    from evaluation_validation import validate_task_order
    graph={'ops':[{'id':1,'op':'MATMUL','pipe':'PIPE_M','cycles':100},
                  {'id':2,'op':'MATMUL','pipe':'PIPE_M','cycles':80}], 'edges':[],'tensors':[]}
    model=Model(graph,block_ops_cap=1);plan=model.plan_from([0,1],[[0,1],[]])
    path=tmp_path/'parent.json';path.write_text(json.dumps(plan),encoding='utf-8')
    parent={'plan_path':str(path),'resource':{'makespan':280,'added_copy_bytes':0,'scheduled_copy_bytes':0,'partition_added':0,'spill_added':0}}
    rows,audit=generate_order_variants(graph,model,[parent])
    assert rows and audit['search_evaluations']<=400 and audit['local_models']==1
    assert all(r['plan']['node_to_subgraph']==plan['node_to_subgraph'] for r in rows)
    assert any(r['coarse']<280 for r in rows)
    for row in rows:validate_task_order(derive_multicore_plan(graph,row['plan']))
    assert json.loads(path.read_text(encoding='utf-8'))==plan
    selected=select_order_variants(rows,limit=1)
    assert len(selected)==1 and selected[0]['coarse']==min(r['coarse'] for r in rows)


def test_selector_prioritizes_distinct_member_partitions_before_second_queue():
    from menu_order_polish import select_order_variants
    def row(pid,assignment,coarse):
        return {'plan_id':pid,'assignment':assignment,'coarse':coarse,'origins':[{'neighborhood':'migration'}]}
    rows=[row('best',[0,0,1],1),row('same-partition',[0,0,1],2),row('other',[0,1,1],9)]
    assert [r['plan_id'] for r in select_order_variants(rows,limit=2)]==['best','other']
