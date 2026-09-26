"""新候选评分适配必须保留原官方资源与搬运语义。"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'solver'))
from scene_a_candidate_score import SceneACandidateScorer


def test_score_reads_parameters_and_matches_original_with_shared_reads(tmp_path):
    config=tmp_path/'config.txt'
    config.write_text('[capacity]\nL1 524288\nUB 131072\n[bandwidth]\nbandwidth 60\n[multicore_scene_a]\ntask_cross_core_wait_cycles 17\ntask_same_core_wait_cycles 3\n',encoding='utf8')
    graph={'ops':[{'id':1,'op':'COPY_IN','pipe':'PIPE_MTE2','cycles':0},
                  {'id':3,'op':'ADD','pipe':'PIPE_V','cycles':9},
                  {'id':4,'op':'ADD','pipe':'PIPE_V','cycles':8}],
           'tensors':[{'id':2,'size':600,'pos':'UB'}],
           'edges':[{'source':1,'target':2},{'source':2,'target':3},{'source':2,'target':4}]}
    plan={'node_to_subgraph':{'3':0,'4':1},'core_schedules':[[0],[1]]}
    from scene_a_fast import official
    from official_protocol import compact_result
    expected=official.evaluate_scene_a(graph,plan,60,{'L1':524288,'UB':131072},17,3)
    scorer=SceneACandidateScorer(graph,config)
    actual=scorer.evaluate(plan)
    assert actual['real']==compact_result(expected)
    assert actual['shared_ddr_peak']>=2
    assert actual['scope']=='candidate_replay_requires_official_confirmation'
    assert scorer.evaluate(plan)['real']==actual['real']
