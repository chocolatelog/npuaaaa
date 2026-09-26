import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'solver'))


def test_tree_refine_is_deterministic_and_keeps_root():
    import json
    from tree_refine import mcts_candidates

    graph = json.loads((ROOT / '通用神经网络处理器下的多核调度问题附件' /
                        'data' / 'case_001.json').read_text(encoding='utf8'))
    plan = json.loads((ROOT / 'tests' / 'fixtures' /
                       'case_001_B_N2.json').read_text(encoding='utf8'))
    first, audit1 = mcts_candidates(graph, plan, 'B', max_states=8, depth=2, width=3)
    second, audit2 = mcts_candidates(graph, plan, 'B', max_states=8, depth=2, width=3)
    assert first == second
    assert audit1 == audit2
    assert first[0] == plan
    assert audit1['expanded'] <= 8
    assert audit1['unique_states'] == len(first)
