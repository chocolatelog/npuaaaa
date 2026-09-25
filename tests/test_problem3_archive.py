import copy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'solver'))
import run_oplist_full as full


@pytest.mark.parametrize('cores', [2, 3, 4, 5])
def test_c_rechecks_b_plan_and_rejects_slower_candidate(tmp_path, monkeypatch, cores):
    source_dir, output_dir = tmp_path / 'source', tmp_path / 'output'
    source_dir.mkdir()
    output_dir.mkdir()
    graph = {'ops': [{'id': 1, 'op': 'COMPUTE', 'pipe': 'PIPE_M', 'cycles': 10}],
             'tensors': [], 'edges': []}
    plan = {'node_to_subgraph': {'1': 0},
            'core_schedules': [[0]] + [[] for _ in range(cores - 1)]}
    candidate = copy.deepcopy(plan)
    candidate['core_schedules'][0], candidate['core_schedules'][1] = [], [0]
    input_path = source_dir / f'case_001_B_N{cores}.json'
    original = json.dumps(plan)
    input_path.write_text(original, encoding='utf-8')
    calls = []

    def evaluate(supplied_graph, supplied_plan, scene, timeout):
        assert supplied_graph == graph and scene == 'C'
        calls.append(scene)
        return {'makespan': 100 if supplied_plan == plan else 120,
                'added_copy_bytes': 10, 'cache_hit_rate': 0.25}

    def generate(supplied_graph, supplied_plan, num_cores, max_candidates):
        assert supplied_plan == plan and num_cores == cores
        return [{'plan': candidate, 'source': 'slower'}], {'status': 'ok'}

    monkeypatch.setattr(full, 'load_graph', lambda _: graph)
    monkeypatch.setattr(full, 'evaluate', evaluate)
    monkeypatch.setattr(full, 'generate_op_candidates', generate)
    row = full.run_task(dict(case='case_001', scene='C', N=cores,
        baseline_dirs=[str(source_dir)], output_dir=str(output_dir), singlecore=200,
        evaluation_seconds=1, generation_seconds=1, max_candidates=6))
    assert calls == ['C', 'C']
    assert row['status'] == 'official'
    assert row['baseline_source'] == str(input_path)
    assert row['baseline']['makespan'] == row['real']['makespan'] == 100
    assert row['speedup'] == row['baseline_speedup'] == 2
    assert row['real']['cache_hit_rate'] == 0.25
    assert not row['candidate_trace'][0]['accepted']
    assert input_path.read_text(encoding='utf-8') == original
    assert json.loads((output_dir / f'case_001_C_N{cores}.json').read_text(encoding='utf-8')) == plan
    assert not (output_dir / input_path.name).exists()


def test_c_summary_does_not_relabel_b_history():
    row = dict(case='case_001', scene='C', N=5, large=False, status='official',
        baseline={'makespan': 100, 'added_copy_bytes': 0},
        real={'makespan': 80, 'added_copy_bytes': 0}, baseline_speedup=4, speedup=5)
    reference = {('case_001', 'B', 5): {'real': {'makespan': 120, 'added_copy_bytes': 0}}}
    result = full.summarize([row], [('case_001', 'C', 5)], reference)
    assert result['all_official_complete']
    assert result['all_official']['wins'] == 1
    assert result['all_official']['mean_speedup'] == 5
    assert result['paired_historical']['tasks'] == 0
    assert result['versus_historical']['tasks'] == 0
