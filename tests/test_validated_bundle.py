from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'solver'))
from validated_bundle import freeze_files, verify_files


def test_bundle_resumes_identical_copy_and_detects_corruption(tmp_path):
    source=tmp_path/'source.txt';source.write_text('原始方案',encoding='utf8')
    output=tmp_path/'bundle'
    records=freeze_files([(source,'plans/a.json')],output)
    assert records==freeze_files([(source,'plans/a.json')],output)
    assert verify_files(output,records)==1
    (output/'plans/a.json').write_text('改变方案',encoding='utf8')
    with pytest.raises(ValueError,match='内容'):
        verify_files(output,records)
    with pytest.raises(ValueError,match='覆盖'):
        freeze_files([(source,'plans/a.json')],output)


def test_bundle_rejects_traversal_before_copy(tmp_path):
    source=tmp_path/'source.txt';source.write_text('内容',encoding='utf8')
    with pytest.raises(ValueError):
        freeze_files([(source,'../escape.txt')],tmp_path/'out')
    assert not (tmp_path/'escape.txt').exists()


def test_bundle_rejects_conflicting_destinations(tmp_path):
    a=tmp_path/'a';b=tmp_path/'b';a.write_text('a');b.write_text('b')
    with pytest.raises(ValueError,match='重复'):
        freeze_files([(a,'same'),(b,'same')],tmp_path/'out')
