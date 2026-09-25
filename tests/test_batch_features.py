"""候选批量特征必须保持逐张量去重和矩阵/向量工作量守恒。"""
import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'solver'))


class BatchFeatureTests(unittest.TestCase):
    def model(self):
        return SimpleNamespace(blocks=[[0],[1],[2]], block_work_m=[10,20,30],
            block_work_v=[5,0,40], tg=[([0],[1,2],False,'UB',60),
                                      ([],[0,1],False,'L1',120),
                                      ([2],[],True,'UB',30)])

    def test_scalar_counts_shared_tensor_once_per_partition(self):
        import batch_features as f
        row = f.scalar_features(self.model(), [0,1,1], [0,1], 2)
        self.assertEqual(row['boundary_bytes'], 390)
        self.assertEqual(row['compute_load'], 50)

    @unittest.skipUnless(importlib.util.find_spec('torch'), '当前解释器没有张量库，另在现有显卡环境验证')
    def test_device_batch_matches_scalar_with_small_chunks(self):
        import torch
        import batch_features as f
        assignments = [[0,1,1],[0,0,0],[0,1,2]]
        owners = [[0,1],[0],[0,1,0]]
        expected = [f.scalar_features(self.model(),a,c,2) for a,c in zip(assignments,owners)]
        for device in ['cpu']+(['cuda'] if torch.cuda.is_available() else []):
            evaluator=f.BatchFeatures(self.model(),2,device=device,batch_size=2,tensor_chunk=1)
            actual=evaluator.evaluate(assignments,owners)
            self.assertEqual(expected, actual)

    @unittest.skipUnless(importlib.util.find_spec('torch'), '当前解释器没有张量库，另在现有显卡环境验证')
    def test_coarse_score_uses_identical_rounding_for_nondivisible_bytes(self):
        import torch
        import batch_features as f
        model = SimpleNamespace(blocks=[[0]], block_work_m=[14404], block_work_v=[0],
                                tg=[([], [0], False, 'UB', 602858)])
        expected = [f.scalar_features(model, [0], [0], 2)]
        for device in ['cpu'] + (['cuda'] if torch.cuda.is_available() else []):
            actual = f.BatchFeatures(model, 2, device=device).evaluate([[0]], [[0]])
            self.assertEqual(expected, actual)


if __name__=='__main__':
    unittest.main()
