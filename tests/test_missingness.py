import unittest
import numpy as np
import torch
from cica.config import ModelConfig
from cica.data import collate
from cica.engine import evaluate, observation_masks
from cica.model import CICA


class MissingnessTests(unittest.TestCase):
    def test_masks_are_batch_and_order_invariant(self):
        ids = [f'utterance_{i}' for i in range(37)]
        for rate in [.3, .7, 1.]:
            whole = observation_masks(ids, 11, drop=('A',), missing_rate=rate)
            batched = np.concatenate([observation_masks(ids[i:i + 4], 11, drop=('A',), missing_rate=rate)
                                      for i in range(0, len(ids), 4)])
            reverse = observation_masks(ids[::-1], 11, drop=('A',), missing_rate=rate)[::-1]
            np.testing.assert_array_equal(whole, batched)
            np.testing.assert_array_equal(whole, reverse)
            self.assertTrue(whole.any(-1).all())
            self.assertFalse(whole[:, 2].any())
        self.assertFalse(np.array_equal(observation_masks(ids, 11, missing_rate=.5),
                                       observation_masks(ids, 23, missing_rate=.5)))

    def test_evaluation_is_invariant_to_batching(self):
        torch.manual_seed(13)
        model = CICA(ModelConfig((5, 4, 3), hidden_dim=12, dropout=0.)).eval()
        records = [dict(id=f'utterance_{i}', y=(-1.) ** i,
                        streams=[torch.randn(2 + i % 3, d) for d in (5, 4, 3)]) for i in range(11)]
        def run(size):
            batches = [collate(records[i:i + size]) for i in range(0, len(records), size)]
            return evaluate(model, batches, 'cpu', metadata=dict(dataset='SMOKE', data_status='SMOKE_TEST'),
                            seed=11, split='test', missing_rate=.7)[1]
        a, b = run(3), run(7)
        for x, y in zip(a, b):
            for m in 'LVA':
                self.assertEqual(x['observed_' + m], y['observed_' + m])
            self.assertAlmostEqual(x['yhat'], y['yhat'], places=5)


if __name__ == '__main__':
    unittest.main()
