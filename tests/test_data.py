import tempfile
import json
from pathlib import Path
import unittest
import numpy as np
import torch
from cica.data import PackedDataset, collate, load_bundle, fit_normalization


def fixture(path, split, shift=0, n=3):
    data = dict(ids=np.array([f'{split}_{i}' for i in range(n)]), y=np.array([-1., 0., 1.]))
    for m in ['L', 'V', 'A']:
        data[m] = np.arange(12, dtype=np.float32).reshape(6, 2) + shift
        data[m + '_offsets'] = np.array([0, 1, 3, 6])
    np.savez(path, **data)


class DataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        for i, s in enumerate(['train', 'validation', 'test']):
            fixture(self.root / (s + '.npz'), s, 100 * i)
        (self.root / 'metadata.json').write_text(json.dumps(dict(dataset='SMOKE', data_status='SMOKE_TEST', text_mode='features')))

    def tearDown(self):
        self.temp.cleanup()

    def test_packing_collation_and_train_only_normalization(self):
        _, ds = load_bundle(self.root, 'features')
        stats = fit_normalization(ds['train'])
        np.testing.assert_allclose(stats['L']['mean'], [5, 6])
        ds['validation'].normalization = stats
        batch = collate([ds['validation'][0], ds['validation'][1]])
        self.assertEqual(batch['streams'][0].shape, (2, 2, 2))
        torch.testing.assert_close(batch['masks'][0], torch.tensor([[True, False], [True, True]]))
        self.assertGreater(float(batch['streams'][0][0, 0, 0]), 20.)

    def test_split_overlap_is_rejected(self):
        fixture(self.root / 'test.npz', 'train')
        with self.assertRaises(ValueError):
            load_bundle(self.root, 'features')

    def test_benchmark_size_is_enforced(self):
        (self.root / 'metadata.json').write_text(json.dumps(dict(dataset='MOSI', data_status='MEASURED', text_mode='features')))
        with self.assertRaises(ValueError):
            load_bundle(self.root, 'features')

    def test_invalid_offsets_are_rejected(self):
        with np.load(self.root / 'train.npz') as f:
            arrays = dict(f)
        arrays['L_offsets'] = np.array([0, 0, 3, 6])
        np.savez(self.root / 'broken.npz', **arrays)
        with self.assertRaises(ValueError):
            PackedDataset(self.root / 'broken.npz', 'features')

    def test_nonfinite_features_are_rejected(self):
        with np.load(self.root / 'train.npz') as f:
            arrays = dict(f)
        arrays['V'][0, 0] = np.nan
        np.savez(self.root / 'broken.npz', **arrays)
        with self.assertRaises(ValueError):
            PackedDataset(self.root / 'broken.npz', 'features')


if __name__ == '__main__':
    unittest.main()
