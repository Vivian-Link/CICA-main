"""Generate a small labelled software-test fixture; no MOSI/MOSEI observations."""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root = Path(args.output)
    if root.exists():
        raise FileExistsError('Smoke data destination already exists')
    root.mkdir(parents=True)
    rng = np.random.default_rng(1234)
    for split, n in [('train', 32), ('validation', 12), ('test', 12)]:
        y = rng.uniform(-3, 3, n).astype('float32')
        y[0] = 0
        arrays = dict(y=y, ids=np.array([f'SMOKE_{split}_{i}' for i in range(n)]))
        for m, d in [('L', 5), ('V', 4), ('A', 3)]:
            lengths = rng.integers(2, 7, n)
            arrays[m + '_offsets'] = np.r_[0, np.cumsum(lengths)].astype('int64')
            x = rng.normal(size=(sum(lengths), d)).astype('float32')
            x[:, 0] += np.repeat(y, lengths)
            arrays[m] = x
        np.savez_compressed(root / (split + '.npz'), **arrays)
    (root / 'metadata.json').write_text(json.dumps(dict(dataset='SMOKE', text_mode='features',
        data_status='SMOKE_TEST', provenance='Generated software-test fixture; not benchmark data'), indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
