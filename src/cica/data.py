"""Safe packed NPZ datasets with explicit sequence boundaries and split identity."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from .config import MODALITIES

COUNTS = {"MOSI": (1284, 229, 686), "MOSEI": (16326, 1871, 4659)}
SPLITS = ("train", "validation", "test")


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


class PackedDataset(Dataset):
    def __init__(self, path, text_mode, normalization=None):
        self.path = Path(path)
        self.text_mode = text_mode
        with np.load(self.path, allow_pickle=False) as z:
            self.arrays = {k: z[k] for k in z.files}
        a = self.arrays
        for required in ['ids', 'y', 'L_offsets', 'V_offsets', 'A_offsets']:
            if required not in a:
                raise ValueError(f'{self.path}: missing {required}')
        if a['ids'].ndim != 1 or a['ids'].dtype.kind not in {'U', 'S'}:
            raise ValueError('ids must be a one-dimensional string array')
        self.ids = [x.decode('utf-8') if isinstance(x, bytes) else str(x) for x in a['ids']]
        if not self.ids or len(set(self.ids)) != len(self.ids) or any(not x.strip() for x in self.ids):
            raise ValueError('Every utterance must have a nonempty unique identifier')
        self.y = np.asarray(a['y'], dtype=np.float32)
        if self.y.shape != (len(self.ids),) or not np.isfinite(self.y).all() or (np.abs(self.y) > 3).any():
            raise ValueError('y must be finite [N] sentiment values in [-3,3]')
        self.input_dims = []
        for m in MODALITIES:
            key = 'input_ids' if m == 'L' and text_mode == 'bert' else m
            if key not in a:
                raise ValueError(f'Missing {key}; no feature dimensions are guessed')
            x, offsets = a[key], a[m + '_offsets']
            if offsets.dtype.kind not in {'i', 'u'} or offsets.shape != (len(self.ids) + 1,):
                raise ValueError('Offsets must be integer [N+1] arrays')
            if offsets[0] != 0 or offsets[-1] != len(x) or not (np.diff(offsets) > 0).all():
                raise ValueError('Offsets must cover nonempty sequences exactly')
            if key == 'input_ids':
                if x.ndim != 1 or x.dtype.kind not in {'i', 'u'} or (x < 0).any():
                    raise ValueError('input_ids must be a packed nonnegative integer vector')
                self.input_dims.append(1)  # replaced with actual backbone width inside CICA
            else:
                if x.ndim != 2 or x.shape[1] == 0 or not np.isfinite(x).all():
                    raise ValueError(f'{m} must be a finite [total_frames,feature_dim] matrix')
                self.input_dims.append(x.shape[1])
        self.normalization = normalization

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        streams = []
        for m in MODALITIES:
            token_stream = m == 'L' and self.text_mode == 'bert'
            key = 'input_ids' if token_stream else m
            offsets = self.arrays[m + '_offsets']
            x = self.arrays[key][offsets[index]:offsets[index + 1]]
            if token_stream:
                streams.append(torch.tensor(x.astype(np.int64), dtype=torch.long))
            else:
                x = x.astype(np.float32)
                if self.normalization and m in self.normalization:
                    stats = self.normalization[m]
                    x = (x - np.asarray(stats['mean'], dtype=np.float32)) / np.asarray(stats['std'], dtype=np.float32)
                streams.append(torch.from_numpy(np.array(x, copy=True)))
        return {'streams': streams, 'y': float(self.y[index]), 'id': self.ids[index]}


def collate(records):
    streams, masks = [], []
    for m in range(3):
        seqs = [r['streams'][m] for r in records]
        streams.append(pad_sequence(seqs, batch_first=True, padding_value=0))
        lengths = torch.tensor([len(x) for x in seqs])
        masks.append(torch.arange(streams[-1].shape[1])[None] < lengths[:, None])
    return {'streams': streams, 'masks': masks, 'y': torch.tensor([r['y'] for r in records]),
            'ids': [r['id'] for r in records]}


def load_bundle(directory, text_mode, *, expected_dataset=None, normalization=None):
    root = Path(directory)
    metadata = json.loads((root / 'metadata.json').read_text(encoding='utf-8'))
    dataset = metadata.get('dataset')
    if dataset not in {*COUNTS, 'SMOKE'} or expected_dataset and dataset != expected_dataset:
        raise ValueError('Dataset identity does not match the selected configuration')
    if metadata.get('data_status') not in {'MEASURED', 'SMOKE_TEST'}:
        raise ValueError('metadata.json requires an explicit MEASURED or SMOKE_TEST data_status')
    if (dataset == 'SMOKE') != (metadata['data_status'] == 'SMOKE_TEST'):
        raise ValueError('SMOKE_TEST data must use the SMOKE dataset identity')
    if metadata.get('text_mode') != text_mode:
        raise ValueError('Dataset text_mode does not match model text_mode')
    splits = {s: PackedDataset(root / (s + '.npz'), text_mode, normalization) for s in SPLITS}
    dims = splits['train'].input_dims
    if any(d.input_dims != dims for d in splits.values()):
        raise ValueError('Feature dimensions differ between splits')
    for i, s in enumerate(SPLITS):
        if dataset in COUNTS and len(splits[s]) != COUNTS[dataset][i]:
            raise ValueError(f'{dataset}/{s}: expected {COUNTS[dataset][i]} utterances, got {len(splits[s])}')
        for t in SPLITS[i + 1:]:
            if set(splits[s].ids) & set(splits[t].ids):
                raise ValueError('Utterance identifiers overlap between data splits')
    return metadata, splits


def fit_normalization(train: PackedDataset):
    stats = {}
    for m in MODALITIES:
        if m == 'L' and train.text_mode == 'bert':
            continue
        x = train.arrays[m]
        # Float64 moment accumulation, computed from training frames only.
        stats[m] = {'mean': np.mean(x, axis=0, dtype=np.float64).tolist(),
                    'std': np.maximum(np.std(x, axis=0, dtype=np.float64), 1e-8).tolist()}
    return stats
