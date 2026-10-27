"""Convert a trusted local MMSA-style feature dictionary to packed NPZ arrays."""
import argparse
import json
import pickle
import shutil
import tempfile
from pathlib import Path
import numpy as np
from cica.data import file_sha256, load_bundle


def packed_features(split, key, lengths_key, n, fallback_lengths=None):
    if key not in split:
        raise ValueError('Missing feature array: ' + key)
    values = split[key]
    lengths = split.get(lengths_key, fallback_lengths)
    ragged = isinstance(values, (list, tuple)) or getattr(values, 'dtype', None) == object
    if len(values) != n:
        raise ValueError('Feature and label counts differ')
    if not ragged and lengths is None:
        raise ValueError(f'Dense padded {key} requires explicit {lengths_key}')
    pieces, offsets = [], [0]
    for i in range(n):
        x = np.asarray(values[i], dtype=np.float32)
        if x.ndim != 2:
            raise ValueError('Feature sequences must be [time,features]')
        length = len(x) if lengths is None else int(lengths[i])
        if length < 1 or length > len(x):
            raise ValueError('Invalid explicit sequence length')
        pieces.append(x[:length]); offsets.append(offsets[-1] + length)
    return np.concatenate(pieces), np.asarray(offsets, dtype=np.int64)


def convert(source, destination, dataset, text_mode):
    # The CLI requires explicit trust before reaching this deserialization operation.
    with Path(source).open('rb') as f:
        payload = pickle.load(f)
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError('Data destination already exists')
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='cica_data_', dir=destination.parent))
    try:
        for name in ['train', 'validation', 'test']:
            s = payload.get(name, payload.get('valid') if name == 'validation' else None)
            if s is None:
                raise ValueError('Missing split: ' + name)
            y = np.asarray(s['regression_labels'], dtype=np.float32).reshape(-1)
            raw_ids = s['id']
            ids = []
            for uid in raw_ids:
                if isinstance(uid, bytes):
                    uid = uid.decode('utf-8')
                elif isinstance(uid, np.ndarray):
                    uid = json.dumps(uid.tolist(), ensure_ascii=False)
                elif isinstance(uid, (list, tuple)):
                    uid = json.dumps(uid, ensure_ascii=False)
                ids.append(str(uid))
            if len(ids) != len(y):
                raise ValueError('Identifier and label counts differ')
            arrays = dict(ids=np.asarray(ids), y=y)
            fallback = None
            if 'text_bert' in s:
                bert = np.asarray(s['text_bert'])
                if bert.ndim != 3 or bert.shape[:2] != (len(y), 3):
                    raise ValueError('text_bert must have shape [N,3,T]')
                attention = bert[:, 1, :]
                if not np.isin(attention, [0, 1]).all():
                    raise ValueError('BERT attention masks must be binary')
                fallback = attention.sum(-1).astype(np.int64)
                expected = np.arange(attention.shape[1])[None] < fallback[:, None]
                if not np.array_equal(attention.astype(bool), expected):
                    raise ValueError('The converter requires right-padded contiguous BERT masks')
            if text_mode == 'bert':
                if fallback is None:
                    raise ValueError('bert mode requires text_bert')
                pieces, offsets = [], [0]
                for i, length in enumerate(fallback):
                    if length < 1 or (bert[i, 2, :length] != 0).any():
                        raise ValueError('Nonempty single-segment BERT input is required')
                    tokens = bert[i, 0, :length]
                    if not np.equal(tokens, np.floor(tokens)).all():
                        raise ValueError('BERT token identifiers must be integers')
                    pieces.append(tokens.astype(np.int64)); offsets.append(offsets[-1] + int(length))
                arrays['input_ids'], arrays['L_offsets'] = np.concatenate(pieces), np.asarray(offsets, dtype=np.int64)
            else:
                arrays['L'], arrays['L_offsets'] = packed_features(s, 'text', 'text_lengths', len(y), fallback)
            for code, key, length_key in [('V', 'vision', 'vision_lengths'), ('A', 'audio', 'audio_lengths')]:
                arrays[code], arrays[code + '_offsets'] = packed_features(s, key, length_key, len(y))
            np.savez_compressed(stage / (name + '.npz'), **arrays)
        (stage / 'metadata.json').write_text(json.dumps(dict(dataset=dataset, text_mode=text_mode,
            data_status='MEASURED', source=Path(source).name, source_sha256=file_sha256(source),
            features=payload.get('feature_metadata', {})), indent=2), encoding='utf-8')
        load_bundle(stage, text_mode, expected_dataset=dataset)
        # Both paths are verified direct children of the selected destination parent.
        assert stage.resolve().parent == destination.resolve().parent
        stage.rename(destination)
    finally:
        if stage.exists():
            assert stage.resolve().parent == destination.resolve().parent
            shutil.rmtree(stage)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--dataset', choices=['MOSI', 'MOSEI'], required=True)
    parser.add_argument('--text-mode', choices=['bert', 'features'], default='bert')
    parser.add_argument('--trust-pickle', action='store_true')
    args = parser.parse_args()
    if not args.trust_pickle:
        parser.error('Converting a pickle requires --trust-pickle for a trusted local input')
    convert(args.input, args.output, args.dataset, args.text_mode)


if __name__ == '__main__':
    main()
