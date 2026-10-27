from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import random
import sys
from dataclasses import asdict
from pathlib import Path
import importlib.metadata
import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from .config import ModelConfig, LossConfig, MODALITIES
from .data import collate, load_bundle, fit_normalization, file_sha256
from .model import CICA
from .losses import objective
from .metrics import metrics


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def seed_everything(seed):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def to_device(batch, device):
    return ([x.to(device) for x in batch['streams']],
            [m.to(device) for m in batch['masks']], batch['y'].to(device))


def loader(dataset, batch_size, *, seed, shuffle=False):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0,
                      collate_fn=collate, generator=torch.Generator().manual_seed(seed))


def observation_masks(ids, seed, *, drop=(), missing_rate=0.0):
    """Sample each utterance independently of batching, order and model variant."""
    if not 0 <= missing_rate <= 1:
        raise ValueError('missing_rate must be in [0,1]')
    drop_ids = [MODALITIES.index(m) for m in drop]
    if len(set(drop_ids)) == 3:
        raise ValueError('Cannot drop all three modalities')
    observed = np.ones((len(ids), 3), dtype=bool)
    observed[:, drop_ids] = False
    allowed = [m for m in range(3) if m not in drop_ids]
    if missing_rate:
        for i, uid in enumerate(ids):
            identity = json.dumps([int(seed), str(uid)], ensure_ascii=False, separators=(',', ':'))
            key = int.from_bytes(hashlib.sha256(identity.encode('utf-8')).digest(), 'little')
            rng = np.random.default_rng(key)
            observed[i] &= rng.random(3) >= missing_rate
            if not observed[i].any():
                observed[i, rng.choice(allowed)] = True
    return observed


def evaluate(model, batches, device, *, metadata, seed, split, drop=(), missing_rate=0.0):
    # Validate settings even if an empty iterator is supplied.
    observation_masks([], seed, drop=drop, missing_rate=missing_rate)
    model.eval()
    records = []
    with torch.no_grad():
        for batch in batches:
            streams, masks, y = to_device(batch, device)
            observed = observation_masks(batch['ids'], seed, drop=drop, missing_rate=missing_rate)
            out = model(streams, masks, torch.from_numpy(observed).to(device))
            values = {k: out[k].detach().cpu().numpy() for k in
                      ['yhat', 'private_y', 'reliability', 'conflict', 'conflict_raw', 'alpha']}
            omega = None if out['omega'] is None else out['omega'].cpu().numpy()
            for i, uid in enumerate(batch['ids']):
                row = dict(data_status=metadata['data_status'], dataset=metadata['dataset'],
                           split=split, seed=str(seed), model=model.config.variant.upper(),
                           utterance_id=uid, y=float(y[i].cpu()), yhat=float(values['yhat'][i]),
                           missing_rate=missing_rate, mask_policy='sha256-seed-utterance-v1')
                for j, m in enumerate(MODALITIES):
                    row['observed_' + m] = int(observed[i, j])
                    for name, key in [('private', 'private_y'), ('r', 'reliability'),
                                      ('C', 'conflict'), ('raw_C', 'conflict_raw'), ('alpha', 'alpha')]:
                        row[name + '_' + m] = float(values[key][i, j]) if observed[i, j] else ''
                    row['omega_' + m] = float(omega[i, j]) if omega is not None else ''
                records.append(row)
    return metrics([r['y'] for r in records], [r['yhat'] for r in records]), records


def write_records(path, records):
    if not records:
        raise ValueError('No prediction records')
    with Path(path).open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def environment():
    versions = {}
    for package in ['torch', 'numpy', 'transformers']:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return dict(python=sys.version, platform=platform.platform(), packages=versions,
                cuda=torch.version.cuda, cuda_available=torch.cuda.is_available(),
                device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')


def train(config, data, output, seeds, variant, device):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Choose a new output directory; existing runs are never overwritten')
    output.mkdir(parents=True, exist_ok=True)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('Seeds must be nonempty and distinct')
    training = config['training']
    if training['epochs'] < 1 or training['patience'] < 1 or training['batch_size'] < 1:
        raise ValueError('epochs, patience and batch_size must be positive')
    if training['gradient_clip'] <= 0 or training['min_delta'] < 0:
        raise ValueError('Invalid clipping or early-stopping tolerance')
    metadata, datasets = load_bundle(data, config['model']['text_mode'], expected_dataset=config['dataset'])
    normalization = fit_normalization(datasets['train'])
    for ds in datasets.values():
        ds.normalization = normalization
    model_spec = dict(config['model'], input_dims=datasets['train'].input_dims, variant=variant)
    weights = LossConfig(**config['loss'])
    manifest = dict(configuration=config, model_configuration=model_spec, seeds=seeds, variant=variant,
                    dataset_metadata=metadata, data_sha256={s: file_sha256(ds.path) for s, ds in datasets.items()},
                    environment=environment(), selection='lowest validation MAE; test evaluated after checkpoint selection')
    write_json(output / 'manifest.json', manifest)
    summary = []
    for seed in seeds:
        seed_everything(seed)
        run = output / f'seed_{seed}'
        run.mkdir()
        model = CICA(ModelConfig(**model_spec)).to(device)
        write_json(run / 'model_spec.json', dict(configuration=asdict(model.config),
            total_parameters=sum(p.numel() for p in model.parameters()),
            trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad)))
        backbone, new = [], []
        for name, parameter in model.named_parameters():
            (backbone if name.startswith('text_backbone.') else new).append(parameter)
        groups = [{'params': new, 'lr': training['learning_rate']}]
        if backbone:
            groups.append({'params': backbone, 'lr': training['encoder_learning_rate']})
        optimizer = torch.optim.AdamW(groups, betas=tuple(training['adam_betas']),
                                    eps=training['adam_epsilon'], weight_decay=training['weight_decay'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                        factor=training['lr_factor'], patience=training['lr_patience'])
        batches = loader(datasets['train'], training['batch_size'], seed=seed, shuffle=True)
        validation = loader(datasets['validation'], training['batch_size'], seed=seed)
        best, stale, history = float('inf'), 0, []
        for epoch in range(1, training['epochs'] + 1):
            model.train()
            accumulated, total_n = {}, 0
            for batch in batches:
                streams, masks, y = to_device(batch, device)
                optimizer.zero_grad(set_to_none=True)
                losses = objective(model(streams, masks, reconstruct=True), y, model.config, weights)
                if not all(bool(torch.isfinite(v)) for v in losses.values()):
                    raise FloatingPointError('Nonfinite training objective')
                losses['total'].backward()
                clip_grad_norm_(model.parameters(), training['gradient_clip'], error_if_nonfinite=True)
                optimizer.step()
                total_n += len(y)
                for key, value in losses.items():
                    accumulated[key] = accumulated.get(key, 0.0) + float(value.detach()) * len(y)
            val, _ = evaluate(model, validation, device, metadata=metadata, seed=seed, split='validation')
            scheduler.step(val['mae'])
            history.append(dict(epoch=epoch, train={k: v / total_n for k, v in accumulated.items()},
                                validation=val, learning_rates=[g['lr'] for g in optimizer.param_groups]))
            # Checkpoint the true minimum. min_delta governs patience only.
            previous_best = best
            if val['mae'] < best:
                best = val['mae']
                torch.save(dict(model=model.state_dict(), model_config=asdict(model.config),
                                loss_config=asdict(weights), training=training, seed=seed, epoch=epoch,
                                normalization=normalization, metadata=metadata, validation=val,
                                optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                                data_sha256=manifest['data_sha256']), run / 'best.pt')
            stale = 0 if val['mae'] < previous_best - training['min_delta'] else stale + 1
            write_json(run / 'history.json', history)
            print(f"seed={seed} epoch={epoch} validation_mae={val['mae']:.6f}", flush=True)
            if stale >= training['patience']:
                break
        checkpoint = torch.load(run / 'best.pt', map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model'])
        for split in ['validation', 'test']:
            score, records = evaluate(model, loader(datasets[split], training['batch_size'], seed=seed),
                                      device, metadata=metadata, seed=seed, split=split)
            write_records(run / (split + '_predictions.csv'), records)
            write_json(run / (split + '_metrics.json'), dict(score, data_status=metadata['data_status'],
                       dataset=metadata['dataset'], model=variant.upper(), seed=seed,
                       selected_epoch=checkpoint['epoch']))
            if split == 'test':
                summary.append(dict(seed=seed, **score))
    aggregates = {}
    for metric in ['mae', 'corr', 'acc2', 'f1', 'acc7']:
        values = [r[metric] for r in summary if r[metric] is not None]
        aggregates[metric] = dict(mean=float(np.mean(values)) if values else None,
                                 sd=float(np.std(values, ddof=1)) if len(values) > 1 else None,
                                 n_valid=len(values))
    write_json(output / 'summary.json', dict(data_status=metadata['data_status'], dataset=metadata['dataset'],
               model=variant.upper(), per_seed=summary, mean_sd=aggregates))


def predict(checkpoint_path, data, output, split, device, drop=(), missing_rate=0.0):
    # Checkpoints are tensor/primitive dictionaries, loaded with the restricted unpickler.
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    c = ModelConfig(**checkpoint['model_config'])
    metadata, datasets = load_bundle(data, c.text_mode,
        expected_dataset=checkpoint['metadata']['dataset'], normalization=checkpoint['normalization'])
    if any(file_sha256(ds.path) != checkpoint['data_sha256'][s] for s, ds in datasets.items()):
        raise ValueError('Prediction dataset differs from the checkpoint dataset')
    model = CICA(c, load_pretrained=False).to(device)
    model.load_state_dict(checkpoint['model'])
    scores, records = evaluate(model, loader(datasets[split], checkpoint['training']['batch_size'],
        seed=checkpoint['seed']), device, metadata=metadata, seed=checkpoint['seed'], split=split,
        drop=drop, missing_rate=missing_rate)
    output = Path(output)
    if output.exists():
        raise FileExistsError('Prediction destination already exists')
    output.parent.mkdir(parents=True, exist_ok=True)
    write_records(output, records)
    write_json(output.with_suffix('.metrics.json'), dict(scores, data_status=metadata['data_status'],
               drop=list(drop), missing_rate=missing_rate, mask_seed=checkpoint['seed']))
