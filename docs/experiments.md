# Experiment execution

## Configuration

`configs/mosi.json` and `configs/mosei.json` contain the model, objective, optimizer, scheduler, checkpoint-selection and random-seed settings. Training reads the seed list from its configuration; `--seeds` selects an alternative list. Resolved settings are saved in the run manifest and checkpoint.

CICA and the uniform control use the same data, settings and seeds. The control retains all trainable modules, vector gating and reconstruction weighting, while assigning a common constant to scalar inference utilities.

## Results

The `results` directory stores the experimental tables in CSV format. `metadata.json` specifies their source-table numbers, metric definitions and units.

| File | Content |
|---|---|
| `benchmark_results.csv` | MOSI and MOSEI benchmark comparisons |
| `matched_control.csv` | CICA and the matched uniform control |
| `component_ablation.csv` | Component ablations |
| `objective_ablation.csv` | Objective-function comparisons |
| `missing_modalities.csv` | Structured and random missingness |

Training exports retain the dataset, model, seed and checkpoint identity. Empty result cells denote statistics absent from the corresponding source table.

## Missing modalities

```bash
python -m cica predict --checkpoint runs/mosi/cica/seed_11/best.pt --data data/mosi --output runs/mosi/drop_text.csv --drop L
python -m cica predict --checkpoint runs/mosi/cica/seed_11/best.pt --data data/mosi --output runs/mosi/text_only.csv --drop V A
python -m cica predict --checkpoint runs/mosi/cica/seed_11/best.pt --data data/mosi --output runs/mosi/random_50.csv --missing-rate 0.5
```

Prediction uses the selected checkpoint without retraining. Cross-modal quantities are recomputed over observed streams, inactive concatenation slots are zero and inference weights are renormalized. A single observed modality has zero relational conflict and unit inference weight. Random masks retain at least one stream. Each mask is keyed by the sampling seed and utterance identifier, independently of batch size, input order and model variant. The sampling seed, missing rate and mask policy are recorded.

## Paired evaluation

```bash
python tools/merge_exports.py --input "runs/mosi/cica/seed_*/validation_predictions.csv" "runs/mosi/cica/seed_*/test_predictions.csv" "runs/mosi/uniform/seed_*/test_predictions.csv" --output runs/mosi/combined.csv
python evaluate_predictions.py --input runs/mosi/combined.csv --output runs/mosi/audit --reference-seed 11
```

The merge tool expands file patterns on Windows and POSIX systems. Select the reference checkpoint using validation data. The evaluator checks utterance and label identity, seed pairing, split separation and exact subgroup-MAE aggregation. Outputs include fixed-reference partitions, threshold sensitivity, reliability diagnostics and seed summaries.

`C_L/C_V/C_A` contain post-sigmoid conflict scores; `raw_C_*` retains pre-sigmoid readouts. `alpha_*` contains saved forward-pass allocations. Separately named `recomputed_alpha` fields are obtained from exported scores.

## Sensitivity analysis

```bash
python tools/run_sweep.py --config configs/mosi.json --data data/mosi --output runs/mosi/sensitivity --factor loss.lambda_rec --device cuda
```

`configs/sensitivity.json` lists the one-factor sweeps and additional training conditions. Each candidate starts an independent run, with normalization and checkpoint selection confined to training and validation data.

## Software fixtures

```bash
python tools/make_smoke_data.py --output data/smoke
python -m cica train --config configs/smoke.json --data data/smoke --output runs/smoke --device cpu
```

The fixture uses the `SMOKE` identity and `SMOKE_TEST` record status. Benchmark datasets retain their own identities and `MEASURED` prediction records. The evaluator keeps these record groups separate.
