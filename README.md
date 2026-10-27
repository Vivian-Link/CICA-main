# CICA

**Conflict-Informed Consensus Arbitration for Reliable Multimodal Sentiment Analysis**

CICA combines cross-modally corroborated evidence with modality-private sentiment cues. A predictive-reliability head evaluates disagreement, class uncertainty and divergence from the consensus distribution. Signed allocation rules then control the influence of private evidence during prediction and its reconstruction priority during training.

The package provides the neural model, the six-term training objective, a parameter-matched uniform control, checkpoint selection, prediction exports, missing-modality evaluation and reliability diagnostics for CMU-MOSI and CMU-MOSEI.

## Installation

Use Python 3.10 or later. Install the PyTorch build appropriate for your hardware, then install the package:

```bash
git clone https://github.com/Vivian-Link/CICA-main.git
cd CICA-main
python -m pip install -e ".[bert]"
python -m unittest discover -s tests -v
```

The BERT integration uses Hugging Face Transformers. The feature-input mode requires only PyTorch and NumPy. Neither installation nor the test suite downloads benchmark datasets. BERT weights are loaded when a BERT-mode training run is started; use a local checkpoint directory for offline operation.

## Data

Prepare one directory per dataset containing `metadata.json`, `train.npz`, `validation.npz` and `test.npz`. Each NPZ stores utterance identifiers, continuous labels and packed modality sequences with explicit offsets. Detailed array shapes, metadata and split rules are documented in [Data preparation](docs/data.md).

For a trusted MMSA-style processed feature file:

```bash
python tools/prepare_mmsa.py --input /path/to/mosi.pkl --output data/mosi --dataset MOSI --text-mode bert --trust-pickle
```

Visual and acoustic feature dimensions are read from the actual arrays. Their extraction versions belong in the metadata. Padding is excluded from attention pooling and training-only normalization. The loader checks finite values, nonempty sequences, identifier uniqueness, disjoint splits and the expected processed split sizes:

| Dataset | Train | Validation | Test |
|---|---:|---:|---:|
| CMU-MOSI | 1,284 | 229 | 686 |
| CMU-MOSEI | 16,326 | 1,871 | 4,659 |

## Training

```bash
python -m cica train --config configs/mosi.json --data data/mosi --output runs/mosi/cica --variant cica --seeds 11 23 37 51 71
python -m cica train --config configs/mosi.json --data data/mosi --output runs/mosi/uniform --variant uniform --seeds 11 23 37 51 71
python -m cica train --config configs/mosei.json --data data/mosei --output runs/mosei/cica --variant cica --seeds 11 23 37 51 71
```

The uniform control preserves the complete parameter structure, reliability-dependent vector gate, objective and reconstruction-weight rule. It replaces only the scalar inference utilities with a common constant, including their use in masked reconstruction passes. Use the same seeds, data and settings for both variants.

Checkpoints minimize validation MAE. Test predictions are exported only after checkpoint selection. Each run stores the resolved model settings, input-data hashes, feature normalization, environment, epoch history, checkpoint, validation/test predictions, per-seed metrics and mean/sample-standard-deviation summaries. Existing run directories are not overwritten.

Configuration overrides use JSON values:

```bash
python -m cica train --config configs/mosi.json --data data/mosi --output runs/mosi/eta0 --set model.eta=0.0
python -m cica train --config configs/mosi.json --data data/mosi --output runs/mosi/no_rec --set loss.lambda_rec=0.0
```

Changing an objective setting starts a new training condition. The supplied seed identifiers are explicit run defaults. Model settings listed in the manuscript and tunable implementation defaults are distinguished in [parameter_sources.json](configs/parameter_sources.json). These include the auxiliary cross-entropy coefficient, numerical epsilon, activation, clipping, early stopping and optimizer details. Save any validated changes in your experiment configuration before running a comparison.

The measured sensitivity-grid settings are listed in `configs/sensitivity.json`; its separate `additional_training_conditions` section defines an eta sweep, without assigning reference scores to it. For example:

```bash
python tools/run_sweep.py --config configs/mosi.json --data data/mosi --output runs/mosi/sensitivity --factor loss.lambda_rec --device cuda
```

## Prediction and missing modalities

```bash
python -m cica predict --checkpoint runs/mosi/cica/seed_11/best.pt --data data/mosi --output runs/mosi/predictions.csv
python -m cica predict --checkpoint runs/mosi/cica/seed_11/best.pt --data data/mosi --output runs/mosi/drop_text.csv --drop L
python -m cica predict --checkpoint runs/mosi/cica/seed_11/best.pt --data data/mosi --output runs/mosi/text_only.csv --drop V A
python -m cica predict --checkpoint runs/mosi/cica/seed_11/best.pt --data data/mosi --output runs/mosi/random_50.csv --missing-rate 0.5
```

Missingness evaluation uses the selected checkpoint without retraining. Cross-modal quantities are recomputed over observed streams, fixed-width inactive slots are zeroed and scalar weights are renormalized. The single-stream case has zero relational conflict and unit inference weight. Random masks retain at least one allowed stream; an empty draw is repaired by retaining one uniformly selected allowed stream. The sampling seed is recorded. An all-missing input is rejected.

## Metrics and reliability diagnostics

MAE and Pearson correlation include neutral references. Acc-2 and weighted binary F1 exclude `y = 0`, with `yhat >= 0` treated as positive. Acc-7 clips to `[-3, 3]` and rounds to the nearest integer using ties to even. Percentages are reported on a 0–100 scale. Auxiliary three-class labels are negative, neutral and positive; they do not replace the benchmark evaluation rules.

Merge complete-input exports, then run the paired evaluator:

```bash
python tools/merge_exports.py --input runs/mosi/cica/seed_*/validation_predictions.csv runs/mosi/cica/seed_*/test_predictions.csv runs/mosi/uniform/seed_*/test_predictions.csv --output runs/mosi/combined.csv
python evaluate_predictions.py --input runs/mosi/combined.csv --output runs/mosi/audit --reference-seed 11
```

The reference checkpoint must be chosen using validation data before test inspection. The evaluator enforces sample identity, seed pairing and the exact count-weighted MAE identity. It provides fixed-reference conflict/reliability partitions, nine threshold combinations, transformed-target MSE, ten-bin target calibration gap, Spearman correlation with private error, a validation-fitted constant reference and seed summaries. Exported `C_L/C_V/C_A` values are post-sigmoid scores; `raw_C_*` fields retain the pre-sigmoid readouts. Saved `alpha_*` values are actual forward-pass allocations. The evaluator's separately named `recomputed_alpha` fields are formula diagnostics.

The merge tool expands file patterns on Windows and POSIX systems.

## Reference results

The values below are the existing manuscript-reported benchmark references. They are archived in [results](results/README.md) with `MANUSCRIPT_REPORTED_REFERENCE` provenance, separately from newly executed training outputs.

| Dataset | Model | MAE | Weighted F1 (%) |
|---|---|---:|---:|
| MOSI | CICA | 0.684 ± 0.006 | 88.17 ± 0.23 |
| MOSI | Matched uniform | 0.706 | 86.74 |
| MOSEI | CICA | 0.487 ± 0.003 | 86.81 ± 0.12 |
| MOSEI | Matched uniform | 0.511 | 85.42 |

Blank dispersion fields in the machine-readable references remain unreported. New benchmark values and checkpoints are produced by training on the prepared features; reference aggregates do not define seed-level predictions.

## Software checks

```bash
python -m unittest discover -s tests -v
python tools/make_smoke_data.py --output data/smoke
python -m cica train --config configs/smoke.json --data data/smoke --output runs/smoke --seeds 11 --device cpu
```

The generated fixture is labelled `SMOKE_TEST` and uses the separate `SMOKE` dataset identity. It checks execution and checkpoint/export behavior; it does not supply benchmark evidence. Unit tests check the signed derivatives, stopped target gradients, reconstruction input isolation, masked slots, padding, stable entropy/JS, loss reductions and parameter matching.

## Layout

```text
src/cica/          model, objective, data validation, training and prediction
configs/           dataset settings and parameter provenance
tools/             feature conversion, export merging and software fixtures
tests/             mathematical and integration checks
results/           manuscript-reported reference records
docs/              data contract and equation-to-code mapping
evaluate_predictions.py  paired and reliability evaluation
```

See [Architecture and numerical conventions](docs/architecture.md) for the equation-to-code mapping and [official PyTorch documentation](https://docs.pytorch.org/docs/stable/index.html) and [BERT documentation](https://huggingface.co/docs/transformers/model_doc/bert) for the underlying APIs.

Maintainer: **Vivian-Link** · **lijiezong@163.com**
