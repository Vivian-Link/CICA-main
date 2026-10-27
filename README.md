# CICA

**Conflict-Informed Consensus Arbitration for Reliable Multimodal Sentiment Analysis**

CICA integrates cross-modal consensus with modality-private sentiment evidence. Predictive reliability and conflict guide evidence allocation during prediction and reconstruction during training. The package includes the model, training and prediction commands, a parameter-matched uniform control, missing-modality evaluation and reliability diagnostics.

## Installation

Use Python 3.10–3.12 and a PyTorch build suitable for your hardware.

```bash
cd CICA-main
python -m pip install -e ".[bert]"
```

The BERT input mode uses Hugging Face Transformers. Precomputed feature inputs use PyTorch and NumPy. For offline training, specify a local BERT checkpoint in the experiment configuration.

## Datasets

| Dataset | Official dataset page | SDK resources |
|---|---|---|
| CMU-MOSI | [CMU-MOSI](https://multicomp.cs.cmu.edu/resources/cmu-mosi-dataset/) | [MOSI dataset definitions](https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK/tree/main/mmsdk/mmdatasdk/dataset/standard_datasets/CMU_MOSI) |
| CMU-MOSEI | [CMU-MOSEI](https://multicomp.cs.cmu.edu/resources/cmu-mosei-dataset/) | [MOSEI dataset definitions](https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK/tree/main/mmsdk/mmdatasdk/dataset/standard_datasets/CMU_MOSEI) |

Features and labels can be obtained through the [CMU Multimodal SDK](https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK). Follow the dataset providers' access and citation instructions.

## Data preparation

Prepare dataset directories containing `metadata.json`, `train.npz`, `validation.npz` and `test.npz`. Array shapes, sequence offsets, identifiers and conversion procedures are documented in [docs/data.md](docs/data.md).

```bash
python tools/prepare_mmsa.py --input /path/to/mosi.pkl --output data/mosi --dataset MOSI --text-mode bert --trust-pickle
```

## Training

```bash
python -m cica train --config configs/mosi.json --data data/mosi --output runs/mosi/cica
python -m cica train --config configs/mosi.json --data data/mosi --output runs/mosi/uniform --variant uniform
python -m cica train --config configs/mosei.json --data data/mosei --output runs/mosei/cica
```

Experiment settings are stored in the configuration files. Each run saves its resolved configuration, data hashes, normalization, history, selected checkpoint, predictions and metric summaries. Checkpoint selection uses validation MAE. Existing run directories are preserved.

## Prediction

```bash
python -m cica predict --checkpoint /path/to/best.pt --data data/mosi --output runs/mosi/predictions.csv
```

The prediction command supports complete inputs and missing modalities. See [docs/experiments.md](docs/experiments.md) for evaluation, paired comparisons and sensitivity runs.

## Tests

```bash
python -m unittest discover -s tests -v
```

Tests cover mathematical identities, gradients, masking, reconstruction inputs, BERT integration, checkpoint behavior and evaluation conventions. Continuous integration runs the suite on every push and pull request.

## Repository layout

```text
src/cica/                 model, losses, data loading, training and prediction
configs/                  experiment configurations
results/                  experimental tables and metric definitions
tools/                    data conversion and experiment utilities
tests/                    model, data and command-line tests
docs/                     architecture, data and experiment documentation
evaluate_predictions.py   paired evaluation and reliability diagnostics
```

The [architecture documentation](docs/architecture.md) maps the formulation to the implementation. Experiment settings and result tables are maintained in their corresponding files.
