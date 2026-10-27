# Data preparation

Use the processed utterance partitions associated with your feature release. Preserve identifiers and labels across every variant. The [CMU Multimodal SDK](https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK) provides the benchmark data interfaces and release information. Dataset access and feature redistribution follow the corresponding providers' terms.

## Packed NPZ schema

Each split contains the following arrays, loaded with `allow_pickle=False`:

| Key | Shape / type | Meaning |
|---|---|---|
| `ids` | `[N]`, Unicode strings | Unique utterance identifiers; disjoint across splits |
| `y` | `[N]`, float | Sentiment intensity in `[-3,3]` |
| `L_offsets` | `[N+1]`, integer | Language sequence boundaries |
| `V_offsets`, `A_offsets` | `[N+1]`, integer | Visual/acoustic sequence boundaries |
| `V`, `A` | `[sum(T_m), d_m]`, finite float | Concatenated unpadded frames |
| `input_ids` | `[sum(T_L)]`, integer | BERT token identifiers in `bert` mode |
| `L` | `[sum(T_L), d_L]`, finite float | Contextual language features in `features` mode |

Offsets start at zero, strictly increase and end at the packed sequence length. There are no empty sequences. Padding is added only by the batch collator. The BERT path uses single-utterance, zero token-type identifiers; any converter input containing nonzero segment identifiers is rejected rather than silently discarded.

Example `metadata.json`:

```json
{
  "dataset": "MOSI",
  "data_status": "MEASURED",
  "text_mode": "bert",
  "features": {
    "language": "actual tokenizer/checkpoint identifier and revision",
    "visual": "actual feature set, extractor version and frame rate",
    "acoustic": "actual feature set, extractor version and frame rate"
  },
  "source": "local feature release identifier"
}
```

The text fields above describe the required metadata; replace them with the actual release information. Using cached BERT features selects `features` mode and does not fine-tune BERT. It therefore changes the optimization and trainable-parameter scope relative to end-to-end BERT mode.

## MMSA-style conversion

`tools/prepare_mmsa.py` accepts a local dictionary with `train`, `valid`/`validation` and `test` entries. Each entry requires `id`, `regression_labels`, `vision` and `audio`, plus:

- `text_bert` of shape `[N,3,T]` for `bert` mode: token IDs, attention masks and token types;
- `text` for `features` mode;
- `vision_lengths` and `audio_lengths` for dense padded arrays; `text_lengths` is required for dense precomputed text arrays unless a BERT attention mask is supplied.

Ragged lists of unpadded feature matrices do not need length arrays. A dense zero vector is a possible valid observation, so sequence length is never guessed by detecting zero rows. The converter rejects missing lengths, empty sequences, nonzero BERT token types, nonfinite observations and incompatible split sizes. It records the source-file hash. Pickle conversion requires `--trust-pickle`, because a pickle can execute code when loaded; use only your trusted local feature release.

Normalization is fitted on training frames, independently per feature. Validation and test use those saved statistics. Feature arrays and benchmark labels are not downloaded or fabricated by the training commands.
