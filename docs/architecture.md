# Architecture and numerical conventions

The modality order is language, visual and acoustic (`L`, `V`, `A`). The hidden state has dimension 128 by default. Model inputs are variable-length encoded feature sequences, or BERT token identifiers for language. The output regression is unconstrained; clipping is applied only by the seven-class evaluation metric.

| Equations | Operation | Code |
|---|---|---|
| 1–4 | Sequence projection, masked attention pooling and layer normalization | `CICA.encode`, `AttentionPool` |
| 5–8 | Feature-wise shared/private partition; tied shared map | `CICA.decouple` |
| 9–15 | Symmetric agreement edges, neighborhood messages, scalar support and consensus gate | `CICA.reason` |
| 16–21 | Private barycenter, cosine disagreement, relational descriptor and raw/bounded conflict | `CICA.reason` |
| 22–29 | Private/consensus heads, normalized entropy and Jensen–Shannon divergence, predictive reliability | `CICA.reason`, `uncertainty_and_divergence` |
| 30–34 | Signed utility, scalar allocation and reliability-conditioned vector correction | `inference_weights`, `CICA.reason` |
| 35–36 | Fused supervision and auxiliary private/consensus supervision | `objective` |
| 37–43 | Leave-one-out recomputation and full-view reconstruction priorities | `CICA.forward`, `reconstruction_weights`, `objective` |
| 44–46 | Conditional signed derivatives | `tests/test_model.py` |
| 47–51 | Shared consistency, squared shared/private cosine, detached reliability target and six-term objective | `objective`, `reliability_target` |
| 52–53 | Exhaustive MAE identity and target calibration gap | `evaluate_predictions.py` |

## Trainable maps

The affine gates and readouts use `nn.Linear`. Maps denoted by `f` or `phi`, feature projections and reconstruction decoders use two linear layers of hidden width `hidden_dim`, GELU, dropout and output layer normalization. Pooling scores use tanh. Additional layer normalization is applied to the gated shared/private states and consensus state as specified in the equations. The final regression and classification maps are linear.

The BERT path obtains `last_hidden_state` using the supplied attention mask, then projects and pools it. Visual/acoustic streams and optional precomputed BERT features use modality-specific feed-forward sequence projections. No visual/acoustic input dimension or extractor version is inferred from a feature-family name. The agreement graph is the consensus operator specified by Equations 9–15.

Only observed sample–modality pairs enter their encoders. Padding is replaced before projection or token lookup and excluded from pooling. An unavailable stream may contain no time steps; its encoded state and fusion slots remain zero. Entropy and Jensen–Shannon terms implement the zero-probability convention before arithmetic, including classes absent from both distributions, without adding probability mass.

## Reductions and gradients

Each equation is evaluated per utterance, then averaged over the batch. L1 reconstruction is a **sum over feature coordinates**, weighted and summed over the three targets. Shared consistency sums squared feature distances over the three unordered pairs. Diversity sums squared scalar cosines over modalities. Reliability supervision sums squared score errors over modalities. Private-head calibration averages over modalities. Changing these sums to coordinate averages changes the objective scale.

Cross-entropy is evaluated from logits with stable log-softmax semantics. Private/consensus class heads use `T_q`; the fused classifier has the output definition given by Equation 34. PyTorch class indices 0/1/2 correspond to the mathematical labels 1/2/3. Cosine denominators use `norm(a) * norm(b) + epsilon`, preserving the displayed equation rather than substituting independent norm clipping.

`exp(-abs(private_y-y)/kappa)` is detached as a complete target expression. Gradients through the learned reliability score remain active. Reconstruction targets `H_m` are not detached: the equation contains no stop-gradient on that target. Full-view reliability/conflict and the reconstruction weights are also differentiable.

Each reconstruction decoder receives a fresh cross-modal pass with its target excluded. Encoded and intrinsic shared/private states can be cached because they contain no cross-modal operations; the private center, messages, consensus, relational descriptors, conflicts, class divergence, reliability, gate and scalar weights are recomputed. Excluded slots are zero after biased maps. Gradient tests verify zero dependence of each decoder input on the excluded state.

## Reproducible execution

Training uses deterministic PyTorch algorithms, fixed explicit seeds, separate optimizer rates for BERT and new modules, gradient clipping and validation-MAE scheduling. The checkpoint with the lowest observed validation MAE is saved. `min_delta` controls early-stopping patience, not which checkpoint is retained. The evaluation set does not enter early stopping or normalization.

Each saved checkpoint contains model settings, normalization, seed, selected epoch, validation metrics, optimizer/scheduler states and hashes of its data files. The prediction command rejects a changed dataset. Checkpoints use tensor/primitive dictionaries and `torch.load(..., weights_only=True)`. Training intentionally starts a new run; automatic resumption is not exposed, so an incomplete optimizer/RNG restoration cannot silently change an experiment.
