"""CICA encoders, evidence graph, reliability head and signed arbitration."""
from __future__ import annotations

import math
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from .config import ModelConfig


def feedforward(n_in: int, hidden: int, n_out: int, dropout: float) -> nn.Module:
    return nn.Sequential(nn.Linear(n_in, hidden), nn.GELU(), nn.Dropout(dropout),
                         nn.Linear(hidden, n_out), nn.LayerNorm(n_out))


def masked_softmax(scores: Tensor, observed: Tensor, dim: int = -1) -> Tensor:
    if not bool(observed.any(dim=dim).all()):
        raise ValueError("Every sample must retain at least one observed modality")
    return scores.masked_fill(~observed, -torch.inf).softmax(dim=dim)


def inference_weights(reliability: Tensor, conflict: Tensor, observed: Tensor,
                      temperature: float, tau: float, uniform: bool = False) -> Tensor:
    scores = (reliability - tau * (1 - reliability) * conflict) / temperature
    if uniform:
        scores = torch.zeros_like(scores)
    return masked_softmax(scores, observed)


def reconstruction_weights(reliability: Tensor, conflict: Tensor,
                           temperature: float, eta: float, floor: float) -> Tensor:
    n = reliability.shape[-1]
    if not 0 < floor < 1 / n:
        raise ValueError("Reconstruction floor must lie between zero and 1/M")
    mass = (reliability * (1 + eta * conflict) / temperature).softmax(-1)
    return floor + (1 - n * floor) * mass


def uncertainty_and_divergence(private_logp: Tensor, consensus_logp: Tensor):
    """Exact normalised entropy and JS; no additive epsilon inside log probabilities."""
    p = private_logp.exp()
    qlog = consensus_logp[:, None, :].expand_as(private_logp)
    q = qlog.exp()
    # Implement 0 log 0 = 0, including categories absent from both distributions.
    # Mask before arithmetic: multiplying an infinite log by zero produces NaN.
    present = (p > 0) | (q > 0)
    mixlog = torch.logaddexp(torch.where(present, private_logp, 0.),
                            torch.where(present, qlog, 0.)) - math.log(2)
    plog = torch.where(p > 0, private_logp, 0.)
    qlog = torch.where(q > 0, qlog, 0.)
    entropy = -(p * plog).sum(-1) / math.log(p.shape[-1])
    divergence = ((p * (plog - mixlog)).sum(-1)
                  + (q * (qlog - mixlog)).sum(-1)) / (2 * math.log(2))
    return entropy, divergence


class AttentionPool(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.map = nn.Linear(hidden, hidden)
        self.score = nn.Linear(hidden, 1, bias=False)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, sequence: Tensor, mask: Tensor) -> Tensor:
        if sequence.ndim != 3 or mask.shape != sequence.shape[:2]:
            raise ValueError("Expected [batch,time,features] and [batch,time] mask")
        mask = mask.bool()
        if sequence.shape[1] == 0:
            return sequence.new_zeros((len(sequence), sequence.shape[-1]))
        sequence = torch.where(mask[..., None], sequence, 0.)
        valid = mask.any(-1)
        safe = mask.clone()
        safe[~valid, 0] = True
        score = self.score(torch.tanh(self.map(sequence))).squeeze(-1)
        attention = masked_softmax(score, safe)
        state = self.norm((attention[..., None] * sequence).sum(1))
        return torch.where(valid[:, None], state, 0.)


class CICA(nn.Module):
    def __init__(self, config: ModelConfig, *, load_pretrained: bool = True):
        super().__init__()
        self.config = config
        d, p = config.hidden_dim, config.dropout
        input_dims = list(config.input_dims)
        self.text_backbone = None
        if config.text_mode == "bert":
            from transformers import AutoConfig, AutoModel
            if load_pretrained:
                self.text_backbone = AutoModel.from_pretrained(
                    config.bert_name, revision=config.bert_revision)
            else:
                if config.bert_config is None:
                    raise ValueError("Checkpoint must contain the BERT architecture configuration")
                spec = dict(config.bert_config)
                kind = spec.pop('model_type')
                self.text_backbone = AutoModel.from_config(AutoConfig.for_model(kind, **spec))
            config.bert_config = self.text_backbone.config.to_dict()
            resolved_revision = getattr(self.text_backbone.config, '_commit_hash', None)
            if resolved_revision:
                config.bert_revision = resolved_revision
            input_dims[0] = self.text_backbone.config.hidden_size
        self.projections = nn.ModuleList([feedforward(v, d, d, p) for v in input_dims])
        self.pooling = nn.ModuleList([AttentionPool(d) for _ in range(3)])
        self.partition = nn.ModuleList([nn.Linear(d, d) for _ in range(3)])
        self.shared_map = feedforward(d, d, d, p)  # tied across L/V/A
        self.private_maps = nn.ModuleList([feedforward(d, d, d, p) for _ in range(3)])
        self.shared_norms = nn.ModuleList([nn.LayerNorm(d) for _ in range(3)])
        self.private_norms = nn.ModuleList([nn.LayerNorm(d) for _ in range(3)])
        self.agreement = nn.Linear(3 * d, d)
        self.beta_map = nn.Linear(2 * d, d)
        self.beta_score = nn.Linear(d, 1, bias=False)
        self.consensus_gate = nn.Linear(2 * d, d)
        self.consensus_map = feedforward(d, d, d, p)
        self.consensus_norm = nn.LayerNorm(d)
        self.inconsistency = feedforward(4 * d + 1, d, d, p)
        self.conflict_head = nn.Linear(d, 1)
        self.private_regressors = nn.ModuleList([nn.Linear(d, 1) for _ in range(3)])
        self.private_classifiers = nn.ModuleList([nn.Linear(d, 3) for _ in range(3)])
        self.consensus_classifier = nn.Linear(d, 3)
        self.reliability_map = feedforward(d + 5, d, d, p)
        self.reliability_head = nn.Linear(d, 1)
        self.fusion_gate = nn.Linear(7 * d, d)  # F_I, three rP slots, three z slots
        self.regressor = nn.Linear(d, 1)
        self.classifier = nn.Linear(d, 3)
        self.decoders = nn.ModuleList([feedforward(d, d, d, p) for _ in range(3)])

    def parameter_inventory(self):
        """Count unique registered tensors, including training-only auxiliary heads."""
        groups = {}
        for name, parameter in self.named_parameters():
            group = groups.setdefault(name.split('.', 1)[0], dict(total=0, trainable=0))
            group['total'] += parameter.numel()
            if parameter.requires_grad:
                group['trainable'] += parameter.numel()
        total = sum(g['total'] for g in groups.values())
        backbone = groups.get('text_backbone', {}).get('total', 0)
        return dict(total_parameters=total,
                    trainable_parameters=sum(g['trainable'] for g in groups.values()),
                    backbone_parameters=backbone, new_module_parameters=total - backbone,
                    projection_input_dims=[p[0].in_features for p in self.projections],
                    modules=groups)

    def encode(self, streams: list[Tensor], masks: list[Tensor],
               observed: Tensor | None = None) -> Tensor:
        states = []
        for m, (x, mask) in enumerate(zip(streams, masks)):
            active = mask.bool().any(-1) if observed is None else observed[:, m]
            indices = active.nonzero(as_tuple=True)[0]
            state = self.projections[m][0].weight.new_zeros((len(x), self.config.hidden_dim))
            if not len(indices):
                states.append(state)
                continue
            x, mask = x.index_select(0, indices), mask.index_select(0, indices).bool()
            if m == 0 and self.text_backbone is not None:
                pad = self.text_backbone.config.pad_token_id
                x = torch.where(mask, x, 0 if pad is None else pad)
                x = self.text_backbone(input_ids=x.long(), attention_mask=mask.long()).last_hidden_state
            else:
                x = torch.where(mask[..., None], x, 0.)
            u = self.projections[m](x)
            states.append(state.index_copy(0, indices, self.pooling[m](u, mask)))
        return torch.stack(states, dim=1)

    def decouple(self, encoded: Tensor):
        shared, private = [], []
        for m in range(3):
            h = encoded[:, m]
            gate = self.partition[m](h).sigmoid()
            shared.append(self.shared_norms[m](gate * self.shared_map(h)))
            private.append(self.private_norms[m]((1 - gate) * self.private_maps[m](h)))
        return torch.stack(shared, 1), torch.stack(private, 1)

    def reason(self, shared: Tensor, private: Tensor, observed: Tensor):
        """All cross-modal quantities are local to this observed set."""
        c = self.config
        b, _, d = shared.shape
        if observed.shape != (b, 3) or not bool(observed.any(-1).all()):
            raise ValueError("observed must be [batch,3] with a nonempty set per sample")
        active = observed[..., None]
        s = torch.where(active, shared, 0.)
        p = torch.where(active, private, 0.)
        count = observed.sum(-1, keepdim=True).to(s.dtype)
        other_count = (count - 1).clamp_min(1)
        pair = observed[:, :, None] & observed[:, None, :]
        pair = pair & ~torch.eye(3, dtype=torch.bool, device=s.device)[None]
        si, sj = s[:, :, None, :], s[:, None, :, :]
        agreement = self.agreement(torch.cat((si + sj, (si - sj).abs(), si * sj), -1)).sigmoid()
        messages = (agreement * sj * pair[..., None]).sum(2) / other_count[..., None]
        xi = self.beta_score(torch.tanh(self.beta_map(torch.cat((s, messages), -1)))).squeeze(-1)
        beta = masked_softmax(xi, observed)
        aggregate = (beta[..., None] * (s + messages)).sum(1)
        mean_shared = s.sum(1) / count
        consensus_gate = self.consensus_gate(torch.cat((aggregate, mean_shared), -1)).sigmoid()
        intuition = self.consensus_norm(self.consensus_map(aggregate) + consensus_gate * mean_shared)
        centre = p.sum(1) / count
        norms = p.norm(dim=-1)
        cosine = torch.einsum('bmd,bnd->bmn', p, p) / (
            norms[:, :, None] * norms[:, None, :] + c.epsilon)
        distance = ((1 - cosine) * pair).sum(-1) / other_count
        centre = centre[:, None, :].expand_as(p)
        z = self.inconsistency(torch.cat((p, centre, (p - centre).abs(), p * centre,
                                          distance[..., None]), -1)) * active
        conflict_raw = self.conflict_head(z).squeeze(-1)
        conflict = conflict_raw.sigmoid()
        conflict = torch.where(count > 1, conflict, torch.zeros_like(conflict)) * observed
        private_y = torch.stack([self.private_regressors[m](p[:, m]).squeeze(-1) for m in range(3)], 1)
        private_logits = torch.stack([self.private_classifiers[m](p[:, m]) for m in range(3)], 1)
        consensus_logits = self.consensus_classifier(intuition)
        private_logp = F.log_softmax(private_logits / c.class_temperature, -1)
        consensus_logp = F.log_softmax(consensus_logits / c.class_temperature, -1)
        uncertainty, divergence = uncertainty_and_divergence(private_logp, consensus_logp)
        rho = torch.stack((conflict, uncertainty, divergence,
                           conflict * (1 - uncertainty), conflict * (1 - divergence)), -1)
        reliability = self.reliability_head(self.reliability_map(torch.cat((z, rho), -1))).squeeze(-1).sigmoid() * observed
        alpha = inference_weights(reliability, conflict, observed, c.temperature, c.tau,
                                  c.variant == "uniform")
        # Mask slots AFTER biased maps so an absent stream cannot re-enter the gate.
        r_stack = (reliability[..., None] * p * active).reshape(b, 3 * d)
        z_stack = (z * active).reshape(b, 3 * d)
        gate = self.fusion_gate(torch.cat((intuition, r_stack, z_stack), -1)).sigmoid()
        reasoning = (alpha[..., None] * p).sum(1)
        fused = intuition + gate * reasoning
        return dict(fused=fused, intuition=intuition, reasoning=reasoning, gate=gate,
                    alpha=alpha, beta=beta, conflict=conflict, conflict_raw=conflict_raw * observed,
                    reliability=reliability, uncertainty=uncertainty * observed,
                    divergence=divergence * observed, z=z, r_stack=r_stack, z_stack=z_stack,
                    private_y=private_y, private_logits=private_logits,
                    consensus_logits=consensus_logits, observed=observed)

    def forward(self, streams: list[Tensor], masks: list[Tensor],
                observed: Tensor | None = None, *, reconstruct: bool = False):
        if len(streams) != 3 or len(masks) != 3:
            raise ValueError("Provide L, V, A streams and padding masks in that order")
        if observed is None:
            observed = torch.stack([m.bool().any(-1) for m in masks], -1)
        observed = observed.to(device=streams[0].device, dtype=torch.bool)
        valid_streams = torch.stack([m.bool().any(-1) for m in masks], -1)
        if observed.shape != valid_streams.shape or bool((observed & ~valid_streams).any()):
            raise ValueError("An observed stream must contain at least one valid time step")
        if not bool(observed.any(-1).all()):
            raise ValueError("An all-missing input is undefined")
        encoded = self.encode(streams, masks, observed)
        shared, private = self.decouple(encoded)
        out = self.reason(shared, private, observed)
        out.update(encoded=encoded, shared=shared, private=private,
                   yhat=self.regressor(out['fused']).squeeze(-1),
                   logits=self.classifier(out['fused']))
        if reconstruct:
            if not bool(observed.all()):
                raise ValueError("The six-term training objective requires complete input")
            recon = []
            for target in range(3):
                retained = observed.clone()
                retained[:, target] = False
                masked = self.reason(shared, private, retained)
                recon.append(self.decoders[target](masked['fused']))
            out['reconstructed'] = torch.stack(recon, 1)
        out['omega'] = reconstruction_weights(out['reliability'], out['conflict'],
            self.config.rec_temperature, self.config.eta, self.config.omega_min) if bool(observed.all()) else None
        return out
