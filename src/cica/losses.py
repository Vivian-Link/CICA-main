import torch
from torch.nn import functional as F
from .config import LossConfig, ModelConfig


def reliability_target(private_y, y, kappa):
    return torch.exp(-(private_y - y[:, None]).abs() / kappa).detach()


def objective(out, y, model: ModelConfig, weights: LossConfig):
    """Feature/modality sums follow the equations; only samples are averaged."""
    if 'reconstructed' not in out or not bool(out['observed'].all()):
        raise ValueError("Complete-view output with reconstruction is required")
    y = y.reshape(-1)
    target_class = torch.where(y < 0, 0, torch.where(y == 0, 1, 2)).long()
    ce = weights.lambda_ce
    task = (out['yhat'] - y).abs().mean() + ce * F.cross_entropy(out['logits'], target_class)
    private_ce = F.cross_entropy(
        (out['private_logits'] / model.class_temperature).reshape(-1, 3),
        target_class[:, None].expand(-1, 3).reshape(-1), reduction='none').reshape(-1, 3)
    calibration = ((out['private_y'] - y[:, None]).abs() + ce * private_ce).mean()
    calibration = calibration + ce * F.cross_entropy(out['consensus_logits'] / model.class_temperature, target_class)
    reconstruction = (out['omega'] * (out['reconstructed'] - out['encoded']).abs().sum(-1)).sum(-1).mean()
    shared, private = out['shared'], out['private']
    consistency = sum((shared[:, i] - shared[:, j]).square().sum(-1).mean()
                      for i in range(3) for j in range(i + 1, 3))
    cosine = (shared * private).sum(-1) / (shared.norm(dim=-1) * private.norm(dim=-1) + model.epsilon)
    diversity = cosine.square().sum(-1).mean()
    target = reliability_target(out['private_y'], y, model.kappa)
    reliability = (out['reliability'] - target).square().sum(-1).mean()
    losses = dict(task=task, cal=calibration, rec=reconstruction, con=consistency,
                  div=diversity, rel=reliability)
    losses['total'] = task + sum(getattr(weights, 'lambda_' + name) * losses[name]
                               for name in ['cal', 'rec', 'con', 'div', 'rel'])
    return losses
