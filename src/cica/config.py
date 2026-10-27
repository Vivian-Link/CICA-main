from __future__ import annotations

from dataclasses import dataclass
import math

MODALITIES = ("L", "V", "A")


@dataclass
class ModelConfig:
    input_dims: tuple[int, int, int]
    text_mode: str = "features"
    bert_name: str = "google-bert/bert-base-uncased"
    bert_revision: str = "main"
    bert_config: dict | None = None
    hidden_dim: int = 128
    dropout: float = 0.3
    temperature: float = 0.5
    tau: float = 0.5
    rec_temperature: float = 1.0
    eta: float = 1.0
    kappa: float = 1.0
    class_temperature: float = 1.0
    omega_min: float = 0.05
    epsilon: float = 1e-8
    variant: str = "cica"

    def __post_init__(self):
        self.input_dims = tuple(self.input_dims)
        if len(self.input_dims) != 3 or any(d <= 0 for d in self.input_dims):
            raise ValueError("input_dims must contain three positive feature dimensions")
        if self.hidden_dim < 2 or self.text_mode not in {"bert", "features"}:
            raise ValueError("Invalid hidden dimension or text mode")
        if self.variant not in {"cica", "uniform"}:
            raise ValueError("variant must be cica or uniform")
        vals = [self.temperature, self.rec_temperature, self.kappa,
                self.class_temperature, self.epsilon, self.tau, self.eta,
                self.omega_min, self.dropout]
        if not all(math.isfinite(v) for v in vals):
            raise ValueError("Model settings must be finite")
        if min(vals[:5]) <= 0 or min(self.tau, self.eta) < 0:
            raise ValueError("Temperatures/scales must be positive; penalties nonnegative")
        if not 0 < self.omega_min < 1 / 3 or not 0 <= self.dropout < 1:
            raise ValueError("Invalid reconstruction floor or dropout")


@dataclass
class LossConfig:
    lambda_ce: float = 0.1
    lambda_cal: float = 0.1
    lambda_rec: float = 0.2
    lambda_con: float = 0.05
    lambda_div: float = 0.01
    lambda_rel: float = 0.1

    def __post_init__(self):
        if any(not math.isfinite(v) or v < 0 for v in vars(self).values()):
            raise ValueError("Loss coefficients must be finite and nonnegative")
