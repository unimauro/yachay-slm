"""Configuración del modelo y del entrenamiento.

Valores por defecto = un SLM DIMINUTO que entrena en laptop. Subir con calma
cuando el pipeline funcione end-to-end.
"""
from dataclasses import dataclass


@dataclass
class ModelConfig:
    vocab_size: int = 8000      # lo fija el tokenizer BPE entrenado
    block_size: int = 256       # longitud de contexto
    dim: int = 256              # dimensión del embedding/modelo
    n_layers: int = 6
    n_heads: int = 8
    dropout: float = 0.1

    @property
    def params_estimados(self) -> int:
        # aproximación grosera para tener una idea del tamaño
        emb = self.vocab_size * self.dim + self.block_size * self.dim
        por_capa = 12 * self.dim * self.dim
        return emb + self.n_layers * por_capa


@dataclass
class TrainConfig:
    batch_size: int = 32
    lr: float = 3e-4
    max_steps: int = 2000
    warmup_steps: int = 100
    eval_every: int = 200
    grad_clip: float = 1.0
    seed: int = 1337
    ckpt: str = "checkpoints/yachay.safetensors"


# Presets fáciles de escalar
PRESETS = {
    "nano":  ModelConfig(dim=128, n_layers=4, n_heads=4, block_size=128),   # ~pocos M
    "micro": ModelConfig(dim=256, n_layers=6, n_heads=8, block_size=256),   # default
    "mini":  ModelConfig(dim=512, n_layers=8, n_heads=8, block_size=512),   # ya pide GPU
}
