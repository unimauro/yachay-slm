"""
nanoGPT en MLX (Apple Silicon).

Esqueleto correcto-como-punto-de-partida. PRIMER PASO en la otra terminal:
`pip install mlx` y verificar un forward pass sobre un batch de juguete
(ver el bloque __main__ al final).
"""
import mlx.core as mx
import mlx.nn as nn

from .config import ModelConfig


class Block(nn.Module):
    """Bloque transformer: atención causal + MLP, con conexiones residuales y pre-LN."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.dim)
        self.attn = nn.MultiHeadAttention(cfg.dim, cfg.n_heads, bias=True)
        self.ln2 = nn.LayerNorm(cfg.dim)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.dim, 4 * cfg.dim),
            nn.GELU(),
            nn.Linear(4 * cfg.dim, cfg.dim),
            nn.Dropout(cfg.dropout),
        )

    def __call__(self, x, mask):
        h = self.ln1(x)
        x = x + self.attn(h, h, h, mask)
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.dim)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = [Block(cfg) for _ in range(cfg.n_layers)]
        self.ln_f = nn.LayerNorm(cfg.dim)
        self.head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)

    def __call__(self, idx):
        B, T = idx.shape
        pos = mx.arange(T)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        x = self.drop(x)
        mask = nn.MultiHeadAttention.create_additive_causal_mask(T).astype(x.dtype)
        for blk in self.blocks:
            x = blk(x, mask)
        x = self.ln_f(x)
        return self.head(x)   # logits (B, T, vocab)

    def loss(self, idx, targets):
        logits = self(idx)
        B, T, V = logits.shape
        return nn.losses.cross_entropy(
            logits.reshape(B * T, V), targets.reshape(B * T), reduction="mean")

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Muestreo autorregresivo. idx: (B, T) de tokens iniciales."""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size:]
            logits = self(idx_cond)[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None:
                v = mx.topk(logits, top_k, axis=-1)
                logits = mx.where(logits < v[:, -1:], -mx.inf, logits)
            probs = mx.softmax(logits, axis=-1)
            nxt = mx.random.categorical(mx.log(probs + 1e-9))[:, None]
            idx = mx.concatenate([idx, nxt], axis=1)
        return idx


if __name__ == "__main__":
    # Smoke test: forward + loss sobre un batch de juguete.
    cfg = ModelConfig(vocab_size=100, block_size=16, dim=64, n_layers=2, n_heads=4)
    model = GPT(cfg)
    x = mx.random.randint(0, cfg.vocab_size, (2, cfg.block_size))
    y = mx.random.randint(0, cfg.vocab_size, (2, cfg.block_size))
    print("logits:", model(x).shape, "| loss:", model.loss(x, y).item())
    print("params estimados:", cfg.params_estimados)
