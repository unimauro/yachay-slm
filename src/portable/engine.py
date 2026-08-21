"""
Motor de inferencia en NumPy puro (sin MLX, sin PyTorch).

Carga los pesos .safetensors entrenados con MLX y reproduce exactamente el
forward de `src/model.py`. Corre en cualquier CPU: Raspberry Pi, x86 viejo,
RK3588, laptops sin GPU... la "capa para equipos viejos" del proyecto.

Uso desde código:
    from src.portable import cargar
    m = cargar("checkpoints/yachay.safetensors")   # lee también el .json y el tokenizer
    print(m.responder("¿por qué el cielo es azul?"))

Uso desde terminal:
    python -m src.portable.run --prompt "¿por qué el cielo es azul?"
"""
import json
import math
import os

import numpy as np


def _gelu(x):
    # GELU exacto (igual que nn.GELU() de MLX): 0.5*x*(1+erf(x/sqrt(2)))
    return 0.5 * x * (1.0 + _erf(x / math.sqrt(2.0)))


def _erf(x):
    # Aproximación de Abramowitz & Stegun 7.1.26 (error < 1.5e-7). numpy puro.
    sign = np.sign(x)
    x = np.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * x)
    y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t
                - 0.284496736) * t + 0.254829592) * t * np.exp(-x * x)
    return sign * y


def _layernorm(x, w, b, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * w + b


def _linear(x, w, b=None):
    # Pesos MLX/PyTorch: w shape (out, in) -> y = x @ w.T + b
    y = x @ w.T
    return y + b if b is not None else y


def _softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


class GPTNumpy:
    def __init__(self, weights, cfg, tokenizer):
        self.W = weights
        self.cfg = cfg
        self.tok = tokenizer
        self.n_layers = cfg["n_layers"]
        self.n_heads = cfg["n_heads"]
        self.dim = cfg["dim"]
        self.block_size = cfg["block_size"]
        self.head_dim = self.dim // self.n_heads

    def _attn(self, x, li):
        B, T, C = x.shape
        p = f"blocks.{li}.attn."
        q = _linear(x, self.W[p + "query_proj.weight"], self.W[p + "query_proj.bias"])
        k = _linear(x, self.W[p + "key_proj.weight"], self.W[p + "key_proj.bias"])
        v = _linear(x, self.W[p + "value_proj.weight"], self.W[p + "value_proj.bias"])
        # (B,T,H,hd) -> (B,H,T,hd)
        q = q.reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = (q * scale) @ k.transpose(0, 1, 3, 2)         # (B,H,T,T)
        mask = np.triu(np.full((T, T), -np.inf, dtype=scores.dtype), k=1)
        scores = scores + mask
        att = _softmax(scores, axis=-1)
        out = att @ v                                          # (B,H,T,hd)
        out = out.transpose(0, 2, 1, 3).reshape(B, T, C)
        return _linear(out, self.W[p + "out_proj.weight"], self.W[p + "out_proj.bias"])

    def _block(self, x, li):
        p = f"blocks.{li}."
        h = _layernorm(x, self.W[p + "ln1.weight"], self.W[p + "ln1.bias"])
        x = x + self._attn(h, li)
        h = _layernorm(x, self.W[p + "ln2.weight"], self.W[p + "ln2.bias"])
        h = _linear(h, self.W[p + "mlp.layers.0.weight"], self.W[p + "mlp.layers.0.bias"])
        h = _gelu(h)
        h = _linear(h, self.W[p + "mlp.layers.2.weight"], self.W[p + "mlp.layers.2.bias"])
        return x + h

    def forward(self, idx):
        """idx: (B, T) enteros -> logits (B, T, vocab)."""
        B, T = idx.shape
        pos = np.arange(T)
        x = self.W["tok_emb.weight"][idx] + self.W["pos_emb.weight"][pos]
        for li in range(self.n_layers):
            x = self._block(x, li)
        x = _layernorm(x, self.W["ln_f.weight"], self.W["ln_f.bias"])
        return _linear(x, self.W["head.weight"])

    def generate(self, ids, max_new_tokens=120, temperature=0.8, top_k=40, eos_id=None, rng=None):
        rng = rng or np.random.default_rng()
        ids = list(ids)
        for _ in range(max_new_tokens):
            cond = np.array([ids[-self.block_size:]], dtype=np.int64)
            logits = self.forward(cond)[0, -1, :]
            logits = logits / max(temperature, 1e-6)
            if top_k:
                kth = np.sort(logits)[-min(top_k, logits.shape[0])]
                logits = np.where(logits < kth, -np.inf, logits)
            probs = _softmax(logits)
            nxt = int(rng.choice(len(probs), p=probs))
            ids.append(nxt)
            if eos_id is not None and nxt == eos_id:
                break
        return ids

    def responder(self, prompt, max_new_tokens=120, temperature=0.8, top_k=40,
                  seed=None, collapse_digits=False):
        rng = np.random.default_rng(seed)
        eos_id = None
        try:
            eos_id = self.tok.tk.token_to_id("<eos>")
        except Exception:
            pass
        start = self.tok.encode(f"<bos>{prompt}\n")
        out = self.generate(start, max_new_tokens, temperature, top_k, eos_id, rng)
        texto = self.tok.decode(out)
        if collapse_digits:
            # Une dígitos que el tokenizer de matemática separa ('1 1 8' -> '118')
            # y aprieta espacios dobles (cosmético).
            import re
            texto = re.sub(r"(?<=\d)\s+(?=\d)", "", texto)
            texto = re.sub(r"[ \t]{2,}", " ", texto)
        return texto


def cargar(ckpt="checkpoints/yachay.safetensors", cfg_path=None, tokenizer_path=None):
    """Carga pesos + config + tokenizer y devuelve un GPTNumpy listo para usar."""
    from safetensors.numpy import load_file
    from src.tokenizer import TokenizerBPE

    weights = {k: v.astype(np.float32) for k, v in load_file(ckpt).items()}

    if cfg_path is None:
        cfg_path = os.path.splitext(ckpt)[0] + ".json"
    with open(cfg_path, encoding="utf-8") as f:
        meta = json.load(f)
    cfg = meta["model"]

    if tokenizer_path is None:
        tokenizer_path = meta.get("tokenizer", "tokenizer.json")
    tok = TokenizerBPE.cargar(tokenizer_path)
    return GPTNumpy(weights, cfg, tok)
