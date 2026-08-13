"""
Loop de entrenamiento en MLX.

Esqueleto correcto-como-punto-de-partida (verificar con mlx instalado).

    python -m src.train --datos data/distill/stem.jsonl --preset micro
"""
import argparse
import json
import os

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from .config import PRESETS, TrainConfig
from .model import GPT
from .tokenizer import TokenizerBPE


def construir_dataset(ruta, tok, block_size):
    """Concatena todo el corpus tokenizado en un solo stream de tokens."""
    ids = []
    with open(ruta, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            texto = f"<bos>{r.get('instruccion','')}\n{r.get('respuesta','')}<eos>"
            ids.extend(tok.encode(texto))
    return np.array(ids, dtype=np.int32)


def batch(data, block_size, bs):
    ix = np.random.randint(0, len(data) - block_size - 1, size=bs)
    x = np.stack([data[i:i + block_size] for i in ix])
    y = np.stack([data[i + 1:i + 1 + block_size] for i in ix])
    return mx.array(x), mx.array(y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datos", required=True)
    ap.add_argument("--tokenizer", default="tokenizer.json")
    ap.add_argument("--preset", default="micro", choices=list(PRESETS))
    args = ap.parse_args()

    tcfg = TrainConfig()
    mcfg = PRESETS[args.preset]
    np.random.seed(tcfg.seed)
    mx.random.seed(tcfg.seed)

    tok = TokenizerBPE.cargar(args.tokenizer)
    mcfg.vocab_size = tok.vocab_size
    data = construir_dataset(args.datos, tok, mcfg.block_size)
    print(f"corpus: {len(data)} tokens | modelo ~{mcfg.params_estimados/1e6:.1f}M params")

    model = GPT(mcfg)
    opt = optim.AdamW(learning_rate=tcfg.lr)
    loss_and_grad = nn.value_and_grad(model, model.loss)

    for step in range(1, tcfg.max_steps + 1):
        x, y = batch(data, mcfg.block_size, tcfg.batch_size)
        loss, grads = loss_and_grad(x, y)
        opt.update(model, grads)
        mx.eval(model.parameters(), opt.state)
        if step % tcfg.eval_every == 0 or step == 1:
            print(f"step {step:5d} | loss {loss.item():.4f}")

    os.makedirs(os.path.dirname(tcfg.ckpt) or ".", exist_ok=True)
    model.save_weights(tcfg.ckpt)
    print(f"pesos guardados en {tcfg.ckpt}")


if __name__ == "__main__":
    main()
