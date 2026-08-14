"""
Loop de entrenamiento en MLX (Apple Silicon).

    python -m src.train --datos data/samples/general.jsonl --preset nano --max_steps 800

Al terminar guarda:
    checkpoints/yachay.safetensors   pesos del modelo
    checkpoints/yachay.json          config (preset, dims, vocab) para inferencia

Los pesos son portables: `src/portable/run.py` los corre en NumPy puro,
sin MLX, en cualquier máquina (Raspberry Pi, x86 viejo, RK3588...).
"""
import argparse
import json
import os
from dataclasses import asdict

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from .config import PRESETS, TrainConfig
from .model import GPT
from .tokenizer import TokenizerBPE


def construir_dataset(ruta, tok):
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
    ap.add_argument("--preset", default="nano", choices=list(PRESETS))
    ap.add_argument("--max_steps", type=int, default=None, help="sobreescribe el preset")
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--out", default="checkpoints/yachay.safetensors")
    ap.add_argument("--val_split", type=float, default=0.1)
    args = ap.parse_args()

    tcfg = TrainConfig()
    if args.max_steps is not None:
        tcfg.max_steps = args.max_steps
    if args.batch_size is not None:
        tcfg.batch_size = args.batch_size
    if args.lr is not None:
        tcfg.lr = args.lr

    mcfg = PRESETS[args.preset]
    np.random.seed(tcfg.seed)
    mx.random.seed(tcfg.seed)

    tok = TokenizerBPE.cargar(args.tokenizer)
    mcfg.vocab_size = tok.vocab_size

    data = construir_dataset(args.datos, tok)
    if len(data) <= mcfg.block_size + 2:
        raise SystemExit(
            f"Corpus muy chico ({len(data)} tokens) para block_size={mcfg.block_size}. "
            f"Usa más datos o un preset con block_size menor.")
    n_val = max(mcfg.block_size + 2, int(len(data) * args.val_split))
    train_data, val_data = data[:-n_val], data[-n_val:]
    print(f"corpus: {len(data)} tokens (train {len(train_data)} / val {len(val_data)}) "
          f"| modelo ~{mcfg.params_estimados/1e6:.2f}M params | preset {args.preset}")

    model = GPT(mcfg)
    opt = optim.AdamW(learning_rate=tcfg.lr)
    loss_and_grad = nn.value_and_grad(model, model.loss)

    def eval_val():
        model.eval()
        losses = []
        for _ in range(10):
            x, y = batch(val_data, mcfg.block_size, tcfg.batch_size)
            losses.append(model.loss(x, y).item())
        model.train()
        return sum(losses) / len(losses)

    model.train()
    for step in range(1, tcfg.max_steps + 1):
        x, y = batch(train_data, mcfg.block_size, tcfg.batch_size)
        loss, grads = loss_and_grad(x, y)
        opt.update(model, grads)
        mx.eval(model.parameters(), opt.state)
        if step % tcfg.eval_every == 0 or step == 1:
            vl = eval_val() if len(val_data) > mcfg.block_size + 1 else float("nan")
            print(f"step {step:5d} | train {loss.item():.4f} | val {vl:.4f}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    model.save_weights(args.out)
    # Config sidecar para que la inferencia sepa las dimensiones exactas.
    cfg_path = os.path.splitext(args.out)[0] + ".json"
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump({"preset": args.preset, "model": asdict(mcfg),
                   "tokenizer": args.tokenizer}, f, ensure_ascii=False, indent=2)
    print(f"pesos guardados en {args.out}")
    print(f"config guardada en {cfg_path}")


if __name__ == "__main__":
    main()
