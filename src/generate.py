"""
Generación / muestreo con un checkpoint entrenado, usando MLX (Apple Silicon).

    python -m src.generate --prompt "¿por qué el cielo es azul?"

Para correr en equipos SIN MLX (Raspberry Pi, x86 viejo, IoT), usa la capa
portátil en NumPy puro:  python -m src.portable.run --prompt "..."
"""
import argparse
import json
import os

import mlx.core as mx

from .config import ModelConfig
from .model import GPT
from .tokenizer import TokenizerBPE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--ckpt", default="checkpoints/yachay.safetensors")
    ap.add_argument("--max_new", type=int, default=120)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--top_k", type=int, default=40)
    args = ap.parse_args()

    ckpt = args.ckpt
    demo = "models/demo/yachay-demo.safetensors"
    if not os.path.exists(ckpt) and os.path.exists(demo):
        print(f"[no encontré {ckpt}; uso el modelo demo incluido: {demo}]")
        ckpt = demo
    args.ckpt = ckpt

    cfg_path = os.path.splitext(args.ckpt)[0] + ".json"
    meta = json.load(open(cfg_path, encoding="utf-8"))
    mcfg = ModelConfig(**meta["model"])
    tok = TokenizerBPE.cargar(meta.get("tokenizer", "tokenizer.json"))

    model = GPT(mcfg)
    model.load_weights(args.ckpt)
    model.eval()

    ids = mx.array([tok.encode(f"<bos>{args.prompt}\n")])
    out = model.generate(ids, args.max_new, temperature=args.temp, top_k=args.top_k)
    texto = tok.decode(out[0].tolist())
    for t in ("<bos>", "<eos>", "<pad>"):
        texto = texto.replace(t, "")
    print(texto.strip())


if __name__ == "__main__":
    main()
