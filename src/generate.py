"""
Generación / muestreo con un checkpoint entrenado.

    python -m src.generate --prompt "por que el cielo es azul?" --preset micro
"""
import argparse

import mlx.core as mx

from .config import PRESETS, TrainConfig
from .model import GPT
from .tokenizer import TokenizerBPE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--tokenizer", default="tokenizer.json")
    ap.add_argument("--preset", default="micro", choices=list(PRESETS))
    ap.add_argument("--max_new", type=int, default=120)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--top_k", type=int, default=40)
    args = ap.parse_args()

    tok = TokenizerBPE.cargar(args.tokenizer)
    mcfg = PRESETS[args.preset]
    mcfg.vocab_size = tok.vocab_size

    model = GPT(mcfg)
    model.load_weights(TrainConfig().ckpt)
    model.eval()

    ids = mx.array([tok.encode(f"<bos>{args.prompt}\n")])
    out = model.generate(ids, args.max_new, temperature=args.temp, top_k=args.top_k)
    print(tok.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
