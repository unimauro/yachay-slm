"""
Chequeo de paridad: el motor portátil (NumPy) debe dar los MISMOS logits que MLX.

Requiere MLX (Apple Silicon) + un checkpoint. Usa el modelo demo por defecto.
    python scripts/check_parity.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

CKPT = sys.argv[1] if len(sys.argv) > 1 else "models/demo/yachay-demo.safetensors"


def main():
    if not os.path.exists(CKPT):
        raise SystemExit(f"No existe {CKPT}. Entrena primero (make demo) o pasa la ruta.")

    import mlx.core as mx
    from src.config import ModelConfig
    from src.model import GPT
    from src.portable import cargar

    meta = json.load(open(os.path.splitext(CKPT)[0] + ".json", encoding="utf-8"))
    mcfg = ModelConfig(**meta["model"])
    mlx_model = GPT(mcfg)
    mlx_model.load_weights(CKPT)
    mlx_model.eval()

    npm = cargar(CKPT)
    vocab = npm.W["tok_emb.weight"].shape[0]

    rng = np.random.default_rng(0)
    idx = rng.integers(0, vocab, size=(1, min(16, mcfg.block_size))).astype(np.int64)
    lo_mlx = np.array(mlx_model(mx.array(idx)))
    lo_np = npm.forward(idx)

    max_diff = np.abs(lo_mlx - lo_np).max()
    argmax_ok = bool((lo_mlx.argmax(-1) == lo_np.argmax(-1)).all())
    print(f"max |diff| logits MLX vs NumPy: {max_diff:.2e}")
    print(f"argmax coincide: {argmax_ok}")
    if max_diff < 1e-3 and argmax_ok:
        print("PARIDAD OK ✅  (el modelo entrenado corre idéntico sin MLX)")
    else:
        raise SystemExit("PARIDAD FALLA ❌")


if __name__ == "__main__":
    main()
