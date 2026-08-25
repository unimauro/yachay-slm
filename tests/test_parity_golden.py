"""Regresión del motor NumPy portable: los logits deben coincidir con el golden.

No requiere MLX ni Rust — corre en CI (solo numpy/safetensors/tokenizers).
Regenera el golden con `python -m scripts.make_golden` si cambias el modelo demo.
"""
import numpy as np

from src.portable.engine import cargar

CKPT = "models/demo/yachay-demo.safetensors"
GOLDEN = "tests/golden_demo_logits.npz"


def test_logits_coinciden_con_golden():
    ref = np.load(GOLDEN)
    m = cargar(CKPT)
    logits = m.forward(ref["ids"])
    assert logits.shape == ref["logits"].shape
    max_diff = float(np.max(np.abs(logits - ref["logits"])))
    assert max_diff < 1e-4, f"motor NumPy divergió del golden: max|diff|={max_diff:.2e}"


def test_argmax_estable():
    ref = np.load(GOLDEN)
    m = cargar(CKPT)
    logits = m.forward(ref["ids"])
    assert np.array_equal(logits.argmax(-1), ref["logits"].argmax(-1))
