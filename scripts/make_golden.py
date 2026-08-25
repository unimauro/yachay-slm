"""
Genera los logits 'golden' de referencia del motor NumPy sobre el modelo demo.

El test tests/test_parity_golden.py los compara para detectar regresiones del
motor portable — sin necesitar MLX ni Rust, así corre en CI (Ubuntu, solo NumPy).

    python -m scripts.make_golden
"""
import numpy as np

from src.portable.engine import cargar

CKPT = "models/demo/yachay-demo.safetensors"
IDS = np.array([[1, 5, 9, 13, 17, 21, 25, 29]])  # secuencia fija dentro del vocab
GOLDEN = "tests/golden_demo_logits.npz"


def main():
    m = cargar(CKPT)
    logits = m.forward(IDS)
    np.savez_compressed(GOLDEN, ids=IDS, logits=logits.astype(np.float32))
    print(f"Golden guardado: {GOLDEN}  shape={logits.shape}")


if __name__ == "__main__":
    main()
