"""
Paridad ENTRE LENGUAJES: el binario Rust debe dar los mismos logits que el
motor NumPy de referencia (que a su vez ya coincide con MLX).

Requiere: haber compilado el binario Rust (cargo build --release en rust/)
y tener el modelo demo (o pásale otro --ckpt).

    python scripts/check_parity_rust.py
"""
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

CKPT = sys.argv[1] if len(sys.argv) > 1 else "models/demo/yachay-demo.safetensors"
RUST_BIN = "rust/target/release/yachay"


def main():
    if not os.path.exists(RUST_BIN):
        raise SystemExit(f"Falta el binario Rust ({RUST_BIN}). Corre: cd rust && cargo build --release")

    from src.portable import cargar
    npm = cargar(CKPT)
    vocab = npm.W["tok_emb.weight"].shape[0]

    # ids de prueba fijos, dentro del vocab
    ids = [2, 45, 10, 88, 3, 7, 100 % vocab, 5]
    ids_csv = ",".join(str(i) for i in ids)

    # logits del motor NumPy (última posición)
    lo_np = npm.forward(np.array([ids], dtype=np.int64))[0, -1, :]

    # logits del binario Rust
    out = subprocess.run(
        [RUST_BIN, "--ckpt", CKPT, "--dump-logits", ids_csv],
        capture_output=True, text=True, check=True)
    lo_rs = np.array([float(x) for x in out.stdout.split()])

    if lo_rs.shape != lo_np.shape:
        raise SystemExit(f"shapes distintas: rust {lo_rs.shape} vs numpy {lo_np.shape}")

    max_diff = float(np.abs(lo_rs - lo_np).max())
    argmax_ok = int(lo_rs.argmax()) == int(lo_np.argmax())
    print(f"max |diff| logits Rust vs NumPy: {max_diff:.2e}")
    print(f"argmax coincide: {argmax_ok} (rust={lo_rs.argmax()} numpy={lo_np.argmax()})")
    # Rust imprime con 6 decimales, así que toleramos ~1e-4
    if max_diff < 5e-4 and argmax_ok:
        print("PARIDAD RUST OK ✅  (el binario Rust corre idéntico al modelo de referencia)")
    else:
        raise SystemExit("PARIDAD RUST FALLA ❌")


if __name__ == "__main__":
    main()
