# yachay (Rust) — inferencia sin Python

Corre el SLM Yachay como un **binario nativo**, sin Python, sin MLX, sin GPU.
Pensado para el despliegue final en equipos viejos e IoT.

## Compilar y correr

```bash
cargo build --release
./target/release/yachay --chat
./target/release/yachay --prompt "¿por qué el cielo es azul?"
```

Por defecto usa el modelo demo del repo (`../models/demo/yachay-demo.safetensors`).
Para otro checkpoint: `--ckpt ruta/al/modelo.safetensors` (espera al lado un
`.json` con la config y el tokenizer, como los que genera `src/train.py`).

Opciones: `--max_new N`, `--temp F`, `--top_k K`, `--seed S`.

## Cómo funciona

- `src/engine.rs` reimplementa el forward pass (embeddings, atención causal
  multi-cabeza, MLP con GELU, LayerNorm) en Rust puro sobre `Vec<f32>`.
- Carga los mismos pesos `.safetensors` entrenados con MLX (crate `safetensors`).
- Tokenizer BPE con el crate nativo `tokenizers` (lee `tokenizer.json`).

Verificado idéntico al motor de referencia NumPy/MLX:

```bash
# desde la raíz del repo
python scripts/check_parity_rust.py     # diff de logits ~1e-6
```

## Cross-compilar a ARM (Raspberry Pi, RK3588)

```bash
rustup target add aarch64-unknown-linux-gnu
cargo build --release --target aarch64-unknown-linux-gnu
```

> El tokenizer usa `oniguruma` (C embebido) para el regex ByteLevel; al
> cross-compilar necesitas un toolchain de C para el target (p. ej. vía `cross`).
