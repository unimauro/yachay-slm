# Backlog — Yachay SLM

## Fase 1 — PoC en laptop (MLX)  ✅ (verificado)
- [x] `pip install mlx` y verificar `python -m src.model` (forward + loss del batch de juguete).
- [x] Datos de ejemplo incluidos (`data/samples/general.jsonl`) para demo sin API.
- [x] Entrenar tokenizer BPE (byte-level, con decoder) y revisar vocab.
- [x] Correr `src.train` preset `nano`, ver que la loss baja (7.2 → 0.03).
- [x] `src.generate` / `src.portable.run`: produce texto del dominio con sentido.
- [x] Split de validación + val loss en el loop de entrenamiento.
- [ ] Métrica de perplejidad explícita + corpus más grande (val loss aún alta por overfit).
- [ ] Decidir/ampliar el DOMINIO del primer SLM serio (general amplio vs. nicho).

## Fase 1b — Portabilidad (equipos viejos / IoT)  ✅
- [x] Motor de inferencia en NumPy puro (`src/portable/`), sin MLX ni GPU.
- [x] Paridad verificada MLX vs NumPy (`scripts/check_parity.py`, diff ~1e-6).
- [x] Modelo demo pre-entrenado versionado (`models/demo/`) para "clonar y correr".
- [x] `requirements-portable.txt` mínimo (numpy/safetensors/tokenizers).
- [ ] Probar en hardware real: Raspberry Pi / RK3588 / x86 viejo.
- [ ] Tokenizer sin dependencia de `tokenizers` (Rust) para equipos muy limitados.

## Fase 4 — Binario Rust (on-device, sin Python)  ✅
- [x] Crate `rust/`: forward pass en Rust puro (safetensors + tokenizers).
- [x] Paridad verificada Rust vs NumPy/MLX (`scripts/check_parity_rust.py`, ~1e-6).
- [x] CLI `yachay --prompt / --chat`, fallback al modelo demo, muestreo top_k/temp/seed.
- [ ] Cross-compilar y probar en ARM real (Raspberry Pi / RK3588) con `cross`.
- [ ] Cuantización int8/4-bit en el binario para bajar RAM y tamaño.

## Fase 2+ (después)
- [ ] Escalar a preset `micro`/`mini` (posible GPU alquilada).
- [ ] Destilación por soft-labels/logits si el teacher lo permite.
- [ ] Cuantización (GGUF/4-bit) para inferencia liviana.
- [ ] Inferencia on-device: candle (Rust) / MLX / llama.cpp; integrar con Toki.

## Decisiones
- Alcance honesto: SLM de dominio estrecho, NO GPT general.
- Empezar diminuto (preset nano) y escalar solo cuando el pipeline funcione.
- Aspiración: pionero SLM open source peruano (verificar antes de decir "el primero").
