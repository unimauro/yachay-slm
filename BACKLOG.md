# Backlog — Yachay SLM

## Fase 1 — PoC en laptop (MLX)
- [ ] `pip install mlx` y verificar `python -m src.model` (forward + loss del batch de juguete).
- [ ] Decidir el DOMINIO del primer SLM (STEM niños / expedientes UNI / legal).
- [ ] Destilar dataset del teacher: `python -m src.distill --dominio "..."` (empezar con ~2000 pares).
- [ ] Entrenar tokenizer BPE (`src.tokenizer`) y revisar vocab.
- [ ] Correr `src.train` preset `nano`, ver que la loss baja.
- [ ] `src.generate`: ¿produce texto del dominio con sentido?
- [ ] Loop de evaluación (held-out) + métrica de perplejidad.

## Fase 2+ (después)
- [ ] Escalar a preset `micro`/`mini` (posible GPU alquilada).
- [ ] Destilación por soft-labels/logits si el teacher lo permite.
- [ ] Cuantización (GGUF/4-bit) para inferencia liviana.
- [ ] Inferencia on-device: candle (Rust) / MLX / llama.cpp; integrar con Toki.

## Decisiones
- Alcance honesto: SLM de dominio estrecho, NO GPT general.
- Empezar diminuto (preset nano) y escalar solo cuando el pipeline funcione.
- Aspiración: pionero SLM open source peruano (verificar antes de decir "el primero").
