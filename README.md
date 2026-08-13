# Yachay SLM — un modelo de lenguaje pequeño, propio, desde cero

> *Yachay* = "conocimiento / aprender" (quechua). Nombre placeholder, fácil de renombrar.

**Objetivo:** entrenar un **Small Language Model (SLM) desde cero**, empezando
**solo en la laptop** (macOS / Apple Silicon), destilando datos de LLMs más
grandes. No busca competir con GPT — busca un modelo **propio, abierto, de dominio
estrecho y ejecutable on-device**. Soberanía, privacidad y costo, no ranking.

## Alcance honesto (anti-overclaiming)

- Desde cero en laptop **sí es viable** si el modelo es **pequeño** (≈1M–125M
  parámetros) y el **dominio es estrecho**. Un modelo general tipo GPT **no**.
- El valor: dueño del modelo, corre en infra chica / hardware embebido
  (encaja con el "modelo propio" de Toki y con inferencia en Rust/candle).
- Aspiración: estar entre los **primeros SLM open source hechos en Perú**.
  Antes de decir "el primero" en público: verificar. Plantearlo como pionero.

## Ruta por fases

| Fase | Dónde | Qué |
|---|---|---|
| **1. PoC** | laptop (MLX) | tokenizer + nanoGPT diminuto + **destilación de datos** de un teacher; entrenar y ver que aprende |
| 2. Escala | GPU alquilada (solo si el PoC lo justifica) | más datos, más parámetros |
| 3. Cuantizar | laptop | GGUF / 4-bit para inferencia liviana |
| 4. On-device | Rust `candle` / MLX / llama.cpp | binario único; cerebro de Toki en SoC (RK3588) |

## Estrategia de datos: destilación

El teacher (un LLM grande, vía OpenRouter / gateway `ai.tunky.net`) **genera un
dataset curado** del dominio objetivo. Empezamos por **destilación por datos
sintéticos** (el teacher produce pares instrucción→respuesta; el student entrena
sobre eso). La destilación por *logits/soft-labels* queda para después (necesita
logits del teacher, difícil por API).

**Decisión pendiente:** ¿qué dominio para el primer SLM?
STEM para niños (Toki) · expedientes/tesauro UNI · legal. El dominio estrecho es
lo que hace factible el "desde cero" en laptop.

## Estructura

```
src/
  config.py      configuración del modelo (dim, capas, cabezas, vocab, block_size)
  model.py       nanoGPT en MLX (embeddings, bloques transformer, cabeza LM)
  tokenizer.py   BPE (entrena/usa un tokenizer propio de vocab chico)
  distill.py     genera el dataset destilando de un teacher (OpenRouter)
  train.py       loop de entrenamiento (MLX)
  generate.py    muestreo / generación de texto
notes/arquitectura.md   decisiones de arquitectura y plan detallado
```

## Empezar (en la otra terminal)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # mlx solo corre en Apple Silicon
# 1) generar datos destilando de un teacher
export OPENROUTER_API_KEY=...             # o usar el gateway ai.tunky.net
python -m src.distill --n 2000 --dominio "STEM para niños 8-12 años" --salida data/distill/stem.jsonl
# 2) entrenar el tokenizer y el modelo
python -m src.tokenizer --entrena data/distill/stem.jsonl
python -m src.train --datos data/distill/stem.jsonl
# 3) probar
python -m src.generate --prompt "por que el cielo es azul?"
```

> ⚠️ El código de `src/` es un **esqueleto correcto-como-punto-de-partida** pero
> **sin verificar con MLX instalado**. Primer paso en la otra terminal: instalar
> mlx y hacer que `model.py` haga un forward pass sobre un batch de juguete.
