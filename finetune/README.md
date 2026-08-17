# Yachay-General — afinar un modelo chico preentrenado (LoRA)

Este track es distinto al "desde cero" (`src/`). Aquí partimos de un modelo
**ya preentrenado y pequeño** (ej. `Qwen2.5-0.5B`, Apache-2.0, ya sabe español)
y lo **afinamos con LoRA** sobre nuestros datos. Resultado: un asistente
**general y multifuncional** que corre en hardware modesto (Raspberry Pi, PC
vieja) — a cambio de que **ya no es "desde cero"**.

Por qué esta vía: un modelo diminuto entrenado desde cero **no puede** ser un
buen asistente general (lo comprobamos: habla fluido pero inventa). Para calidad
general con pocos parámetros, la única ruta realista es partir de un preentrenado.

Requiere Apple Silicon (MLX):

```bash
pip install -r requirements-finetune.txt
```

## Flujo completo

```bash
# 1) datos: nuestro JSONL {instruccion, respuesta} -> formato chat de mlx-lm
python finetune/prepare_data.py --datos data/distill/alpaca_es.jsonl --salida finetune/data

# 2) afinar con LoRA (rápido en M1/M2)
python -m mlx_lm.lora \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --train --data finetune/data \
  --iters 200 --batch-size 4 --num-layers 8 \
  --adapter-path finetune/adapters

# 3) probar (con el adapter aplicado)
python -m mlx_lm.generate \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --adapter-path finetune/adapters \
  --prompt "Explica la fotosíntesis para un niño." --max-tokens 200

# 4) fusionar el adapter en el modelo (un modelo independiente ya afinado)
python -m mlx_lm.fuse \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --adapter-path finetune/adapters \
  --save-path finetune/yachay-general
```

## Correr en equipos viejos / IoT (Raspberry Pi, RK3588, PC vieja)

El modelo afinado usa la arquitectura de Qwen (no la nuestra), así que el motor
NumPy/Rust de `src/portable` **no** lo corre. El estándar para hardware chico es
**GGUF + llama.cpp** (CPU, cuantizado a 4 bits):

```bash
# convertir a GGUF (con llama.cpp) y cuantizar a Q4_K_M
python convert_hf_to_gguf.py finetune/yachay-general --outfile yachay-general.gguf
./llama-quantize yachay-general.gguf yachay-general-q4.gguf Q4_K_M

# correr en el dispositivo (pocos cientos de MB en RAM)
./llama-cli -m yachay-general-q4.gguf -p "Explica la fotosíntesis para un niño."
```

> Un `Qwen2.5-0.5B` cuantizado a 4 bits pesa ~350 MB y corre en una Raspberry Pi.
> Para lo MÁS chico (microcontroladores) sigue estando el track "desde cero" de
> nicho en `src/`.
