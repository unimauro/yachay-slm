# Yachay SLM — un modelo de lenguaje pequeño, propio, desde cero

> *Yachay* = "conocimiento / aprender" (quechua).

Entrena un **Small Language Model (SLM) desde cero** en tu laptop, y **córrelo
en cualquier equipo** — incluidos **equipos viejos e IoT** (Raspberry Pi, x86
antiguo, RK3588) — con un motor de inferencia en **NumPy puro**, sin GPU, sin
MLX, sin PyTorch.

No busca competir con GPT. Busca un modelo **propio, abierto, chico y
ejecutable on-device**. Soberanía, privacidad y costo — no ranking.

```
Entrenas en Mac (rápido, con MLX)  ──►  .safetensors  ──►  corre en cualquier CPU (NumPy)
                                                             Raspberry Pi · x86 viejo · RK3588 · IoT
```

## Empezar en 2 minutos

### A) Solo probar el modelo demo incluido (cualquier equipo, sin GPU)

Funciona en Linux/Windows/Mac, incluso en una Raspberry Pi. **No necesita MLX
ni API key** — el repo trae un modelo demo pre-entrenado (~4.5 MB):

```bash
git clone https://github.com/unimauro/yachay-slm.git
cd yachay-slm

python3 -m venv .venv && source .venv/bin/activate   # recomendado
pip install -r requirements-portable.txt

python -m src.portable.run --chat                    # chat interactivo
# o una sola pregunta:
python -m src.portable.run --prompt "¿por qué el cielo es azul?"
```

> Escribe `salir` para terminar el chat. Si no activaste el venv, usa
> `./.venv/bin/python` en lugar de `python`.

> El demo es un modelo **diminuto** (~1M parámetros) entrenado sobre pocos
> ejemplos: sirve para ver que el pipeline funciona, no para respuestas serias.
> Entrena el tuyo con tus datos ↓

### B) Entrenar tu propio modelo desde cero (Mac Apple Silicon)

```bash
make setup          # venv + MLX + dependencias
make demo           # entrena un modelo diminuto con los datos de ejemplo incluidos
make chat           # conversa con lo que acabas de entrenar
```

Sin `make`:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m src.tokenizer --entrena data/samples/general.jsonl
python -m src.train --datos data/samples/general.jsonl --preset nano --max_steps 1500
python -m src.portable.run --chat
```

## Dos tracks, dos objetivos

El proyecto tiene **dos caminos complementarios** (porque "general + diminuto +
desde cero" no coexisten — es física del modelo):

| Track | Qué es | Para qué | Calidad |
|---|---|---|---|
| 🌱 **Yachay-Nano** (`src/`) | modelo **desde cero**, propio y diminuto | IoT chico / juguetes de **nicho** | domina un dominio acotado |
| 🧠 **Yachay-General** (`finetune/`) | **afinar** (LoRA) un modelo chico preentrenado (Qwen2.5-0.5B) | asistente **general** en Raspberry Pi / PC vieja | multifuncional, útil de verdad |

Un modelo diminuto desde cero **habla pero no sabe** (aprende gramática, no
hechos). Para un asistente general con pocos parámetros, la ruta realista es
**partir de un preentrenado y afinarlo** — sigue siendo abierto y tuyo, corre en
hardware modesto, pero ya no es "desde cero". Ver **[finetune/README.md](finetune/README.md)**.

## La "capa para equipos viejos" (portabilidad)

El entrenamiento usa **MLX** (solo Apple Silicon), pero la **inferencia** vive
en `src/portable/`: una reimplementación del modelo en **NumPy puro** que carga
los mismos pesos `.safetensors`. Solo depende de `numpy`, `safetensors` y
`tokenizers`.

- Corre en **cualquier CPU**: Raspberry Pi, PCs viejas, ARM Linux, RK3588…
- Sin GPU, sin CUDA, sin MLX, sin PyTorch.
- Está **verificado**: los logits del motor NumPy coinciden con los de MLX
  (`make test` / `scripts/check_parity.py`, diff ~1e-6).

Así puedes entrenar en una Mac y desplegar el cerebro en un juguete educativo o
un equipo reciclado.

### Rust: el binario para el dispositivo (aún más liviano)

Para el despliegue final en IoT/embebido hay una implementación en **Rust puro**
(`rust/`) que carga los mismos pesos y **no necesita Python en absoluto**:

```bash
cd rust && cargo build --release
./target/release/yachay --chat
./target/release/yachay --prompt "¿cuántos planetas hay?"
```

- Un **binario estático** de pocos MB: sin intérprete, sin `pip`, sin venv.
- Cross-compila a ARM (Raspberry Pi, RK3588) con `cargo build --target ...`.
- Verificado idéntico al motor de referencia (`scripts/check_parity_rust.py`).

Regla práctica: **Python/MLX para entrenar y prototipar; Rust para desplegar**
en el hardware chico.

## Estrategia de datos: destilación

El demo entrena con los datos de `data/samples/general.jsonl` (incluidos, sin
API). Para datos propios a escala, un LLM *teacher* genera un dataset curado del
dominio que quieras:

```bash
export OPENROUTER_API_KEY=...        # o TEACHER_BASE_URL + TEACHER_TOKEN (gateway propio)
python -m src.distill --n 2000 --dominio "STEM para niños 8-12 años" \
    --salida data/distill/stem.jsonl
python -m src.tokenizer --entrena data/distill/stem.jsonl
python -m src.train --datos data/distill/stem.jsonl --preset micro
```

Es destilación por **datos sintéticos** (el teacher produce pares
instrucción→respuesta; el student entrena sobre eso). La destilación por
*logits/soft-labels* queda para después.

## Presets (escalar con calma)

| Preset | dim | capas | ctx | ~params | Dónde |
|---|---|---|---|---|---|
| `nano`  | 128 | 4 | 128 | ~1M   | laptop / demo |
| `micro` | 256 | 6 | 256 | ~5M   | laptop |
| `mini`  | 512 | 8 | 512 | ~25M  | ya pide GPU |

Empieza en `nano` hasta que el pipeline funcione end-to-end; escala solo cuando
lo justifique.

## Estructura

```
src/
  config.py           config del modelo + presets (nano/micro/mini)
  model.py            nanoGPT en MLX (entrenamiento)
  tokenizer.py        BPE propio (vocab chico, byte-level)
  distill.py          genera dataset destilando de un teacher (OpenRouter/gateway)
  train.py            loop de entrenamiento (MLX); guarda .safetensors + config .json
  generate.py         generación con MLX
  portable/           ◄ inferencia en NumPy puro (equipos viejos / IoT)
    engine.py         forward + muestreo, carga .safetensors sin MLX
    run.py            CLI: --prompt / --chat
data/samples/         datos de ejemplo (demo sin API)
models/demo/          modelo demo pre-entrenado (clonar y correr)
scripts/check_parity.py   verifica NumPy == MLX
notes/arquitectura.md     decisiones de arquitectura
```

## Alcance honesto (anti-overclaiming)

- SLM de **dominio estrecho** o corpus chico: **viable** desde cero en laptop.
  Un modelo general tipo GPT: **no**.
- El valor real: dueño del modelo, corre en infra chica / hardware embebido.
- Aspiración: estar entre los primeros SLM open source hechos en Perú. Antes de
  decir "el primero" en público: verificar. Plantearlo como pionero.

## Ruta por fases

| Fase | Dónde | Qué |
|---|---|---|
| **1. PoC** ✅ | laptop (MLX) | tokenizer + nanoGPT + destilación + entrenar y ver que aprende |
| **1b. Portátil** ✅ | cualquier CPU | motor NumPy, correr sin MLX (equipos viejos / IoT) |
| **4. On-device** ✅ | Rust puro (`rust/`) | binario nativo sin Python; paridad verificada; cross-compila a ARM |
| 2. Escala | GPU alquilada (si el PoC lo justifica) | más datos, más parámetros |
| 3. Cuantizar | laptop / Rust | int8 / 4-bit para inferencia aún más liviana |

## Licencia

MIT — ver [LICENSE](LICENSE). Úsalo, modifícalo y compártelo libremente.
