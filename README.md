# Yachay SLM — un modelo de lenguaje pequeño, propio, desde cero

> *Yachay* = "conocimiento / aprender" (quechua). · 🌐 **[English README](README.en.md)**

Entrena un **Small Language Model (SLM) desde cero** en tu laptop, y **córrelo
en cualquier equipo** — incluidos **equipos viejos e IoT** (Raspberry Pi, x86
antiguo, RK3588) — con un motor de inferencia en **NumPy puro**, sin GPU, sin
MLX, sin PyTorch.

No busca competir con GPT. Busca un modelo **propio, abierto, chico y
ejecutable on-device**. Soberanía, privacidad y costo — no ranking.

📍 **Roadmap:** [sitio web](https://unimauro.github.io/yachay-slm/) · [ROADMAP.md](ROADMAP.md)

## ⚡ Usarlo ahora mismo (sin entrenar nada)

Clona, instala una vez y prueba. Los modelos ya vienen en `models/` — **no hay
que descargar pesos ni pedir API key**.

```bash
git clone https://github.com/unimauro/yachay-slm.git && cd yachay-slm
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-portable.txt          # numpy, safetensors, tokenizers
```

| Quiero… | Comando | Necesita |
|---|---|---|
| 💬 **Chatear** con el demo (cualquier CPU) | `python -m src.portable.run --chat` | portable |
| ➕ **Aritmética** de verdad (91.3%) | `python -m src.portable.run --ckpt models/nano-math/yachay-math.safetensors --collapse-digits --chat` | portable |
| 🎓 **Mate UNI exacta** (deriva, integra, límites…) | `python -m src.mate --chat` | `pip install -r requirements-mate.txt` |
| 📈 **Graficar** una función | `python -m src.mate --prompt "grafica x^2 - 4"` | mate |
| 📷 **Foto → resuelve** (OCR) | `python -m src.mate --imagen problema.png` | `pip install -r requirements-ocr.txt` + tesseract |
| 🎙️ **Voz** (oír + hablar) | `python -m voz.talk --text "¿cuánto es 347 más 285?" --tts say` | ver [voz/README.md](voz/README.md) |

> Escribe `salir` para salir de cualquier chat. Detalle por SO en
> **[RUN.md](RUN.md)**. ¿Cómo funciona cada pieza? Sigue leyendo ↓

### 🌱 Primer modelo 100% propio: Yachay-Nano de matemática

Un modelo **desde cero, soberano** (arquitectura propia + datos generados por
código, sin Qwen ni GPT) que hace **aritmética** con solo **0.87M parámetros** y
corre en cualquier CPU. Este sí **calcula** de verdad (aprende a sumar/multiplicar
dígito a dígito — algo que una regla fija no hace):

```bash
python -m src.portable.run --ckpt models/nano-math/yachay-math.safetensors \
    --collapse-digits --chat
# > ¿Cuánto es 347 más 285?
#   347 + 285 = 632.
```

**Precisión honesta**, medida sobre el test held-out **completo de 2000
problemas** (números nunca vistos), reproducible con `src/eval_math.py`:

| global | suma | división | multiplicación | resta |
|:---:|:---:|:---:|:---:|:---:|
| **91.3%** | 94.5% | 93.9% | 91.5% | 85.7% |

Reproducible: `src/gen_math.py` (datos) + `src/train.py` + `src/eval_math.py`.
Es el primer ladrillo del track Nano.

**🎙️ Con voz (escuchar + hablar):** piezas locales conectadas al cerebro —
Whisper (STT) + Piper/`say` (TTS). Un tutor que oye la pregunta y responde
hablando, todo on-device. Ver **[voz/README.md](voz/README.md)**.

```bash
python -m voz.talk --text "¿cuánto es 347 más 285?" --tts say
```

### 🎓 Nivel 2 — matemática universitaria EXACTA (traduce → SymPy)

Para cálculo/álgebra la respuesta debe ser **exacta**, así que no la calcula una
red: el enunciado se **traduce** a una línea de código SymPy, y **SymPy resuelve
exacto**. Derivadas (incl. parciales), integrales **definidas e indefinidas**,
límites, ecuaciones, factorización, expansión, simplificación y **series de
Taylor**… nivel UNI, 100% local.

```bash
pip install -r requirements-mate.txt      # sympy, matplotlib
python -m src.mate --prompt "Deriva x^3*sen(x) respecto a x."
#   SymPy:  diff(x**3*sin(x),x)
#   =       x**3*cos(x) + 3*x**2*sin(x)
python -m src.mate --chat
```

**Sobre la traducción — honestidad ganada en una auditoría adversarial.** Esta
tarea (enunciado → una línea de SymPy) es una transformación de texto casi
biyectiva. Un **traductor determinista** de ~40 líneas (`src/traducir.py`,
reglas + regex) la resuelve al **100.0%** en el test y **nunca alucina** — es el
camino **por defecto** de `src.mate`.

Entrenamos además un **modelo neuronal** de 0.87M para la misma tarea
(`models/nano-sympy/`). Es un **experimento honesto**: llega al **99.1%** pero
**no supera** al traductor determinista, y falla justo donde haría falta
"entender" (p. ej. el límite clásico `sen(x)/x`, donde alucina). Se mantiene y
se puede probar con `--modelo`, pero **no lo vendemos como "el modelo aprendió
matemática"**: el logro real es el patrón **traductor + herramienta exacta**.

```bash
python -m src.eval_sympy            # traductor determinista → 100.0%
python -m src.eval_sympy --modelo   # modelo neuronal → 99.1% (experimento)
```

> El evaluador también reporta el **solape train/test** (≈68% de los pares del
> test aparecen en el train: el espacio generable es pequeño). El traductor
> determinista no entrena, así que no le afecta; para el modelo neuronal, ese %
> de la métrica es memorización, no generalización. Lo decimos abiertamente.

**📈 Gráficas:** `python -m src.mate --prompt "grafica x^2 - 4"` dibuja la
función con matplotlib y marca las **raíces exactas** (calculadas por SymPy).

**📷 OCR (foto → resuelve):** lee el enunciado desde una imagen con **Tesseract**
(abierto, offline) y lo resuelve. Cierra el círculo *ver → resolver → hablar*.

```bash
pip install -r requirements-ocr.txt       # + binario tesseract (ver el archivo)
python -m src.mate --imagen problema.png
#   OCR:    Deriva x*3*sen(x) respecto a x.
#   ⚠ corr: Deriva x^3*sen(x) respecto a x.   (heurística '*'→'^'; verifica)
#   =       x**3*cos(x) + 3*x**2*sin(x)
```

> Alcance honesto: el OCR es fiable con texto **impreso/tipeado**; el `^`
> (exponente) suele leerse como `*`, así que se aplica una heurística **visible**
> y se muestra siempre lo leído para que lo verifiques. Manuscrito y notación 2D
> no están garantizados.

> **Seguridad del `eval()`:** el código SymPy se valida antes de ejecutarse con
> una **lista blanca de AST** (`src/sympy_solve.py`): se rechaza cualquier acceso
> a atributos (`.__class__`…), comprensiones o nombres fuera de SymPy — las vías
> del escape clásico de sandboxes en Python. Verificado contra payloads de RCE.

```
Entrenas en Mac (rápido, con MLX)  ──►  .safetensors  ──►  corre en cualquier CPU (NumPy)
                                                             Raspberry Pi · x86 viejo · RK3588 · IoT
```

## Empezar en 2 minutos

> 🖥️ **¿Windows, Linux o macOS?** Guía por sistema operativo en **[RUN.md](RUN.md)**
> (ejecutar funciona en los tres; entrenar es solo Mac Apple Silicon).

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
> Un modelo diminuto desde cero **habla pero no sabe**. Entrena el tuyo ↓

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
| 🌱 **Yachay-Nano** (`src/`) | modelo **desde cero**, propio y diminuto | IoT chico / juguetes de **nicho** | domina un dominio acotado (mate 91.3%) |
| 🧠 **Yachay-General** (`finetune/`) | **afinar** (LoRA) un modelo chico preentrenado (Qwen2.5-0.5B) | asistente **general** en Raspberry Pi / PC vieja | multifuncional, útil de verdad |

Para un asistente general con pocos parámetros, la ruta realista es **partir de
un preentrenado y afinarlo** — sigue siendo abierto y tuyo, corre en hardware
modesto, pero ya no es "desde cero". Ver **[finetune/README.md](finetune/README.md)**.

## La "capa para equipos viejos" (portabilidad)

El entrenamiento usa **MLX** (solo Apple Silicon), pero la **inferencia** vive
en `src/portable/`: una reimplementación del modelo en **NumPy puro** que carga
los mismos pesos `.safetensors`. Solo depende de `numpy`, `safetensors` y
`tokenizers`.

- Corre en **cualquier CPU**: Raspberry Pi, PCs viejas, ARM Linux, RK3588…
- Sin GPU, sin CUDA, sin MLX, sin PyTorch.
- **Verificado**: los logits del motor NumPy coinciden con los de MLX y con el
  binario Rust. El diff medido es **~6–8e-6** (`scripts/check_parity.py` y
  `scripts/check_parity_rust.py`); nota que el binario Rust imprime 6 decimales.

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
API). Para datos propios a escala, un LLM *teacher* genera un dataset curado:

```bash
export OPENROUTER_API_KEY=...        # o TEACHER_BASE_URL + TEACHER_TOKEN (gateway propio)
python -m src.distill --n 2000 --dominio "STEM para niños 8-12 años" \
    --salida data/distill/stem.jsonl
python -m src.tokenizer --entrena data/distill/stem.jsonl
python -m src.train --datos data/distill/stem.jsonl --preset micro
```

> **Licencia de los datos:** el código es MIT, pero **los datos no todos**. Los
> de `nano-math`/`nano-sympy` son 100% autogenerados (MIT). Si además destilas o
> descargas datos de terceros (p. ej. Alpaca en `data/distill/`, que está
> **gitignoreado y no se versiona**), respeta su licencia — ver
> **[data/README.md](data/README.md)**.

## Presets (escalar con calma)

| Preset | dim | capas | ctx | ~params | Dónde |
|---|---|---|---|---|---|
| `nano`  | 128 | 4 | 128 | ~1M   | laptop / demo |
| `micro` | 256 | 6 | 256 | ~5M   | laptop |
| `mini`  | 512 | 8 | 512 | ~25M  | ya pide GPU |

## Estructura

```
src/
  config.py           config del modelo + presets (nano/micro/mini)
  model.py            nanoGPT en MLX (entrenamiento)
  tokenizer.py        BPE propio (vocab chico, byte-level, dígitos separables)
  train.py            loop de entrenamiento (MLX); guarda .safetensors + config .json
  gen_math.py         genera datos de aritmética (soberanos, por código)
  gen_sympy.py        genera pares enunciado→código SymPy
  traducir.py         ◄ traductor DETERMINISTA español→SymPy (por defecto en mate)
  sympy_solve.py      motor SymPy exacto con sandbox por lista blanca de AST
  mate.py             Nivel 2: traduce→SymPy, resuelve exacto, y grafica
  grafica.py          gráficas con matplotlib (raíces exactas de SymPy)
  ocr.py              lee un problema desde una foto (Tesseract, offline)
  eval_math.py        precisión de aritmética por operación
  eval_sympy.py       precisión NL→SymPy + solape train/test
  portable/           ◄ inferencia en NumPy puro (equipos viejos / IoT)
voz/                  escuchar (Whisper) + hablar (Piper/say)
rust/                 binario de inferencia en Rust puro (sin Python)
models/               demo + nano-math + nano-sympy (clonar y correr)
finetune/             LoRA sobre Qwen2.5-0.5B (track General)
scripts/check_parity*.py  verifican NumPy == MLX == Rust
```

## Alcance honesto (anti-overclaiming)

- SLM de **dominio estrecho** o corpus chico: **viable** desde cero en laptop.
  Un modelo general tipo GPT desde cero: **no**.
- El nano-math **sí calcula** aritmética (91.3%). El nano-sympy **no "entiende"
  matemática**: la traducción la hace mejor una regla determinista, y así lo
  reportamos. El valor está en el sistema (cerebro chico + herramienta exacta).
- Los números publicados se reproducen con los scripts de `eval_*`; la paridad
  medida es ~6–8e-6, no un gate de 1e-6.
- Aspiración: estar entre los primeros SLM open source hechos en Perú. Antes de
  decir "el primero" en público: verificar. Se plantea como pionero, no líder.

## Licencia

Código bajo **MIT** — ver [LICENSE](LICENSE). Datos: ver [data/README.md](data/README.md).
