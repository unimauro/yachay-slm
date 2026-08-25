# Yachay SLM — a small, sovereign language model, from scratch

> *Yachay* = "knowledge / to learn" (Quechua). · 🌐 **[README en español](README.md)**

Train a **Small Language Model (SLM) from scratch** on your laptop, and **run it
on any machine** — including **old computers and IoT** (Raspberry Pi, old x86,
RK3588) — with a **pure-NumPy** inference engine: no GPU, no MLX, no PyTorch.

It isn't trying to beat GPT. It aims for a model that is **your own, open, tiny
and runnable on-device**. Sovereignty, privacy and cost — not leaderboards.

📍 **Roadmap:** [website](https://unimauro.github.io/yachay-slm/) · [ROADMAP.md](ROADMAP.md)

### 🌱 First 100%-own model: Yachay-Nano for math

A **from-scratch, sovereign** model (own architecture + code-generated data, no
Qwen, no GPT) that does **arithmetic** with only **0.87M parameters** and runs on
any CPU. This one **actually computes** (it learns to add/multiply digit by
digit — something a fixed rule cannot do):

```bash
python -m src.portable.run --ckpt models/nano-math/yachay-math.safetensors \
    --collapse-digits --chat
# > How much is 347 plus 285?
#   347 + 285 = 632.
```

**Honest accuracy**, measured on the **full 2000-problem held-out test** (numbers
never seen in training), reproducible with `src/eval_math.py`:

| overall | add | division | multiplication | subtraction |
|:---:|:---:|:---:|:---:|:---:|
| **91.3%** | 94.5% | 93.9% | 91.5% | 85.7% |

Reproducible: `src/gen_math.py` (data) + `src/train.py` + `src/eval_math.py`.
It's the first brick of the Nano track.

**🎙️ With voice (listen + speak):** local pieces wired to the brain — Whisper
(STT) + Piper/`say` (TTS). A tutor that hears the question and answers out loud,
all on-device. See **[voz/README.md](voz/README.md)**.

```bash
python -m voz.talk --text "how much is 347 plus 285?" --tts say
```

### 🎓 Level 2 — EXACT university-level math (translate → SymPy)

For calculus/algebra the answer must be **exact**, so a neural net doesn't compute
it: the problem is **translated** into one line of SymPy code, and **SymPy solves
it exactly**. Derivatives (incl. partial), **definite and indefinite** integrals,
limits, equations, factoring, expansion, simplification and **Taylor series**…
university level, 100% local.

```bash
pip install -r requirements-mate.txt      # sympy, matplotlib
python -m src.mate --prompt "Deriva x^3*sen(x) respecto a x."
#   SymPy:  diff(x**3*sin(x),x)
#   =       x**3*cos(x) + 3*x**2*sin(x)
python -m src.mate --chat
```

**On the translation — honesty earned in an adversarial audit.** This task
(problem → one SymPy line) is a nearly bijective string transform. A **~40-line
deterministic translator** (`src/traducir.py`, rules + regex) solves it at
**100.0%** on the test and **never hallucinates** — it is the **default** path of
`src.mate`.

We also trained a **0.87M neural model** for the same task (`models/nano-sympy/`).
It's an **honest experiment**: it reaches **99.1%** but **does not beat** the
deterministic translator, and it fails exactly where "understanding" would be
needed (e.g. the classic limit `sin(x)/x`, where it hallucinates). It stays and
can be tried with `--modelo`, but we **do not sell it as "the model learned
math"**: the real achievement is the **translator + exact tool** pattern.

```bash
python -m src.eval_sympy            # deterministic translator → 100.0%
python -m src.eval_sympy --modelo   # neural model → 99.1% (experiment)
```

> The evaluator also reports **train/test overlap** (~68% of test pairs appear in
> train: the generable space is small). The deterministic translator doesn't
> train, so it's unaffected; for the neural model, that share of the metric is
> memorization, not generalization. We say so openly.

**📈 Plots:** `python -m src.mate --prompt "grafica x^2 - 4"` draws the function
with matplotlib and marks the **exact roots** (computed by SymPy).

**📷 OCR (photo → solve):** reads the problem from an image with **Tesseract**
(open, offline) and solves it. Closes the loop *see → solve → speak*.

```bash
pip install -r requirements-ocr.txt       # + tesseract binary (see the file)
python -m src.mate --imagen problema.png
#   OCR:    Deriva x*3*sen(x) respecto a x.
#   ⚠ corr: Deriva x^3*sen(x) respecto a x.   ('*'→'^' heuristic; verify)
#   =       x**3*cos(x) + 3*x**2*sin(x)
```

> Honest scope: OCR is reliable with **printed/typed** text; the `^` (exponent)
> is often read as `*`, so a **visible** heuristic is applied and the read text is
> always shown for you to verify. Handwriting and 2D notation aren't guaranteed.

> **`eval()` safety:** the SymPy code is validated before running with an **AST
> allow-list** (`src/sympy_solve.py`): any attribute access (`.__class__`…),
> comprehensions, or names outside SymPy are rejected — the classic Python
> sandbox-escape vectors. Verified against RCE payloads.

```
Train on a Mac (fast, with MLX)  ──►  .safetensors  ──►  runs on any CPU (NumPy)
                                                          Raspberry Pi · old x86 · RK3588 · IoT
```

## Get started in 2 minutes

> 🖥️ **Windows, Linux or macOS?** Per-OS guide in **[RUN.md](RUN.md)** (running
> works on all three; training is Mac Apple Silicon only).

### A) Just try the bundled demo model (any machine, no GPU)

Works on Linux/Windows/Mac, even a Raspberry Pi. **No MLX, no API key** — the
repo ships a pre-trained demo model (~4.5 MB):

```bash
git clone https://github.com/unimauro/yachay-slm.git
cd yachay-slm

python3 -m venv .venv && source .venv/bin/activate   # recommended
pip install -r requirements-portable.txt

python -m src.portable.run --chat                    # interactive chat
# or a single question:
python -m src.portable.run --prompt "why is the sky blue?"
```

> Type `salir` to quit the chat. If you didn't activate the venv, use
> `./.venv/bin/python` instead of `python`.

> The demo is a **tiny** model (~1M params) trained on few examples: it shows the
> pipeline works, not for serious answers. A tiny from-scratch model **talks but
> doesn't know**. Train your own ↓

### B) Train your own model from scratch (Mac Apple Silicon)

```bash
make setup          # venv + MLX + dependencies
make demo           # trains a tiny model on the bundled sample data
make chat           # chat with what you just trained
```

Without `make`:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m src.tokenizer --entrena data/samples/general.jsonl
python -m src.train --datos data/samples/general.jsonl --preset nano --max_steps 1500
python -m src.portable.run --chat
```

## Two tracks, two goals

The project has **two complementary paths** (because "general + tiny + from
scratch" cannot coexist — it's model physics):

| Track | What it is | For | Quality |
|---|---|---|---|
| 🌱 **Yachay-Nano** (`src/`) | **from-scratch**, own and tiny model | small IoT / **niche** toys | masters a narrow domain (math 91.3%) |
| 🧠 **Yachay-General** (`finetune/`) | **fine-tune** (LoRA) a small pretrained model (Qwen2.5-0.5B) | **general** assistant on Raspberry Pi / old PC | multi-purpose, genuinely useful |

For a general assistant with few parameters, the realistic route is **starting
from a pretrained model and fine-tuning it** — still open and yours, runs on
modest hardware, but no longer "from scratch." See **[finetune/README.md](finetune/README.md)**.

## The "old-hardware layer" (portability)

Training uses **MLX** (Apple Silicon only), but **inference** lives in
`src/portable/`: a pure-**NumPy** reimplementation of the model that loads the
same `.safetensors` weights. It only depends on `numpy`, `safetensors` and
`tokenizers`.

- Runs on **any CPU**: Raspberry Pi, old PCs, ARM Linux, RK3588…
- No GPU, no CUDA, no MLX, no PyTorch.
- **Verified**: NumPy engine logits match MLX and the Rust binary. Measured diff
  is **~6–8e-6** (`scripts/check_parity.py`, `scripts/check_parity_rust.py`);
  note the Rust binary prints 6 decimals.

### Rust: the on-device binary (even lighter)

For final IoT/embedded deployment there is a **pure Rust** implementation
(`rust/`) that loads the same weights and **needs no Python at all**:

```bash
cd rust && cargo build --release
./target/release/yachay --chat
./target/release/yachay --prompt "how many planets are there?"
```

- A **static binary** of a few MB: no interpreter, no `pip`, no venv.
- Cross-compiles to ARM (Raspberry Pi, RK3588) with `cargo build --target ...`.
- Verified identical to the reference engine (`scripts/check_parity_rust.py`).

Rule of thumb: **Python/MLX to train and prototype; Rust to deploy** on small
hardware.

## Data strategy: distillation

The demo trains on `data/samples/general.jsonl` (bundled, no API). For your own
data at scale, a *teacher* LLM generates a curated dataset:

```bash
export OPENROUTER_API_KEY=...        # or TEACHER_BASE_URL + TEACHER_TOKEN (own gateway)
python -m src.distill --n 2000 --dominio "STEM for kids 8-12" \
    --salida data/distill/stem.jsonl
python -m src.tokenizer --entrena data/distill/stem.jsonl
python -m src.train --datos data/distill/stem.jsonl --preset micro
```

> **Data licensing:** the code is MIT, but **not all the data is**. The
> `nano-math`/`nano-sympy` data is 100% self-generated (MIT). If you additionally
> distill or download third-party data (e.g. Alpaca in `data/distill/`, which is
> **gitignored and not versioned**), respect its license — see
> **[data/README.md](data/README.md)**.

## Presets (scale calmly)

| Preset | dim | layers | ctx | ~params | Where |
|---|---|---|---|---|---|
| `nano`  | 128 | 4 | 128 | ~1M   | laptop / demo |
| `micro` | 256 | 6 | 256 | ~5M   | laptop |
| `mini`  | 512 | 8 | 512 | ~25M  | needs a GPU |

## Honest scope (anti-overclaiming)

- A **narrow-domain** SLM or small corpus: **viable** from scratch on a laptop.
  A general GPT-like model from scratch: **no**.
- nano-math **does compute** arithmetic (91.3%). nano-sympy **does not
  "understand" math**: a deterministic rule translates better, and we report it
  that way. The value is the system (small brain + exact tool).
- Published numbers reproduce with the `eval_*` scripts; measured parity is
  ~6–8e-6, not a 1e-6 gate.
- Aspiration: to be among the first open-source SLMs made in Peru. Before saying
  "the first" publicly: verify. Framed as a pioneer, not a leader.

## License

Code under **MIT** — see [LICENSE](LICENSE). Data: see [data/README.md](data/README.md).
