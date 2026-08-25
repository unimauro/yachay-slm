# Roadmap — Yachay SLM

> 🌐 **Sitio web (GitHub Pages):** https://unimauro.github.io/yachay-slm/

**Tesis:** entrenar en una laptop, correr en una PC antigua. Un modelo propio,
abierto, que cabe donde ningún otro entra. Soberanía, privacidad y costo — no
ranking.

## Dos tracks, dos objetivos

«General + diminuto + desde cero» no coexisten (es física del modelo). Por eso
el proyecto avanza en dos caminos complementarios:

- 🌱 **Yachay-Nano** (`src/`, `rust/`) — modelo **desde cero**, propio y diminuto
  (~1–7M). Domina un **nicho** acotado y cabe en el hardware más chico (IoT,
  microcontroladores, PCs antiguas). **Primer modelo real:** Nano de matemática,
  **91.3%** de precisión (test held-out de 2000, 0.87M params), datos 100%
  autogenerados (`models/nano-math/`). **Nivel 2 (mate UNI):** el enunciado se
  traduce a SymPy y este calcula exacto. La traducción la hace **mejor una regla
  determinista (100%, `src/traducir.py`)** que el modelo neuronal (99.1%, un
  experimento honesto que se conserva pero no supera al regex). Además: capa de
  **voz** (Whisper + Piper) y **gráficas** (matplotlib) — ver README.
- 🧠 **Yachay-General** (`finetune/`) — **afinar** (LoRA) un modelo chico
  preentrenado (Qwen2.5-0.5B). Multifuncional de verdad, corre en Raspberry Pi /
  PC vieja. Sigue siendo abierto y tuyo, pero no «desde cero».

## Fases

Leyenda: ✅ hecho · 🔨 en curso · 🔭 horizonte · dificultad 🟢 baja / 🟡 media / 🔴 alta

| # | Fase | Estado | Qué | Entregable |
|---|------|--------|-----|------------|
| 0 | Prueba de concepto | ✅ | nanoGPT en MLX, tokenizer BPE, destilación, demo que aprende | tokenizer + modelo diminuto + demo offline sin API key 🟢 |
| 1 | Portabilidad on-device | ✅ | inferencia en NumPy puro y binario Rust, sin MLX/GPU | paridad MLX↔NumPy↔Rust, diff de logits ~1e-6 🟡 |
| 2 | Asistente general (fine-tune) | ✅ | LoRA sobre Qwen2.5-0.5B con datos en español | adapter 5.6 MB · ~1 GB RAM · (val loss indicativa, sin log versionado aún) 🟡 |
| 3 | Hardware viejo, de verdad | 🔭 | fusionar adapter + GGUF 4-bit + llama.cpp en Raspberry Pi/RK3588 | ~350 MB (objetivo) · cross-compile ARM · guía de despliegue 🟡 |
| 4 | Datos de dominio | 🔭 | destilación dirigida + datasets HF por dominio (STEM, legal, etc.) | un juguete = un dominio · calidad alta donde importa 🟡 |
| 5 | Aplicaciones con RAG local | 🔭 | extensión de Chrome + add-in de Office con **RAG local** sobre tus documentos, 100% en tu máquina | índice local · privacidad total · asistente sobre tu propia info 🟡 |
| 6 | Comunidad y registro | 🔭 | aportes de datos de la comunidad → reentrenar barato → redistribuir | datos comunitarios → modelo mejorado, publicado en registro abierto 🟢 |
| 7 | Napster para modelos (red P2P) | 🔭 | usar la **CPU/GPU ociosa** de la comunidad para **entrenar y servir** el modelo, distribuido y **controlado** por un coordinador | cómputo ocioso → entrenamiento P2P → modelo de todos 🔴 |
| 8 | El cerebro de Toki | 🔭 | integrar el SLM en juguetes educativos / IoT como cerebro on-device | SoC (RK3588) · binario único · sin nube, sin costo por token 🔴 |

## La idea grande: de un modelo propio a una red de la comunidad

El sueño no es un modelo — es que las propias máquinas, en red, contribuyan con
respuestas. Aterrizado del más barato al más ambicioso:

1. **Datos comunitarios** — aportar/curar pares de calidad. Barato, empieza ya.
2. **Registro de modelos** — publicar cada versión para bajar y correr.
3. **Inferencia distribuida** — nodos respondiendo consultas.
4. **Entrenamiento descentralizado — «Napster para modelos»** — aprovechar la
   CPU/GPU ociosa de voluntarios para entrenar el modelo de forma distribuida y
   **controlada** (un coordinador orquesta, los nodos aportan cómputo). Como
   Napster compartía archivos, pero para entrenar un LLM abierto.

**Honestidad de alcance:** el punto 4 de arriba (entrenamiento descentralizado)
es el más difícil (investigación pura), pero **ya se ha hecho** — *hivemind /
DiLoCo*, *Petals* (inferencia distribuida),
*INTELLECT-1* (entrenado de forma descentralizada). Es un precedente honesto, no
una promesa. Para SLMs diminutos aporta poco (corren enteros en una máquina
chica); tiene sentido cuando el modelo crece. El valor que sí llega **pronto** es
1–2: **datos de la comunidad → reentrenar barato → redistribuir**.

## Cómo contribuir

Es abierto (MIT). Clona, córrelo, propón datos o mejoras. Ver
[`README.md`](README.md) y [`finetune/README.md`](finetune/README.md).

```bash
git clone https://github.com/unimauro/yachay-slm
cd yachay-slm
pip install -r requirements-portable.txt
python -m src.portable.run --chat
```

---

*Alcance honesto: un SLM de dominio estrecho o afinado desde un preentrenado es
viable en laptop; un modelo general tipo GPT desde cero, no. Aspiración: estar
entre los primeros SLM open source hechos en Perú — antes de decir «el primero»,
verificar.*
