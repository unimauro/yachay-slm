# Roadmap — Yachay SLM

> 🌐 **Versión visual y compartible:** https://claude.ai/code/artifact/93c7014e-fca2-4320-b07f-a29f0f66cadf

**Tesis:** entrenar en una laptop, correr en un juguete. Un modelo propio,
abierto, que cabe donde ningún otro entra. Soberanía, privacidad y costo — no
ranking.

## Dos tracks, dos objetivos

«General + diminuto + desde cero» no coexisten (es física del modelo). Por eso
el proyecto avanza en dos caminos complementarios:

- 🌱 **Yachay-Nano** (`src/`, `rust/`) — modelo **desde cero**, propio y diminuto
  (~1–7M). Domina un **nicho** acotado y cabe en el hardware más chico (IoT,
  microcontroladores, juguetes de propósito único).
- 🧠 **Yachay-General** (`finetune/`) — **afinar** (LoRA) un modelo chico
  preentrenado (Qwen2.5-0.5B). Multifuncional de verdad, corre en Raspberry Pi /
  PC vieja. Sigue siendo abierto y tuyo, pero no «desde cero».

## Fases

Leyenda: ✅ hecho · 🔨 en curso · 🔭 horizonte · dificultad 🟢 baja / 🟡 media / 🔴 alta

| # | Fase | Estado | Qué | Entregable |
|---|------|--------|-----|------------|
| 0 | Prueba de concepto | ✅ | nanoGPT en MLX, tokenizer BPE, destilación, demo que aprende | tokenizer + modelo diminuto + demo offline sin API key 🟢 |
| 1 | Portabilidad on-device | ✅ | inferencia en NumPy puro y binario Rust, sin MLX/GPU | paridad MLX↔NumPy↔Rust, diff de logits ~1e-6 🟡 |
| 2 | Asistente general (fine-tune) | ✅ | LoRA sobre Qwen2.5-0.5B con datos en español | val loss 2.81 → 1.28 · ~1 GB RAM · adapter 5.6 MB 🟡 |
| 3 | Hardware viejo, de verdad | 🔨 | fusionar adapter + GGUF 4-bit + llama.cpp en Raspberry Pi/RK3588 | ~350 MB · cross-compile ARM · guía de despliegue 🟡 |
| 4 | Datos de dominio | 🔭 | destilación dirigida + datasets HF por dominio (STEM, legal, etc.) | un juguete = un dominio · calidad alta donde importa 🟡 |
| 5 | Aplicaciones con RAG local | 🔭 | extensión de Chrome + add-in de Office con **RAG local** sobre tus documentos, 100% en tu máquina | índice local · privacidad total · asistente sobre tu propia info 🟡 |
| 6 | Comunidad y registro | 🔭 | aportes de datos de la comunidad → reentrenar barato → redistribuir | datos comunitarios → modelo mejorado, publicado en registro abierto 🟢 |
| 7 | Red descentralizada | 🔭 | nodos de la comunidad sirviendo respuestas (estilo *Petals*) | inferencia distribuida · cerebros repartidos 🔴 |
| 8 | El cerebro de Toki | 🔭 | integrar el SLM en juguetes educativos / IoT como cerebro on-device | SoC (RK3588) · binario único · sin nube, sin costo por token 🔴 |

## La idea grande: de un modelo propio a una red de la comunidad

El sueño no es un modelo — es que las propias máquinas, en red, contribuyan con
respuestas. Aterrizado del más barato al más ambicioso:

1. **Datos comunitarios** — aportar/curar pares de calidad. Barato, empieza ya.
2. **Registro de modelos** — publicar cada versión para bajar y correr.
3. **Inferencia distribuida** — nodos respondiendo consultas.
4. **Entrenamiento descentralizado** — repartido entre voluntarios. Horizonte lejano.

**Honestidad de alcance:** para SLMs diminutos, descentralizar el *entrenamiento*
aporta poco (la gracia es que corren enteros en una máquina chica). El valor que
sí llega pronto es 1–2: **datos de la comunidad → reentrenar barato →
redistribuir**. Eso ya es contribuir de verdad.

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
