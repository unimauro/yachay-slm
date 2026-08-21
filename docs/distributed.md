# ¿Es factible un LLM distribuido y controlado? — Evidencia

> Análisis honesto de la **Fase 7** del [roadmap](../ROADMAP.md): usar CPU/GPU
> ociosa de la comunidad para **entrenar y servir** un modelo de forma
> distribuida y controlada («Napster para modelos»).

**Respuesta corta:** sí es factible — no es ciencia ficción, ya se ha hecho —
pero es un esfuerzo serio de investigación-ingeniería, requiere GPUs voluntarias
(no solo CPU) y solo cobra sentido cuando el modelo es grande.

## Precedentes reales (ya existe)

| Proyecto | Qué demostró | Enlace |
|---|---|---|
| **Petals** (BigScience) | Inferencia distribuida de modelos grandes (BLOOM-176B, Llama) sobre máquinas voluntarias por internet. | https://github.com/bigscience-workshop/petals |
| **hivemind** | Librería base para deep learning descentralizado sobre internet (DHT, all-reduce tolerante a fallos). | https://github.com/learning-at-home/hivemind |
| **DiLoCo** (DeepMind) | Entrenamiento con **comunicación baja**: entrenar local muchos pasos y sincronizar rara vez (~500× menos tráfico). | https://arxiv.org/abs/2311.08105 |
| **DeMo** (Nous Research) | Optimizador de momento desacoplado que reduce drásticamente la comunicación entre nodos. | https://arxiv.org/abs/2411.19870 |
| **OpenDiLoCo** (Prime Intellect) | Implementación abierta de DiLoCo; entrenamiento entre continentes. | https://github.com/PrimeIntellect-ai/OpenDiLoCo |
| **INTELLECT-1 / INTELLECT-2** (Prime Intellect) | Modelos de 10B y 32B **entrenados de verdad** con GPUs descentralizadas por el mundo. | https://www.primeintellect.ai/ |
| **Folding@home / BOINC** | El modelo «dona tu cómputo ocioso» a escala exaflops (prueba la parte social/logística). | https://foldingathome.org/ |

## Las tres versiones, de más fácil a más difícil

1. **Inferencia distribuida** 🟢 — la comunidad *sirve* un modelo compartido
   (estilo Petals). **Factible hoy**, buen primer paso.
2. **Entrenamiento distribuido controlado** 🟡 — coordinador + GPUs voluntarias con
   sincronización infrecuente (DiLoCo/DeMo). **Factible pero difícil**; hay libs
   abiertas (`hivemind`, `OpenDiLoCo`).
3. **P2P puro sin coordinador** 🔴 — sin confianza, totalmente descentralizado. La
   **verificación** (evitar nodos maliciosos) es frontera de investigación.

> La palabra **«controlado»** es la clave: un **coordinador central** que reparte
> trabajo, agrega y valida (semi-descentralizado) esquiva los problemas más duros
> del P2P sin confianza.

## Los obstáculos reales

1. **Ancho de banda.** Entrenar normal sincroniza gradientes en *cada* paso (GB,
   requiere NVLink/InfiniBand). El internet de casa no puede → se resuelve con
   sync infrecuente (DiLoCo/DeMo). Es *el* problema central, con solución parcial.
2. **CPU vs GPU (honesto).** Para *entrenar*, la **CPU ociosa aporta poco**;
   entrenar necesita GPU. El cómputo CPU rinde mejor en **inferencia** y
   **procesar datos**, no en training.
3. **Heterogeneidad y churn.** Nodos lentos y que entran/salen (como peers de
   Napster). Requiere tolerancia a fallos y checkpoints.
4. **Confianza / verificación.** Un nodo malicioso puede enviar gradientes
   envenenados. Necesita redundancia, validación y reputación. Investigación abierta.
5. **Tamaño del modelo.** Un SLM *diminuto* cabe en una sola máquina → repartir su
   entrenamiento **no aporta**. Tiene sentido cuando el modelo crece
   (cientos de M / miles de M de parámetros).

## Camino realista para Yachay

1. **Inferencia distribuida primero** (Petals/hivemind) — factible ya.
2. **Entrenamiento controlado con GPUs voluntarias** (DiLoCo-style, sync
   infrecuente, coordinador) — meta grande, apoyada en `OpenDiLoCo`/`hivemind`.
3. **Capa de confianza/verificación** — después.

**Coherencia del proyecto:** la gracia de Yachay-Nano es que corre entero en una
máquina chica; la red distribuida **solo cobra sentido si apuntamos a un modelo
comunitario más grande**. Son dos ambiciones distintas — conviene mantenerlas
separadas, como en el roadmap.

## No partimos de cero

`hivemind`, `OpenDiLoCo`, `Petals` y el trabajo de Nous Research (DeMo/DisTrO)
son **open source**. La ruta es integrarlos y adaptarlos, no reinventarlos.
