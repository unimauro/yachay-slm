# Arquitectura y plan

## Modelo (nanoGPT)
- Decoder-only transformer, atención causal, pre-LN, GELU, embeddings de token + posición aprendidos.
- Tamaños en `src/config.py` (presets nano/micro/mini). Empezar en **nano**.
- Implementado en **MLX** (nativo Apple Silicon). Alternativa portable: PyTorch (MPS/CPU).

## Datos = destilación
- Teacher (LLM grande, vía OpenRouter/ai.tunky.net) genera pares instrucción→respuesta del dominio.
- Formato JSONL: `{"instruccion": "...", "respuesta": "..."}`.
- Student entrena por next-token sobre `<bos>instr\nresp<eos>`.
- Empezar hard-label (datos). Soft-label (logits) = fase posterior.

## Por qué es viable en laptop
- Modelo diminuto (pocos M params) + dominio estrecho + datos destilados de calidad
  = aprende el dominio sin necesitar el corpus web gigante ni GPUs de datacenter.
- El "desde cero" real: arquitectura + tokenizer + entrenamiento propios; los datos
  se apalancan de un teacher (destilación), que es la parte cara de hacer bien.

## Riesgos / notas
- El código de `src/` es punto de partida, sin verificar con mlx instalado.
- Definir el dominio ANTES de destilar (cambia todo el dataset).
- Anti-overclaiming en la comunicación pública del proyecto.
