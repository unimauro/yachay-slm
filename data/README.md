# Datos — procedencia y licencias

El **código** de Yachay SLM es MIT. Los **datos** no todos comparten esa
licencia; aquí se documenta cada fuente para evitar confusiones.

## Lo que SÍ se versiona en el repo

| Ruta | Qué es | Licencia |
|---|---|---|
| `data/samples/general.jsonl` | ejemplos pequeños para el demo | MIT (propio) |

## Lo que NO se versiona (regenerable o de terceros)

Estos archivos están en `.gitignore` y **no viajan en el clon**. Se regeneran o
se descargan localmente.

| Ruta | Qué es | Cómo obtenerlo | Licencia |
|---|---|---|---|
| `data/nano/math*.jsonl` | aritmética autogenerada | `python -m src.gen_math …` | MIT (propio, 100% por código) |
| `data/nano/sympy*.jsonl` | pares enunciado→SymPy | `python -m src.gen_sympy …` | MIT (propio, 100% por código) |
| `data/distill/*.jsonl` | destilados / terceros | script de destilación o descarga | **según la fuente** ⚠️ |

## ⚠️ Datos de terceros (Alpaca u otros)

Si generas o descargas `data/distill/alpaca_es.jsonl` (una traducción al español
del dataset **Stanford Alpaca**), ten presente que **no es MIT**:

- Stanford Alpaca se publica bajo **CC BY-NC 4.0** (uso **no comercial**).
- Fue generado con salidas de **OpenAI**, sujetas a los términos de OpenAI.

Por eso **no se incluye en este repositorio**. Si lo usas para entrenar, los
pesos resultantes pueden heredar esas restricciones: **no los publiques como si
fueran MIT sin revisar la licencia de origen**. Atribuye a los autores de Alpaca
y respeta la cláusula NC.

Los datos verdaderamente soberanos del proyecto (nano-math y nano-sympy) son
100% autogenerados por código y **no dependen de ninguna fuente externa**.
