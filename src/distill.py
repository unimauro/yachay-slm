"""
Destilación de datos: un LLM teacher genera un dataset curado del dominio.

Estrategia: destilación por DATOS SINTÉTICOS (hard-label). El teacher produce
pares instrucción→respuesta; el student (nuestro SLM) entrena sobre eso.

Usa OpenRouter (o el gateway ai.tunky.net). Configurar:
    export OPENROUTER_API_KEY=...          # o
    export TEACHER_BASE_URL=https://ai.tunky.net/api  TEACHER_TOKEN=...

Uso:
    python -m src.distill --n 2000 --dominio "STEM para niños 8-12 años" \
        --salida data/distill/stem.jsonl
"""
import argparse
import json
import os
import time

import requests

BASE_URL = os.getenv("TEACHER_BASE_URL", "https://openrouter.ai/api/v1")
TOKEN = os.getenv("TEACHER_TOKEN") or os.getenv("OPENROUTER_API_KEY", "")
MODELO_TEACHER = os.getenv("TEACHER_MODEL", "google/gemini-2.0-flash-001")

SYS = (
    "Eres un generador de datos de entrenamiento. Devuelve SOLO JSON válido: "
    "una lista de objetos {{\"instruccion\": ..., \"respuesta\": ...}}. "
    "Dominio: {dominio}. Respuestas correctas, claras y apropiadas. Español."
)


def pide_lote(dominio, k=10):
    """Pide k pares al teacher. Devuelve lista de dicts."""
    prompt = (f"Genera {k} pares instrucción-respuesta variados y de calidad "
              f"sobre: {dominio}. Solo el JSON, sin texto extra.")
    r = requests.post(
        f"{BASE_URL}/chat/completions",
        headers={"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"},
        json={
            "model": MODELO_TEACHER,
            "messages": [
                {"role": "system", "content": SYS.format(dominio=dominio)},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.9,
        },
        timeout=90,
    )
    r.raise_for_status()
    txt = r.json()["choices"][0]["message"]["content"]
    txt = txt.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    datos = json.loads(txt)
    return [d for d in datos if d.get("instruccion") and d.get("respuesta")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000, help="pares objetivo")
    ap.add_argument("--dominio", required=True)
    ap.add_argument("--salida", default="data/distill/dataset.jsonl")
    ap.add_argument("--k", type=int, default=10, help="pares por llamada")
    args = ap.parse_args()

    if not TOKEN:
        raise SystemExit("Falta OPENROUTER_API_KEY / TEACHER_TOKEN en el entorno.")

    os.makedirs(os.path.dirname(args.salida) or ".", exist_ok=True)
    vistos, total = set(), 0
    with open(args.salida, "a", encoding="utf-8") as f:
        while total < args.n:
            try:
                for d in pide_lote(args.dominio, args.k):
                    clave = d["instruccion"].strip().lower()[:120]
                    if clave in vistos:
                        continue
                    vistos.add(clave)
                    f.write(json.dumps(d, ensure_ascii=False) + "\n")
                    f.flush()
                    total += 1
                print(f"{total}/{args.n}")
            except Exception as e:
                print(f"[reintento tras error: {e}]")
                time.sleep(3)
    print(f"Listo: {total} pares en {args.salida}")


if __name__ == "__main__":
    main()
