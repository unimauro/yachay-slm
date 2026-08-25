"""
Evalúa la PRECISIÓN del Nano de matemática: genera la respuesta a cada problema
del set de test y compara el resultado numérico con el correcto.

    python -m src.eval_math --ckpt checkpoints/yachay-math.safetensors \
        --test data/nano/math_test.jsonl --n 400

Reporta precisión global y por operación (+, -, ×, ÷).
"""
import argparse
import json
import re

from .portable import cargar


def _collapse(texto):
    """Une dígitos separados por espacios ('1 1 8' -> '118'), efecto del tokenizer."""
    return re.sub(r"(?<=\d)\s+(?=\d)", "", texto)


def resultado(texto):
    """Extrae el número que sigue al último '=' (el resultado de la operación)."""
    texto = _collapse(texto)
    ms = re.findall(r"=\s*(-?\d+)", texto)
    return ms[-1] if ms else None


def tipo(respuesta):
    for op, nombre in [("+", "suma"), ("-", "resta"), ("×", "mult"), ("÷", "div")]:
        if op in respuesta:
            return nombre
    return "otro"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="models/nano-math/yachay-math.safetensors")
    ap.add_argument("--test", default="data/nano/math_test.jsonl")
    ap.add_argument("--n", type=int, default=400, help="cuántos problemas evaluar")
    args = ap.parse_args()

    m = cargar(args.ckpt)
    filas = [json.loads(l) for l in open(args.test, encoding="utf-8") if l.strip()][: args.n]

    ok = 0
    por_tipo = {}
    ejemplos = []
    for r in filas:
        esperado = resultado(r["respuesta"])
        # greedy (top_k=1) para respuesta determinista
        gen = m.responder(r["instruccion"], max_new_tokens=32, temperature=1.0, top_k=1, seed=0)
        pred = resultado(gen)
        correcto = (pred is not None and pred == esperado)
        t = tipo(r["respuesta"])
        d = por_tipo.setdefault(t, [0, 0])
        d[1] += 1
        if correcto:
            ok += 1
            d[0] += 1
        elif len(ejemplos) < 8:
            ejemplos.append((r["instruccion"], esperado, pred))

    n = len(filas)
    print(f"\n=== PRECISIÓN Nano-mate ({n} problemas) ===")
    print(f"GLOBAL: {ok}/{n} = {100*ok/n:.1f}%\n")
    for t, (c, tot) in sorted(por_tipo.items()):
        print(f"  {t:6s}: {c:4d}/{tot:<4d} = {100*c/tot:.1f}%")
    if ejemplos:
        print("\nEjemplos que falló:")
        for instr, esp, pred in ejemplos:
            print(f"  · {instr}  → esperado {esp}, dijo {pred}")


if __name__ == "__main__":
    main()
