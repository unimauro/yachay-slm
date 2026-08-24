"""
Genera datos para el NIVEL 2: traducir un problema de matemática en español a una
línea de código SymPy. 100% por código (SymPy + plantillas propias), soberano.

El modelo aprende a TRADUCIR (no a calcular); SymPy hace el cálculo exacto.

    python -m src.gen_sympy --n 30000 --salida data/nano/sympy.jsonl
    python -m src.gen_sympy --n 1500 --salida data/nano/sympy_test.jsonl --seed 99

Formato: {"instruccion": <problema en español>, "respuesta": <código SymPy>}
"""
import argparse
import json
import os
import random

import sympy as sp

x = sp.Symbol("x")


def natural(expr):
    """Muestra la expresión de forma 'de estudiante': ^ para potencias, sen/ln, ·."""
    s = str(expr)
    s = s.replace("**", "^")
    s = s.replace("sin", "sen").replace("log", "ln").replace("sqrt", "raíz")
    return s


def _poly(r, gmin=1, gmax=3):
    deg = r.randint(gmin, gmax)
    e = 0
    for k in range(deg + 1):
        c = r.randint(-6, 6)
        if k == deg and c == 0:
            c = r.choice([-1, 1, 2, 3])
        e += c * x ** k
    return sp.expand(e)


def derivar(r):
    base = _poly(r, 1, 3)
    factor = r.choice([1, sp.sin(x), sp.cos(x), sp.exp(x), x])
    e = sp.expand(base * factor) if factor == 1 or factor == x else base * factor
    return f"Deriva {natural(e)} respecto a x.", f"diff({e}, x)"


def integrar(r):
    tipo = r.choice(["poly", "poly", "trig", "exp"])
    if tipo == "poly":
        e = _poly(r, 0, 3)
    elif tipo == "trig":
        e = r.randint(1, 5) * r.choice([sp.sin(x), sp.cos(x)])
    else:
        e = r.randint(1, 5) * sp.exp(x)
    return f"Integra {natural(e)} respecto a x.", f"integrate({e}, x)"


def resolver(r):
    a, b = r.randint(-6, 6), r.randint(-6, 6)
    e = sp.expand((x - a) * (x - b))
    return f"Resuelve la ecuación {natural(e)} = 0.", f"solve(Eq({e}, 0), x)"


def factorizar(r):
    a, b = r.randint(-6, 6), r.randint(-6, 6)
    e = sp.expand((x - a) * (x - b))
    return f"Factoriza {natural(e)}.", f"factor({e})"


def expandir(r):
    a, b = r.randint(-6, 6), r.randint(-6, 6)
    e = (x + a) * (x + b)
    return f"Expande {natural(e)}.", f"expand({e})"


def limite(r):
    clasicos = [
        (sp.sin(x) / x, 0), ((1 - sp.cos(x)) / x, 0),
        (sp.exp(x) - 1, 0), (_poly(r, 1, 2), r.randint(-3, 3)),
    ]
    e, p = r.choice(clasicos)
    return f"Calcula el límite de {natural(e)} cuando x tiende a {p}.", f"limit({e}, x, {p})"


GENERADORES = [derivar, derivar, integrar, integrar, resolver, resolver,
               factorizar, expandir, limite, limite]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30000)
    ap.add_argument("--salida", default="data/nano/sympy.jsonl")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    r = random.Random(args.seed)
    os.makedirs(os.path.dirname(args.salida) or ".", exist_ok=True)
    vistos, total = set(), 0
    with open(args.salida, "w", encoding="utf-8") as f:
        intentos = 0
        while total < args.n and intentos < args.n * 30:
            intentos += 1
            try:
                instr, code = r.choice(GENERADORES)(r)
            except Exception:
                continue
            if instr in vistos:
                continue
            vistos.add(instr)
            f.write(json.dumps({"instruccion": instr, "respuesta": code}, ensure_ascii=False) + "\n")
            total += 1
    print(f"Listo: {total} pares NL→SymPy en {args.salida}")


if __name__ == "__main__":
    main()
