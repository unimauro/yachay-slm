"""
Generador de datos de MATEMÁTICA en español — 100% por código (sin ningún modelo
externo). Datos soberanos para entrenar Yachay-Nano en un nicho procedimental que
un modelo diminuto SÍ puede dominar.

Uso:
    python -m src.gen_math --n 40000 --salida data/nano/math.jsonl
    python -m src.gen_math --n 2000 --salida data/nano/math_test.jsonl --seed 99

Formato de salida: JSONL con {"instruccion": ..., "respuesta": ...}
La respuesta siempre incluye la operación y el resultado, p.ej. "47 + 68 = 115."
"""
import argparse
import json
import os
import random

NOMBRES = ["Ana", "Luis", "María", "José", "Carla", "Pedro", "Rosa", "Juan",
           "Sofía", "Diego", "Lucía", "Marco", "Elena", "Tomás"]
OBJETOS = ["caramelos", "manzanas", "libros", "galletas", "canicas", "soles",
           "lápices", "panes", "flores", "monedas", "stickers", "globos"]


def _fmt(op_txt, resultado):
    return f"{op_txt} = {resultado}."


def suma(r):
    a, b = r.randint(0, 999), r.randint(0, 999)
    q = r.choice([f"¿Cuánto es {a} más {b}?", f"{a} + {b}", f"Calcula {a} + {b}.",
                  f"Suma {a} y {b}."])
    return q, _fmt(f"{a} + {b}", a + b)


def resta(r):
    a, b = r.randint(0, 999), r.randint(0, 999)
    if b > a:
        a, b = b, a
    q = r.choice([f"¿Cuánto es {a} menos {b}?", f"{a} - {b}", f"Calcula {a} - {b}.",
                  f"Resta {b} de {a}."])
    return q, _fmt(f"{a} - {b}", a - b)


def multiplica(r):
    a = r.randint(0, 12)
    b = r.choice([r.randint(0, 12), r.randint(0, 99)])
    q = r.choice([f"¿Cuánto es {a} por {b}?", f"{a} x {b}", f"Calcula {a} × {b}.",
                  f"Multiplica {a} por {b}."])
    return q, _fmt(f"{a} × {b}", a * b)


def divide(r):
    b = r.randint(1, 12)
    c = r.randint(0, 99)
    a = b * c
    q = r.choice([f"¿Cuánto es {a} entre {b}?", f"{a} / {b}", f"Calcula {a} ÷ {b}.",
                  f"Divide {a} entre {b}."])
    return q, _fmt(f"{a} ÷ {b}", c)


def problema_suma(r):
    n, o = r.choice(NOMBRES), r.choice(OBJETOS)
    a, b = r.randint(1, 100), r.randint(1, 100)
    q = f"{n} tiene {a} {o} y consigue {b} más. ¿Cuántos {o} tiene ahora?"
    return q, f"{a} + {b} = {a + b}. Ahora tiene {a + b} {o}."


def problema_resta(r):
    n, o = r.choice(NOMBRES), r.choice(OBJETOS)
    a = r.randint(5, 100)
    b = r.randint(1, a)
    q = f"{n} tiene {a} {o} y regala {b}. ¿Cuántos {o} le quedan?"
    return q, f"{a} - {b} = {a - b}. Le quedan {a - b} {o}."


def problema_mult(r):
    o = r.choice(OBJETOS)
    a, k = r.randint(1, 20), r.randint(2, 9)
    q = f"En cada caja hay {a} {o}. Si hay {k} cajas iguales, ¿cuántos {o} hay en total?"
    return q, f"{a} × {k} = {a * k}. Hay {a * k} {o} en total."


def problema_div(r):
    n, o = r.choice(NOMBRES), r.choice(OBJETOS)
    k = r.randint(2, 9)
    c = r.randint(1, 20)
    a = k * c
    q = f"{n} reparte {a} {o} en partes iguales entre {k} amigos. ¿Cuántos {o} recibe cada uno?"
    return q, f"{a} ÷ {k} = {c}. Cada uno recibe {c} {o}."


GENERADORES = [suma, suma, resta, resta, multiplica, multiplica, divide, divide,
               problema_suma, problema_resta, problema_mult, problema_div]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40000, help="pares a generar")
    ap.add_argument("--salida", default="data/nano/math.jsonl")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    r = random.Random(args.seed)
    os.makedirs(os.path.dirname(args.salida) or ".", exist_ok=True)

    vistos, total = set(), 0
    with open(args.salida, "w", encoding="utf-8") as f:
        intentos = 0
        while total < args.n and intentos < args.n * 20:
            intentos += 1
            gen = r.choice(GENERADORES)
            instr, resp = gen(r)
            if instr in vistos:
                continue
            vistos.add(instr)
            f.write(json.dumps({"instruccion": instr, "respuesta": resp}, ensure_ascii=False) + "\n")
            total += 1
    print(f"Listo: {total} pares de matemática en {args.salida}")


if __name__ == "__main__":
    main()
