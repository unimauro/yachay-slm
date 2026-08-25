"""
Evalúa el Nivel 2 (NL→SymPy): ¿el código traducido, al ejecutarse en SymPy, da la
MISMA respuesta que el código correcto?

    # traductor determinista (por defecto):
    python -m src.gen_sympy --n 1500 --salida data/nano/sympy_test.jsonl --seed 99
    python -m src.eval_sympy

    # modelo neuronal (experimento):
    python -m src.eval_sympy --modelo --ckpt models/nano-sympy/yachay-sympy.safetensors

Reporta: precisión de respuesta (resultado correcto) y match exacto de código.
Además, sobre el mismo test, mide el solape con el train para no confundir
memorización con generalización.
"""
import argparse
import json
import os

from .sympy_solve import resolver_texto
from .traducir import traducir as traducir_reglas


def _solape(test_filas, train_path):
    """% de pares (instrucción,respuesta) del test que aparecen literales en el train."""
    if not os.path.exists(train_path):
        return None
    train = set()
    for l in open(train_path, encoding="utf-8"):
        if l.strip():
            d = json.loads(l)
            train.add((d["instruccion"], d["respuesta"]))
    en_train = sum((r["instruccion"], r["respuesta"]) in train for r in test_filas)
    return 100 * en_train / len(test_filas), en_train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", action="store_true", help="evaluar el modelo neuronal en vez del traductor determinista")
    ap.add_argument("--ckpt", default="models/nano-sympy/yachay-sympy.safetensors")
    ap.add_argument("--test", default="data/nano/sympy_test.jsonl")
    ap.add_argument("--train", default="data/nano/sympy.jsonl", help="para medir solape train/test")
    ap.add_argument("--n", type=int, default=1500)
    args = ap.parse_args()

    filas = [json.loads(l) for l in open(args.test, encoding="utf-8") if l.strip()][: args.n]

    if args.modelo:
        from .portable import cargar
        from .mate import traducir_modelo
        m = cargar(args.ckpt)
        traducir = lambda instr: traducir_modelo(m, instr)
        etiqueta = "modelo neuronal"
    else:
        traducir = traducir_reglas
        etiqueta = "traductor determinista"

    ok_ans, ok_code = 0, 0
    fallos = []
    for r in filas:
        code = traducir(r["instruccion"]) or ""
        code_ok = (code == r["respuesta"])
        pred = resolver_texto(code)
        real = resolver_texto(r["respuesta"])
        err = pred.startswith("[error") or pred.startswith("[código rechazado")
        ans_ok = (not err) and (pred == real)
        ok_code += code_ok
        ok_ans += ans_ok
        if not ans_ok and len(fallos) < 8:
            fallos.append((r["instruccion"], r["respuesta"], code))

    n = len(filas)
    print(f"\n=== NIVEL 2 (NL→SymPy) — {etiqueta} — {n} problemas ===")
    print(f"Respuesta correcta (ejecutada en SymPy): {ok_ans}/{n} = {100*ok_ans/n:.1f}%")
    print(f"Código idéntico al esperado:             {ok_code}/{n} = {100*ok_code/n:.1f}%")

    sol = _solape(filas, args.train)
    if sol is not None:
        pct, cnt = sol
        print(f"\n[honestidad] Solape train/test: {cnt}/{n} = {pct:.1f}% de los pares del test "
              f"están literales en el train ({args.train}).")
        print("  → El traductor determinista no entrena, así que el solape no lo afecta;")
        print("    para el modelo neuronal, ese % de la métrica es memorización, no generalización.")

    if fallos:
        print("\nEjemplos que falló:")
        for instr, esp, code in fallos:
            print(f"  · {instr}\n      esperado: {esp}\n      generó:   {code}")


if __name__ == "__main__":
    main()
