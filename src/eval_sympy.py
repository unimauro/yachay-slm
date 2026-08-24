"""
Evalúa el Nivel 2 (NL→SymPy): ¿el código que genera el modelo, al ejecutarse en
SymPy, da la MISMA respuesta que el código correcto?

    python -m src.eval_sympy --ckpt checkpoints/yachay-sympy.safetensors \
        --test data/nano/sympy_test.jsonl --n 300

Reporta: precisión de respuesta (resultado correcto) y match exacto de código.
"""
import argparse
import json

from .mate import traducir
from .portable import cargar
from .sympy_solve import resolver_texto


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/yachay-sympy.safetensors")
    ap.add_argument("--test", default="data/nano/sympy_test.jsonl")
    ap.add_argument("--n", type=int, default=300)
    args = ap.parse_args()

    m = cargar(args.ckpt)
    filas = [json.loads(l) for l in open(args.test, encoding="utf-8") if l.strip()][: args.n]

    ok_ans, ok_code = 0, 0
    fallos = []
    for r in filas:
        code = traducir(m, r["instruccion"])
        code_ok = (code == r["respuesta"])
        # respuesta exacta: ejecutar ambos en SymPy y comparar el resultado
        pred = resolver_texto(code)
        real = resolver_texto(r["respuesta"])
        ans_ok = (not pred.startswith("[error")) and (pred == real)
        ok_code += code_ok
        ok_ans += ans_ok
        if not ans_ok and len(fallos) < 8:
            fallos.append((r["instruccion"], r["respuesta"], code))

    n = len(filas)
    print(f"\n=== NIVEL 2 (NL→SymPy) — {n} problemas ===")
    print(f"Respuesta correcta (ejecutada en SymPy): {ok_ans}/{n} = {100*ok_ans/n:.1f}%")
    print(f"Código idéntico al esperado:             {ok_code}/{n} = {100*ok_code/n:.1f}%")
    if fallos:
        print("\nEjemplos que falló:")
        for instr, esp, code in fallos:
            print(f"  · {instr}\n      esperado: {esp}\n      generó:   {code}")


if __name__ == "__main__":
    main()
