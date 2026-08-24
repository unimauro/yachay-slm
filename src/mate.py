"""
Yachay Mate (Nivel 2) — resuelve matemática nivel UNI con respuesta EXACTA.

Pipeline:  problema en español → [modelo traduce] → código SymPy → [SymPy calcula] → respuesta exacta

    python -m src.mate --prompt "Deriva x^3*sen(x) respecto a x."
    python -m src.mate --chat

Todo local y soberano: el cerebro es tu modelo; el cálculo lo hace SymPy.
"""
import argparse
import re

from .portable import cargar
from .sympy_solve import resolver_texto

MODELO = "models/nano-sympy/yachay-sympy.safetensors"


def traducir(m, problema):
    """El modelo traduce el problema a una línea de código SymPy."""
    out = m.responder(problema, max_new_tokens=64, temperature=1.0, top_k=1, seed=0)
    code = out.split("\n", 1)[1] if "\n" in out else out
    for t in ("<bos>", "<eos>", "<pad>"):
        code = code.replace(t, "")
    code = re.sub(r"(?<=\d)\s+(?=\d)", "", code)  # une dígitos separados
    return code.strip()


def resolver_problema(m, problema):
    code = traducir(m, problema)
    return code, resolver_texto(code)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=MODELO)
    ap.add_argument("--prompt")
    ap.add_argument("--chat", action="store_true")
    args = ap.parse_args()

    m = cargar(args.ckpt)
    print(f"[Yachay Mate — traduce→SymPy | modelo {m.n_layers} capas, dim {m.dim}]")

    def responder(p):
        code, ans = resolver_problema(m, p)
        return f"  SymPy:  {code}\n  =       {ans}"

    if args.chat:
        print("Escribe un problema (o 'salir'). Ej: Integra x^2 respecto a x.")
        while True:
            try:
                p = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if p.lower() in {"salir", "exit", "quit"}:
                break
            if p:
                print(responder(p))
    elif args.prompt:
        print(responder(args.prompt))
    else:
        raise SystemExit("Da --prompt o --chat")


if __name__ == "__main__":
    main()
