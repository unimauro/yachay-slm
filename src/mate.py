"""
Yachay Mate (Nivel 2) — resuelve matemática nivel UNI con respuesta EXACTA.

Pipeline:  problema en español → [traductor] → código SymPy → [SymPy calcula] → respuesta exacta

    python -m src.mate --prompt "Deriva x^3*sen(x) respecto a x."
    python -m src.mate --chat
    python -m src.mate --prompt "grafica x^2 - 4"      # dibuja con matplotlib

Por defecto traduce con reglas DETERMINISTAS (`src.traducir`): 100% en el test,
sin alucinar. El modelo neuronal (experimento, 0.87M) se activa con `--modelo` —
lo mantenemos por transparencia, aunque no supera al traductor determinista.

Todo local y soberano: el cálculo exacto lo hace SymPy.
"""
import argparse
import re

from .sympy_solve import resolver_texto
from .traducir import a_sympy, traducir as traducir_reglas

MODELO = "models/nano-sympy/yachay-sympy.safetensors"
GRAFICA_VERBOS = ("grafica", "gráfica", "graficar", "grafícame", "dibuja", "traza", "plot", "grafic")


def _es_grafica(texto):
    p = texto.strip().lower()
    return any(p.startswith(v) for v in GRAFICA_VERBOS)


def traducir_modelo(m, problema):
    """El modelo neuronal traduce el problema a una línea de código SymPy (experimento)."""
    out = m.responder(problema, max_new_tokens=64, temperature=1.0, top_k=1, seed=0)
    code = out.split("\n", 1)[1] if "\n" in out else out
    for t in ("<bos>", "<eos>", "<pad>"):
        code = code.replace(t, "")
    code = re.sub(r"\s+", "", code)  # el tokenizer de dígitos deja huecos
    return code.strip()


def resolver_problema(problema, m=None):
    # ¿pide una gráfica? -> matplotlib (local, exacto), sin traducción
    if _es_grafica(problema):
        expr = problema.strip()
        for v in GRAFICA_VERBOS:
            if expr.lower().startswith(v):
                expr = expr[len(v):]
                break
        for filler in ("la función", "la funcion", "la gráfica de", "y =", "y=", ":"):
            expr = expr.replace(filler, "")
        expr = a_sympy(expr.strip(" ¿?."))
        from .grafica import graficar
        ruta = graficar(expr, salida="grafica.png")
        return f"plot({expr})", f"gráfica guardada en {ruta}"

    # Traducción: determinista por defecto; modelo neuronal si se pidió (--modelo).
    if m is not None:
        code = traducir_modelo(m, problema)
    else:
        code = traducir_reglas(problema)
        if code is None:
            return None, "[enunciado no reconocido — prueba 'Deriva/Integra/Resuelve/Factoriza/Expande/Calcula el límite ...']"
    return code, resolver_texto(code)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt")
    ap.add_argument("--chat", action="store_true")
    ap.add_argument("--modelo", action="store_true",
                    help="usar el modelo neuronal (experimento) en vez del traductor determinista")
    ap.add_argument("--ckpt", default=MODELO)
    args = ap.parse_args()

    m = None
    if args.modelo:
        from .portable import cargar
        m = cargar(args.ckpt)
        print(f"[Yachay Mate — modelo neuronal {m.n_layers} capas, dim {m.dim} → SymPy]")
    else:
        print("[Yachay Mate — traductor determinista → SymPy]")

    def responder(p):
        code, ans = resolver_problema(p, m=m)
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
