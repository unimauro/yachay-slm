"""
Gráficas de matemática — dibuja una función con matplotlib (local, exacto).

Es la pieza "visual" del sistema Yachay: nada de IA pesada de imágenes, solo
matplotlib (abierto, corre en cualquier lado). SymPy da las raíces exactas.

    python -m src.grafica "x**2 - 4"
    python -m src.grafica "sin(x)" --min -6.28 --max 6.28 --out seno.png

Desde código:
    from src.grafica import graficar
    graficar("x**2 - 4", salida="parabola.png")
"""
import argparse

import matplotlib
matplotlib.use("Agg")  # sin ventana, guarda a archivo (headless / servidor)
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

x = sp.Symbol("x")


def graficar(expr_str, x_min=-10.0, x_max=10.0, salida="grafica.png", n=800):
    """Dibuja expr_str (en x) y marca las raíces reales. Devuelve la ruta del PNG."""
    expr = sp.sympify(expr_str, locals={"x": x})
    f = sp.lambdify(x, expr, "numpy")

    xs = np.linspace(x_min, x_max, n)
    with np.errstate(all="ignore"):
        ys = f(xs)
    ys = np.array(ys, dtype=float)
    ys[np.abs(ys) > 1e6] = np.nan  # corta asíntotas

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=130)
    ax.plot(xs, ys, color="#0E6D5E", linewidth=2.2)

    # ejes por el origen
    ax.axhline(0, color="#888", linewidth=1)
    ax.axvline(0, color="#888", linewidth=1)
    ax.grid(True, alpha=0.25)

    # raíces reales exactas (SymPy) dentro del rango
    try:
        raices = [float(r) for r in sp.solve(expr, x) if r.is_real]
        raices = [r for r in raices if x_min <= r <= x_max]
        if raices:
            ax.plot(raices, [0] * len(raices), "o", color="#B5830F", markersize=7, zorder=5)
    except Exception:
        pass

    try:
        ax.set_title(rf"$y = {sp.latex(expr)}$", fontsize=14)
    except Exception:
        ax.set_title(f"y = {expr}", fontsize=13)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    fig.tight_layout()
    fig.savefig(salida)
    plt.close(fig)
    return salida


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("expr", help='expresión en x, ej: "x**2 - 4"')
    ap.add_argument("--min", type=float, default=-10.0)
    ap.add_argument("--max", type=float, default=10.0)
    ap.add_argument("--out", default="grafica.png")
    args = ap.parse_args()
    ruta = graficar(args.expr, args.min, args.max, args.out)
    print(f"Gráfica guardada en {ruta}")


if __name__ == "__main__":
    main()
