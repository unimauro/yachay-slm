"""
Motor de matemática EXACTA (nivel universitario) con SymPy.

El modelo Yachay traduce un problema en español a una línea de código SymPy;
este módulo la ejecuta de forma segura y devuelve la respuesta exacta.

    from src.sympy_solve import resolver
    resolver("diff(x**3*sin(x), x)")   # -> "x**3*cos(x) + 3*x**2*sin(x)"

CLI:
    python -m src.sympy_solve "integrate(x**2, x)"
"""
import sympy as sp

# Símbolos comunes disponibles para el código generado.
_SYMS = {s: sp.Symbol(s) for s in ["x", "y", "z", "n", "t", "a", "b", "c", "k"]}

# Espacio de nombres permitido: solo SymPy (sin builtins peligrosos).
_ALLOWED = {
    "diff": sp.diff, "integrate": sp.integrate, "limit": sp.limit, "solve": sp.solve,
    "simplify": sp.simplify, "factor": sp.factor, "expand": sp.expand, "Eq": sp.Eq,
    "sin": sp.sin, "cos": sp.cos, "tan": sp.tan, "asin": sp.asin, "acos": sp.acos,
    "atan": sp.atan, "sinh": sp.sinh, "cosh": sp.cosh, "exp": sp.exp, "log": sp.log,
    "ln": sp.log, "sqrt": sp.sqrt, "Abs": sp.Abs, "factorial": sp.factorial,
    "pi": sp.pi, "E": sp.E, "oo": sp.oo, "I": sp.I, "Rational": sp.Rational,
    "Matrix": sp.Matrix, "Symbol": sp.Symbol, "summation": sp.summation,
    "Sum": sp.Sum, "series": sp.series, "gcd": sp.gcd, "lcm": sp.lcm,
    "binomial": sp.binomial, "Derivative": sp.Derivative, "Integral": sp.Integral,
    "true": True, "false": False,
}
_NS = {**_ALLOWED, **_SYMS, "__builtins__": {}}


def resolver(codigo: str):
    """Ejecuta una expresión SymPy y devuelve el resultado (objeto SymPy)."""
    return eval(codigo, _NS)  # noqa: S307 — espacio de nombres restringido a SymPy


def resolver_texto(codigo: str) -> str:
    """Como resolver(), pero devuelve el resultado como string legible."""
    try:
        res = resolver(codigo)
        return str(res)
    except Exception as e:
        return f"[error al evaluar SymPy: {e}]"


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        raise SystemExit('Uso: python -m src.sympy_solve "diff(x**2, x)"')
    print(resolver_texto(sys.argv[1]))
