"""
Traductor DETERMINISTA español → código SymPy (sin ML).

Honestidad: la tarea "traducir un enunciado de mate a una línea de SymPy" es una
transformación de texto casi biyectiva. Un puñado de reglas la resuelve al 100%
y sin alucinar — supera a un modelo neuronal de 0.87M en el mismo test. Por eso
este traductor es el camino POR DEFECTO en `src.mate`; el modelo neuronal queda
como experimento (`--modelo`). El verdadero valor del sistema está en la pieza
exacta (SymPy), no en meter una red donde una regla determinista es mejor.

    from src.traducir import traducir
    traducir("Deriva x^3*sen(x) respecto a x.")   # -> "diff(x**3*sin(x), x)"

Devuelve None si el enunciado no encaja con ningún patrón conocido.
"""
import re


def a_sympy(expr: str) -> str:
    """Notación 'de estudiante' -> SymPy: ^->**, sen->sin, ln->log, raíz->sqrt, ·->*."""
    expr = expr.strip().rstrip(".").strip()
    expr = expr.replace("^", "**").replace("·", "*")
    expr = expr.replace("raíz", "sqrt").replace("raiz", "sqrt")
    expr = expr.replace("sen", "sin").replace("ln", "log")
    return expr.strip()


# (patrón, cómo construir el código SymPy). El ORDEN importa: los patrones más
# específicos (integral definida, con variable) van antes que los generales.
_VAR = r"([a-z])"  # variable de derivación/integración (x, y, z, t…)

_PATRONES = [
    # Integral DEFINIDA: "Integra x^2 respecto a x entre 0 y 1."
    (re.compile(rf"^integra\s+(.*?)\s+respecto\s+a\s+{_VAR}\s+entre\s+(-?\d+)\s+y\s+(-?\d+)\.?$", re.I | re.S),
     lambda m: f"integrate({a_sympy(m.group(1))}, ({m.group(2)}, {m.group(3)}, {m.group(4)}))"),
    # Serie de Taylor: "Calcula la serie de Taylor de sen(x) en x = 0 hasta orden 5."
    (re.compile(r"^calcula\s+la\s+serie\s+de\s+taylor\s+de\s+(.*?)\s+en\s+x\s*=\s*(-?\d+)\s+hasta\s+orden\s+(\d+)\.?$", re.I | re.S),
     lambda m: f"series({a_sympy(m.group(1))}, x, {m.group(2)}, {m.group(3)})"),
    # Derivada (parcial si la variable no es x): "Deriva x^2*y respecto a y."
    (re.compile(rf"^deriva\s+(.*?)\s+respecto\s+a\s+{_VAR}\.?$", re.I | re.S),
     lambda m: f"diff({a_sympy(m.group(1))}, {m.group(2)})"),
    # Integral indefinida con variable: "Integra 3*cos(x) respecto a x."
    (re.compile(rf"^integra\s+(.*?)\s+respecto\s+a\s+{_VAR}\.?$", re.I | re.S),
     lambda m: f"integrate({a_sympy(m.group(1))}, {m.group(2)})"),
    (re.compile(r"^resuelve\s+la\s+ecuaci[oó]n\s+(.*?)\s*=\s*0\.?$", re.I | re.S),
     lambda m: f"solve(Eq({a_sympy(m.group(1))}, 0), x)"),
    (re.compile(r"^factoriza\s+(.*?)\.?$", re.I | re.S),
     lambda m: f"factor({a_sympy(m.group(1))})"),
    (re.compile(r"^expande\s+(.*?)\.?$", re.I | re.S),
     lambda m: f"expand({a_sympy(m.group(1))})"),
    (re.compile(r"^simplifica\s+(.*?)\.?$", re.I | re.S),
     lambda m: f"simplify({a_sympy(m.group(1))})"),
    (re.compile(r"^calcula\s+el\s+l[ií]mite\s+de\s+(.*?)\s+cuando\s+x\s+tiende\s+a\s+(-?\d+)\.?$", re.I | re.S),
     lambda m: f"limit({a_sympy(m.group(1))}, x, {m.group(2)})"),
]


def traducir(instruccion: str):
    """Traduce un enunciado en español a una línea de código SymPy, o None si no encaja."""
    s = instruccion.strip()
    for pat, construir in _PATRONES:
        m = pat.match(s)
        if m:
            return construir(m)
    return None


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        raise SystemExit('Uso: python -m src.traducir "Deriva x^2 respecto a x."')
    code = traducir(" ".join(sys.argv[1:]))
    print(code if code else "[no reconocido]")
