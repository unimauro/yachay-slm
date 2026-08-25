"""
Motor de matemática EXACTA (nivel universitario) con SymPy.

El modelo Yachay traduce un problema en español a una línea de código SymPy;
este módulo la valida y la ejecuta, devolviendo la respuesta exacta.

    from src.sympy_solve import resolver
    resolver("diff(x**3*sin(x), x)")   # -> "x**3*cos(x) + 3*x**2*sin(x)"

CLI:
    python -m src.sympy_solve "integrate(x**2, x)"

Seguridad: NO usamos eval() a ciegas. Antes de evaluar, el código pasa por un
validador de AST con lista blanca (`_validar`): solo se permiten llamadas a
funciones SymPy conocidas, símbolos declarados, números y operadores
aritméticos. Se RECHAZA cualquier acceso a atributos (`.__class__`…),
comprensiones, subíndices, o nombres fuera de la lista — que son las vías del
escape clásico de sandboxes en Python. Aun así se ejecuta con `__builtins__`
vacío como segunda capa.
"""
import ast

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

# Nombres invocables/legibles permitidos en el código.
_NOMBRES_OK = set(_ALLOWED) | set(_SYMS)

# Nodos de AST permitidos. Nota lo que NO está: Attribute (bloquea .__class__),
# Subscript, ListComp/GeneratorExp/DictComp/SetComp, Lambda, Starred, comparaciones
# encadenadas raras… todas vías de escape. Solo expresiones aritméticas + llamadas.
_NODOS_OK = (
    ast.Expression, ast.Call, ast.Name, ast.Load, ast.Constant, ast.keyword,
    ast.BinOp, ast.UnaryOp, ast.Tuple, ast.List,
    # operadores aritméticos y unarios
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod, ast.FloorDiv,
    ast.USub, ast.UAdd,
)


def _validar(codigo: str) -> ast.Expression:
    """Parsea y valida el código con lista blanca de AST. Lanza ValueError si no pasa."""
    try:
        arbol = ast.parse(codigo, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"sintaxis inválida: {e}") from e
    for nodo in ast.walk(arbol):
        if not isinstance(nodo, _NODOS_OK):
            raise ValueError(f"construcción no permitida: {type(nodo).__name__}")
        if isinstance(nodo, ast.Name) and nodo.id not in _NOMBRES_OK:
            raise ValueError(f"nombre no permitido: {nodo.id!r}")
        if isinstance(nodo, ast.Constant) and not isinstance(nodo.value, (int, float, complex, str, bool)):
            raise ValueError(f"constante no permitida: {nodo.value!r}")
    return arbol


def resolver(codigo: str):
    """Valida y ejecuta una expresión SymPy; devuelve el resultado (objeto SymPy)."""
    arbol = _validar(codigo)
    return eval(compile(arbol, "<sympy>", "eval"), _NS)  # noqa: S307 — validado por _validar + NS restringido


def resolver_texto(codigo: str) -> str:
    """Como resolver(), pero devuelve el resultado como string legible."""
    try:
        res = resolver(codigo)
        return str(res)
    except ValueError as e:
        return f"[código rechazado: {e}]"
    except Exception as e:
        return f"[error al evaluar SymPy: {e}]"


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        raise SystemExit('Uso: python -m src.sympy_solve "diff(x**2, x)"')
    print(resolver_texto(sys.argv[1]))
