"""El traductor determinista español→SymPy debe producir código exacto."""
import pytest

from src.sympy_solve import resolver_texto
from src.traducir import traducir

CASOS = [
    ("Deriva x^2 respecto a x.", "diff(x**2, x)"),
    ("Deriva x^3*sen(x) respecto a x.", "diff(x**3*sin(x), x)"),
    ("Integra 3*cos(x) respecto a x.", "integrate(3*cos(x), x)"),
    ("Resuelve la ecuación x^2 - 4 = 0.", "solve(Eq(x**2 - 4, 0), x)"),
    ("Factoriza x^2 - 9.", "factor(x**2 - 9)"),
    ("Expande (x + 2)*(x + 3).", "expand((x + 2)*(x + 3))"),
    ("Calcula el límite de sen(x)/x cuando x tiende a 0.", "limit(sin(x)/x, x, 0)"),
]


@pytest.mark.parametrize("enunciado,esperado", CASOS)
def test_traduccion_exacta(enunciado, esperado):
    assert traducir(enunciado) == esperado


@pytest.mark.parametrize("enunciado,esperado", CASOS)
def test_resultado_ejecutable(enunciado, esperado):
    # el código traducido debe ejecutarse en SymPy sin error/rechazo
    r = resolver_texto(traducir(enunciado))
    assert not r.startswith("[error") and not r.startswith("[código rechazado")


def test_no_reconocido_devuelve_none():
    assert traducir("hola qué tal") is None
