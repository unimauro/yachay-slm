"""El sandbox de SymPy debe ejecutar matemática legítima y RECHAZAR escapes.

Este test es la garantía viva de que el eval() no permite RCE (ver la auditoría
que encontró el escape original por introspección de objetos).
"""
import pytest

from src.sympy_solve import resolver_texto

LEGITIMOS = {
    "diff(x**3*sin(x), x)": "x**3*cos(x) + 3*x**2*sin(x)",
    "integrate(x**2, x)": "x**3/3",
    "solve(Eq(x**2 - 4, 0), x)": "[-2, 2]",
    "limit(sin(x)/x, x, 0)": "1",
    "factor(x**2 - 9)": "(x - 3)*(x + 3)",
}

ATAQUES = [
    "().__class__.__base__.__subclasses__()",
    "[c for c in ().__class__.__base__.__subclasses__() if c.__name__=='Popen']",
    "__import__('os').system('echo pwned')",
    "open('/etc/passwd').read()",
    "(1).__class__",
    "globals()",
    "eval('1+1')",
]


@pytest.mark.parametrize("codigo,esperado", LEGITIMOS.items())
def test_matematica_legitima_funciona(codigo, esperado):
    assert resolver_texto(codigo) == esperado


@pytest.mark.parametrize("payload", ATAQUES)
def test_escapes_son_rechazados(payload):
    r = resolver_texto(payload)
    assert r.startswith("[código rechazado") or r.startswith("[error"), \
        f"payload NO fue bloqueado: {payload!r} -> {r!r}"
