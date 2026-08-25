"""Tests del OCR. Las funciones puras (limpieza, heurística de exponentes) corren
siempre; la lectura real de imagen se salta si no hay Tesseract/pytesseract."""
import shutil

import pytest

from src.ocr import _limpiar, corregir_exponentes


def test_limpiar_normaliza():
    assert _limpiar("Deriva  x   ×  2\nrespecto a x.") == "Deriva x * 2 respecto a x."


@pytest.mark.parametrize("entrada,esperado,cambia", [
    ("Deriva x*3*sen(x) respecto a x.", "Deriva x^3*sen(x) respecto a x.", True),
    ("(x+1)*2", "(x+1)^2", True),          # paréntesis seguido de dígito -> exponente
    ("Integra 3*cos(x) respecto a x.", "Integra 3*cos(x) respecto a x.", False),  # 3*c es mult.
    ("2*x + 1", "2*x + 1", False),         # no toca multiplicación real
])
def test_corregir_exponentes(entrada, esperado, cambia):
    salida, hubo = corregir_exponentes(entrada)
    assert salida == esperado
    assert hubo == cambia


@pytest.mark.skipif(shutil.which("tesseract") is None, reason="Tesseract no instalado")
def test_lee_imagen_render():
    """Renderiza un enunciado impreso, lo lee por OCR y comprueba el pipeline."""
    pytest.importorskip("pytesseract")
    PIL = pytest.importorskip("PIL")
    from PIL import Image, ImageDraw, ImageFont
    import tempfile, os

    from src.ocr import leer, corregir_exponentes
    from src.traducir import traducir
    from src.sympy_solve import resolver_texto

    img = Image.new("RGB", (1000, 150), "white")
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 44)
    except Exception:
        font = ImageFont.load_default()
    d.text((30, 50), "Integra 3*cos(x) respecto a x.", fill="black", font=font)
    with tempfile.TemporaryDirectory() as tmp:
        ruta = os.path.join(tmp, "p.png")
        img.save(ruta)
        texto, _ = corregir_exponentes(leer(ruta))
    assert "Integra" in texto and "cos(x)" in texto
    assert resolver_texto(traducir(texto)) == "3*sin(x)"
