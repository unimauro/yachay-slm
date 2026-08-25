"""
OCR local: lee un problema de matemática desde una foto/captura.

Es la pieza "ojos" del sistema Yachay: nada de nube ni APIs — usa **Tesseract**
(abierto, offline, corre en cualquier lado). Convierte una imagen con un
enunciado escrito en notación de estudiante ("Deriva x^2 respecto a x.") en texto
que el traductor determinista + SymPy pueden resolver.

    python -m src.ocr foto.png
    python -m src.mate --imagen foto.png     # OCR -> traduce -> SymPy

Alcance honesto: funciona bien con texto **impreso/tipeado** (capturas, PDFs,
libros). La escritura a mano y la notación 2D (fracciones apiladas, raíces con
vínculo) son mucho más difíciles y no están garantizadas.
"""
import argparse
import re


def _preprocesar(img):
    """Escala de grises + upscale suave: ayuda a Tesseract con texto chico."""
    from PIL import Image
    img = img.convert("L")
    if max(img.size) < 1000:
        f = 2
        img = img.resize((img.width * f, img.height * f), Image.LANCZOS)
    return img


def _limpiar(texto: str) -> str:
    """Normaliza la salida cruda del OCR a una línea de enunciado."""
    t = texto.replace("\n", " ")
    t = t.replace("×", "*").replace("·", "*").replace("−", "-").replace("–", "-")
    t = re.sub(r"[ \t]{2,}", " ", t)
    return t.strip()


def corregir_exponentes(texto: str):
    """Tesseract suele leer el '^' (superíndice) como '*'. Corrige 'x*3' -> 'x^3'
    (una base seguida de un dígito). Es HEURÍSTICO: devuelve (texto, hubo_cambio)
    para que el llamador lo muestre y el usuario verifique. No toca '3*x' (mult.)."""
    nuevo = re.sub(r"([a-zA-Z)])\*(\d)", r"\1^\2", texto)
    return nuevo, (nuevo != texto)


def leer(ruta: str, lang: str = "spa+eng") -> str:
    """Devuelve el texto reconocido en la imagen (una línea, limpia)."""
    import pytesseract
    from PIL import Image

    img = _preprocesar(Image.open(ruta))
    crudo = pytesseract.image_to_string(img, lang=lang)
    return _limpiar(crudo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("imagen")
    ap.add_argument("--lang", default="spa+eng")
    args = ap.parse_args()
    print(leer(args.imagen, lang=args.lang))


if __name__ == "__main__":
    main()
