"""
Capa de VOZ para Yachay — escuchar (STT) y hablar (TTS), todo local y soberano.

Arquitectura (piezas abiertas que se conectan al cerebro Yachay):
    micrófono/archivo → [STT: Whisper] → texto
                        → [Yachay: modelo propio] → respuesta
                        → [TTS: Piper / say] → audio

Modos de entrada:
    --text "¿cuánto es 25 más 18?"     escribes el texto (sin micrófono)
    --audio pregunta.wav                transcribe un archivo de audio (Whisper)
    --live [segundos]                   graba del micrófono y transcribe

Backends de voz (TTS):
    --tts say      voz nativa de macOS (local, instantáneo — para probar en tu Mac)
    --tts piper    Piper (voces en español, corre en Raspberry Pi — despliegue)
    --tts none     no habla, solo imprime

Ejemplos:
    python -m voz.talk --text "¿cuánto es 347 más 285?" --tts say
    python -m voz.talk --audio pregunta.wav --tts piper --out respuesta.wav
    python -m voz.talk --live 4 --tts say

Requisitos: ver voz/requirements.txt (STT/mic son opcionales según el modo).
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile

MODELO = os.getenv("YACHAY_CKPT", "models/nano-math/yachay-math.safetensors")
PIPER_VOICE = os.getenv("PIPER_VOICE", "es_ES-sharvard-medium.onnx")


# ---------- STT (voz -> texto) ----------
def transcribir(audio_path, modelo="base"):
    """Whisper local vía faster-whisper. Devuelve el texto reconocido."""
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        raise SystemExit("Falta faster-whisper. Instala:  pip install faster-whisper")
    m = WhisperModel(modelo, device="cpu", compute_type="int8")
    segs, _ = m.transcribe(audio_path, language="es")
    return " ".join(s.text for s in segs).strip()


def grabar(segundos, salida):
    """Graba del micrófono a un WAV (sounddevice)."""
    try:
        import sounddevice as sd
        import soundfile as sf
    except ImportError:
        raise SystemExit("Falta sounddevice/soundfile. Instala:  pip install sounddevice soundfile")
    sr = 16000
    print(f"🎙️  Grabando {segundos}s... habla ahora.")
    audio = sd.rec(int(segundos * sr), samplerate=sr, channels=1)
    sd.wait()
    sf.write(salida, audio, sr)
    return salida


# ---------- Yachay (texto -> respuesta) ----------
def responder(pregunta):
    from src.portable import cargar
    m = cargar(MODELO)
    return m.responder(pregunta, max_new_tokens=48, temperature=1.0, top_k=1,
                       seed=0, collapse_digits=True)


# ---------- TTS (texto -> voz) ----------
def hablar(texto, backend, out=None):
    if backend == "none":
        return
    if backend == "say":
        if not shutil.which("say"):
            print("[aviso] 'say' solo existe en macOS. Usa --tts piper o none.")
            return
        if out:
            subprocess.run(["say", "-o", out, texto])
        else:
            subprocess.run(["say", texto])
        return
    if backend == "piper":
        if not shutil.which("piper"):
            raise SystemExit("Falta Piper. Ver voz/README.md para instalarlo y bajar una voz.")
        wav = out or tempfile.mktemp(suffix=".wav")
        p = subprocess.Popen(["piper", "--model", PIPER_VOICE, "--output_file", wav],
                             stdin=subprocess.PIPE, text=True)
        p.communicate(texto)
        if not out:  # reproducir
            player = "afplay" if shutil.which("afplay") else ("aplay" if shutil.which("aplay") else None)
            if player:
                subprocess.run([player, wav])
        return


def limpiar(texto):
    """Quita <bos>/<eos> y la pregunta eco; deja solo lo que conviene decir."""
    for t in ("<bos>", "<eos>", "<pad>"):
        texto = texto.replace(t, "")
    # nos quedamos con la parte de la respuesta (después del primer salto de línea)
    partes = texto.strip().split("\n", 1)
    return (partes[1] if len(partes) > 1 else partes[0]).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", help="pregunta como texto (sin micrófono)")
    ap.add_argument("--audio", help="archivo de audio a transcribir")
    ap.add_argument("--live", nargs="?", const=4, type=int, help="grabar N segundos del micrófono (def 4)")
    ap.add_argument("--stt-model", default="base", help="modelo Whisper: tiny/base/small")
    ap.add_argument("--tts", default="say", choices=["say", "piper", "none"])
    ap.add_argument("--out", help="guardar la respuesta hablada en un WAV")
    args = ap.parse_args()

    # 1) obtener la pregunta (texto, archivo o micrófono)
    if args.text:
        pregunta = args.text
    elif args.audio:
        pregunta = transcribir(args.audio, args.stt_model)
    elif args.live is not None:
        wav = grabar(args.live, tempfile.mktemp(suffix=".wav"))
        pregunta = transcribir(wav, args.stt_model)
    else:
        raise SystemExit("Da --text, --audio o --live")

    print(f"🗣️  Pregunta: {pregunta}")

    # 2) Yachay responde
    cruda = responder(pregunta)
    respuesta = limpiar(cruda)
    print(f"🤖 Yachay: {respuesta}")

    # 3) hablar
    hablar(respuesta, args.tts, args.out)
    if args.out:
        print(f"🔊 Audio guardado en {args.out}")


if __name__ == "__main__":
    main()
