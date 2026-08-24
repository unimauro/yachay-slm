# Yachay con voz — escuchar y hablar (local y soberano)

Convierte a Yachay en un **tutor que escucha y habla**, conectando piezas
abiertas y locales al cerebro (tu propio modelo):

```
micrófono/archivo → [Whisper: STT] → texto → [Yachay] → respuesta → [Piper/say: TTS] → audio
```

Todo corre **on-device**, sin nube. Cada pieza es chica y abierta.

## Instalar

```bash
pip install -r voz/requirements.txt      # STT (Whisper) + micrófono opcional
```

Para hablar (TTS) elige un backend:
- **macOS (demo rápida):** usa el comando nativo `say` (ya viene con el sistema).
- **Piper (despliegue, Raspberry Pi, voz en español):**
  ```bash
  # binario: https://github.com/rhasspy/piper/releases
  # voz español: https://huggingface.co/rhasspy/piper-voices  (es_ES / es_MX)
  # deja el .onnx junto y exporta PIPER_VOICE=ruta/voz.onnx
  ```

## Usar

```bash
# escribir la pregunta (sin micrófono), Yachay la dice en voz alta (macOS)
python -m voz.talk --text "¿cuánto es 347 más 285?" --tts say

# transcribir un archivo de audio y responder
python -m voz.talk --audio pregunta.wav --tts piper --out respuesta.wav

# grabar del micrófono 4 segundos y responder
python -m voz.talk --live 4 --tts say

# solo texto, sin hablar
python -m voz.talk --text "58 + 67" --tts none
```

Opciones: `--stt-model tiny|base|small` (más grande = más preciso, más lento),
`--out archivo.wav` (guarda la respuesta hablada).

Por defecto usa el **Nano de matemática** (`models/nano-math/`). Cambia el cerebro
con `YACHAY_CKPT=ruta/modelo.safetensors`.

## Verificado

Round-trip probado sin micrófono: se generó una pregunta hablada, **Whisper** la
transcribió (*"¿Cuánto es 58 más 67?"*) y **Yachay** respondió correctamente
(*"58 + 67 = 125"*). El habla se verificó con `say` (audio generado).

## En Raspberry Pi / PC vieja

- STT: usa `whisper.cpp` con el modelo `tiny`/`base` (más liviano que faster-whisper).
- TTS: **Piper** corre muy bien en Raspberry Pi con voces en español (~20-60 MB).
- Cerebro: el Nano corre en NumPy o en el binario Rust (`rust/`).

Resultado: un tutor que escucha y habla, **sin internet y sin costo por uso**.
