# Cómo ejecutar Yachay en tu máquina — Windows · Linux · macOS

Guía por sistema operativo. **Probar / ejecutar** el modelo funciona en los tres
(trae un modelo demo incluido). **Entrenar desde cero** solo corre en Mac Apple
Silicon (MLX). Aquí está todo el detalle.

## Qué funciona en cada sistema

| Tarea | Windows | Linux | macOS |
|---|:---:|:---:|:---:|
| **Ejecutar** el modelo (motor NumPy) | ✅ | ✅ | ✅ |
| **Ejecutar** el modelo (binario Rust) | ✅ | ✅ | ✅ |
| **Entrenar** desde cero (MLX) | ❌¹ | ❌¹ | ✅ (Apple Silicon) |
| **Fine-tune** general (mlx-lm) | ❌¹ | ❌¹ | ✅ (Apple Silicon) |

¹ MLX es exclusivo de Apple Silicon. En Windows/Linux se puede **ejecutar** cualquier
modelo ya entrenado (con NumPy o Rust); para *entrenar* ahí haría falta un backend
en PyTorch (está en el backlog).

Requisito común para la vía Python: **Python 3.9 o más nuevo**.

---

## 🅰️ Ejecutar el modelo (vía Python + NumPy) — los tres sistemas

Trae un modelo demo, no necesita GPU ni API key.

### Windows (PowerShell)

```powershell
git clone https://github.com/unimauro/yachay-slm
cd yachay-slm
py -3 -m venv .venv
.venv\Scripts\Activate.ps1        # si PowerShell lo bloquea, usa:  .venv\Scripts\activate.bat
pip install -r requirements-portable.txt
python -m src.portable.run --chat
```

> Instala Python desde https://python.org y marca **"Add python.exe to PATH"**.

### Linux

```bash
git clone https://github.com/unimauro/yachay-slm
cd yachay-slm
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-portable.txt
python -m src.portable.run --chat
```

> Si falta venv:  `sudo apt install python3-venv python3-pip`  (Debian/Ubuntu).

### macOS

```bash
git clone https://github.com/unimauro/yachay-slm
cd yachay-slm
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-portable.txt
python -m src.portable.run --chat
```

**Uso:** escribe tu pregunta y Enter; `salir` para terminar. Para una sola pregunta:

```
python -m src.portable.run --prompt "¿por qué el cielo es azul?"
```

---

## 🅱️ Ejecutar el modelo (binario Rust) — los tres sistemas

Más rápido y sin Python en el equipo. Necesitas el toolchain de Rust
(https://rustup.rs — un instalador para los tres sistemas).

```bash
# desde la raíz del repo
cd rust
cargo build --release
```

Correr el binario:

- **Linux / macOS:** `./target/release/yachay --chat`
- **Windows (PowerShell):** `.\target\release\yachay.exe --chat`

Una sola pregunta: agrega `--prompt "tu pregunta"`.

> **Cross-compilar a ARM** (Raspberry Pi, RK3588): ver [`rust/README.md`](rust/README.md).

---

## 🍎 Entrenar tu propio modelo (solo macOS · Apple Silicon)

Requiere MLX, que solo corre en chips Apple (M1/M2/M3…).

```bash
git clone https://github.com/unimauro/yachay-slm
cd yachay-slm
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # incluye MLX

make demo        # entrena un modelo diminuto con datos de ejemplo (sin API key)
make chat        # conversa con lo que entrenaste
```

El fine-tune general (Yachay-General) también es solo Mac; ver
[`finetune/README.md`](finetune/README.md).

---

## Problemas comunes

- **Windows: `python` no se reconoce** → reinstala Python marcando "Add to PATH",
  o usa `py` en lugar de `python`.
- **Windows: PowerShell bloquea el activate** → ejecuta
  `Set-ExecutionPolicy -Scope Process RemoteSigned` y reintenta, o usa
  `.venv\Scripts\activate.bat`.
- **Linux: `error: externally-managed-environment`** al instalar con pip → siempre
  usa el venv (`source .venv/bin/activate`) antes de `pip install`.
- **`command not found: cargo`** → instala Rust desde https://rustup.rs y abre una
  terminal nueva.
- **La primera respuesta tarda** → es normal: carga el modelo en memoria una vez.
