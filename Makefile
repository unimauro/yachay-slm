# Yachay SLM — atajos. Uso: make <objetivo>
# Detecta un venv en .venv si existe; si no, usa python del sistema.
PY := $(shell [ -x .venv/bin/python ] && echo .venv/bin/python || echo python3)
PIP := $(shell [ -x .venv/bin/pip ] && echo .venv/bin/pip || echo pip3)

DATA ?= data/samples/general.jsonl
PRESET ?= nano
STEPS ?= 1500

.PHONY: help
help:
	@echo "Objetivos:"
	@echo "  make setup            venv + dependencias de ENTRENAMIENTO (Mac Apple Silicon, MLX)"
	@echo "  make setup-portable   solo dependencias de INFERENCIA (cualquier equipo, sin MLX)"
	@echo "  make demo             entrena un modelo diminuto end-to-end con datos incluidos (sin API key)"
	@echo "  make chat             chat interactivo con el modelo (motor portátil NumPy)"
	@echo "  make ask Q=\"...\"      una pregunta al modelo (motor portátil)"
	@echo "  make train            entrena (DATA=... PRESET=nano|micro|mini STEPS=...)"
	@echo "  make tokenizer        entrena el tokenizer BPE sobre DATA"
	@echo "  make distill DOM=...  destila datos de un teacher (requiere OPENROUTER_API_KEY)"
	@echo "  make test             smoke test del modelo + chequeo de paridad MLX vs NumPy"
	@echo "  make clean            borra checkpoints y tokenizer locales"

.PHONY: setup
setup:
	python3 -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt

.PHONY: setup-portable
setup-portable:
	python3 -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements-portable.txt

.PHONY: tokenizer
tokenizer:
	$(PY) -m src.tokenizer --entrena $(DATA) --salida tokenizer.json

.PHONY: demo
demo: tokenizer
	$(PY) -m src.train --datos $(DATA) --preset $(PRESET) --max_steps $(STEPS)
	@echo "\nListo. Prueba:  make chat   (o)   make ask Q=\"¿por qué el cielo es azul?\""

.PHONY: train
train:
	$(PY) -m src.train --datos $(DATA) --preset $(PRESET) --max_steps $(STEPS)

.PHONY: chat
chat:
	$(PY) -m src.portable.run --chat

.PHONY: ask
ask:
	$(PY) -m src.portable.run --prompt "$(Q)"

.PHONY: distill
distill:
	$(PY) -m src.distill --dominio "$(DOM)" --salida data/distill/dataset.jsonl

.PHONY: test
test:
	$(PY) -m src.model
	$(PY) scripts/check_parity.py

.PHONY: clean
clean:
	rm -rf checkpoints tokenizer.json
