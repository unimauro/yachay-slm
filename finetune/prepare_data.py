"""
Prepara datos para fine-tuning con mlx-lm.

Convierte nuestro JSONL {instruccion, respuesta} al formato de chat que espera
mlx-lm (una carpeta con train.jsonl y valid.jsonl, cada línea con "messages").

Uso:
    python finetune/prepare_data.py --datos data/distill/alpaca_es.jsonl \
        --salida finetune/data --val 0.05
"""
import argparse
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datos", required=True, help="JSONL con instruccion/respuesta")
    ap.add_argument("--salida", default="finetune/data", help="carpeta destino")
    ap.add_argument("--val", type=float, default=0.05, help="fracción de validación")
    ap.add_argument("--sistema", default="Eres un asistente útil que responde en español de forma clara y correcta.")
    args = ap.parse_args()

    filas = []
    with open(args.datos, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            instr = (r.get("instruccion") or "").strip()
            resp = (r.get("respuesta") or "").strip()
            if not instr or not resp:
                continue
            filas.append({
                "messages": [
                    {"role": "system", "content": args.sistema},
                    {"role": "user", "content": instr},
                    {"role": "assistant", "content": resp},
                ]
            })

    if not filas:
        raise SystemExit("No se encontraron pares válidos en la entrada.")

    n_val = max(1, int(len(filas) * args.val))
    val, train = filas[:n_val], filas[n_val:]

    os.makedirs(args.salida, exist_ok=True)
    for nombre, datos in [("train", train), ("valid", val)]:
        ruta = os.path.join(args.salida, f"{nombre}.jsonl")
        with open(ruta, "w", encoding="utf-8") as out:
            for d in datos:
                out.write(json.dumps(d, ensure_ascii=False) + "\n")
        print(f"{nombre}: {len(datos)} ejemplos -> {ruta}")


if __name__ == "__main__":
    main()
