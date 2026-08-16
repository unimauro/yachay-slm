"""
Ingesta datasets de Hugging Face al formato propio {instruccion, respuesta}.

Alternativa (o complemento) a la destilación de un teacher: bajar miles de pares
de calidad ya existentes en español para entrenar un modelo con datos de verdad.

Requiere:  pip install datasets

Uso:
    # dataset conocido (mapeo automático de campos)
    python -m src.hf_dataset --dataset alpaca-es --n 5000 --salida data/distill/alpaca_es.jsonl

    # cualquier dataset de HF, indicando los campos a mano
    python -m src.hf_dataset --repo tatsu-lab/alpaca --instr_field instruction \
        --resp_field output --input_field input --n 2000 --salida data/distill/x.jsonl

Ver datasets en español: https://huggingface.co/datasets?language=language:es
"""
import argparse
import json
import os

# Registro de datasets conocidos: alias -> (repo, split, campos)
CONOCIDOS = {
    # Alpaca traducido al español (~52k pares instruction/input/output)
    "alpaca-es": {
        "repo": "bertin-project/alpaca-spanish",
        "split": "train",
        "instr": "instruction", "resp": "output", "input": "input",
    },
    # Dolly 15k curado multilingüe (tiene español)
    "dolly-es": {
        "repo": "argilla/databricks-dolly-15k-curated-multilingual",
        "split": "es",
        "instr": "instruction", "resp": "response", "input": "context",
    },
}


def _texto_instr(instr, inp):
    instr = (instr or "").strip()
    inp = (inp or "").strip()
    if inp:
        return f"{instr}\n{inp}"
    return instr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(CONOCIDOS), help="alias conocido")
    ap.add_argument("--repo", help="repo de HF (si no usas --dataset)")
    ap.add_argument("--split", default="train")
    ap.add_argument("--instr_field", default="instruction")
    ap.add_argument("--resp_field", default="output")
    ap.add_argument("--input_field", default=None, help="campo opcional de contexto/input")
    ap.add_argument("--n", type=int, default=5000, help="máximo de pares a guardar")
    ap.add_argument("--min_len", type=int, default=8, help="descarta respuestas muy cortas")
    ap.add_argument("--salida", default="data/distill/hf.jsonl")
    args = ap.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit("Falta la librería 'datasets'. Instala:  pip install datasets")

    if args.dataset:
        c = CONOCIDOS[args.dataset]
        repo, split = c["repo"], c["split"]
        f_instr, f_resp, f_input = c["instr"], c["resp"], c["input"]
    else:
        if not args.repo:
            raise SystemExit("Da --dataset (alias) o --repo (repo de HF).")
        repo, split = args.repo, args.split
        f_instr, f_resp, f_input = args.instr_field, args.resp_field, args.input_field

    print(f"Bajando {repo} [{split}] ...")
    ds = load_dataset(repo, split=split, streaming=True)

    os.makedirs(os.path.dirname(args.salida) or ".", exist_ok=True)
    vistos, total = set(), 0
    with open(args.salida, "w", encoding="utf-8") as out:
        for row in ds:
            instr = _texto_instr(row.get(f_instr), row.get(f_input) if f_input else None)
            resp = (row.get(f_resp) or "").strip()
            if not instr or len(resp) < args.min_len:
                continue
            clave = instr.lower()[:120]
            if clave in vistos:
                continue
            vistos.add(clave)
            out.write(json.dumps({"instruccion": instr, "respuesta": resp}, ensure_ascii=False) + "\n")
            total += 1
            if total % 500 == 0:
                print(f"{total}/{args.n}")
            if total >= args.n:
                break
    print(f"Listo: {total} pares en {args.salida}")


if __name__ == "__main__":
    main()
