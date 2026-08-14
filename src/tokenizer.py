"""
Tokenizer BPE propio (vocab chico) usando la librería `tokenizers` de HuggingFace.

Entrenar:
    python -m src.tokenizer --entrena data/distill/stem.jsonl --vocab 8000
Usar en código:
    tok = TokenizerBPE.cargar("tokenizer.json"); ids = tok.encode("hola mundo")
"""
import argparse
import json
import os


class TokenizerBPE:
    def __init__(self, tk):
        self.tk = tk

    @staticmethod
    def entrenar(textos, vocab_size=8000, salida="tokenizer.json"):
        from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
        tk = Tokenizer(models.BPE(unk_token="<unk>"))
        tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
        tk.decoder = decoders.ByteLevel()   # reconstruye texto legible al decodificar
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"],
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        )
        tk.train_from_iterator(textos, trainer=trainer)
        tk.save(salida)
        print(f"Tokenizer entrenado ({tk.get_vocab_size()} tokens) -> {salida}")
        return TokenizerBPE(tk)

    @staticmethod
    def cargar(ruta="tokenizer.json"):
        from tokenizers import Tokenizer
        return TokenizerBPE(Tokenizer.from_file(ruta))

    def encode(self, texto):
        return self.tk.encode(texto).ids

    def decode(self, ids):
        return self.tk.decode(ids)

    @property
    def vocab_size(self):
        return self.tk.get_vocab_size()


def _iter_textos(ruta):
    """Lee un JSONL de destilación y produce texto (instrucción + respuesta)."""
    with open(ruta, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            yield f"{r.get('instruccion','')}\n{r.get('respuesta','')}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--entrena", required=True, help="JSONL de datos destilados")
    ap.add_argument("--vocab", type=int, default=8000)
    ap.add_argument("--salida", default="tokenizer.json")
    args = ap.parse_args()
    TokenizerBPE.entrenar(_iter_textos(args.entrena), args.vocab, args.salida)
