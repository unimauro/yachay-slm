"""
CLI de inferencia portátil (NumPy puro, sin MLX). Para equipos viejos e IoT.

    python -m src.portable.run --prompt "¿por qué el cielo es azul?"
    python -m src.portable.run --chat        # modo interactivo

Requisitos mínimos: numpy, safetensors, tokenizers  (NO requiere mlx).
"""
import argparse
import os

DEMO = "models/demo/yachay-demo.safetensors"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/yachay.safetensors")
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--chat", action="store_true", help="modo interactivo")
    ap.add_argument("--max_new", type=int, default=120)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--top_k", type=int, default=40)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--collapse-digits", action="store_true",
                    help="une dígitos separados (para modelos de matemática)")
    args = ap.parse_args()

    ckpt = args.ckpt
    if not os.path.exists(ckpt) and os.path.exists(DEMO):
        print(f"[no encontré {ckpt}; uso el modelo demo incluido: {DEMO}]")
        ckpt = DEMO

    from src.portable import cargar
    m = cargar(ckpt)
    print(f"[modelo portátil cargado: {m.n_layers} capas, dim {m.dim}, "
          f"vocab {m.W['tok_emb.weight'].shape[0]} | NumPy puro]")

    def responder(p):
        return m.responder(p, args.max_new, args.temp, args.top_k, args.seed,
                           collapse_digits=args.collapse_digits)

    if args.chat:
        print("Modo chat. Escribe 'salir' para terminar.")
        while True:
            try:
                p = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if p.lower() in {"salir", "exit", "quit"}:
                break
            if p:
                print(responder(p))
    else:
        if not args.prompt:
            raise SystemExit("Da un --prompt o usa --chat")
        print(responder(args.prompt))


if __name__ == "__main__":
    main()
