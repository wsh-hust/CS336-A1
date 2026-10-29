"""Interactive sampling: type a prompt in the console, get a continuation."""
import argparse
import os
import pickle
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cs336_basics.generate import generate
from cs336_basics.model import TransformerLM
from cs336_basics.tokenizer import Tokenizer

EOS = "<|endoftext|>"


def load_model(ckpt_path, tokenizer_path, device):
    model = TransformerLM(
        vocab_size=10000,
        context_length=256,
        d_model=512,
        num_layers=4,
        num_heads=16,
        d_ff=1344,
        theta=10000.0,
    )
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    with open(tokenizer_path, "rb") as f:
        saved = pickle.load(f)
    tok = Tokenizer(saved["vocab"], saved["merges"], [EOS])
    return model, tok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="out/ts_20k/ckpt_final.pt")
    parser.add_argument("--tokenizer", default="data/ts_vocab_10k.pkl")
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok = load_model(args.ckpt, args.tokenizer, device)
    print(f"loaded {args.ckpt} on {device} | temp={args.temperature} top_p={args.top_p}")
    print("输入 prompt 后回车生成；直接回车或输入 exit 退出。")

    while True:
        try:
            prompt = input("\nprompt> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not prompt or prompt.lower() in {"exit", "quit"}:
            break

        t0 = time.perf_counter()
        text = generate(
            model,
            tok,
            prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            context_length=256,
            device=device,
        )
        elapsed = time.perf_counter() - t0
        print(text.split(EOS)[0])
        print(f"[{elapsed:.1f}s, <= {args.max_new_tokens} new tokens]")


if __name__ == "__main__":
    main()
