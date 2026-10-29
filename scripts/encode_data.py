"""Encode a UTF-8 text file into a uint16 token-id .npy file."""
import argparse
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cs336_basics.tokenizer import Tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--tokenizer", required=True, help="pickle with {'vocab':..., 'merges':...}")
    parser.add_argument("--output", required=True)
    parser.add_argument("--special", nargs="*", default=["<|endoftext|>"])
    args = parser.parse_args()

    with open(args.tokenizer, "rb") as f:
        saved = pickle.load(f)

    tok = Tokenizer(saved["vocab"], saved["merges"], args.special)

    with open(args.input, encoding="utf-8") as f:
        tokens = np.fromiter(tok.encode_iterable(f), dtype=np.uint16)

    np.save(args.output, tokens)
    print(f"tokens={len(tokens):,} dtype={tokens.dtype} max_id={int(tokens.max())}")
    print(f"saved -> {args.output}")


if __name__ == "__main__":
    main()
