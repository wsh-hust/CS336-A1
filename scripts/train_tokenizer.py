"""Train a byte-level BPE tokenizer, record time/memory, and serialize the result."""
import argparse
import os
import pickle
import sys
import threading
import time

import psutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cs336_basics.train_bpe import train_bpe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--vocab_size", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--special", nargs="*", default=["<|endoftext|>"])
    args = parser.parse_args()

    stop = threading.Event()
    peak = [0]
    proc = psutil.Process()

    def sample():
        while not stop.wait(0.5):
            peak[0] = max(peak[0], proc.memory_info().rss)

    thread = threading.Thread(target=sample, daemon=True)
    thread.start()

    t0 = time.perf_counter()
    vocab, merges = train_bpe(args.input, args.vocab_size, args.special)
    elapsed = time.perf_counter() - t0

    stop.set()
    thread.join()

    with open(args.output, "wb") as f:
        pickle.dump({"vocab": vocab, "merges": merges}, f)

    longest = max(vocab.values(), key=len)
    print(f"elapsed={elapsed:.1f}s peak_rss={peak[0] / 2**30:.2f}GiB")
    print(f"vocab={len(vocab)} merges={len(merges)}")
    print(f"longest_token={longest!r} bytes={len(longest)}")
    print(f"saved -> {args.output}")


if __name__ == "__main__":
    main()
