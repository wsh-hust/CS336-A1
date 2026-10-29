# CS336 Assignment 1: Transformer Language Model from Scratch

A from-scratch implementation of a byte-level BPE tokenizer and a GPT-style
Transformer language model, including the full training loop and sampling
code. No `transformers` / `tiktoken` dependencies: everything is built on
`torch` plus the `regex` package.

Trained on TinyStories (10K vocabulary) on a single 6 GB laptop GPU.

## Results

| Metric | Value |
|---|---|
| BPE vocabulary / merges | 10,000 / 9,743 (vocab includes 256 byte tokens + 1 special token) |
| Tokenizer compression (TinyStories valid) | 4.041 bytes/token |
| Longest learned token | `b' accomplishment'` (15 bytes) |
| BPE training time / peak RSS | 696 s / 10.8 GiB (2.07 GiB corpus) |
| Encoding throughput | ~158k tokens/s (~0.65 MB/s) |
| Training | 20,000 steps, 163.8M tokens, 1 h 53 min |
| Training throughput | 24.1k tokens/s, 5.0 GB VRAM (RTX 3060 Laptop) |
| Validation loss / perplexity | 1.4610 / 4.31 |
| Bits per byte | 0.52 |
| Inference (no KV cache) | ~100 tokens/s, 200 tokens in 1–2 s |
| Tests | 46 passed, 2 skipped (resource-limit tests unavailable on Windows) |

## Repository layout

Each file below maps to one exercise of the assignment.

| File | What it contains | Assignment part |
|---|---|---|
| `cs336_basics/train_bpe.py` | Byte-level BPE training. Pre-tokenizes with the GPT-2 regex, splits on special tokens as hard boundaries, and uses an inverted index (`pair -> word ids`) plus `pair_count` so each merge only touches the words that contain that pair. Ties are broken by comparing the merged byte strings. | 2.4 `train_bpe` |
| `cs336_basics/train_bpe_outtime.py` | Naive reference version (rescans every word on every merge). Kept only as a baseline for profiling the optimized implementation. | 2.4 / 2.5(b) profiling |
| `cs336_basics/tokenizer.py` | `Tokenizer`: `encode`, `decode`, `encode_iterable`, and the BPE merging loop that applies merges in learned order. Handles overlapping special tokens by matching the longest first. | 2.6 `tokenizer` |
| `cs336_basics/model.py` | Building blocks: `Linear`, `Embedding`, `RMSNorm`, `swish`/`SwiGLU`, rotary position embeddings, numerically stable `softmax`, `scaled_dot_product_attention`, causal multi-head self-attention, pre-norm `TransformerBlock`, and `TransformerLM` (4 layers, 512 dim, 16 heads, 1344 FFN, 22.7M params). | Chapter 3 |
| `cs336_basics/nn_utils.py` | `cross_entropy_loss` (max-shifted log-sum-exp), `SGD`, `AdamW` (decoupled weight decay, bias correction), cosine learning-rate schedule with warmup, and global L2 gradient clipping. | Chapter 4 |
| `cs336_basics/data.py` | `get_batch`: samples contiguous `context_length` windows from a memory-mapped `uint16` token array and returns `(x, y)` shifted by one. | 5.1 `data_loading` |
| `cs336_basics/checkpoint.py` | `save_checkpoint` / `load_checkpoint`, storing model weights, optimizer state and iteration for exact resumption. | 5.2 `checkpointing` |
| `cs336_basics/train.py` | Training loop: LR schedule, gradient clipping, periodic validation, CSV logging (loss, lr, grad norm, tokens, wall clock, perplexity), periodic checkpoints, `--resume`. | 5.3 + 7.2 |
| `cs336_basics/generate.py` | Decoding: temperature scaling, top-p (nucleus) sampling, EOS stopping, context window truncation, greedy path for temperature 0. | Chapter 6 `decoding` |
| `cs336_basics/inference.py` | Interactive CLI: loads a checkpoint plus tokenizer and continues prompts typed on stdin. | 7.3 `generate` support |
| `scripts/train_tokenizer.py` | Trains a BPE tokenizer from a text file, samples peak RSS in a background thread, and serializes `{"vocab", "merges"}` with pickle. | 2.5 `train_bpe_tinystories` |
| `scripts/encode_data.py` | Encodes a text file into a `uint16` `.npy` token array using `np.fromiter` over a streaming encoder. | 2.7(d) |
| `tests/adapters.py`, `tests/test_tokenizer.py` | Adapters wiring the course tests to this implementation, plus tokenizer fixes. | test harness |

`figs/` holds the reference figures used while studying the assignment.

## Quick start

```bash
# install (uv)
uv sync

# 1. train a 10K BPE tokenizer (~12 min, ~11 GiB RAM)
python scripts/train_tokenizer.py \
  --input data/TinyStoriesV2-GPT4-train.txt \
  --vocab_size 10000 \
  --output data/ts_vocab_10k.pkl

# 2. encode the corpus into uint16 token ids
python scripts/encode_data.py \
  --input data/TinyStoriesV2-GPT4-train.txt \
  --tokenizer data/ts_vocab_10k.pkl \
  --output data/ts_train.npy

# 3. train the LM (20k steps ~ 2 h on an RTX 3060 Laptop)
python -m cs336_basics.train \
  --train_data_path data/ts_train.npy \
  --valid_data_path data/ts_valid.npy \
  --out_dir out/ts_20k \
  --max_iters 20000 --warmup_iters 1000 \
  --lr 1e-3 --min_lr 1e-4 \
  --batch_size 32 --context_length 256 \
  --d_model 512 --num_layers 4 --num_heads 16 --d_ff 1344 \
  --no_wandb

# 4. interact with the model
python -m cs336_basics.inference --ckpt out/ts_20k/ckpt_final.pt
```

`data/` and `out/` are ignored by git; the commands above regenerate them.

## Implementation notes

- **BPE without a full rescan.** Each merge removes the invalidated index
  entries for the affected words, rewrites those words in place, and
  registers the new pairs, so work is proportional to the number of words
  containing the merged pair rather than to the whole corpus.
- **Special tokens as hard boundaries.** `<|endoftext|>` is split out before
  pre-tokenization, so it never participates in merges, but still occupies a
  single token id at encode time.
- **`uint16` token arrays.** A 10K vocabulary fits in 16 bits, so the 540M
  token corpus takes 1.08 GB instead of 4.3 GB with `int64`, and `np.load(...,
  mmap_mode="r")` lets the loader page in only the batches it needs.
- **Untied output head.** Embedding and output projection are separate
  (22.7M params). Tying them would save 5.1M params and is a natural next
  experiment.
- **Per-token vs per-byte loss.** Token-level loss is not comparable across
  tokenizers, so results are also reported as bits per byte
  (`loss / (ln 2 * bytes_per_token)`).

## Known limitations

- Decoding recomputes the full prefix for every token (no KV cache), so
  latency grows quadratically with context length.
- Trained for 163.8M tokens (half of the reference budget); validation loss
  is 1.461, slightly above the 1.45 target.
- No document-level attention masking: documents are concatenated with an
  `<|endoftext|>` separator and attend across boundaries.
