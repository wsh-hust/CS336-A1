# CS336 作业一：从零实现 Transformer 语言模型

从零实现的字节级 BPE 分词器与 GPT 风格 Transformer 语言模型，包含完整的
训练循环和采样代码。不依赖 `transformers` / `tiktoken`，只使用 `torch` 和
`regex`。

在单张 6GB 笔记本显卡上、使用 TinyStories 数据集（10K 词表）完成训练。

## 结果

| 指标 | 数值 |
|---|---|
| BPE 词表 / 合并次数 | 10,000 / 9,743（词表含 256 个字节 token + 1 个特殊 token） |
| 分词器压缩比（TinyStories valid） | 4.041 bytes/token |
| 最长的合并 token | `b' accomplishment'`（15 字节） |
| BPE 训练耗时 / 峰值内存 | 696 秒 / 10.8 GiB（语料 2.07 GiB） |
| 编码吞吐 | 约 158k token/s（约 0.65 MB/s） |
| 语言模型训练 | 20,000 步，163.8M token，1 小时 53 分 |
| 训练吞吐 | 24.1k token/s，显存 5.0 GB（RTX 3060 Laptop） |
| 验证集损失 / 困惑度 | 1.4610 / 4.31 |
| 每字节比特数 | 0.52 |
| 推理速度（无 KV cache） | 约 100 token/s，200 token 用时 1–2 秒 |
| 测试 | 46 通过，2 跳过（Windows 不支持资源限制类测试） |

## 代码文件与作业对应关系

每个文件对应作业中的一项习题。

| 文件 | 内容 | 对应习题 |
|---|---|---|
| `cs336_basics/train_bpe.py` | 字节级 BPE 训练。用 GPT-2 正则做预分词，把特殊 token 当作硬边界；用倒排索引（`pair -> 单词 id`）配合 `pair_count`，每次合并只更新包含该 token 对的单词。出现次数相同时比较合并后的字节串决定先后。 | 2.4 `train_bpe` |
| `cs336_basics/train_bpe_outtime.py` | 朴素参考实现（每次合并都重新扫描全部单词），仅作为 profiling 时对比的基线保留。 | 2.4 / 2.5(b) 性能分析 |
| `cs336_basics/tokenizer.py` | `Tokenizer`：`encode`、`decode`、`encode_iterable`，以及按训练顺序应用合并的 BPE 循环；特殊 token 按“长的优先”匹配，避免短串抢先命中。 | 2.6 `tokenizer` |
| `cs336_basics/model.py` | 模型基础构件：`Linear`、`Embedding`、`RMSNorm`、`SwiGLU`、旋转位置编码 RoPE、数值稳定的 `softmax`、`scaled_dot_product_attention`、因果多头自注意力、Pre-Norm 的 `TransformerBlock`，以及 `TransformerLM`（4 层、512 维、16 头、FFN 1344，约 22.7M 参数）。 | 第 3 章 |
| `cs336_basics/nn_utils.py` | `cross_entropy_loss`（减最大值后再做 log-sum-exp）、`SGD`、`AdamW`（解耦权重衰减 + 偏差修正）、带预热的余弦学习率调度、全局 L2 梯度裁剪。 | 第 4 章 |
| `cs336_basics/data.py` | `get_batch`：在内存映射（mmap）的 `uint16` token 数组上随机取连续窗口，返回错开一位的 `(x, y)`。 | 5.1 `data_loading` |
| `cs336_basics/checkpoint.py` | `save_checkpoint` / `load_checkpoint`，保存模型权重、优化器状态和迭代步数，支持精确续训。 | 5.2 `checkpointing` |
| `cs336_basics/train.py` | 训练循环：学习率调度、梯度裁剪、定期验证、CSV 日志（loss、lr、梯度范数、token 数、墙钟时间、困惑度）、定期保存检查点、`--resume` 断点续训。 | 5.3 + 7.2 |
| `cs336_basics/generate.py` | 解码：温度缩放、top-p（核采样）、遇到 EOS 停止、上下文截断，温度为 0 时退化为贪心解码。 | 第 6 章 `decoding` |
| `cs336_basics/inference.py` | 交互式命令行：加载检查点和分词器，读取终端输入的 prompt 并续写。 | 7.3 `generate` 配套 |
| `scripts/train_tokenizer.py` | 从文本文件训练 BPE 分词器，用后台线程采样峰值内存，并把 `{"vocab", "merges"}` 序列化为 pickle。 | 2.5 `train_bpe_tinystories` |
| `scripts/encode_data.py` | 用流式编码器把文本文件编码成 `uint16` 的 `.npy` token 数组。 | 2.7(d) |
| `tests/adapters.py`、`tests/test_tokenizer.py` | 把课程测试接到本实现的适配层，以及分词器的测试修正。 | 测试 |

`figs/` 存放学习过程中使用的参考插图。

## 快速开始

```bash
# 安装依赖（uv）
uv sync

# 1. 训练 10K 词表的 BPE 分词器（约 12 分钟，峰值内存约 11 GiB）
python scripts/train_tokenizer.py \
  --input data/TinyStoriesV2-GPT4-train.txt \
  --vocab_size 10000 \
  --output data/ts_vocab_10k.pkl

# 2. 把语料编码成 uint16 的 token id 数组
python scripts/encode_data.py \
  --input data/TinyStoriesV2-GPT4-train.txt \
  --tokenizer data/ts_vocab_10k.pkl \
  --output data/ts_train.npy

# 3. 训练语言模型（20k 步，RTX 3060 Laptop 约 2 小时）
python -m cs336_basics.train \
  --train_data_path data/ts_train.npy \
  --valid_data_path data/ts_valid.npy \
  --out_dir out/ts_20k \
  --max_iters 20000 --warmup_iters 1000 \
  --lr 1e-3 --min_lr 1e-4 \
  --batch_size 32 --context_length 256 \
  --d_model 512 --num_layers 4 --num_heads 16 --d_ff 1344 \
  --no_wandb

# 4. 与模型交互
python -m cs336_basics.inference --ckpt out/ts_20k/ckpt_final.pt
```

`data/` 和 `out/` 已被 git 忽略，按上面的命令可自行重新生成。

## 实现要点

- **BPE 不做全量重扫。** 每次合并时先撤销受影响单词的旧 token 对贡献，原地改写这些
  单词，再登记新的 token 对。计算量正比于“包含该 token 对的单词数”，而不是整个语料。
- **特殊 token 是硬边界。** `<|endoftext|>` 在预分词前就被切出来，不参与任何合并，
  但在编码时仍占一个独立的 token id。
- **`uint16` token 数组。** 10K 词表用 16 位就够，5.4 亿 token 的语料只占 1.08 GB
  （用 `int64` 需要 4.3 GB）；配合 `np.load(..., mmap_mode="r")`，加载时只把需要
  的批次读进内存。
- **输出头与词嵌入不共享权重。** 当前实现两者独立（共 22.7M 参数）；共享可省下
  5.1M 参数，是一个很自然的后续实验。
- **per-token 损失不可跨词表比较。** 因此同时报告每字节比特数
  （`loss / (ln 2 * bytes_per_token)`），保证换词表后仍可比。

## 已知局限

- 解码时每个新 token 都重新前向整条前缀（没有 KV cache），延迟随上下文长度平方增长。
- 只训练了 163.8M token（参考预算的一半），验证损失 1.461，略高于 1.45 的目标。
- 没有文档级注意力掩码：文档之间用 `<|endoftext|>` 分隔，但仍会跨文档互相注意。
