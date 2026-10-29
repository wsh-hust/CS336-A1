"""
训练模型
"""
import argparse
import csv
import math
import os
import time
import wandb  # 建议安装：pip install wandb

import numpy as np
import torch

from cs336_basics.checkpoint import load_checkpoint, save_checkpoint
from cs336_basics.data import get_batch
from cs336_basics.model import TransformerLM
from cs336_basics.nn_utils import (
    AdamW,
    clip_gradient_norm,
    cross_entropy_loss,
    get_lr_cosine_schedule,
)

LOG_HEADER = ["iteration", "split", "loss", "lr", "grad_norm", "tokens", "elapsed_s", "perplexity"]


def log_row(path, row):
    is_new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(LOG_HEADER)
        writer.writerow(row)


def total_grad_norm(parameters):
    grads = [p.grad.detach() for p in parameters if p.grad is not None]
    if not grads:
        return 0.0
    return torch.sqrt(sum((g * g).sum() for g in grads)).item()


@torch.no_grad()
def evaluate(model, data, args):
    model.eval()
    rng_state = np.random.get_state()
    np.random.seed(args.seed + 1)  # 固定验证集采样，保证曲线可比
    total = 0.0
    for _ in range(args.eval_batches):
        x, y = get_batch(data, args.batch_size, args.context_length, args.device)
        total += cross_entropy_loss(model(x), y).item()
    np.random.set_state(rng_state)
    model.train()
    return total / args.eval_batches


def main():
    parser = argparse.ArgumentParser()
    # --- 模型基础超参数 ---
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--d_ff", type=int, default=2048)
    parser.add_argument("--vocab_size", type=int, default=10000)

    # --- 消融开关 ---
    # 移除 RoPE (NoPE)：theta=None 时 TransformerBlock 不初始化旋转位置编码
    parser.add_argument("--no_rope", action="store_true", help="Disable Rotary Positional Embeddings")

    # --- 优化器超参数 ---
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--min_lr", type=float, default=6e-5)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--max_norm", type=float, default=1.0)
    parser.add_argument("--max_iters", type=int, default=10000)
    parser.add_argument("--warmup_iters", type=int, default=1000)
    parser.add_argument("--rope_theta", type=float, default=10000.0)
    parser.add_argument("--seed", type=int, default=0)

    # --- 日志 ---
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--eval_interval", type=int, default=200)
    parser.add_argument("--eval_batches", type=int, default=10)
    parser.add_argument("--checkpoint_interval", type=int, default=1000)
    parser.add_argument("--log_csv", type=str, default=None)

    # --- 路径与系统 ---
    parser.add_argument("--train_data_path", type=str, required=True)
    parser.add_argument("--valid_data_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="out")
    parser.add_argument("--resume", type=str, default=None, help="从指定 checkpoint 恢复训练")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # --- WandB 设置 ---
    parser.add_argument("--wandb_project", type=str, default="cs336-pretraining")
    parser.add_argument("--run_name", type=str, default=None, help="WandB 实验名称")
    parser.add_argument("--no_wandb", action="store_true", help="关闭 wandb，只输出到控制台")

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)

    # 1. 加载数据 (使用 mmap)
    # 数据需用 np.save 存成 dtype=uint16 的 .npy 文件
    if not os.path.exists(args.train_data_path):
        raise FileNotFoundError(f"Training data not found at {args.train_data_path}")
    if not os.path.exists(args.valid_data_path):
        raise FileNotFoundError(f"Validation data not found at {args.valid_data_path}")

    train_data = np.load(args.train_data_path, mmap_mode="r")
    val_data = np.load(args.valid_data_path, mmap_mode="r")

    if len(train_data) <= args.context_length + 1:
        raise ValueError(f"Training data too short for context_length={args.context_length}")
    if len(val_data) <= args.context_length + 1:
        raise ValueError(f"Validation data too short for context_length={args.context_length}")

    print(f"训练集大小: {len(train_data)} tokens")
    print(f"验证集大小: {len(val_data)} tokens")

    # 2. 处理消融实验逻辑
    # 如果 no_rope 为 True，则 theta 设为 None，TransformerBlock 内部就不会初始化 RoPE
    actual_rope_theta = None if args.no_rope else args.rope_theta

    # 3. 初始化模型
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        theta=actual_rope_theta,
        device=args.device,
    ).to(args.device)

    print(f"Model Config: RoPE={not args.no_rope}")

    # 4. 初始化优化器
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay,
    )

    # 5. 检查点恢复逻辑
    start_iter = 0
    ckpt_path = os.path.join(args.out_dir, "ckpt.pt")
    if args.resume is not None:
        start_iter = load_checkpoint(args.resume, model, optimizer)
        print(f"Resuming from iteration {start_iter}")

    # 6. 初始化 WandB 监控
    use_wandb = not args.no_wandb
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.run_name,
            config=vars(args),
        )

    log_path = args.log_csv or os.path.join(args.out_dir, "log.csv")
    t_begin = time.time()
    last_log_time = t_begin
    last_log_tokens = 0
    tokens_seen = 0

    # 7. 主训练循环
    for it in range(start_iter, args.max_iters):
        # A. 更新学习率
        lr = get_lr_cosine_schedule(it, args.lr, args.min_lr, args.warmup_iters, args.max_iters)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # B. 训练步
        model.train()
        x, y = get_batch(train_data, args.batch_size, args.context_length, args.device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = cross_entropy_loss(logits, y)

        loss.backward()

        grad_norm = total_grad_norm(model.parameters()) if it % args.log_interval == 0 else float("nan")

        # 梯度裁剪
        clip_gradient_norm(model.parameters(), args.max_norm)

        optimizer.step()

        tokens_seen += x.numel()

        # C. 训练日志
        if it % args.log_interval == 0:
            now = time.time()
            dt = max(now - last_log_time, 1e-9)
            tok_per_s = (tokens_seen - last_log_tokens) / dt
            loss_val = loss.item()
            print(f"Iter {it}: train_loss {loss_val:.4f}, lr {lr:.2e}, grad_norm {grad_norm:.3f}, {tok_per_s:,.0f} tok/s")
            log_row(log_path, [it, "train", f"{loss_val:.6f}", f"{lr:.6e}", f"{grad_norm:.6f}", tokens_seen, f"{now - t_begin:.1f}", ""])
            last_log_time, last_log_tokens = now, tokens_seen

        # D. 验证
        if it % args.eval_interval == 0 or it == args.max_iters - 1:
            v_loss = evaluate(model, val_data, args)
            ppl = math.exp(v_loss)
            print(f"Iter {it}: val_loss {v_loss:.4f}, perplexity {ppl:.2f}")
            log_row(log_path, [it, "val", f"{v_loss:.6f}", f"{lr:.6e}", "", "", f"{time.time() - t_begin:.1f}", f"{ppl:.3f}"])
            if use_wandb:
                wandb.log({"train/loss": loss.item(), "val/loss": v_loss, "lr": lr, "iter": it + 1})

        # E. 保存检查点
        if (it + 1) % args.checkpoint_interval == 0 or it == args.max_iters - 1:
            save_checkpoint(model, optimizer, it + 1, ckpt_path)

    # 训练结束保存最终模型
    save_checkpoint(model, optimizer, args.max_iters, os.path.join(args.out_dir, "ckpt_final.pt"))
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
