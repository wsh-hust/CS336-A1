import torch
from .model import softmax


@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens=256,
             temperature=1.0, top_p=1.0, context_length=256,
             eos_token="<|endoftext|>", device=None):
    if not 0.0 < top_p <= 1.0:
        raise ValueError(f"top_p must be in (0, 1], got {top_p}")

    model.eval()
    if device is None:
        device = next(model.parameters()).device

    eos_id = None
    if eos_token is not None:
        eos_id = tokenizer.token_to_id.get(eos_token.encode("utf-8"))

    ids = tokenizer.encode(prompt)
    if not ids:
        raise ValueError("prompt must encode to at least one token")

    for _ in range(max_new_tokens):
        # 取最后生成的context_length个token作为输入，并添加batch维度为1
        x = torch.tensor(ids[-context_length:], dtype=torch.long, device=device).unsqueeze(0)

        # model(x) 输出 (1, T, V)：每个位置对整个词表的打分。[0, -1]：第 0 个 batch、最后一个时间步
        logits = model(x)[0, -1, :].float()             # 只取最后位置的分布 (V,)

        # τ→0 时 softmax 会趋近 one-hot，等价于取最大 logit 的 token。这里直接 argmax，既避免除零，也省一次 softmax
        if temperature <= 0:
            next_id = int(logits.argmax().item())
        else:
            # 温度缩放再softmax，可以减小或者增大分布的宽度，控制生成的文本的随机性
            probs = softmax(logits / temperature, dim=-1)
            if top_p < 1.0:
                # 把概率从大到小排。sorted_probs 是排好的概率，order 是“排序后第 j 个位置对应原词表的哪个 id”
                sorted_probs, order = torch.sort(probs, descending=True)
                # 累计和：cum[j] = sorted_probs[0] + ... + sorted_probs[j]。
                cum = torch.cumsum(sorted_probs, dim=-1)
                # 保留"累积概率刚超过 p"的最小集合：cum−自身 ≤ p 的才保留
                # cum是前缀和，remove就是到i之前的所有元素的概率和是否大于top_p
                remove = cum - sorted_probs > top_p
                # 如果到i之前就大于top_p，就将i设为0
                sorted_probs = sorted_probs.masked_fill(remove, 0.0)
                # 重新归一化，使概率和为1
                sorted_probs = sorted_probs / sorted_probs.sum()
                # order[j] = 原词表中概率第 j 大的那个 token 的 id
                # torch.multinomial(probs, 1)      # 加权随机抽样，概率大的更容易被抽中，返回在sorted_probs中的下标
                next_id = int(order[torch.multinomial(sorted_probs, 1)].item())
            else:
                # 普通采样
                next_id = int(torch.multinomial(probs, 1).item())

        ids.append(next_id)
        if eos_id is not None and next_id == eos_id:
            break
    return tokenizer.decode(ids)