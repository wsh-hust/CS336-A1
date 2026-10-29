import torch
import torch.nn as nn
import math
from collections.abc import Iterable


"""
交叉熵损失函数
假设经过transformerlm处理后，某个token得到的vocab大小的输出分数是o，则损失：
公式：loss = -log(softmax(o_y)) = -log p_y
     = -log( exp(o_y) / sum_k exp(o_k) )
     = log(sum_k exp(o_k)) - log(exp(o_y))
     = log(sum_k exp(o_k)) - o_y
假设输出o中最大值是m，为了数值稳定性，我们对o_k进行归一化，即o_k = o_k - m
则有：
log(sum_k exp(o_k)) = log(sum_k exp(o_k - m+m))
     = log(sum_k exp(o_k - m)) + m
loss = log(sum_k exp(o_k - m)) - o_y + m
其中y是真实标签，o_y是模型输出的logits
对于一个长度为i的序列，经过transformerLM模块后，得到(B,S,D)的矩阵，然后再通过linear层，得到一个(B,S,V)的矩阵，其中V是词汇表大小，这个矩阵的每一行就是对应输入token对该token的下一个token的预测得分。然后在交叉熵损失函数中，根据真实标签y(也就是正确的下一个token的id)计算loss
为什么是负 log？
因为要最大化概率，而不是最小化概率。
p_y = 1      -> loss = 0       完全正确
p_y = 0.5    -> loss = 0.693   还有改进空间
看讲义,公式推导
"""
def cross_entropy_loss(logits, targets):
    """
    计算交叉熵损失函数
    """
    # 首先取出o_y，也就是正确标签对应位置的得分logits
    target_logits = torch.gather(logits, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)

    # 计算每行logits的最大值
    max_logits = torch.max(logits, dim=-1, keepdim=True).values
    
    # 对logits进行数值稳定性处理,这样求exp时不会溢出
    logits = logits - max_logits
    
    # 计算loss
    # 首先计算log(sum_k exp(o_k - m))，max_logits提取出来后形状是 (B, S, 1)，求和项形状是 (B, S)，相加时需先 squeeze
    log_sum_exp = max_logits.squeeze(-1) + torch.log(torch.sum(torch.exp(logits), dim=-1))
    # 计算loss
    loss = log_sum_exp - target_logits
    return loss.mean()


"""
优化器：
SGD：
更新: θ ← θ − α·g
优点:
  - 实现最简单，无额外状态（显存 ≈ 参数 + 梯度）
  - 超参少，只有 α；配上 momentum 后泛化往往很好
缺点:
  - 所有参数、所有方向共用一个 α：曲率大的方向步子过大 → 来回震荡；
    曲率小的方向步子过小 → 收敛慢（病态/条件数问题）
  - 对 α 和参数尺度极敏感，α 取 1e1~1e3 会直接发散（题 4.2.1 的实验）
  - 只用当前 batch 的梯度，噪声大，无动量平滑

Adam
更新:
  m_t = β₁·m_(t−1) + (1−β₁)·g
  v_t = β₂·v_(t−1) + (1−β₂)·g²
  m̂ = m_t/(1−β₁ᵗ),  v̂ = v_t/(1−β₂ᵗ)        (偏差修正)
  θ ← θ − α·m̂/(√v̂ + ε)
优点:
  - 动量（一阶矩）平滑梯度噪声
  - 逐参数自适应步长（二阶矩）：梯度大的维度自动减小步长，
    梯度小/稀疏的维度自动放大，对参数尺度差异鲁棒
  - 调参相对省心，是 Transformer 训练的默认选择
缺点:
  - 每个参数额外存 m、v：优化器状态显存约为参数量的 2 倍
    （float32 下参数+梯度+m+v ≈ 16 bytes/参数）
  - 超参更多：β₁、β₂、ε、λ
  - 某些视觉任务上泛化不如 SGD + momentum
  - 把 L2 正则 λθ 加进梯度时，衰减会被 1/√v̂ 缩放，
    参数梯度大 → 衰减被削弱，正则效果不一致

AdamW
更新: 与 Adam 相同，只把权重衰减从梯度里拿出来，直接作用在参数上
  θ ← θ − α·λ·θ          (实现上: p.mul_(1 − α·λ))
  ... 再算 m、v，做自适应更新 θ ← θ − α_t·m/(√v + ε)
  (题目里 α_t = α·√(1−β₂ᵗ)/(1−β₁ᵗ)，t 从 1 开始)
优点:
  - 继承 Adam 的全部优点（动量 + 自适应）
  - 衰减强度不再被 1/√v̂ 缩放，正则效果稳定可控
  - 大语言模型（GPT-3、LLaMA 等）的标准选择
缺点:
  - 与 Adam 相同的显存开销和超参数数量
"""

# 这是优化后的sgd，是带学习率衰减的 SGD
# 多了一个时间步数 t，用于计算学习率衰减因子
# 早期: 步长大 → 快速下降
# 后期: 步长小 → 减少在最优解附近的震荡，做更细的收敛
class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        """
        params: 传入模型需要优化的参数（通常是 model.parameters()）。
        defaults: 一个字典，存储默认超参数（如学习率 lr）。
        调用 super().__init__ 后，PyTorch 会将参数组织在 self.param_groups 中
        """
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        # 可以自己设置额外参数，务必把 `params` 传给基类的 `__init__` 方法，基类会把这些参数保存下来供 `step` 使用。你可以根据优化器的需要接受额外参数（例如学习率就是一个常见参数），并把它们以字典的形式传给基类构造函数，字典的键是你为这些参数起的名字（字符串）。
        # params 是“优化谁”（张量本身），defaults 是“怎么优化”（超参数的默认值）。两者进入基类后存放的位置和用途完全不同。
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self):
        loss = None
        # 遍历每个参数组，从组里取当前的 lr。 为什么：支持“embedding 用 1e-4、主干用 1e-3”这种分组配置；另外学习率调度器（cosine schedule）改的就是 group["lr"]，所以每步都要重新从组里读，不能缓存成 self.lr。
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                # 如果参数没有梯度，直接跳过
                if p.grad is None:
                    continue
                
                # 获取该参数对应的状态字典（用于记录步数 t）
                state = self.state[p]
                if len(state) == 0:
                    state["t"] = 0
                
                t = state["t"]
                grad = p.grad.data
                
                # 执行 PDF 中的更新公式 (20): p = p - (lr / sqrt(t+1)) * grad
                p.data -= lr / math.sqrt(t + 1) * grad
                
                # 更新步数
                state["t"] += 1
        return loss




"""
AdamW 优化器
在adam的基础上，添加了权重衰减，避免了梯度消失问题，提高了模型的泛化能力。
什么是权重衰减？
权重衰减是一种正则化技术，用于防止模型过拟合。在训练过程中，每个参数的梯度会被乘以一个权重衰减因子，这个因子通常是一个小的常数（例如 0.01）。这个因子的目的是惩罚参数的大小，从而防止模型学习过拟合的特征。
也是希望参数不要太大
看讲义

"""
class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        # 1. 基本参数检查​
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")

        # 2. 将超参数存入 defaults 字典​
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
    @torch.no_grad()
    def step(self):
        """执行单步优化更新"""
        loss = None

        for group in self.param_groups:
            # 分组读，因为每个参数组可能有不同的超参数
            beta1, beta2 = group['betas']
            eps = group['eps']
            lr = group['lr']
            wd = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                state = self.state[p]

                # 3. 状态初始化 (第一次运行步时执行)​
                if len(state) == 0:
                    state['step'] = 0
                    # m: 一阶矩 (梯度的指数移动平均)​
                    state['m'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # v: 二阶矩 (梯度平方的指数移动平均)​
                    state['v'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                m, v = state['m'], state['v']
                state['step'] += 1
                t = state['step']

                # 先计算权重衰减量
                # theta = theta - alpha * lambda * theta
                if wd != 0:
                    p.mul_(1 - lr * wd)

                # 4. 更新矩估计 (Algorithm 1)​
                # m = beta1 * m + (1 - beta1) * g​
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                # v = beta2 * v + (1 - beta2) * g^2​
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2) 

                # 5. 计算偏差校正后的学习率 alpha_t​
                # 这一步是为了消除初始值为 0 带来的偏移​
                bias_correction1 = 1 - beta1 ** t
                bias_correction2 = 1 - beta2 ** t
                step_size = lr * (math.sqrt(bias_correction2) / bias_correction1)

                # 6. 更新参数：theta = theta - alpha_t * m / (sqrt(v) + eps)​
                denom = v.sqrt().add_(eps)
                # 这是一个专门为优化器设计的复合算子，名字可以拆解为：add (加) + constant (常数) + div (除)。​
                # p.addcdiv_(tensor1, tensor2, value=1.0)。 p=p+value×( tensor1 / tensor2 )​
                p.addcdiv_(m, denom, value=-step_size)
                
        return loss



"""
学习率调度schedule
带预热的余弦退火学习率调度
画一下函数图像：先线性增加到max，然后余弦退火到min，之后都保持在min
"""
def get_lr_cosine_schedule(
    it: int, 
    max_learning_rate: float, 
    min_learning_rate: float, 
    warmup_iters: int, 
    cosine_cycle_iters: int
) -> float:
    """
    计算带预热的余弦退火学习率。
    
    it: 当前迭代次数 (t)
    max_learning_rate: 最大学习率 (alpha_max)
    min_learning_rate: 最小学习率 (alpha_min)
    warmup_iters: 预热步数 (T_w)
    cosine_cycle_iters: 总退火步数 (T_c)
    """
    
    # 1. 预热阶段：线性增长
    if it < warmup_iters:
        return max_learning_rate * it / warmup_iters
    
    # 2. 退火后阶段：保持最小学习率
    if it > cosine_cycle_iters:
        return min_learning_rate
    
    # 3. 余弦退火阶段
    # 计算当前在退火阶段的进度 (0.0 到 1.0)
    decay_ratio = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
    
    # 计算余弦系数：从 1.0 降到 0.0。公式就是这样子的
    # math.cos(math.pi * decay_ratio) 的范围是 [1, -1]
    # 我们需要映射到 [0, 1] 范围，所以乘以 0.5 并加上 0.5
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    
    # 最终学习率 = 最小值 + 系数 * (最大值 - 最小值)
    return min_learning_rate + coeff * (max_learning_rate - min_learning_rate)



"""
梯度裁剪:在每次反向传播之后、优化器步进（Step）之前
当网络层数过多时，求出loss并对底层参数进行求导计算梯度时，因为会通过很多个链式求导，如果这些中间项都大于1，就会导致梯度爆炸
L(a)=L/B*B/C.../a
对梯度进行裁剪，防止梯度爆炸或消失
如果梯度的范数大于阈值，就用阈值除以梯度的范数，得到新的梯度
如果梯度的范数小于阈值，就用梯度本身，得到新的梯度
最后，用新的梯度更新参数
在每次反向传播之后、执行优化器步进之前，对梯度的范数施加一个上限。给定（所有参数的）梯度 `g`，我们计算其 `ℓ₂`-范数 `‖g‖₂`。如果该范数小于最大值 `M`，则保持 `g` 不变；否则，我们按因子 `M / (‖g‖₂ + ε)` 缩小 `g`

L2范数，也称欧几里得范数，是衡量向量长度或大小的标准方法，通过向量各元素平方和的平方根来计算

梯度裁剪把更新幅度限制在可控范围内，相当于给参数更新装了限速器。
它不改变更新方向：所有梯度乘同一个系数，方向不变，只是步长变短
"""
def clip_gradient_norm(parameters: Iterable[torch.nn.Parameter], max_norm: float):
    """
    实现全局梯度裁剪 (Global Norm Clipping)。
    参数:
        parameters: 模型的所有参数 (model.parameters())
        max_norm: 允许的最大梯度 L2 范数 (M)
    """
    # 1. 过滤掉没有梯度的参数（防止对 None 对象操作）​
    params_with_grad = [p for p in parameters if p.grad is not None]
    if not params_with_grad:
        return
    
    # 2. 计算全局 L2 范数 (Global L2 Norm)​
    total_norm = 0.0
    for p in params_with_grad:
        # 使用 .detach() 极其重要：​
        # 梯度裁剪是在计算完导数后进行的数值操作，我们不希望“计算范数”的过程也被记入计算图。​
        # torch.norm(..., p=2) 算出当前层梯度的 L2 范数 L_i​
        param_norm = torch.norm(p.grad.detach(), p=2)
        # 将各层范数的平方累加 (L_total = sqrt(sum(L_i^2)))​
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    
    # 3. 检查是否触发裁剪​
    eps = 1e-6  # 防止除零的稳定性常数​
    if total_norm > max_norm:
        # 计算统一的缩放系数​
        clip_coef = max_norm / (total_norm + eps)
        # 4. 原地 (in-place) 修改每个参数的梯度​
        # 使用 mul_ 直接修改内存，不产生临时副本，节省显存​
        for p in params_with_grad:
            p.grad.detach().mul_(clip_coef)



    