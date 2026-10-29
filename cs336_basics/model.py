import math
import torch
import torch.nn as nn
"""
参数初始化：
Linear 权重:
  分布:  N(mean = 0, variance = 2 / (d_in + d_out))均值为0，方差为2/(d_in+d_out)，
  等价:  std = sqrt( 2 / (d_in + d_out) ) std是标准差，用于控制权重的分布范围
  截断:  范围 [ -3*std, +3*std ]
 
Embedding 权重:
  分布:  N(mean = 0, variance = 1)，即标准正态
  等价:  std = 1
  截断:  范围 [ -3, +3 ]
 
RMSNorm 权重:
  全 1 向量
"""

"""
全连接层
    1. init: 定义一个W矩阵
    2. forward: 如输入x进行矩阵乘法
"""
class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        # 注意这里初始化W矩阵，是(out_features, in_features)，而不是(in_features, out_features)，因为pytorch里面是行优先的，这样在进行输入向量x和权重矩阵W的乘法时，x是行向量，W转置后不改变数据的实际存储方式，只是改变视图，这样可以方便地进行矩阵乘法操作，每次取一行进行乘法
        # 为什么要存为 (out, in) 而不是 (in, out)？​
        # 内存局部性（Locality）：在 (out, in) 布局下，权重矩阵的每一行对应输出空间的一个“神经元”。当计算机执行矩阵乘法时，它可以连续读取一整行内存到缓存（Cache）中，极大地提高了运算速度。
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        std = math.sqrt(2.0 / (in_features + out_features))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3 * std, b=3 * std) # 截断:  范围 [ -3*std, +3*std ]

    def forward(self, x):
        return x @ self.weight.T 


"""
嵌入层
    1. init: 定义一个嵌入矩阵，大小为(num_embeddings, embedding_dim)
    2. forward: 根据输入token_ids进行索引，返回对应的嵌入向量
"""
class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        )
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)
 
    def forward(self, token_ids):
        # token_ids的形状为(batch_size, seq_len)
        # self.weight的形状为(num_embeddings, embedding_dim)
        # token_ids的每个元素是一个整数，对应一个嵌入向量的索引
        # 根据token_ids的每个元素，从self.weight中提取对应的嵌入向量
        # 最终返回一个形状为(batch_size, seq_len, embedding_dim)的张量
        return self.weight[token_ids] # W[ids] 等价于 W[ids, :]：ids 只索引第 0 维，后面的维度用 : 原样保留



"""
均方根层归一化
是对 LayerNorm 的简化。它对每个 token 的 d_model 维向量独立归一化
LayerNorm(x) = (x - mean(x)) / sqrt(var(x) + eps) * g + b
RMSNorm(x)   =  x            / sqrt(mean(x^2) + eps) * g
"""
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x):
        in_dtype = x.dtype
        x = x.to(torch.float32) 
        ms = (x * x).mean(dim=-1, keepdim=True)
        rms = torch.sqrt(ms + self.eps)
        result = x / rms * self.weight
        return result.to(in_dtype)


"""
SiLU 激活函数
"""
def silu(x):
    return x * torch.sigmoid(x)



"""
SwiGLU
这个模块就相当于原始Transformer结构中的FFN，用SiLU激活函数代替了ReLU激活函数，并引入了门控机制
SiLU(W1 x)  -> 候选特征，同时充当“门”
W3 x        -> 被门控的信息
两者逐元素相乘 -> 门决定每个维度通过多少
"""
class SwiGLU(nn.Module):
    def __init__(self, d_model: int,d_ff: int, device=None, dtype=None):
        super().__init__()
        self.W1=Linear(d_model,d_ff,device=device,dtype=dtype)
        self.W2=Linear(d_ff,d_model,device=device,dtype=dtype)
        self.W3=Linear(d_model,d_ff,device=device,dtype=dtype)

    def forward(self, x):
        gate=silu(self.W1(x)) # 即silu(W1(x)),通过一个linear就是乘一个矩阵
        signal=self.W3(x) # 原始FFN是直接输出self.W2(silu(self.W1(x))),这里加了一个门控机制
        return self.W2(gate*signal) # *是逐元素乘法，对每个维度进行对应乘法



"""
ROPE 层嵌入

注意和原始的位置编码的区别：
sinusoidal PE: 用 d_model
RoPE:          用每个 head 的 d_k = d_model / n_heads
ROPE是对分头后的QK矩阵做旋转(如果是对分头前的矩阵做旋转，那么分头后，后面的矩阵可能就对附近的单词不敏感，后面的头就是分针，时针)

RoPE 不是把位置向量加到输入上，而是根据 token 的位置，把 query 和 key 向量“旋转”一个角度。旋转后的 query 和 key 做点积时，结果只依赖它们的相对位置，不依赖绝对位置。
它只作用于 Q 和 K，不作用于 V，而且没有可学习参数。
注意这个计算公式：；；；，对于一个单词的向量表示，在作用这个旋转角度时，前面的维度组变换的角度大，而后面的维度组变换的角度小。
这使得RoPE能够捕获到单词的长距离依赖关系，同时能够保持单词的局部依赖关系。
角度大能够感知细小的依赖关系，而角度小能够感知大的依赖关系。
相当于前面是秒针，后面是分针，时针
同时这样即使序列很长，也不会包含一个角度周期，因为要让所有维度组都旋转一个倍数。
在数学上，RoPE 描述为一个巨大的分块对角矩阵。但在工程实践中，为了压榨 GPU 性能，我们绝对不会构建这个稀疏矩阵，而是利用 张量广播 和 成对变换 来实现。

####################### 看这就是小c的文档介绍 ######################
"""
# 一个问题：为什么需要token_position呢？只需要知道序列长度不就可以了吗？
class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        """     
        theta: 基准频率 (通常为 10000)
        d_k: 每个 Head 的维度 (必须是偶数)
        max_seq_len: 最大序列长度​
        """
        super().__init__()
        self.d_k = d_k
        # 1. 计算频率频率 omega_k = theta^(-2k / d)​
        # 只需要计算 d_k/2 个频率，因为旋转是按照维度成对进行的​
        # arange(0, d_k, 2) 产生 [0, 2, 4, ..., d_k-2]，对应公式中的2k-2(k从1开始)
        powers = torch.arange(0, d_k, 2, device=device).float() / d_k
        freqs = 1.0 / (theta ** powers)  # 形状: (d_k/2,)​
        
        # 2. 创建位置序列 [0, 1, ..., max_seq_len - 1]
        # 每一行对应一个，也就是一个token序列，第一个pos为0，第二个pos为1，以此类推​
        t = torch.arange(max_seq_len, device=device).float()  # 形状: (max_seq_len,)​

        # 3. 计算所有位置的所有角度 (外积)​
        # freqs_matrix 形状: (max_seq_len, d_k/2)​
        # 这个的第i行第j列，就是对应序列第i个token第j组维度的旋转角度
        freqs_matrix = torch.outer(t, freqs)
        
        # 4. 预计算 cos 和 sin 并作为 buffer 注册​
        # 使用 persistent=False 确保这些缓存不会被保存在 state_dict 中 (因为可以随时重新生成)​
        self.register_buffer("cos_cached", freqs_matrix.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs_matrix.sin(), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor):
        # token_positions 形状: (..., Seq)​,是每个token的位置编号
        # 比如输入"我爱你"：token_ids       = [45, 892, 77]
        #                  token_positions = [0, 1, 2]
        # x就是我爱你的三个token的向量表示
        # 1. 提取 cos/sin (..., Seq, d_k/2)​
        cos = self.cos_cached[token_positions]
        sin = self.sin_cached[token_positions]

        # 2. 维度对齐​
        # 只有当 x 是 4D (含 Head 维) 且 cos 是 3D (含 Batch 维) 时，才需要手动插入 Head 维。​
        # 对于 test_rope 这种 3D x vs 2D cos 的情况，PyTorch 会自动左侧补 1，无需操作。​
        if x.ndim > cos.ndim and cos.ndim >= 3:
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)

        # 确保类型一致​
        cos = cos.to(x.dtype)
        sin = sin.to(x.dtype)

        # 3. 拆分​。分成偶数维度和奇数维度两组，其实就是一对维度的第一项和第二项
        x_even = x[..., 0::2]   # 偶数维
        x_odd = x[..., 1::2]    # 奇数维
        output = torch.empty_like(x)    
        output[..., 0::2] = x_even * cos - x_odd * sin
        output[..., 1::2] = x_even * sin + x_odd * cos

        return output
"""
ROPE是作用在分头后的QK矩阵上的，对分头前的矩阵做旋转，那么分头后，后面的矩阵可能就对附近的单词不敏感。因为根据计算公式，越靠前的维度组旋转的角度越大，所以越靠前的维度组对位置变化越敏感。
如果在分头前用 d_model 的整套频率表去旋转，那么越靠后的 head 会拿到越低的频率，对近距离位置变化越不敏感。
标准 RoPE 的做法是：分头之后，每个 head 在自己的 d_k 维子空间里，使用完整的一套 d_k 频率表。
"""
"""
一个例子：
假设有一个token，在对应序列的位置是第5个，向量表示为：[x0,x1,x2,x3]
那么对应的，在init计算得到的freqs_matrix矩阵中，第5行就是对应序列第5个token的旋转角度。而该token向量为：[x0,x1,x2,x3]。分组：[(x0,x1),(x2,x3)]
那么第0组的旋转角度就是freqs_matrix[4,0]，第1组的旋转角度就是freqs_matrix[4,1]。
假设这两个旋转角度分别为a，b
旋转矩阵R(@) = [[cos(a), -sin(a)],
                  [sin(a), cos(a)]]                       
那么作用在该向量上旋转后得到的结果应该是：[x0cos(a)-x1sin(a), x0sin(a)+x1cos(a), x2cos(b)-x3sin(b), x2sin(b)+x3cos(b)]
规律就是对于每一组维度(t0,t1),旋转角度为c，那么结果就是：[t0cos(c)-t1sin(c), t0sin(c)+t1cos(c)]。所以每一组都是这样的变换。

回到举的这个例子上：
该token为：[(x0,x1),(x2,x3)]，旋转角度分别为：[a,b]，要得到：[x0cos(a)-x1sin(a), x0sin(a)+x1cos(a), x2cos(b)-x3sin(b), x2sin(b)+x3cos(b)]。
先根据角度矩阵得到对应的cos和sin矩阵
得到：cos：[cos(a), cos(b)]和sin：[sin(a), sin(b)]

然后因为对每一组维度的操作都是一样的：前一个值*cos减去后一个值*sin，然后前一个值*sin加上后一个值*cos。
所以可以分开来，先计算前一个结果，再计算后一个结果，最后合并起来。

首先把token的偶数维(每一个维度组的前一个)和奇数维(每一个维度组的后一个)抽出来
得到：偶数维矩阵：[x0,x2]和奇数维矩阵：[x1,x3]。
然后用cos矩阵与偶数维矩阵相乘，用sin矩阵与奇数维矩阵相乘，两者相减
得到：[x0cos(a)-x1sin(a), x2cos(b)-x3sin(b)]。
这就是结果的每一组维度的前半部分。也就是最终结果矩阵的偶数维，放在out[0::2]上。

同理，再把sin矩阵和偶数维矩阵相乘，用cos矩阵和奇数维矩阵相乘，两者相加
得到：[x0sin(a)+x1cos(a), x2sin(b)+x3cos(b)]。
这就是结果的后半部分。也就是最终结果矩阵的奇数维，放在out[1::2]上。

两者拼凑就得到最终的结果了：
out = [x0cos(a)-x1sin(a), x0sin(a)+x1cos(a), x2cos(b)-x3sin(b), x2sin(b)+x3cos(b)]

标准 RoPE 分别作用在分头后的 Q 和 K 上。
每个 head 在自己的 d_k 子空间里，使用完整的一套 d_k 频率表，
所以每个 head 都同时拥有高频（局部敏感）和低频（长程）维度。
 
如果在分头前用 d_model 的整套频率表旋转，
频率会按维度位置分段：
前面的 head 拿到高频，后面的 head 拿到低频，
后面的 head 对邻近位置的区分能力会变弱。

具体例子
d_k = 4
第一组维度：k=0: inv_freq = 1 / 10000^(0/4) = 1
第二组维度：k=1: inv_freq = 1 / 10000^(2/4) = 0.01
位置 0 到 3 的旋转角度(每个值是一组维度的旋转角度)：
角度=位置*频率
位置 0: [0 * 1,     0 * 0.01] = [0,    0   ]
位置 1: [1 * 1,     1 * 0.01] = [1,    0.01]
位置 2: [2 * 1,     2 * 0.01] = [2,    0.02]
位置 3: [3 * 1,     3 * 0.01] = [3,    0.03]
"""




"""
实现softmax函数
原始的softmax公式是e^vi / sum(e^vj)，但是可能溢出
当数值较大时 `exp(vi)` 可能变为 inf（此时 `inf / inf = NaN`）。因为 softmax 操作对给所有输入加上任意常数 `c` 保持不变。所以可以利用这一性质来获得数值稳定性——通常的做法是从 `v` 的所有元素中减去 `v` 中的最大项，使新的最大项为 0。e的0等于1，而其它项此时都小于0，取e后都小于1,不会溢出
"""
def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        # 为了数值稳定性，减去指定维度上的最大值
        # dim=-1 通常是 Transformer 中的隐藏层或词表维度
        x_max = torch.max(x, dim=dim, keepdim=True).values
        x_stable = x - x_max # 防止溢出
        
        # 计算指数
        exp_x = torch.exp(x_stable)
        
        # 计算分母的各指数之和
        sum_exp = torch.sum(exp_x, dim=dim, keepdim=True)
        
        # 计算最终结果
        return exp_x / sum_exp


"""
缩放点积注意力
实现：
scores = Q @ K^T / sqrt(d_k)
weights = softmax(scores)
out = weights @ V
"""
def scaled_dot_product_attention(q,k,v,mask=None):
    # 首先计算注意力分数：
    scores=q @ k.transpose(-1,-2)
    # 然后再除以sqrt(d_k)，除 √dk 防止 dk 大时 softmax 饱和
    d_k=q.shape[-1]
    scores=scores / math.sqrt(d_k)
    # 掩码作用在softmax之前，除根号d_k之后
    if mask is not None:
        scores=scores.masked_fill(mask==0, float('-inf'))
    # 计算注意力权重,沿最后一维就行，也就是对每一个token对其它所有token的注意力分数进行softmax操作
    weights=softmax(scores, dim=-1)
    # 计算注意力输出
    out=weights @ v
    return out


"""
多头自注意力
实现：
1. 因为是自注意力，所有Q，K，V都来自同一个x，经过不同的矩阵Wq，Wk，Wv得到
2. 对Q,K,V进行分头
3. 对分头后的Q，K矩阵进行RoPE编码
4. 对每个头进行缩放点积注意力
5. 对所有头的输出进行拼接
"""
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads,theta=None,max_seq_len=None,device=None,dtype=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.Wq=Linear(d_model, d_model,device=device,dtype=dtype)
        self.Wk=Linear(d_model, d_model,device=device,dtype=dtype)
        self.Wv=Linear(d_model, d_model,device=device,dtype=dtype)
        # 多头分开求完后，再拼接起来，最后经过一个线性层进行融合
        self.Wo=Linear(d_model, d_model,device=device,dtype=dtype)

        if theta is not None and max_seq_len is not None:
            self.rope=RotaryPositionalEmbedding(theta,self.d_k,max_seq_len,device=device)
        else:
            self.rope=None

    # 因为是自注意力机制，q，k，v都来自同一个x
    def forward(self, x,token_positions=None, mask=None):
        # 输入x的形状为(batch_size, seq_len, d_model)
        B,S,D=x.shape
        seq_len=x.shape[-2]
        # 计算掩码mask
        if mask is None:
            mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool))
        # 首先通过Wq，Wk，Wv得到Q,K,V
        Q=self.Wq(x)
        K=self.Wk(x)
        V=self.Wv(x)

        # 此时Q，K，V矩阵的维度依然是(batch_size, seq_len, d_model)
        # 接下来对Q,K,V进行分头
        Q=Q.view(B,S,self.num_heads,self.d_k)
        K=K.view(B,S,self.num_heads,self.d_k)
        V=V.view(B,S,self.num_heads,self.d_k)

        # 此时Q，K，V矩阵的维度为(batch_size, seq_len, num_heads, d_k)
        # 还需要转置一下，将num_heads维度放到第二维，变为(B,H,S,d_k)
        Q=Q.transpose(1,2)
        K=K.transpose(1,2)
        V=V.transpose(1,2)

        # 对分头后的Q，K矩阵进行RoPE编码
        if self.rope is not None:
            if token_positions is None:
                token_positions=torch.arange(S,device=x.device)
            Q=self.rope(Q,token_positions)
            K=self.rope(K,token_positions)

        # 编码分头后的Q，K后，就可以进行注意力点积了
        out=scaled_dot_product_attention(Q,K,V,mask)

        # 然后对所有头的输出进行拼接
        # 此时的out维度是(B,H,S,d_k)
        out=out.transpose(1,2) # 又转置一次，变为(B,S,H,d_k)
        out=out.contiguous().view(B,S,self.d_model) # 合并所有头的输出，变为(B,S,d_model)
        out=self.Wo(out) # 最后通过线性层进行融合
        return out



"""
TransformerBlock的实现
一个TransformerBlock包含一个多头自注意力层和一个全连接层，同时两个层都有残差连接和层归一化
最后的模型会包含多个TransformerBlock
"""
class TransformerBlock(nn.Module):
    def __init__(self, d_model,d_ff, num_heads,theta=None,max_seq_len=None,device=None,dtype=None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_heads = num_heads
        self.norm1=RMSNorm(d_model,device=device,dtype=dtype)
        self.norm2=RMSNorm(d_model,device=device,dtype=dtype)
        self.attention=MultiHeadSelfAttention(d_model, num_heads,theta,max_seq_len,device,dtype)
        # 前面实现的SwiGLU全连接层就是FFN模块的实现
        self.ffn=SwiGLU(d_model, d_ff,device=device,dtype=dtype)

    def forward(self, x):
        x=x+self.attention(self.norm1(x))
        x=x+self.ffn(self.norm2(x))
        return x



"""
transformer的综合实现
接收的是(Batch,SeqLen),其中Seqlen是对应序列的token id序列
"hello world"
  -> ["hello", " world"]          # BPE 切出的 token
  -> [15496, 995]                 # 转化为对应的token id
  -> 然后输入TransformerLM
  -> Embedding.weight[[15496, 995]]
  -> (2, d_model) 的向量矩阵

"""
class TransformerLM(nn.Module):
    def __init__(self, d_model,d_ff, num_heads,theta=None,context_length=None,vocab_size=None,num_layers=None,device=None,dtype=None):
        super().__init__()
        self.embedding=Embedding(vocab_size, d_model,device=device,dtype=dtype)
        self.blocks=nn.ModuleList([TransformerBlock(d_model,d_ff, num_heads,theta,context_length,device,dtype) for _ in range(num_layers)])
        self.final_norm=RMSNorm(d_model,device=device,dtype=dtype)
        self.final_linear=Linear(d_model, vocab_size,device=device,dtype=dtype)

    def forward(self, x):
        x=self.embedding(x)
        for block in self.blocks:
            x=block(x)
        x=self.final_norm(x)
        x=self.final_linear(x)
        return x # 输出的是logits矩阵






