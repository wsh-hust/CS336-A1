import torch
import numpy as np
import numpy.typing as npt

"""
原始数据文本经过tokenizer处理之后，得到的不是“很多文件”而是一长串 token ID,比如：
x = [5, 12, 9, 30, 7, 88, 3, 5, 5, 40, ...]   (长度 n,可能是几亿个)
不同文档(不同故事、不同网页)直接首尾相接，中间插一个 <|endoftext|> 的 token ID 当“句号”，告诉模型“这里一个文档结束了”。这样做的目的：训练时随便从序列任何位置切窗口都能得到合法的训练样本，不用按文档分组、处理变长等问题。
随机取可能就会拿到一句话的后半截，以及后一句话的前半截，这样也可以。
这是“拼接 + 随机采样”方案的核心设计，而且是大模型预训练的标准做法(GPT-2、LLaMA 都是这么干的)。分几层来看为什么没问题：
窗口里的每个位置仍然是合法的训练样本
模型的任务是“给定左边的前文，预测下一个 token”,它对“前文属于哪句话、哪篇文档”没有任何要求。
因果注意力保证了“后半截前文”不会污染预测
你可能担心：新文档开头的前文里挂着上一篇的尾巴，会不会干扰？不会，因为注意力是单向的
 “一句话的后半截”更是根本不算问题
即使不跨文档，窗口切在句子中间也是常态(窗口长 m=256,起点随机)。语言建模本来就要求模型能在任意长度前文下预测下一个 token——用户输入半句话时模型也得接得上。所以“前文是否完整”从来不是训练样本有效性的条件。
"""


def get_batch(
    dataset: npt.NDArray, 
    batch_size: int, 
    context_length: int, 
    device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    从 Numpy 数组数据集中随机采样一个批次。
    
    参数:
        dataset: 1D Numpy数组 (Token IDs)，是原始文本经过分词后转化为的token id序列
        batch_size: 批大小 (B)
        context_length: 上下文长度 (m)
        device: 设备字符串 ('cpu', 'cuda', 'mps')
    """
    # 首先看dataset的长度，也就是token id序列一共有多少个，确认选择数据时的下标范围
    # 从0开始，到len_dataset-context_length,这个范围是可以取出context_length个token id的，但是因为还要有对应的答案id组，而答案id组的最后一个是序列id组的下一个token id，所以还有留出一个token id给答案id组的最后一个token id
    # 所以可以取的下标范围是[0, len_dataset-context_length-1]
    len_dataset = len(dataset)

    # 随机产生 batch_size 个起始位置
    # np.random.randint 在 [0, len_dataset-context_length-1] 之间产生随机整数
    # 因为是左闭右开区间所以要加1
    index = np.random.randint(0, len_dataset-context_length, batch_size)
    
    # 根据索引提取输入和目标
    # x: dataset[i : i+m]
    # y: dataset[i+1 : i+m+1]
    # 先在 CPU 上提取数据，然后一次性转为 Tensor
    # Python 切片永远是左闭右开:dataset[i : i + context_length] 取出的是下标 i, i+1, …, i+context_length-1
    x_stack = [dataset[i : i + context_length] for i in index]
    y_stack = [dataset[i + 1 : i + context_length + 1] for i in index]
    
    # 转换为 PyTorch 张量并移动到指定设备
    # 注意：dataset 通常是 int32 或 int64，转为 torch 后通常使用 torch.long (int64)
    x = torch.from_numpy(np.array(x_stack, dtype=np.int64)).to(device)
    y = torch.from_numpy(np.array(y_stack, dtype=np.int64)).to(device)
    
    return (x, y)

