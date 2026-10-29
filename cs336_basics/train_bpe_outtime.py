from collections import Counter
import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

"""
- `input_path: str`：存放 BPE tokenizer 训练数据的文本文件路径。
- `vocab_size: int`：一个正整数，定义最终词表的最大规模（包括初始字节词表、由合并产生的词表项，以及所有特殊 token）。
- `special_tokens: list[str]`：要加入词表的字符串列表。训练时把它们视为硬边界，阻止跨越其跨度的合并，但在统计合并次数时不把它们计入。

**输出**

- `vocab: dict[int, bytes]`：tokenizer 的词表，从 int（词表中的 token ID）到 bytes（token 的字节序列）的映射。
- `merges: list[tuple[bytes, bytes]]`：训练产生的 BPE 合并列表。列表中每一项是一个由字节对组成的元组 `<token1>, <token2>`，表示 `<token1>` 与 `<token2>` 发生了合并。合并列表应按创建顺序排列。
"""
def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    # 读取文件
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    # 对文本进行划分，按照每个文章的结束符进行划分，得到各个文章
    if special_tokens:
        split_pat = "(" + "|".join(re.escape(st) for st in special_tokens) + ")"
        chunks = [c for c in re.split(split_pat, text) if c and c not in special_tokens]
    else:
        chunks = [text]

    # 要合并的token对数量
    epoch=vocab_size-len(special_tokens)-256
    vocab={}
    merges=[]
    next_id=256
    # 首先初始化词表，这256个字节的 token 分别对应 ASCII 码表中的前 256 个字符，就已经可以编码所有可能的字符了，任何一个字符，无论中文还是英文，都可以通过utf-8变成1-4个字节,也就是0-255，在此时的词表中都有token对应，但是这样对于输入序列，转换为token后太多了，导致计算量大，所有把常见的挨着的token合并为一个新token
    for i in range(256):
        vocab[i]=bytes([i])
    for st in special_tokens:
        vocab[next_id]=st.encode("utf-8")
        next_id+=1

    # 对得到的所有文章进行预分词(得到low，lower，lowest这样的词)，然后统计词频（综合所有文章，得到字典：low:82,lower:100,lowest:150这样的词以及对应的词频，方便后续合并token对）
    words_count=Counter()
    for chunk in chunks:
        for token in re.findall(PAT, chunk):
            token=token.encode("utf-8")  # 得到单词对应字节序列，如b'low'
            words_count[tuple(token)]+=1 # 这里tuple会把字节序列转换为int，比如low的utf-8表示是b'\x6c\x6f\x77',通过tuple后转换为int后是（108，111，119），而108在此时的词表中对应的正好是b'l'，所以这样其实是把一个单词拆分成了token id，用int表示，方便后续合并token对。
    # 此时的words_count是一个字典，键是token id，值是对应的token的词频，比如low：(108,111,119):82

    # 接下来逐个合并相邻token对，每次合并此时出现次数最多的那一对
    for epoch in range(epoch): # 合并epoch次，每次合并出现次数最多的那一对
        pair_count=Counter()
        for word, freq in words_count.items():
            if len(word)==1:
                continue
            for p in zip(word, word[1:]):
                pair_count[p] += freq

        if not pair_count:
            break # 如果没有token对，说明已经合并完成，全部都是单个token，跳出循环

        # 找到出现次数最多的那一对token，合并为一个新token
        # 先按照出现次数排序，再按照token的字典序排序，pair_count里面放的是token id，所以要先转换为token，再排序
        best_pair=max(pair_count, key=lambda p: (pair_count[p], (vocab[p[0]], vocab[p[1]])))

        # 合并best_pair[0]和best_pair[1]为一个新token
        a,b=best_pair
        new_token=vocab[a]+vocab[b]  # 比如a=b'l',b=b'o',则new_token=b'lo'
        # 把新token加入词表
        vocab[next_id]=new_token
        # 把合并的token对加入到merges中
        merges.append((vocab[a], vocab[b]))

        # 合并得到新的token后，更新words_count，各个单词中如果有best_pair[0]和best_pair[1]，就用new_token的id(也就是next_id)来替换
        new_words_count={}
        for word in words_count:
            new_word=[]
            if len(word)==1:
                new_word.append(word[0])
            else:
                i=0
                while i<len(word):
                #for i in range(len(word)):  不能用for循环，python里面虽然后面i+=2，但是i还是会取自动增加1的值
                    if i<len(word)-1 and word[i]==a and word[i+1]==b:
                        new_word.append(next_id)
                        i+=2
                    else:
                        new_word.append(word[i])
                        i+=1
            # 统计完成后，填入到新的words_count中
            new_words_count[tuple(new_word)]=words_count[word]
        words_count=new_words_count
        next_id+=1

    return vocab, merges
