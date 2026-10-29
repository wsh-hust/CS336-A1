import json
import base64
import regex as re
from collections.abc import Iterable

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        """
        special_tokens就是一些不能被 BPE 合并的 token，比如 <s>、</s>、<pad> 等，要单独摘出来，直接用vocab中对应的token id来表示。不能用于bpe算法，不然会被拆开，比如<pad>，如果不单独摘出来，就会被拆分成<,pad,>,变成3个token id
        - `vocab: dict[int, bytes]`
        - `merges: list[tuple[bytes, bytes]]`
        - `special_tokens: list[str] | None = None`
        """
        super().__init__()

                # 对于special_tokens，直接用vocab中对应的token id来表示
        # bpe训练时处理方式是：        split_pat = "(" + "|".join(re.escape(st) for st in special_tokens) + ")"
        # chunks = [c for c in re.split(split_pat, text) if c and c not in special_tokens]
        # 但是encode这里不能这么做，应该上面是直接跳过了special_tokens，而encode这里需要保留special_tokens的token id
        self.vocab = vocab
        self.special_tokens = special_tokens or []
        if self.special_tokens:
            pat = "(" + "|".join(
                re.escape(st) for st in sorted(self.special_tokens, key=len, reverse=True)
            ) + ")"  # 注意还需要对special_tokens进行排序，长的先匹配，短的后匹配，避免匹配到短的special_tokens
            self.special_re = re.compile(pat)
        else:
            self.special_re = None
        
        self.id_to_token = vocab
        self.merges = merges
        self.token_to_id = {token: id for id, token in vocab.items()}
        # 用来查找匹配token对的优先级，先合并优先级高的
        self.merges_to_idx = {merge: i for i, merge in enumerate(merges)}


    def bpe_encode_text(self, text: str) -> list[int]:
        """
        - `text: str` 这里传进来的text就是单纯的一段文本，不含特殊词，可以完全按照bpe算法把它转换为一个token id列表
        - `return: list[int]`
        """
        an=[]
        for word in re.finditer(PAT, text):  # 遍历每个单词，逐个转换
            word = word.group().encode("utf-8") # 转换为字节,此时是b'low'
            word = [bytes([i]) for i in word]  # 转换为[b'l', b'o', b'w']
            # 然后查merges，看是否有应该合并的token对，优先级按顺序
            while len(word)>1: # 只要还包含有两个token，就有可能继续合并
                min_idx=len(self.merges_to_idx)
                pos=-1
                for npos in range(len(word)-1):
                    now_pair=(word[npos],word[npos+1])
                    if(now_pair in self.merges_to_idx):
                        if pos==-1:
                            pos=npos
                        else:
                            pos=npos if min_idx>self.merges_to_idx[now_pair] else pos
                        min_idx=min(min_idx,self.merges_to_idx[now_pair])
                # 如果已经没有可以合并的token对，就跳出循环
                if(min_idx==len(self.merges_to_idx)):
                    break
                # 合并
                word[pos]=word[pos]+word[pos+1]
                del word[pos+1]
            for token in word:
                an.append(self.token_to_id[token]) # 用extend来扩展列表，因为token_to_id返回的是一个列表，而不是一个元素
        return an


    def encode(self, text: str) -> list[int]:
        """
        - `text: str`
        - `return: list[int]`
        """
        # 这样根据special_tokens，将text拆分，得到的是包含special_tokens的
        # 比如：text = "Hello<|endoftext|>world"
        #  拆分为：-> ['Hello', '<|endoftext|>', 'world']（没有not in special token）
        if self.special_re is None:
            texts = [text]
        else:
            texts = ([c for c in re.split(self.special_re, text) if c])

        ans=[]
        for text in texts:
            if text in self.special_tokens:
                # 不能用text = self.token_to_id[text]，因为token是bytes，不能用str来当键
                text=text.encode('utf-8')
                # 前面bpe算法已经把special_tokens添加有对应的token id了，这里直接用
                ans.append(self.token_to_id[text])
            else:
                ans.extend(self.bpe_encode_text(text))
        return ans


    def decode(self, ids: list[int]) -> str:
        """
        - `ids: list[int]`
        - `return: str`
        """
        return b"".join([self.id_to_token[t] for t in ids]).decode("utf-8", errors="replace")

    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        """
        内存高效的迭代编码器。
        
        参数:
            iterable: 一个可迭代的字符串对象（例如文件句柄）。
        返回:
            一个生成器，逐个产出编码后的 ID。用于处理无法一次性读入内存的大文件。
        """
        for chunk in iterable:
            # 对每一块文本进行编码，并通过 yield 吐出结果
            yield from self.encode(chunk)


        
        
            
