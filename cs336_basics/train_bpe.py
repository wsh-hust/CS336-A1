from collections import Counter
import regex as re
from collections import defaultdict

# 正则表达式，用于划分单词
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    # 读取文件
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    # 对文本进行划分，按照每个文章的结束符进行划分，得到各个文章
    # 不包含special_tokens，因为special_tokens是特殊token，不需要被划分到文章中
    # 并且之后在编码时，一个special_token对应一个token id，不会拆分
    if special_tokens:
        split_pat = "(" + "|".join(re.escape(st) for st in special_tokens) + ")"
        chunks = [c for c in re.split(split_pat, text) if c and c not in special_tokens]
    else:
        chunks = [text]

    # 对每个文章进行分词，得到所有单词, 并统计每个单词的词频
    # 比如：low:82，lower:102
    words_count = Counter()
    for chunk in chunks: # 遍历每个文章
        for word in re.finditer(PAT, chunk): # 遍历每个单词
            word = word.group().encode("utf-8") # 转换为字节:b'low'
            words_count[tuple(word)] += 1 # 统计词频，必须用tuple，不能用list，不能作为字典的键。 同时这个地方的tuple也会把bytes转换为int，而初始的词表token id和token(字节)是一一对应的，所以这个时候就把每个单词转换为对应的token id表示了，比如low：(108,111,119)：82
    # words_count是一个字典，键是token id，值是对应的token的词频，比如low：(108,111,119):82
    # 因为有时间限制，所以要使用倒排索引，统计涉及到某个单词对的所有单词，用idx记录,后续更新时根据idx找到对应的需要更新的单词
    # idx_to_word是一个字典，键是idx，值是对应的token id组,比如low是第5个单词
    # 则在idx_to_word中，5:(108,111,119)
    # word_to_idx是一个字典，键是token id组，值是对应的idx,比如low的idx是5
    # 则在word_to_idx中，(108,111,119):5
    # idx_to_word的作用是在之后合并一对token时，根据索引找到对应的单词的token id组
    # 比如合并了108和111，用新的token id 257表示，则通过index字典的键(108,111)找到包含该token对的所有单词的idx，也就是low对应的5，然后拿着这个5在idx_to_word中找到对应的token id组，也就是(108,111,119)，再把(108,111,119)合并为(257,119)
    # word_to_idx的作用是在统计一对token时，为token id组添加对应单词的idx
    # 一个token id组对应一个单词，比如low：(108,111,119)
    idx_to_word = {}
    word_to_idx = {}
    i=0
    for word, freq in words_count.items():
        idx_to_word[i]=word
        word_to_idx[word]=i
        i+=1

    # 索引结构，根据token对找到包含该token对的所有单词的idx
    index=defaultdict(set)

    # 初始化pair_count，用于找最大词频的token对
    # 其元素是：(token id,token id):freq
    # 比如： (108,111):82
    pair_count = Counter()
    for word, freq in words_count.items():
        if len(word)==1:
            continue
        for p in zip(word,word[1:]):
            pair_count[p] += freq
            index[p].add(word_to_idx[word])

    # 初始化词表，包含初始的256个token
    # 词表的结构是token id -> token
    vocab={i:bytes([i]) for i in range(256)}
    merge=[]
    # 需要合并的轮数
    epoch=vocab_size-256-len(special_tokens)
    next_token_id=256
    for word in special_tokens:
        vocab[next_token_id]=word.encode("utf-8")
        next_token_id+=1

    for _ in range(epoch):
        # 如果已经合并完全，也就是剩下的单词都可以用一个token表示，那么就停止合并
        if not pair_count:
            break

        # 找到最大词频的token对
        best_pair=max(pair_count, key=lambda p: (pair_count[p], (vocab[p[0]], vocab[p[1]])))

        # 将新token加入到词表中：新token id：新token
        a,b=best_pair
        new_token=vocab[a]+vocab[b]
        # 更新词表，加入新合并的token，并记录合并顺序
        vocab[next_token_id]=new_token
        merge.append((vocab[a],vocab[b]))

        # 找到包含该token对的所有单词的idx，同时移除这条已经失效的索引
        # 该语句从字典 index 中取出并删除键 best_pair 对应的值，赋给 idxs；如果 best_pair 不存在，就返回一个空的 set()，并且不报错
        # 实现更新通过两步：
        # 首先根据合并的token id对，在index中找到包含对应token id对的所有单词的idx，再根据这个idx在idx_to_word中找到对应的token id组，然后把这些单词(也就是这些token id组)产生的所有相邻token对的贡献都撤销，freq也撤销，freq的值通过words_count[word]找到,word就是token id组
        # 然后更新，按照best_pair的token id对，更新所有包含该token id对的单词的token id组，把(a,b)替换为new_token_id，然后计算新的token id组产生的所有相邻token对的贡献，同时还要更新pair_count和index中的记录。
        # 比如合并了108和111，用新的token id 257表示，则通过index字典的键(108,111)找到包含该token对的所有单词的idx，也就是low对应的5，然后拿着这个5在idx_to_word中找到对应的token id组，也就是(108,111,119)，再把(108,111,119)合并为(257,119)
        # 然后更新：index中添加(257,119)的记录，对应值是5；idx_to_word中更新5的记录，对应值是(257,119);pair_count中更新(257,119)的记录，对应值是82
        # 所以每一轮合并操作是：首先在pair_count中找到最大词频的token id对，然后根据这个token id对，在index中找到包含该token id对的所有单词的idx，再根据这个idx在idx_to_word中找到对应的token id组，然后把这些单词(也就是这些token id组)产生的所有相邻token对的贡献都撤销，然后更新idx，idx_to_word,pair_count，word_count。
        idxs=index.pop(best_pair,set())
        for idx in idxs:
            word=idx_to_word[idx] # 合并前的旧词（tuple）
            freq=words_count.pop(word) # 取出旧词词频，并把旧词从words_count中删掉

            # 先撤销旧词所有相邻token对产生的pair_count和index贡献
            # 这里必须按出现次数逐次撤销，不能先去重，否则aaa这类重复pair会少减
            for p in zip(word,word[1:]):
                pair_count[p] -= freq
                if pair_count[p]==0:
                    del pair_count[p]
                index[p].discard(idx)
                if not index[p]:
                    del index[p]

            # 执行合并：从左到右，相邻的(a,b)替换为new_token，npos指向当前检查位置
            new_word=[]
            npos=0
            while npos < len(word):
                if npos<len(word)-1 and word[npos]==a and word[npos+1]==b:
                    new_word.append(next_token_id) # a和b合并为新token，一次前进两步，自然处理aaa这类重叠情况
                    npos+=2
                else:
                    new_word.append(word[npos]) # 单个int用append，int列表用extend添加
                    npos+=1
            new_word=tuple(new_word) # 列表不能做字典的键，所以要转换为元组

            # 更新单词映射和词频
            idx_to_word[idx]=new_word
            # del word_to_idx[word]
            # word_to_idx[new_word]=idx #不用更新，因为合并token id对的时候已经用不到了，该字典只是用来构建索引，遍历每一个单词统计每一对token id对频率时用来构建该索引的映射单词列表
            words_count[new_word] += freq

            # 最后登记新词产生的所有相邻token对
            for p in zip(new_word,new_word[1:]):
                pair_count[p] += freq
                index[p].add(idx)

        next_token_id+=1

    return vocab,merge


"""
一个例子：
tuple(b'low')=(108,111,119) 一个单词在算法过程中基本都是用对应token id组表示的
只有在vocab中token id对应的是token(字节表示),merge中记录的也是合并的token对，同样是字节表示，比如(b'l',b'o')
对于一个单词low，假设它在文章出现的次数是82次，在各个数据结构中的表示：
vocab:
{108:b'l',111:b'o',119:b'w'}

words_count:        
{(108,111,119):82}

idx_to_word:   
{5:(108,111,119)}

word_to_idx:
{(108,111,119):[5]}

pair_count:  # 只算low的贡献,其实还有其它包含lo,ow的单词的贡献,假设忽略
{(108,111):82,(111,119):82}

index:   # 同样只算low的贡献，实际可能还有其它单词的idx
{(108,111):[5]},(111,119):[5]}

假设现在pair_count中(108,111)的词频最高，合并(108,111)，具体操作是：
首先根据108，111这两个token id，在vocab中找到对应的字符(字节表示)，分别是'l'和'o'，然后把这两个字符合并为一个新的token id 257，用257表示合并后的token。
添加到词表vocab中：257:b'lo'
然后添加到merge中：(b'l',b'o')

然后根据index中的(108,111)对应的索引列表找到包含该token id对的单词索引，也就是5，进行增量修改(倒排索引)
得到idx为5后，通过idx_to_word[idx]找到对应的单词，也就是token id组，即单词low：(108,111,119)，接下来就需要把(108,111,119)合并为(257,119)

但是在合并之前，先把该单词的贡献清除
首先遍历low的每一个token id对，(108,111)和(111,119)，把该单词对(108,111)和(111,119)的贡献都撤销，贡献值在words_count中:low:(108,111,119):82
所以把pair_count中(108,111)和(111,119)的值都减去82
index里面索引也删，后面更新index时再添加

然后合并得到新的表示：合并(108,111)，改为257：
（108,111,119）---> (257,119)
这个时候low的token id组就是(257,119)了，要改涉及到的所有数据结构：
words_count:
{(108,111,119):82} ---> {(257,119):82}

idx_to_word:
{5:(108,111,119)} ---> {5:(257,119)}

word_to_idx:
{(108,111,119):[5]} ---> {(257,119):[5]}

pair_count: 前面去除了旧的low的贡献，现在添加新的：
{(108,111):82,(111,119):82} ---> {(257,119):82} (这个82只是low的贡献，如果还有其它单词包含这个token id对，也要添加)

index: 
{(108,111):[5]},(111,119):[5]} --->{(257,119):[5]}


几个问题：
1.UTF-8，unicode编码，a.encode('utf-8')?
2.为什么要用bpe？词级编码？字符级编码？
3.解码时为什么要按照merge中的token对顺序来解码？为什么越前面的token对越优先？
4.如果一个token id对在merge中没有记录，那么它在解码时是否会被忽略？

"""

