
## 1. What is BPE, and how does it differ from word-level and character-level tokenization? Draw a merge tree for a concrete example (you can look for one in the data, or make your own. Do not reuse an existing example!). Make sure to also include the counts for your merges.



BPF(Byte Pair Encoding), is a subword tokenization algorithm. It creates a vocabulary by repeatedly merging the most frequent neighboring pair.

The basic procedure is:
1. Start with individual characters or bytes.
2. Count all adjacent symbol pairs.
3. Merge the most frequent pair into a new symbol.
4. Recount the pairs after the merge.
5. Repeat until the desired vocabulary size is reached.

Suppose the corpus contains: [momo,momo, momi]

Step 1: Split into characters
m o m o
m o m o
m o m i

Step 2: Count pairs and merge the most common one
m o : 5
o m : 3
m i : 1

The most common pair is m o, so merge it: m + o → mo

step3: The corpus becomes:
mo mo
mo mo
mo m i

... repeated 

finally: The corpus becomes:[momo, momo, mo m i]


Word-level tokenization:
Word-level tokenization splits text into complete words:
The cat sat
→ ["The", "cat", "sat"]

Character-level tokenization
Character-level tokenization splits text into individual characters:
The cat
→ ["T", "h", "e", " ", "c", "a", "t"]

| Tokenization method | Advantages | Disadvantages |
|---|---|---|
| **Word-level** | Short token sequences; common words retain their full meaning; | Requires a very large vocabulary; cannot handle unseen words well; |
| **Character-level** | Very small vocabulary; can represent almost any text;  | Produces long sequences; individual characters contain limited meaning; requires more computation and makes learning word-level patterns harder |
| **BPE / subword-level** | Balances vocabulary size and sequence length; handles rare and unseen words by combining subword units;  |  tokens are not always linguistically meaningful |


The merge tree for momo is:
             momo
            /    \
          mo      mo
         /  \    /  \
        m    o  m    o

m + o       ──[5 occurrences]──> mo
mo + mo     ──[2 occurrences]──> momo

## 2.How does vocabulary size affect compression ratio (tokens per character) and sequence length for a fixed text sample? Show measurements from both of your tokenizers on English text.

I used the fixed English news_text sample from the nanochat project’s [`scripts/tok_eval.py`](/Users/zhengxuzhang/Code/nanochat/scripts/tok_eval.py)

The sequence length is the number of tokens produced by the tokenizer:
\[
\text{sequence length} = \text{number of tokens}
\]

The compression measure is:
\[
\text{tokens per character}
=
\frac{\text{number of tokens}}{\text{number of characters}}
\]


| Vocabulary size | Sequence length | Tokens per character | Bytes per token* |
|---:|---:|---:|---:|
| 8,192 | 483 tokens | 0.2679 | 3.77 |
| 32,768 | 406 tokens | 0.2252 | 4.48 |

可见，同样的文本中：
- 8,192 词表需要 483 个 token
- 32,768 词表只需要 406 个 token


因此，32,768 词表使序列长度减少：
\[
\frac{483-406}{483}\times100\% \approx 15.9\%
\]

Therefore, the larger vocabulary provides better compression because it contains more tokens and allows more merge operations. As a result, longer and more frequent character sequences can be combined into a single token.




## 3. Why does vocabulary size matter for the downstream language model?
In your answer, consider the embedding matrix size, the softmax denominator, and the challenge of learning reliable representations for rare tokens.

Embedding matrix
如果词表大小为 \(V\)，embedding dimension 为 \(d\)，输入 embedding 矩阵大小是：
\[
V \times d
\]词表越大，embedding 参数越多。同时，输出层通常也有一个大小约为 \(V \times d\) 的矩阵，因此显存、内存和训练成本都会增加。

Softmax denominator
语言模型计算下一个 token 的概率：
\[
P(y=i|x)=
\frac{\exp(z_i)}
{\sum_{j=1}^{V}\exp(z_j)}
\]词表越大，softmax 分母中需要考虑的候选 token 越多，输出 logits 和 softmax 的计算量也越大。

Rare tokens
更大的词表可以包含更长的词或子词，因此序列更短。但是，一些 token 可能非常罕见，训练过程中出现次数很少，模型难以学习可靠的表示和概率。
较小的词表会把罕见词拆成更常见的子词。这些子词出现次数更多，因此通常更容易学习，但会产生更长的序列。

Sequence length
较大的词表通常会产生更短的 token 序列，从而减少 Transformer 的序列计算量，并让固定 context window 覆盖更多原始文本。

因此，词表大小是一个权衡：
- 更大的词表：序列更短，但 embedding、softmax 和 rare-token 问题更严重。
- 更小的词表：参数和 softmax 更小，但序列更长，计算量可能更大。