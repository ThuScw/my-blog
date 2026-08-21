---
title: "【LLM】从零开始学大语言模型 | 5 | Encoder-Decoder和掩码"
date: 2026-08-21T07:00:00+08:00
draft: false
categories: ["LLM"]
tags: ["LLM","Encoder","Decoder","Casual Mask"]
---

# Encoder-Decoder 架构与 Mask 机制：从翻译模型到自回归生成

在理解了 Attention、Multi-Head Attention、FFN 以及 Transformer Block 之后，我们已经拥有了一个可以反复堆叠的基础模块。下一步要回答的问题是：这些模块如何组成一个完整的模型？在 2017 年的论文《Attention Is All You Need》中，Transformer 被设计成一个 Encoder-Decoder 架构，最初主要用于机器翻译这样的序列到序列任务。也就是说，输入是一段序列，输出也是一段序列，而且输入和输出长度可以不同。

这篇文章将从整体架构、Decoder 的特殊结构、Mask 机制、训练时的 Loss 与反向传播，以及多层 Decoder 中 Cross-Attention 的 KV 来源几个方面，把 Encoder-Decoder Transformer 的工作方式完整梳理一遍。


## 一、从单个 Transformer Block 到 Encoder-Decoder 架构

### 1. Encoder 与 Decoder 的分工

可以把 Encoder-Decoder 结构理解成一个翻译官的工作台。

左侧是 Encoder，也就是编码器。它的任务是深度阅读源语言。例如输入一句中文：“我 / 爱 / AI”，Encoder 会通过多层网络逐步理解这些词之间的关系：谁是主语，谁是谓语，哪个词修饰哪个词，整句话的整体语义是什么。经过若干层 Encoder Block 之后，模型得到的不是一串孤立的词向量，而是一组包含上下文信息的特征向量。可以把它看作 Encoder 对源语言句子进行抽象压缩后得到的“语义表示”。

右侧是 Decoder，也就是解码器。它的任务是逐字生成目标语言。例如它要生成英文：“I / love / AI”。Decoder 不能像 Encoder 那样一次性随意看完整句目标语言的全部内容，因为在真正生成文本时，模型只能依赖已经生成的内容。Decoder 需要一边参考 Encoder 对源语言的理解，一边根据当前已经生成的词预测下一个词。

因此，Encoder 更像是在做理解，Decoder 更像是在做生成。Encoder 的输出是 Decoder 生成时的重要参考。

### 2. Encoder Block 与标准 Transformer Block 的关系

一个常见的问题是：Encoder Block 是不是就是之前讨论过的标准 Transformer Block？

答案是肯定的。标准 Transformer Block 通常包含两个核心部分：Self-Attention 和 FFN。Encoder Block 的结构与此基本一致。Encoder 中的 Self-Attention 不加因果掩码，因此每个位置可以看到整个输入序列。换句话说，Encoder 在处理中文句子“我 / 爱 / AI”时，“爱”这个位置既可以关注前面的“我”，也可以关注后面的“AI”。这对于理解整句语义是合理的，因为输入句子是已经完整给出的，不存在“未来信息泄露”的问题。

Decoder Block 则不同。Decoder 也基于 Transformer Block，但它有两个关键变化。第一，Decoder 的 Self-Attention 需要加入 Causal Mask，防止当前位置看到未来词。第二，Decoder 在 Self-Attention 之后还加入了 Cross-Attention，用于读取 Encoder 的输出。

所以可以这样理解：Encoder Block 是标准 Transformer Block；Decoder Block 是在标准 Transformer Block 基础上增加因果掩码和 Cross-Attention 的变体。它们本质上仍然遵循 Transformer 的基本结构，但任务目标不同，因此可见范围和模块组成也不同。

### 3. 翻译任务中的语义压缩与还原

在机器翻译任务中，Encoder 的作用是把一种语言逐层抽象成一组高维向量。这个抽象过程不仅仅是词义的简单映射。随着网络层数增加，模型会逐步编码词法、句法、语义搭配，甚至某些语境层面的信息。可以形象地说，Encoder 将源语言压缩成一个“语义表示”。

Decoder 则负责把这个语义表示还原成另一种语言。它不是一次性把整句话直接写出来，而是在每一步生成时都参考 Encoder 提供的语义信息，并结合自己已经生成的历史词，预测下一个最合适的词。

需要强调的是，Transformer 最初确实是为机器翻译设计的。后来人们发现，这套架构不仅能做翻译，还能做摘要、问答、文本生成、代码生成等多种任务。于是 Encoder-only、Decoder-only、Encoder-Decoder 等不同变体逐渐发展出来。但从源头看，Encoder-Decoder 是原论文中的经典结构。



## 二、Decoder Block 的内部结构：为什么需要 Cross-Attention

### 1. Masked Self-Attention：理解已经生成的内容

一个标准的 Decoder Block 通常包含三个子层。第一个是 Masked Self-Attention，也就是带因果掩码的自注意力。

在这一层中，Query、Key、Value 都来自 Decoder 自身的输入。假设 Decoder 正在生成英文句子 “I / love / AI”。当模型处理到 “love” 这个位置时，它应该知道前面已经有 “I”，因此可以利用 “I” 来判断语法和语义。但它不应该看到后面的 “AI”，因为在真实生成过程中，“AI” 还没有被生成。

因此，Masked Self-Attention 的作用是让 Decoder 内部已经生成的词之间互相建立联系，同时严格禁止它们看到未来位置。

### 2. Cross-Attention：连接源语言和目标语言

Decoder Block 的第二个子层是 Cross-Attention，这是 Encoder 与 Decoder 之间的桥梁。

在 Cross-Attention 中，Query 来自 Decoder，Key 和 Value 来自 Encoder。可以写成：

$$\text{CrossAttention}(Q_{dec}, K_{enc}, V_{enc}) =\text{softmax}\left(\frac{Q_{dec}K_{enc}^T}{\sqrt{d_k}}\right)V_{enc}$$

这里的含义非常直观。Decoder 在生成当前词时，会产生一个“需求”，也就是 Query。这个 Query 会去 Encoder 输出的 Key 中检索：源语言中哪些位置与当前生成任务最相关？检索到的权重再用于加权 Encoder 的 Value，从而得到当前生成步骤所需要的源语言信息。

例如翻译“我 / 爱 / AI”时，如果 Decoder 当前要生成英文中的 “AI”，它的 Query 会更强烈地匹配 Encoder 中 “AI” 对应的表示。这样模型就知道，这里应该翻译出与 “AI” 相关的内容，而不是凭空生成别的词。

Cross-Attention 是 Encoder-Decoder 架构中非常关键的一层。没有它，Decoder 就只能依赖自己已有的输入，而无法参考源语言句子。

### 3. FFN：独立变换与知识整合

Decoder Block 的第三个子层仍然是 FFN，即前馈神经网络。它的作用与 Encoder Block 中的 FFN 类似：对每个位置的表示做进一步非线性变换，并增强模型的表达能力。

可以把 Attention 理解为“不同位置之间交换信息”，而 FFN 理解为“每个位置在拿到信息之后进行独立加工”。两者交替堆叠，使 Transformer 既能建模长距离依赖，又能对局部表示进行深层变换。



## 三、Mask 机制：让模型只看见该看见的信息

### 1. Padding Mask：处理批内变长序列

在 Attention 中，每个词默认可以看到序列中的其他词。但在实际训练中，有些信息是不应该被看到的。Mask 机制的作用就是告诉模型：哪些位置可以关注，哪些位置必须忽略。

最常见的一种 Mask 是 Padding Mask。

训练时，多个句子通常会被组成一个 batch。为了形成形状规整的张量，同一个 batch 内的序列长度必须一致。但真实句子长短不同，例如一句长 10 个 token，另一句长 5 个 token。为了补齐，短句子后面会加入特殊符号 `<PAD>`。

这些 `<PAD>` 本身没有语义。如果 Attention 仍然把它们当作正常词参与计算，就会污染模型表示。因此，Padding Mask 的作用是在计算 Attention 时，让所有位置对 `<PAD>` 的注意力权重为 0。

实现方式通常是在 Attention score 上将 `<PAD>` 对应的位置填成负无穷。经过 Softmax 之后，这些位置的权重会趋近于 0。

### 2. Causal Mask：防止模型偷看未来

比 Padding Mask 更核心的是 Causal Mask，也常被称为因果掩码或前瞻掩码。它是 Decoder 能够进行自回归生成的关键机制。

在训练文本生成或翻译任务时，模型预测第 3 个词时，不应该看到第 4、5、6 个词。如果能看到，模型就相当于在考试时直接看到了答案。训练阶段的 Loss 可能会非常低，但到了推理阶段，未来词根本不存在，模型会立刻失效。

因此，在 Decoder 的 Self-Attention 中，必须强制每个位置只能看到自己和之前的位置，不能看到之后的位置。

这里容易产生一个理解偏差。Causal Mask 不是“前面的词被后面的词清零”，而是“前面的词看不到后面的词”。更准确地说：

- 第 1 个词只能看到自己；
- 第 2 个词可以看到第 1 个词和自己；
- 第 3 个词可以看到第 1、2 个词和自己；
- 第 4 个词可以看到前面所有词和自己。

以句子 “I / love / AI” 为例。处理 “I” 时，它不能看到 “love” 和 “AI”。处理 “AI” 时，它可以看到 “I”“love” 和 “AI”。这正是“因果”的含义：当前状态只能由过去决定，不能由未来决定。

### 3. 数学实现：用负无穷和 Softmax 屏蔽未来

Causal Mask 的实现非常巧妙。它不是真的删除未来位置，而是在 Softmax 之前把未来位置的 Attention score 变成负无穷。

假设序列长度为 4，未加 Mask 的 Attention score 矩阵是一个 $4 \times 4$ 方阵。Causal Mask 可以写成：

$$\text{Mask} = \begin{bmatrix} 0 & -\infty & -\infty & -\infty \\\\ 0 & 0 & -\infty & -\infty \\\\ 0 & 0 & 0 & -\infty \\\\ 0 & 0 & 0 & 0 \end{bmatrix} $$

然后将其加到 Attention score 上：

$$ \text{Scores}_{\text{masked}} = \frac{QK^T}{\sqrt{d_k}} + \text{Mask} $$

接着执行 Softmax：

$$\text{Weights} = \text{softmax}(\text{Scores}_{\text{masked}})$$

由于$\exp(-\infty) = 0$,所以未来位置对应的注意力权重会变成 0。这样，在后续用注意力权重对 Value 加权求和时，未来信息就不会进入当前位置的表示中。

用一个 PyTorch 风格的示例来表达：

```python
import torch
import torch.nn.functional as F

# scores 的形状通常为 [batch_size, num_heads, seq_len, seq_len]
seq_len = scores.size(-1)

# 生成上三角为 True 的布尔矩阵，代表未来位置
causal_mask = torch.triu(
    torch.ones(seq_len, seq_len, dtype=torch.bool, device=scores.device),
    diagonal=1
)

# 将未来位置填充为负无穷
scores = scores.masked_fill(causal_mask, float("-inf"))

# 正常计算 Softmax
attn_weights = F.softmax(scores, dim=-1)
```

在实际工程中，为了避免某些极端情况下出现数值问题，有时也会用一个很大的负数代替 `-inf`，例如 `-1e9`。但从原理上说，目标是让未来位置的 Softmax 权重变成 0。

### 4. 每一层 Decoder 都需要 Causal Mask

还有一个重要问题：Causal Mask 是只在某一层使用，还是每一层 Decoder 都要使用？

答案是：只要 Decoder Block 中包含 Self-Attention，这一层的 Self-Attention 就必须使用 Causal Mask。

原因很简单。Decoder 的每一层都在处理序列信息。无论第 1 层还是第 12 层，只要某一层允许当前位置看到未来位置，未来信息就会通过这一层流入后续表示。即使其他层加了 Mask，也无法保证整体因果性。因此，所有 Decoder Self-Attention 层都必须遵守同一规则：当前位置只能聚合历史位置的信息。

也就是说，Causal Mask 不是最后输出前才加的一次性处理，而是贯穿整个 Decoder 前向传播过程的内置机制。



## 四、训练流程与梯度传播：端到端如何成立

### 1. Encoder 和 Decoder 是分开训练，还是同步训练？

一个很自然的疑问是：既然 Decoder 的 Cross-Attention 会使用 Encoder 的 Key 和 Value，那是不是应该先训练 Encoder，等 Encoder 训练好之后，再训练 Decoder？

在标准 Encoder-Decoder Transformer 中，答案是否定的。Encoder 和 Decoder 不是分阶段训练的，而是串联在一起端到端同步训练。

整个模型只有一个最终输出，也只有一个最终 Loss。训练时，输入源语言句子进入 Encoder，Encoder 输出语义表示；Decoder 接收目标语言的移位序列，并结合 Encoder 输出生成预测；最后用 Decoder 的预测结果计算 Loss。反向传播时，梯度会依次更新 LM Head、Decoder、Cross-Attention，再通过 Cross-Attention 流入 Encoder，更新 Encoder 的参数。

所以，不存在“Encoder 先训练完成，再给 Decoder 使用”的过程。Encoder 和 Decoder 是在同一次前向传播和反向传播中共同学习的。

### 2. Encoder 有自己独立的 Loss 吗？

在标准翻译任务中，Encoder 没有自己独立的 Loss。

Loss 来自 Decoder 最终输出的预测。具体来说，Decoder 的输出经过 LM Head 得到词汇表上的概率分布，然后与真实目标词计算交叉熵损失。这个 Loss 是整个模型的训练信号。

Encoder 的参数之所以会更新，不是因为 Encoder 有单独的 Loss，而是因为 Decoder 的 Loss 可以通过 Cross-Attention 反向传播到 Encoder。换句话说，Encoder 学习的是“如何产生对 Decoder 生成更有帮助的表示”。

这一点很重要。很多人会误以为 Encoder 负责理解，所以应该有一个专门衡量理解质量的 Loss；Decoder 负责生成，所以有另一个生成 Loss。在原论文的标准机器翻译设置中并不是这样。整个系统作为一个整体被优化。

当然，在后来的许多模型中，比如 BERT 这样的 Encoder-only 模型，确实会设计专门针对 Encoder 的预训练目标。但如果讨论的是原始 Transformer 的 Encoder-Decoder 翻译模型，那么最终 Loss 来自 Decoder 的输出。

### 3. Cross-Attention 中的 Key 和 Value来自 Encoder 的哪一层？

另一个关键问题是：Decoder 使用 Encoder 的 Key 和 Value 时，到底使用的是 Encoder 哪一层的输出？

在标准 Transformer 中，Decoder 所有层的 Cross-Attention 通常都使用 Encoder 最顶层，也就是最后一层的输出。

假设 Encoder 有 6 层。第 1 层可能更多捕捉局部词法和浅层搭配，第 6 层则包含更抽象、更完整的全局语义表示。Decoder 在生成目标语言时，需要的是 Encoder 对整句源语言最成熟的理解结果，因此使用最后一层输出是合理的选择。

这意味着 Encoder 最后一层的输出可以理解为一个共享的语义数据库。Decoder 的第 1 层可以通过 Cross-Attention 查询它，Decoder 的第 6 层也可以继续查询它。不同层的 Decoder 会从不同角度使用这份信息，但它们访问的是同一个来源。

在某些后续研究或变体模型中，也可能存在使用 Encoder 中间层输出、分层对齐、逐层连接等设计。但在经典 Transformer 的标准实现中，通常使用 Encoder 最后一层输出作为所有 Decoder 层 Cross-Attention 的 Key 和 Value 来源。

### 4. LM Head 是什么，它如何参与训练？

Decoder 最后一层输出的向量还不能直接变成单词。它需要经过 LM Head。LM Head 通常可以看作一个线性层：

$$\text{logits} = hW^T$$

其中 $h$ 是 Decoder 最后一层的隐藏状态，形状可以理解为 `[batch_size, seq_len, d_model]`。$W$ 的形状与词汇表大小相关，例如 `[vocab_size, d_model]`。经过 LM Head 后，每个位置都会得到一个长度为词汇表大小的向量，称为 logits。

如果词汇表大小是 30000，那么模型在每个位置上会输出 30000 个分数。对这些分数做 Softmax，就得到下一个词的概率分布：

$$p = \text{softmax}(\text{logits})$$

训练时，模型预测的概率分布会与真实目标词比较。假设真实目标词对应类别为 $y$，模型预测概率为 $p_y$，交叉熵损失的一项就是：

$$L = -\log p_y$$

如果模型给正确答案的概率很高，Loss 就低；如果概率很低，Loss 就高。

### 5. 梯度如何从 LM Head 传回 Decoder 和 Encoder？

反向传播是整个训练机制中最关键的部分。我们可以把它理解成一次误差追责和参数更新过程。

首先是 LM Head。Loss 对 LM Head 的参数求导，更新 LM Head 的权重，使它下次更有可能给正确词更高的分数。

如果 softmax 和交叉熵联合计算，输出层常见的梯度形式非常简洁。设真实目标为 one-hot 向量 $y$，预测概率为 $p$，则 logits 上的梯度大致为：

$$\frac{\partial L}{\partial \text{logits}} = p - y$$

这个梯度会继续传回 Decoder 最后一层的隐藏状态 $h$。也就是说，Decoder 每一层都要根据梯度调整自己的参数，使得最终输出更符合目标序列。

接下来是最关键的一步：梯度如何进入 Encoder？

答案就在 Cross-Attention 中。Cross-Attention 的计算形式是：

$$\text{Attention}(Q_{dec}, K_{enc}, V_{enc})$$

Loss 不仅会对 $Q_{dec}$ 求导，也会对 $K_{enc}$ 和 $V_{enc}$ 求导。由于 $K_{enc}$ 和 $V_{enc}$ 来自 Encoder 的输出，因此梯度会通过这条路径流入 Encoder。

具体来说，梯度先进入 Encoder 最后一层，然后继续向前传播，依次更新 Encoder 的第 6 层、第 5 层，直到第 1 层。这样，Encoder 的所有参数也会因为 Decoder 的 Loss 而得到更新。

这就是端到端训练的含义。模型中的各个部分并不是孤立训练的，而是通过一次完整的计算图连接在一起。Loss 从最终输出开始，沿计算图反向传播，更新所有可训练参数。



## 五、多层 Decoder、共享 KV 与整体前向传播

### 1. Decoder 不止一层

还有一个容易混淆的问题：Decoder 是不是只有一层？

不是。原版 Transformer 中，Encoder 和 Decoder 通常都是 6 层。现代模型可能更深，例如 12 层、24 层，甚至更多。

Decoder 的每一层都有其作用。较浅的层可能更多处理局部语法和初步语义特征，较深的层则逐步提炼出更抽象的表示。生成任务不是一次简单映射，而是需要多层逐步细化。

### 2. 所有 Decoder 层使用的 Encoder KV 是否一致？

在标准设计中，是的。所有 Decoder 层的 Cross-Attention 都使用 Encoder 最后一层输出得到的 Key 和 Value。

这样设计的原因有两个。第一，Encoder 最后一层表示最完整。它是 Encoder 经过多层抽象后的结果，比中间层更适合指导生成。第二，所有 Decoder 层共享同一份 Encoder 输出，可以使结构清晰，训练稳定。

可以这样理解：Encoder 最后一层输出是一份完整笔记。Decoder 的每一层都会翻阅这份笔记，只是每一层关注的问题不同。第 1 层可能用它确定基本语义方向，第 6 层可能用它校准最终生成细节。

### 3. 训练时是一遍跑完，而不是逐词慢慢跑

在训练阶段，Decoder 通常不是一步一步串行生成整句话，而是采用 teacher forcing 的方式并行计算。

假设真实目标序列是：

$$[\text{<BOS>}, I, love, AI]$$

其中，$\text{<BOS>}$是句子开始标记。$Decoder 的输入通常是目标序列的移位版本，例如：

$$[\text{<BOS>}, I, love]$$

模型需要在对应位置分别预测：

$$[I, love, \text{<EOS>}]$$

其中，$\text{<EOS>}$是句子结束标记。

由于 Causal Mask 的存在，每个位置只能看到自己和之前的位置。因此，虽然整个目标序列一次性输入了模型，但每个位置仍然不会看到自己的未来。这样既能并行训练，又能保证生成逻辑。

所以，可以把训练时的前向传播概括为：

1. 源语言输入 Encoder；
2. Encoder 输出最终语义表示；
3. 目标语言移位序列输入 Decoder；
4. Decoder 每一层的 Masked Self-Attention 都使用 Causal Mask；
5. Decoder 每一层的 Cross-Attention 都读取 Encoder 最后一层输出；
6. Decoder 最后一层输出进入 LM Head；
7. LM Head 输出 logits；
8. 根据 logits 和真实目标词计算 Loss；
9. 反向传播更新 LM Head、Decoder 和 Encoder。

### 4. 推理时才是逐词生成

训练时为了效率，可以一次性并行计算多个位置。但在推理阶段，模型没有完整的未来目标序列，只能自回归生成。

推理过程通常是：

1. 输入起始符号 `<BOS>`；
2. 模型预测下一个 token；
3. 把预测出的 token 加入输入序列；
4. 再次送入模型预测下一个 token；
5. 重复这个过程，直到生成结束符号 `<EOS>` 或达到最大长度。

在这个过程中，Causal Mask 仍然起作用。但由于推理时输入序列本来就是从左到右逐步增长的，模型自然只能看到历史词。工程实现中，为了提高速度，还会使用 KV Cache 等技术缓存已经计算过的 Key 和 Value，避免重复计算。



## 六、总结

Encoder-Decoder Transformer 的核心思想可以概括为一句话：Encoder 负责理解输入序列，Decoder 负责在因果约束下生成输出序列。

Encoder Block 与标准 Transformer Block 基本一致，它通过 Self-Attention 和 FFN 对输入进行全局建模。Decoder Block 则在此基础上增加了两个关键机制：Masked Self-Attention 防止当前位置看到未来词，Cross-Attention 使 Decoder 能够读取 Encoder 的语义表示。

Mask 机制是 Transformer 工程中不可忽视的一环。Padding Mask 用于忽略 batch 中补齐的无意义 token；Causal Mask 用于保证自回归生成的正确性。Causal Mask 的实现依赖一个简单但重要的数学事实：Softmax 中负无穷对应的权重为 0。因此，只要在 Softmax 前把未来位置填成负无穷，就能让模型在数学上彻底屏蔽未来信息。

在训练方面，Encoder 和 Decoder 是端到端同步训练的。标准翻译任务中，Encoder 没有独立 Loss，最终 Loss 来自 Decoder 经过 LM Head 后的预测结果。反向传播时，梯度先更新 LM Head 和 Decoder，再通过 Cross-Attention 对 $K_{enc}$ 和 $V_{enc}$ 的依赖传回 Encoder，从而更新 Encoder 的全部参数。

在标准结构中，Decoder 有多层，而且所有 Decoder 层的 Cross-Attention 通常都使用 Encoder 最后一层输出作为 Key 和 Value。这个输出可以看作 Encoder 对源语言的最完整理解。Decoder 每一层都会查询它，用于不同层次的生成决策。

掌握这些内容之后，Transformer 原版架构的主要脉络就清晰了：输入经过 Encoder 编码，输出经过 Decoder 解码；Decoder 用 Causal Mask 保证因果性，用 Cross-Attention 连接输入；整个模型通过一个最终 Loss 端到端训练。这也是理解后续 BERT、GPT、T5 等不同 Transformer 变体的重要基础。