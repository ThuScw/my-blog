---
title: "【LLM】从零开始学大语言模型 | SSM篇 | 7 | Mamba（上）：选择性机制"
date: 2026-09-07T09:00:00+08:00
draft: false
categories: ["LLM"]
tags: ["LLM","SSM","Mamba"]
---
# Mamba（上）：动机、架构与选择性机制

前六篇我们从连续时间的状态空间方程出发，走过了离散化、HiPPO 理论、S4 的 Block 结构，以及 S4 赖以高效的 DPLR、双线性变换、Cauchy 核与 FFT 加速。S4 已经能在长序列任务上全面超过 Transformer，但它在语言建模上始终差一口气。根本原因在于它是线性时不变（LTI）系统：训练完成后，矩阵 $\mathbf{B}$、$\mathbf{C}$、$\mathbf{A}$ 就固定了，对所有输入一视同仁。

本篇进入 SSM 发展史上最关键的一步——Mamba。我们仍然按“自顶向下”的顺序展开：先看它要解决什么问题，再看它的整体架构，最后放大到一个 Mamba Block 的内部，逐个拆解五个组件，并把选择性机制背后的数学讲透。并行扫描与硬件感知优化这两块工程内容留到下一篇。

## 一、Mamba 要解决什么问题

### 1. 序列建模需要的两种能力

如果把人类阅读长文的过程抽象出来，一个优秀的序列模型需要同时具备两种能力。

**第一是压缩能力（Compression）。** 模型要能把长达几十万字的背景信息压缩并维持在一个固定大小的“脑容量”（隐藏状态）里，而且不能出现“脑容量爆炸”，也就是显存 OOM。

**第二是内容感知与选择性（Content-Awareness / Selectivity）。** 面对海量信息，模型要能根据当前看到的内容，动态决定什么该牢牢记住，什么可以左耳进右耳出。

举个具体的例子。当读到“请注意，接下来的密码是 8492”时，模型应该高度专注，把“8492”刻进记忆；而当读到“今天天气不错，顺便说一句”这类废话时，模型应该降低关注度，保住之前记下的核心内容不被覆盖。

### 2. 现有模型各自的困境

**Transformer** 通过自注意力机制（$QK^\top$）天然具备很强的内容感知能力，注意力权重完全由当前 Query 和 Key 的内容决定。它的代价是灾难性的 $O(L^2)$ 复杂度，长文本的压缩成本极高。

**经典 SSM（如 S4）** 具备很强的压缩能力，推理显存只有 $O(1)$，但它是线性时不变的。矩阵 $\mathbf{B}$、$\mathbf{C}$、$\mathbf{A}$ 在训练好之后就固定了，对所有输入一视同仁。它像一台没有感情的录音机，匀速记录一切，做不到“遇到关键词就竖起耳朵”。

### 3. Mamba 的目标

Mamba 要打破 LTI 的枷锁，在保留 SSM 线性复杂度 $O(L)$ 与常数级推理显存的前提下，赋予 SSM 内容感知（也就是选择性）的能力，使它在语言建模等需要精确信息检索的任务上能够与 Transformer 正面竞争。

## 二、Mamba 的整体架构

### 1. 宏观结构：Decoder-only 堆叠

Mamba 的宏观架构很纯粹，是一个典型的 Decoder-only 堆叠结构，与 GPT 等 Transformer 模型高度一致。数据流如下：

1. 输入 token 序列，形状为 $(B, L)$，其中 $B$ 是批大小，$L$ 是序列长度。
2. 经过 Token Embedding，得到 $(B, L, D)$，$D$ 是模型隐藏维度。
3. 依次通过若干个 Mamba Block，每一层的输出形状都保持 $(B, L, D)$。
4. 最后经过 RMSNorm 与 LM Head，预测下一个 token 的概率分布。

与 Transformer Block（Attention 加 FFN 两个子层）不同的是，这里的每一个 Mamba Block 都是一个高度集成的计算单元，它把序列混合（Sequence Mixing）和通道混合（Channel Mixing）融合在了一起。序列混合负责让不同时间步交换信息，通道混合负责让同一时间步的不同特征通道交换信息。

### 2. 归一化层与残差连接的位置

一个容易被忽略但很重要的细节是归一化层的位置。Mamba 官方实现采用的是 Pre-Norm 结构，归一化层（RMSNorm）放在每个 Mamba Block 的输入端，残差连接则绕过整个 Block。

单个 Block 的完整数据流是这样的：先把输入 $\mathbf{x} \in \mathbb{R}^{B \times L \times D}$ 保存一份；接着对 $\mathbf{x}$ 做 RMSNorm；归一化后的结果进入 Mamba Block 内部（包含扩展、卷积、Selective SSM、门控、投影）；Block 的输出再与之前保存的那份 $\mathbf{x}$ 相加，得到下一个 Block 的输入。

整个模型的堆叠顺序因此是：RMSNorm 到 Mamba Block，再到残差相加，然后进入下一个 RMSNorm 与下一个 Mamba Block，如此循环。这与现代 Transformer（如 LLaMA）的 Pre-LN 结构一致。归一化层位于每个 Block 的入口处，两个 Block 之间不再额外插入归一化层。

### 3. 非线性是在哪里引入的

Mamba 和 S4 一样，遵循“核心线性、外部非线性”的原则。

在 Block 内部，Selective SSM 的核心状态演化方程 $h_t = \bar{\mathbf{A}}_t h_{t-1} + \mathbf{B}_t \mathbf{x}_t$ 依然是纯线性的。非线性来自 Block 内部的 SiLU 激活函数（放在卷积之后）以及门控机制。

在 Block 之间，多个 Mamba Block 纵向堆叠时通过残差连接和归一化层相连。归一化层本身带有可学习的缩放和平移参数，配合残差结构，保证了深层网络的非线性表达能力和训练稳定性。

## 三、Mamba Block 的组件拆解

现在放大看一个 Mamba Block 的内部。假设输入为 $\mathbf{x} \in \mathbb{R}^{B \times L \times D}$（$D$ 是模型隐藏维度，例如 1024），它的数据流分为扩展、分支处理、合并、投影四个阶段，共涉及五个组件：线性扩展、一维因果卷积、选择性 SSM、门控、线性投影与残差。

### 1. 线性扩展（Linear Expansion）

输入是 $\mathbf{x} \in \mathbb{R}^{B \times L \times D}$，经过一个线性层得到 $\mathbf{x}_{exp} = \text{Linear}(\mathbf{x})$，输出形状为 $(B, L, 2ED)$，其中 $E$ 是扩展因子，通常取 2。

这一步的作用有两个：为后续的双分支结构提供充足的特征通道；把特征空间切成两半，一半用于核心的序列处理（SSM），一半用于门控（Gating）。

### 2. 一维因果卷积（1D Causal Conv）

输入是切分后的两路特征 $\mathbf{x}_{ssm}, \mathbf{x}_{gate} \in \mathbb{R}^{B \times L \times ED}$。计算方式是 $\mathbf{x}'_{ssm} = \text{SiLU}(\text{Conv1D}(\mathbf{x}_{ssm}))$，门控分支同理。

这里的卷积在数学操作上与 S4 中的一维因果卷积完全一致：都是用一个小的卷积核（例如 kernel size 取 4）沿着序列维度做滑动窗口计算。所谓“因果”，指卷积核只覆盖当前时间步及之前的时间步（$t, t-1, t-2, t-3$），绝对不包含未来信息 $t+1$。

加入它的理由是这样：SSM 擅长捕捉长距离依赖，但对于极短期的局部模式（比如相邻两三个字符的组合、局部语法结构），标准的小核卷积计算效率更高、更直接。这一层卷积弥补了 SSM 在极短距离建模上的不足，同时通过 SiLU 提供初步的非线性特征提取。从 CNN 的视角理解这件事是合适的。

### 3. 选择性 SSM（Selective SSM）

这是 Mamba 区别于所有前代 SSM 的核心。输入是 $\mathbf{x}'_{ssm} \in \mathbb{R}^{B \times L \times ED}$，核心逻辑是放弃固定的 $\mathbf{B}$、$\mathbf{C}$、$\Delta$，让它们变成当前输入 $\mathbf{x}'_{ssm}$ 的函数。

对于序列中的每一个时间步 $t$，各量按下面的方式计算。

**第一，离散化步长 $\Delta_t$：**

$$\Delta_t = \text{softplus}(\mathbf{W}_\Delta \mathbf{x}'_{ssm, t} + \mathbf{b}_\Delta)$$

其中 $\mathbf{W}_\Delta$ 负责把 $ED$ 维特征投影成 $\Delta$。官方实现用一次低秩投影来生成它：先把特征压到约 $ED/16$ 维，再投回 $ED$ 维。因此 $\Delta_t$ 是每个通道独立的一个标量，参数开销很小。softplus 保证 $\Delta_t > 0$。

**第二，输入投影矩阵 $\mathbf{B}_t$：**

$$\mathbf{B}_t = \mathbf{W}_B \mathbf{x}'_{ssm, t}$$

其中 $\mathbf{W}_B$ 把 $ED$ 维特征投影到状态维度 $N$，因此 $\mathbf{B}_t \in \mathbb{R}^{N}$。

**第三，输出投影矩阵 $\mathbf{C}_t$：**

$$\mathbf{C}_t = \mathbf{W}_C \mathbf{x}'_{ssm, t}$$

同样地，$\mathbf{C}_t \in \mathbb{R}^{N}$。

**第四，状态转移矩阵 $\bar{\mathbf{A}}_t$（ZOH 离散化）：**

$$\bar{\mathbf{A}}_t = \exp(\Delta_t \mathbf{A})$$

这里的 $\mathbf{A}$ 回到了 S4 中初始化好的固定对角矩阵。原因是 $\Delta_t$ 已经是动态的了，$\mathbf{A}$ 只需要提供一个基础的衰减基底。

**第五，状态更新与输出（循环视角）：**

$$h_t = \bar{\mathbf{A}}_t h_{t-1} + \mathbf{B}_t \mathbf{x}'_{ssm, t}$$

$$y_{ssm, t} = \mathbf{C}_t^\top h_t$$

引入选择性之后，系统从线性时不变（LTI）变成了线性时变（Linear Time-Variant, LTV）。

### 4. 选择性如何带来内容感知

选择性的落点就在 $\Delta_t$、$\mathbf{B}_t$、$\mathbf{C}_t$ 这三个量依赖于当前输入 $\mathbf{x}_t$。三个量各自承担一个明确的分工：

- $\Delta_t$ 表示写入强度，决定是覆盖旧记忆还是保留旧记忆。
- $\mathbf{B}_t$ 表示写入方式，决定把输入投影到状态空间的哪个方向。
- $\mathbf{C}_t$ 表示读取方式，决定从状态空间中提取哪部分信息作为输出。

之所以能体现内容感知，是因为 $\mathbf{x}_t$ 是当前时间步的语义表示（Embedding）。

当 $\mathbf{x}_t$ 代表一个重要的关键词（例如“密码”）时，网络通过学习到的 $\mathbf{W}_\Delta$ 会输出一个较大的 $\Delta_t$，强制模型记住这个关键词：覆写旧状态，写入新信息。

当 $\mathbf{x}_t$ 代表一个无关紧要的标点符号时，网络会输出一个较小的 $\Delta_t$，模型忽略它：保留旧状态，不写入新信息。

模型根据输入内容动态调整记忆策略，这就是内容感知能力的具体形态。

### 5. $\Delta_t$ 为什么表示写入强度

上面的结论可以从数学上严格推导出来。为了简化分析，假设 $\mathbf{A}$ 是标量 $-a$（$a > 0$，对应一个衰减模式），则 $\bar{A}_t = e^{-\Delta_t a}$。在 ZOH 离散化下，输入矩阵的离散形式为 $\bar{\mathbf{B}}_t = \left(\frac{1 - e^{-\Delta_t a}}{a}\right)\mathbf{B}_t$。

**情况一：$\Delta_t$ 很大（例如取 10）。** 此时 $\bar{A}_t = e^{-10a} \approx 0$，状态更新变为 $h_t \approx \bar{\mathbf{B}}_t x_t$。旧状态 $h_{t-1}$ 被乘以接近 0 的系数，被完全丢弃，新输入 $x_t$ 完全覆写了状态。这就是“强写入”：遇到重要信息时清空旧记忆，写入新信息。

**情况二：$\Delta_t$ 很小（例如取 0.01）。** 此时 $\bar{A}_t = e^{-0.01a} \approx 1$，同时 $\bar{\mathbf{B}}_t \approx \Delta_t \mathbf{B}_t \approx 0.01\mathbf{B}_t$，也趋近于 0。状态更新变为 $h_t \approx h_{t-1}$。旧状态几乎被完整保留，新输入对状态的影响微乎其微。这就是“不写入、忽略”：遇到无关信息时保留旧记忆，跳过当前输入。

把两种情况放在一起看，$\Delta_t$ 同时控制了旧记忆的衰减速度和新信息的写入强度，它相当于一个统一了遗忘门与写入强度的旋钮。大的 $\Delta_t$ 对应“这个很重要，记住它”，小的 $\Delta_t$ 对应“这个不重要，跳过”。

### 6. 门控机制（Gating）

输入是 SSM 的输出 $\mathbf{y}_{ssm} \in \mathbb{R}^{B \times L \times ED}$ 以及门控分支的信号 $\mathbf{x}'_{gate}$，计算方式为：

$$\mathbf{y}_{out} = \mathbf{y}_{ssm} \odot \text{SiLU}(\mathbf{x}'_{gate})$$

这里的 $\odot$ 是逐元素相乘，相乘的两个张量形状一致，都在特征维度（Channel dimension）上按位置对应。对每个样本、每个时间步、每个特征通道 $i$：

$$y_{out, i} = y_{ssm, i} \times \text{SiLU}(x_{gate, i})$$

这里选用乘法的原因是：加法只能提供偏置（Shift），即 $y = x + b$ 这种形式；乘法可以提供缩放（Scaling）和门控（Gating）。门控分支相当于一个阀门，它经过 SiLU 后的取值决定了主分支（SSM 输出）的哪些特征通道应该被放大、哪些应该被抑制（乘以接近 0 的值）。相较加法，乘法具有更强的非线性表达能力和信息筛选能力。

其中的 SiLU（Sigmoid Linear Unit，也叫 Swish）是一个非线性激活函数：

$$\text{SiLU}(x) = x \cdot \sigma(x)$$

其中 $\sigma(x)$ 是 Sigmoid 函数。

逐元素相乘加上 SiLU 引入的通道混合与非线性，很好地替代了 Transformer 中的 FFN（前馈神经网络）。

### 7. 线性投影与残差

输入是 $\mathbf{y}_{out} \in \mathbb{R}^{B \times L \times ED}$，计算方式为：

$$\mathbf{y} = \text{Linear}(\mathbf{y}_{out}) + \mathbf{x}$$

输出 $\mathbf{y} \in \mathbb{R}^{B \times L \times D}$，传递给下一个 Block。这个加法就是前面提到的残差连接，它把 Block 输入的那份 $\mathbf{x}$ 直接带到输出。

## 四、可学习参数与动态变量

理解了组件之后，还有一个非常关键的问题需要澄清：既然每个时间步的 $\mathbf{B}_t$、$\mathbf{C}_t$、$\Delta_t$ 都不一样，参数量会不会因此暴涨？

### 1. 每步的取值不同，参数却没有变多

需要严格区分可学习参数（Weights）和动态生成的中间变量（Activations）。

在 S4 中，可学习参数是固定的矩阵 $\mathbf{A}$、$\mathbf{B}$、$\mathbf{C}$（每层一套），每个时间步使用的都是同一套值。

在 Mamba 中，可学习参数是投影矩阵 $\mathbf{W}_B$、$\mathbf{W}_C$、$\mathbf{W}_\Delta$（每层一套）。这些投影矩阵的参数量与 S4 中 $\mathbf{B}$、$\mathbf{C}$ 的参数量处于同一数量级。

而 $\mathbf{B}_t$、$\mathbf{C}_t$、$\Delta_t$ 是前向传播时实时计算出来的中间结果，它们不需要作为参数存储和更新。因此，“每个时间步都不一样”并不会让参数量爆炸。

### 2. 权重矩阵与动态变量的作用范围

在一个 Mamba Block 内，$\mathbf{W}_\Delta$、$\mathbf{W}_B$、$\mathbf{W}_C$ 是固定的可学习权重矩阵，对该 Block 内的所有时间步共享。

$\Delta_t$、$\mathbf{B}_t$、$\mathbf{C}_t$ 是动态变量。权重矩阵共享，但每个时间步的输入 $\mathbf{x}_t$ 不同，所以计算出来的结果在每个时间步都不同：

- $\Delta_t = \text{softplus}(\mathbf{W}_\Delta \mathbf{x}_t)$，每步不同。
- $\mathbf{B}_t = \mathbf{W}_B \mathbf{x}_t$，每步不同。
- $\mathbf{C}_t = \mathbf{W}_C \mathbf{x}_t$，每步不同。
- 基础矩阵 $\mathbf{A}$ 是固定的，通常初始化为 HiPPO 对角阵，不随时间步变化；但离散化后的 $\bar{\mathbf{A}}_t = \exp(\Delta_t \mathbf{A})$ 因为 $\Delta_t$ 逐时间步变化，所以也是每步不同的。

### 3. 参数量级的对比

把两边的参数构成放在一起对比会更清楚。

S4 的可学习参数包括：稠密矩阵 $\mathbf{A}$，参数量 $O(N^2)$；向量 $\mathbf{B}$，参数量 $O(N)$；向量 $\mathbf{C}$，参数量 $O(N)$；标量步长 $\Delta$，参数量为 1。

Mamba 的可学习参数包括：基础矩阵 $\mathbf{A}$，参数量 $O(N)$；投影矩阵 $\mathbf{W}_B$，参数量 $O(ED \times N)$；投影矩阵 $\mathbf{W}_C$，参数量 $O(ED \times N)$；投影矩阵 $\mathbf{W}_\Delta$（低秩形式），参数量 $O((ED)^2/16)$。

核心转变在于：S4 中 $\mathbf{B}$、$\mathbf{C}$、$\Delta$ 是直接存储的固定参数；Mamba 中它们由投影矩阵与当前输入实时算出，属于中间变量，不作为参数保存。

### 4. $\mathbf{A}$ 矩阵的角色

$\mathbf{A}$ 的初始化用的是 HiPPO-LegS 的对角近似，特征值取 $-1, -2, \ldots, -N$。官方代码的写法是：

```python
A = -torch.arange(1, d_state + 1).float()   # [-1, -2, -3, ..., -N]
A = A.repeat(d_inner, 1)                    # 对每个通道重复
```

$\mathbf{A}$ 被定义为 `nn.Parameter`，训练过程中会被梯度下降更新，所以它在 Mamba 中依然是一个可学习参数。不过与 S4 相比，它的角色被弱化了。

在 S4 中，$\mathbf{A}$ 是核心参数，它的精确取值直接决定模型的长程记忆能力，训练过程中 $\mathbf{A}$ 的变化非常关键。

在 Mamba 中，由于引入了选择性机制（$\Delta_t$ 动态变化），$\mathbf{A}$ 只提供一个基础的衰减率谱，也就是“有哪些频率的衰减模式”，而实际的衰减强度由 $\Delta_t$ 动态调制。

可以这样理解两者的分工：$\mathbf{A}$ 像一块固定的调色板，提供了 $N$ 种不同衰减率的“颜色”（$-1, -2, \ldots, -N$）；$\Delta_t$ 像画家的手，根据当前输入的内容动态决定用多少颜料，也就是衰减多快。训练结束后，$\mathbf{A}$ 会有微小的更新，但它的核心结构（对角、负实数、均匀分布）基本保持不变。

## 五、选择性的代价：卷积视角失效

引入选择性之后，Mamba 获得了内容感知能力，但付出了一个致命的数学代价。

前面的篇章讲过，S4 之所以能用 FFT 加速训练，前提是系统满足 LTI：卷积核 $\bar{K}$ 只依赖于参数，不依赖于输入，因此可以提前算好。

在 Mamba 中，因为 $\mathbf{B}_t$、$\mathbf{C}_t$、$\bar{\mathbf{A}}_t$ 都依赖于当前输入，卷积核变成了输入依赖的：

$$\bar{K}_{t, \tau} = \mathbf{C}_t \left( \prod_{j=\tau+1}^{t} \bar{\mathbf{A}}_j \right) \mathbf{B}_\tau$$

卷积视角因此彻底失效，S4 赖以生存的 FFT 加速无法使用。如果退回朴素的状态递推（也就是用 for 循环一步步算 $h_t$），训练复杂度会退化到 $O(L)$ 的串行计算，GPU 的并行算力被完全浪费，Mamba 会变得比 Transformer 还慢。

Mamba 是怎样在不使用卷积的情况下，在 GPU 上实现高效并行训练的？这就涉及它在工程和算法层面的两大创新：并行扫描（Parallel Scan）与硬件感知（Hardware-aware）内存优化。下一篇会详细拆解这两块内容。

## 六、$\Delta_t$ 的设计与训练：架构与学习的分工

读到 $\Delta_t = \text{softplus}(\mathbf{W}_\Delta \mathbf{x}_t)$ 这个形式时，一个自然的追问是：这个被定义出来的 $\Delta_t$ 为什么会满足我们需要的条件？它是试出来的吗，还是“训练会解决这个问题”？

准确的回答包含三层：$\Delta_t$ 的设计有明确的数学动机；它的具体数值确实由训练决定；而这个设计之所以可行，来自设计者有意选择的归纳偏置。

### 1. 第一层：设计有明确的数学动机

回到 $\Delta$ 在状态空间模型中的原始含义：它是离散化步长，也就是“我们在多长的时间间隔内观察连续系统”。

在 LTI 系统（S4）中，$\Delta$ 是固定的，意味着以恒定速率采样所有输入。在 Selective SSM（Mamba）中，$\Delta$ 变成输入的函数，意味着根据内容动态调整采样速率。

这个设计选择对应控制论与数值计算中一个成熟的概念：自适应步长（Adaptive Step Size）。求解微分方程时，系统的变化越剧烈，越需要仔细地更新状态；变化越平缓，就可以快速跳过。Mamba 把同样的思想引入序列建模：遇到信息密度高的 token（关键词），模型可以更充分地关注并写入当前输入；遇到信息密度低的 token（标点、停用词），模型快速跳过，保留已有状态。

把这一步与前面关于 $\Delta_t$ 的推导对齐之后，方向的读法是明确的：大的 $\Delta_t$ 对应重置旧状态、把当前输入写进去，小的 $\Delta_t$ 对应保留旧状态、跳过当前输入。让 $\Delta$ 成为输入的函数这个结构性决策，来自对状态空间模型数学结构的理解，属于设计者主动做出的选择。

### 2. 第二层：具体数值由训练决定

设计者决定的是 $\Delta_t$ 的形式：它是输入的线性函数，再经过 softplus 保证为正。而 $\mathbf{W}_\Delta$ 的具体数值通过反向传播学习得到。

训练会找到合适的 $\mathbf{W}_\Delta$。在语言建模任务中，遇到 “the”、“a”、“.” 这类低信息量 token 时，$\mathbf{W}_\Delta \mathbf{x}_t$ 输出较小的值，$\Delta_t$ 随之较小，模型倾向于忽略；遇到 “password”、“therefore”、数字这类高信息量 token 时，$\mathbf{W}_\Delta \mathbf{x}_t$ 输出较大的值，$\Delta_t$ 随之较大，模型倾向于强写入。

训练并不需要发明 $\Delta_t$ 这个概念，它只需要在已有的框架内找到最优的映射关系。

### 3. 第三层：这个设计为什么能工作

接下来的问题是：为什么把动态性放在 $\Delta$ 上，而没有放在 $\mathbf{A}$ 或状态维度上？这涉及归纳偏置（Inductive Bias）的选择。设计者做了下面这些判断。

**让 $\Delta$ 动态变化。** $\Delta$ 控制时间步长，是最自然的“注意力强度”旋钮。改变 $\Delta$ 会同时影响遗忘速率和写入强度，一个参数解决两个问题。

**让 $\mathbf{B}$、$\mathbf{C}$ 动态变化。** $\mathbf{B}$ 控制写入方向，$\mathbf{C}$ 控制读取方向。让它们随内容变化，等价于动态选择记忆的子空间。

**不让 $\mathbf{A}$ 动态变化。** $\mathbf{A}$ 提供基础衰减率谱，是系统的骨架。如果 $\mathbf{A}$ 也随输入变化，系统会变得极不稳定，而且计算复杂度暴增。

**选用 softplus。** $\Delta$ 需要无上界，因为步长可以非常大；如果改用 sigmoid，取值会被限制在 $(0, 1)$ 区间内。

**选用线性投影。** 它简单、高效、参数少，实验表明已经足够表达。

这些设计选择共同构成了 Mamba 的归纳偏置。它们来自设计者基于对问题的理解所做的预先设定，训练负责填充的只是其中的具体数值。好的归纳偏置让训练更容易找到好的解，差的归纳偏置会让训练事倍功半。

### 4. 与 Transformer 的 Attention 对比

“设计加训练”这个模式在 Transformer 中同样存在。

设计者决定的结构：注意力权重为 $\text{softmax}(QK^\top / \sqrt{d})$；Mamba 中写入强度为 $\Delta_t = \text{softplus}(W_\Delta x_t)$。

这样设计的理由：注意力用点积衡量相关性，用 softmax 归一化为概率分布；Mamba 用时间步长控制记忆衰减，用 softplus 保证为正。

训练学习的内容：Transformer 学习 $W_Q, W_K, W_V$ 的具体数值；Mamba 学习 $W_\Delta, W_B, W_C$ 的具体数值。

对应的归纳偏置：Transformer 假设“相关性可以用向量内积衡量”；Mamba 假设“记忆策略可以用时间步长动态调节”。

在 Transformer 中，很少有人会追问“为什么用点积 $QK^\top$，而不用更简单的 $Q + K$”这类问题。设计者基于“用内积衡量相关性”的直觉做出了这个选择，然后由训练负责学习具体的 $W_Q$、$W_K$。Mamba 的 $\Delta_t$ 设计遵循完全相同的逻辑。

可以这样总结：$\Delta_t$ 的设计来自设计者基于状态空间模型数学结构所选择的、具有物理意义的“旋钮”（时间步长），训练则负责学习什么时候把旋钮拧大、什么时候拧小。这就是深度学习中架构设计与参数学习的分工。

## 七、小结

本篇的内容可以归纳为以下几点。

- Mamba 的目标是在保留 SSM 线性训练复杂度和 $O(1)$ 推理显存的前提下，为 SSM 补上内容感知能力，方法是让 $\mathbf{B}$、$\mathbf{C}$、$\Delta$ 成为当前输入的函数，把系统从 LTI 变成 LTV。
- Mamba 的宏观结构是 Decoder-only 堆叠，每个 Mamba Block 内部同时完成序列混合与通道混合，归一化层（RMSNorm）采用 Pre-Norm，放在每个 Block 的入口。
- 一个 Mamba Block 由五个组件构成：线性扩展、一维因果卷积、选择性 SSM、门控、线性投影与残差。状态演化本身是线性的，非线性由卷积后的 SiLU 与门控引入。
- 选择性 SSM 中，每步的 $\Delta_t$、$\mathbf{B}_t$、$\mathbf{C}_t$ 都不同，但可学习参数只有每层一套的投影矩阵，参数量没有爆炸。基础矩阵 $\mathbf{A}$ 仍然可学习，但角色被弱化为提供基础衰减率谱。
- $\Delta_t$ 同时控制旧记忆的衰减速度与新信息的写入强度：$\Delta_t$ 大时状态被覆写，$\Delta_t$ 小时状态被保留。它的形式由设计者选定，具体数值由训练学习。
- 选择性的代价是卷积核变成输入依赖，S4 的 FFT 加速失效。Mamba 用并行扫描算法与硬件感知优化来填补这个空缺，这是下一篇的主题。
