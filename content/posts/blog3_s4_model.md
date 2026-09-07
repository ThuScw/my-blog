---
title: "【LLM】从零开始学大语言模型 | SSM篇 | 3 | S4 Model 简洁"
date: 2026-09-06T11:00:00+08:00
draft: false
categories: ["LLM"]
tags: ["LLM","SSM","S4"]
---
# S4 Model 简洁：一个 SSM Block 内部到底在做什么

上一篇文章介绍了 SSM 的数学基础和典型 Block 结构。本文将聚焦于 S4（Structured State Space Sequence Model）这个 SSM 家族中第一个真正取得工程成功的模型，逐个拆解 S4 Block 中每个组件的任务、输入、输出和设计目的。

本文的讲解方式类似讲解 Transformer Block：先讲整体架构，再逐个组件说明"它是什么、它做什么、它为什么存在"，不涉及数学优化和工程优化的细节。


## 一、S4 的整体定位

S4 于 2021 年提出，是第一个在长序列建模任务上全面超越 Transformer 的 SSM 模型。它在 Long Range Arena（LRA）基准上取得了全面领先的成绩——按原始 LRA 六项任务的算术平均来看，S4 约为 81.84%，而当时的 Transformer 变体约为 45%。

> 数据注记：有的总结材料里会把 S4 在 LRA 上的平均准确率写成约 86.1%。这个数字和上文按六项明细算出的 81.84% 并不完全吻合，可能来源于不同的报告版本、是否包含特定任务变体、加权平均还是算术平均等差异。读者如果在写作或复习时引用这一数据，建议优先以逐任务明细为准，并注意区分“某一版本汇总的平均值”和“从明细算出的算术平均”。

S4 的核心贡献不在于改变了 SSM 的基本方程，而在于解决了两个实际问题：

- **如何初始化矩阵 $\mathbf{A}$：** 通过 HiPPO 理论推导出理论上最优的 $\mathbf{A}$ 矩阵，使 SSM 从一开始就具备长程记忆能力。
- **如何高效计算：** 通过 DPLR（对角加低秩）结构和 Z 变换，将卷积核计算的复杂度从 $O(LN^2)$ 压缩到 $O((L+N)\log(L+N))$。

但这些优化细节将在后续的优化篇中讲解。本文先专注于 S4 Block 的架构本身。

一个完整的 S4 模型由 $L$ 个 S4 Block 纵向堆叠而成（$L$ 通常为 4、6、8 或 12），每个 Block 的结构完全相同，但参数独立。


## 二、S4 Block 的整体数据流

S4 Block 的输入是一个三维张量 $\mathbf{X} \in \mathbb{R}^{B \times L \times d}$，其中 $B$ 是 batch size，$L$ 是序列长度，$d$ 是特征维度（如 256、512、768）。输出 $\mathbf{Y}$ 与输入同形状，用于残差连接和传递给下一个 Block。

S4 Block 的核心设计是"双路门控"架构：输入被分成两路，左路经过 SSM 核心进行序列建模，右路作为门控信号提供非线性调制。两条路径最终合并输出。

整体数据流可以概括为：

1. 输入 $\mathbf{X}$ 经过线性投影，维度从 $d$ 扩展到 $2d$。
2. $2d$ 维特征被分成两半：左路 $\mathbf{X}_A$（$d$ 维）和右路 $\mathbf{X}_B$（$d$ 维）。
3. 左路经过 SSM 核心层和非线性激活，得到 $\mathbf{H}_A'$。
4. 右路经过门控激活，得到门控权重。
5. $\mathbf{H}_A'$ 与门控权重逐元素相乘，得到 $\mathbf{H}_{gated}$。
6. $\mathbf{H}_{gated}$ 经过输出线性投影，维度恢复为 $d$。
7. 与原始输入做残差连接，再经过归一化，得到输出 $\mathbf{Y}$。

接下来逐个拆解每个组件。


## 三、组件 1：输入线性投影

### 做什么

将输入特征维度从 $d$ 扩展到 $2d$：

$$\mathbf{X}' = \mathbf{X}\mathbf{W}_{in} + \mathbf{b}_{in}$$

其中 $\mathbf{W}_{in} \in \mathbb{R}^{d \times 2d}$。

### 为什么需要它

扩展到 $2d$ 是为了给后续的双路分裂做准备。左路需要 $d$ 维进入 SSM 核心，右路需要 $d$ 维作为门控信号。如果不做扩展，分裂后每路只有 $d/2$ 维，表达能力不足。

扩展因子选择 2（而非 Transformer FFN 中常见的 4），是因为 SSM 核心本身已经提供了强大的序列建模能力，不需要像 FFN 那样通过大扩展比来增加参数容量。


## 四、组件 2：双路分裂

### 做什么

将 $\mathbf{X}' \in \mathbb{R}^{B \times L \times 2d}$ 沿特征维度切成两半：

$$\mathbf{X}' = [\mathbf{X}_A \| \mathbf{X}_B]$$

其中 $\mathbf{X}_A, \mathbf{X}_B \in \mathbb{R}^{B \times L \times d}$。

### 为什么需要它

这种设计来自门控线性单元（Gated Linear Unit, GLU）的思想。左路 $\mathbf{X}_A$ 承载序列建模的任务，右路 $\mathbf{X}_B$ 承载特征选择的任务。两路并行处理后通过逐元素相乘合并，使得模型既能捕捉上下文关系，又能动态过滤特征。


## 五、组件 3：S4 核心层（左路）

这是 S4 Block 的灵魂，负责序列混合（Sequence Mixing），让每个 token 能够感知整个序列的上下文信息。S4 核心层由两个子组件构成。

### 子组件 3.1：一维因果卷积（1D Causal Conv）

**做什么。** 对左路输入 $\mathbf{X}_A$ 做一维因果卷积，卷积核大小 $k$ 通常为 3 或 5。

对于序列中第 $t$ 个位置，输出为：

$$\mathbf{H}_{local}[t] = \sum_{j=0}^{k-1} \mathbf{W}[j] \odot \mathbf{X}_A[t - j]$$

其中 $\mathbf{W}[j]$ 是卷积核的第 $j$ 个权重向量，$\odot$ 表示逐元素相乘（depthwise 卷积，每个通道独立卷积），当 $t-j < 0$ 时用零填充。

**为什么需要它。** SSM 核心擅长捕捉中远距离的依赖关系，但对于长度为 2 或 3 的极短期局部模式（例如相邻字符的组合），用一个简单的小核 CNN 来捕捉效率更高。这个小核卷积作为 SSM 核心的补充，填补了极短距离建模的微小空白。

**因果性的体现。** "因果"意味着计算第 $t$ 个位置的输出时，只使用 $t, t-1, \ldots, t-k+1$ 位置的输入，绝不使用未来的信息。这保证了自回归语言模型的正确性。

### 子组件 3.2：S4 状态空间层

**做什么。** 这是 S4 核心层的核心。它接收 $\mathbf{X}_A$，通过 SSM 方程对其进行序列建模，输出 $\mathbf{H}_{ssm}$。

具体分为三步：

**第一步，离散化。** 根据当前参数 $\mathbf{A}$、$\mathbf{B}$、$\mathbf{C}$、$\Delta$，计算离散化的矩阵 $\bar{\mathbf{A}}$ 和 $\bar{\mathbf{B}}$。

**第二步，计算卷积核。** 计算长度为 $L$ 的卷积核序列 $\bar{K}_i = \mathbf{C}\bar{\mathbf{A}}^i\bar{\mathbf{B}}$，$i = 0, 1, \ldots, L-1$。

**第三步，执行因果卷积。** 将输入 $\mathbf{X}_A$ 与卷积核 $\bar{K}$ 做一维因果卷积：$\mathbf{H}_{ssm} = \mathbf{X}_A * \bar{K}$。

在训练时，这三步整体以卷积方式在 GPU 上并行计算。在推理时，第二步和第三步被折叠为循环递推，逐步处理新 token。

**为什么需要它。** 这是 S4 Block 区别于普通 CNN 或 RNN 的核心组件。它通过精心设计的矩阵 $\mathbf{A}$（由 HiPPO 理论推导）来维护一个极小的隐藏状态，用这个状态来压缩和摘要整个历史序列的信息。卷积核的长度等于序列长度 $L$，意味着模型在理论上可以看到序列中任意远的信息。

**为什么 S4 能做而 RNN 做不到。** 关键在于卷积核 $\bar{K}_i = \mathbf{C}\bar{\mathbf{A}}^i\bar{\mathbf{B}}$ 的计算。由于 SSM 核心是线性的且时不变，我们可以预计算整个卷积核，然后用 FFT 一次性并行完成卷积。传统 RNN 由于内部的非线性，无法这样做。

### 子组件 3.3：合并局部与全局

**做什么。** 将一维因果卷积的输出和 S4 状态空间层的输出相加：

$$\mathbf{H}_A = \mathbf{H}_{local} + \mathbf{H}_{ssm}$$

**为什么需要它。** 将短程的局部特征和长程的全局特征融合在一起，为后续的非线性处理提供更丰富的表示。

### 3.4 非线性激活

**做什么。** 对合并后的结果施加 GELU 激活函数：

$$\mathbf{H}_A' = \text{GELU}(\mathbf{H}_A)$$

GELU 的定义为 $\text{GELU}(x) = x \cdot \Phi(x)$，其中 $\Phi$ 是标准正态分布的累积分布函数。

**为什么需要它。** SSM 核心是纯线性的，如果不加非线性，多层 SSM 堆叠后的表达能力等价于单层线性变换。非线性激活为模型提供了拟合复杂函数的能力。


## 六、组件 4：门控机制（右路）

### 做什么

右路 $\mathbf{X}_B$ 经过一个非线性激活函数（如 Sigmoid 或 SiLU），生成门控权重：

$$\text{Gate} = \sigma(\mathbf{X}_B)$$

然后与左路的输出逐元素相乘：

$$\mathbf{H}_{gated} = \mathbf{H}_A' \odot \text{Gate}$$

### 为什么需要它

门控机制是 SSM Block 中 Channel Mixing（通道混合）的核心，它对应于 Transformer 中 FFN 的角色。

门控的作用是动态控制每个特征通道的信息流。对于每个 token 的每个特征维度，门控值接近 1 意味着该维度的信息被保留，接近 0 意味着被抑制。这种逐元素的调制让模型能够根据上下文选择性地放大或缩小不同特征通道的贡献。

相比于 Transformer 中 FFN 的两层全连接网络，门控机制的计算量更小（只涉及逐元素运算，不涉及跨维度的矩阵乘法），但同样能提供强大的非线性表达能力。


## 七、组件 5：输出线性投影

### 做什么

将门控后的结果通过一个线性层映射：

$$\mathbf{Y}' = \mathbf{H}_{gated}\mathbf{W}_{out} + \mathbf{b}_{out}$$

其中 $\mathbf{W}_{out} \in \mathbb{R}^{d \times d}$。

### 为什么需要它

线性投影层提供了一次跨特征维度的信息混合，将门控后的特征进行线性组合，增强特征的表达能力。同时将维度恢复为 $d$，为残差连接做准备。


## 八、组件 6：残差连接与归一化

### 做什么

将 Block 的输出与原始输入相加，然后做归一化：

$$\mathbf{Y} = \text{LayerNorm}(\mathbf{X} + \mathbf{Y}')$$

### 为什么需要它

**残差连接：** 保证信息在深层网络中稳定传递。即使某个 Block 暂时学不到有用的东西，残差连接确保原始信息不会被破坏（输出至少等于输入）。同时，残差连接在反向传播时提供了一条"梯度高速公路"，保证梯度能够稳定回传到浅层。

**归一化：** 将每个 token 的特征向量的数值分布稳定在合理范围内，防止某些维度过大或过小，保证深层网络的训练稳定性。


## 九、S4 Block 与 Transformer Block 的组件对应

为了帮助有 Transformer 背景的读者建立直觉，将 S4 Block 的每个组件与 Transformer Block 做对应：

**Transformer Block 的组件：**

- Multi-Head Attention → 负责序列混合
- FFN → 负责通道混合（非线性特征变换）
- LayerNorm → 稳定训练
- 残差连接 → 保持信息传递和梯度流动

**S4 Block 的组件：**

- S4 核心层（SSM + 小核卷积） → 负责序列混合
- 门控机制 → 负责通道混合
- LayerNorm → 稳定训练
- 残差连接 → 保持信息传递和梯度流动

S4 Block 的独特之处在于它将序列混合和通道混合整合到了一个紧凑的双路结构中，而不是像 Transformer 那样分成两个独立的子层。这种设计使得 S4 Block 的参数量和计算量更小，同时通过 SSM 核心获得了对长序列的高效建模能力。


## 十、完整数据流总结

一个 S4 Block 的完整计算过程如下：

1. **输入：** $\mathbf{X} \in \mathbb{R}^{B \times L \times d}$
2. **线性投影：** $\mathbf{X}' = \mathbf{X}\mathbf{W}_{in}$，$\mathbf{X}' \in \mathbb{R}^{B \times L \times 2d}$
3. **分裂：** $\mathbf{X}_A, \mathbf{X}_B = \text{split}(\mathbf{X}')$，各 $\in \mathbb{R}^{B \times L \times d}$
4. **左路处理：**
   - 一维因果卷积：$\mathbf{H}_{local} = \text{CausalConv1D}(\mathbf{X}_A)$
   - S4 状态空间层：$\mathbf{H}_{ssm} = \text{S4Core}(\mathbf{X}_A)$
   - 合并：$\mathbf{H}_A = \mathbf{H}_{local} + \mathbf{H}_{ssm}$
   - 非线性激活：$\mathbf{H}_A' = \text{GELU}(\mathbf{H}_A)$
5. **右路处理：** $\text{Gate} = \sigma(\mathbf{X}_B)$
6. **门控合并：** $\mathbf{H}_{gated} = \mathbf{H}_A' \odot \text{Gate}$
7. **输出投影：** $\mathbf{Y}' = \mathbf{H}_{gated}\mathbf{W}_{out}$
8. **残差与归一化：** $\mathbf{Y} = \text{LayerNorm}(\mathbf{X} + \mathbf{Y}')$
9. **输出：** $\mathbf{Y} \in \mathbb{R}^{B \times L \times d}$


## 十一、S4 的实验表现

S4 在 Long Range Arena（LRA）基准的六个任务上全面领先。LRA 是 2021 年 Google Research 提出的基准，专门用来考察序列模型在长程依赖上的能力：

- **ListOps**（序列长度 2000，嵌套算术表达式）：S4 达到 58.35%，Transformer 只有 36.37%。
- **Text**（序列长度 1000-4000，长文本情感分类）：S4 达到 76.02%，Transformer 只有 64.27%。
- **Retrieval**（序列长度 4000-8000，文档相关性判断）：S4 达到 87.09%，Transformer 只有 57.46%。
- **Image**（序列长度 1024，像素级图像分类）：S4 达到 88.65%，Transformer 只有 42.44%。
- **Pathfinder**（序列长度 1024，视觉路径追踪）：S4 达到 94.20%，Transformer 只有 71.40%。
- **Path-X**（序列长度 16000，超长视觉路径追踪）：S4 达到 86.73%，Transformer 完全崩溃为 0%。

特别是在 Path-X 任务上，序列长度达到 16000，不仅标准 Transformer 变体崩溃至 0%，包括 Performer（ListOps 18.01 / Text 65.40 / Retrieval 53.82 / Image 42.77 / Pathfinder 77.05 / Path-X 0.00）和 BigBird（ListOps 36.07 / Text 64.02 / Retrieval 59.24 / Image 40.83 / Pathfinder 74.87 / Path-X 0.00）在内的高效注意力方法也在 Path-X 上达到 0%，而 S4 仍然达到了 86.73%。这充分证明了 SSM 在超长序列建模上的优势。

> 补充说明：有的总结材料里会提到 S4 在 LRA 上的平均准确率约为 86.1%、Transformer 约为 45%。按上面六项明细算出的算术平均，S4 约为 81.84%、Transformer 约为 45.32%，两者并不完全一致。造成这种差异的可能原因包括不同报告版本、是否包含特定任务变体、加权平均还是算术平均等；读者在引用时建议以逐任务明细为准，并注意区分“某一版本的平均值”和“从明细算出的算术平均”。


## 十二、总结

S4 Block 的设计哲学可以用一句话概括：用最紧凑的结构同时完成序列混合和通道混合。

- SSM 核心层负责序列混合，让每个 token 感知整个上下文。它内部是纯线性的，保证了卷积视角的可用性和高效并行训练。
- 门控机制负责通道混合，提供非线性特征变换能力。它在 SSM 核心之外工作，不破坏核心的线性性。
- 一维因果卷积补充极短期的局部建模能力。
- 残差连接和归一化保证深层网络的训练稳定性。

S4 的成功证明了 SSM 可以在长序列任务上超越 Transformer。但它也有一个根本性的局限：S4 是线性时不变（LTI）系统，对所有输入使用完全相同的记忆策略，无法根据内容动态选择"记什么、忘什么"。这个局限性在后续的 Mamba 模型中得到了解决。
