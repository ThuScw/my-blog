---
title: "【LLM】从零开始学大语言模型 | SSM篇 | 8 | Mamba（下）：并行扫描与硬件感知优化"
date: 2026-09-07T10:00:00+08:00
draft: false
categories: ["LLM"]
tags: ["LLM","SSM","Mamba"]
---
# Mamba（下）：并行扫描与硬件感知优化

上一篇我们给 SSM 装上了“选择性”，让它变成线性时变（LTV）系统，代价是卷积视角失效、S4 赖以生存的 FFT 加速无法使用。如果退回最朴素的 RNN 循环视角，用 for 循环一步步算 $h_t$，GPU 上训练 Mamba 会比 Transformer 慢得多。

本篇从两个层面拆解 Mamba 的应对办法：数学优化，把串行递推变成可并行的计算；工程优化，消除 GPU 上的内存带宽瓶颈。最后给出 Mamba 与 Transformer、S4 的整体对比。

## 一、问题的起点：串行递推

无法使用 FFT 之后，我们只能回到递推公式：

$$h_t = \bar{\mathbf{A}}_t h_{t-1} + \bar{\mathbf{B}}_t x_t$$

这个式子存在严格的串行依赖：算 $h_2$ 必须等 $h_1$ 算完，算 $h_3$ 必须等 $h_2$ 算完。在拥有上万个计算核心的 GPU 上，串行执行等于浪费了绝大部分算力。

破局的思路是：寻找一种数学方法，把看似串行的递推转化为可以并行计算的形式。答案就是并行扫描（Parallel Scan），而它成立的前提是操作必须满足结合律（Associativity）。

## 二、并行扫描（Parallel Scan）

### 1. 把递推写成仿射变换

为了符号简洁，令 $u_t = \bar{\mathbf{B}}_t x_t$，它是一个已知向量。递推式变为：

$$h_t = \bar{\mathbf{A}}_t h_{t-1} + u_t$$

这在数学上是一个仿射变换（Affine Transformation）。我们可以把每一步的“转移”抽象为一个二元组 $(\mathbf{A}_t, u_t)$，它的作用是把前一个状态 $h$ 映射到新状态：

$$(\mathbf{A}_t, u_t) \circ h = \mathbf{A}_t h + u_t$$

### 2. 定义二元运算符 $\otimes$

如果连续执行两步（先执行第 1 步，再执行第 2 步），状态的变化是：

$$h_2 = \mathbf{A}_2 h_1 + u_2 = \mathbf{A}_2 (\mathbf{A}_1 h_0 + u_1) + u_2 = (\mathbf{A}_2 \mathbf{A}_1) h_0 + (\mathbf{A}_2 u_1 + u_2)$$

这说明两个连续的步骤可以合并为一个新的步骤。据此定义二元运算符 $\otimes$：

$$(\mathbf{A}_2, u_2) \otimes (\mathbf{A}_1, u_1) = (\mathbf{A}_2 \mathbf{A}_1, \quad \mathbf{A}_2 u_1 + u_2)$$

关键在于，这个 $\otimes$ 运算满足结合律，即 $C \otimes (B \otimes A) = (C \otimes B) \otimes A$。下面给出严格证明。

### 3. 结合律的证明

**定义。** 二元运算符 $\otimes$ 定义为：

$$(\mathbf{A}_2, u_2) \otimes (\mathbf{A}_1, u_1) \triangleq (\mathbf{A}_2 \mathbf{A}_1, \quad \mathbf{A}_2 u_1 + u_2)$$

其中 $\mathbf{A}_1, \mathbf{A}_2$ 是 $N \times N$ 矩阵，$u_1, u_2$ 是 $N$ 维向量。

**待证命题。** 对任意三个元素 $(\mathbf{A}, u)$、$(\mathbf{B}, v)$、$(\mathbf{C}, w)$，有：

$$\left((\mathbf{C}, w) \otimes (\mathbf{B}, v)\right) \otimes (\mathbf{A}, u) = (\mathbf{C}, w) \otimes \left((\mathbf{B}, v) \otimes (\mathbf{A}, u)\right)$$

**计算左边。** 先算 $(\mathbf{C}, w) \otimes (\mathbf{B}, v) = (\mathbf{C}\mathbf{B}, \quad \mathbf{C}v + w)$。再把结果与 $(\mathbf{A}, u)$ 做 $\otimes$：

$$(\mathbf{C}\mathbf{B}, \quad \mathbf{C}v + w) \otimes (\mathbf{A}, u) = \left((\mathbf{C}\mathbf{B})\mathbf{A}, \quad (\mathbf{C}\mathbf{B})u + (\mathbf{C}v + w)\right)$$

化简得到：

$$\text{左边} = \left(\mathbf{C}\mathbf{B}\mathbf{A}, \quad \mathbf{C}\mathbf{B}u + \mathbf{C}v + w\right)$$

**计算右边。** 先算 $(\mathbf{B}, v) \otimes (\mathbf{A}, u) = (\mathbf{B}\mathbf{A}, \quad \mathbf{B}u + v)$。再把 $(\mathbf{C}, w)$ 与结果做 $\otimes$：

$$(\mathbf{C}, w) \otimes (\mathbf{B}\mathbf{A}, \quad \mathbf{B}u + v) = \left(\mathbf{C}(\mathbf{B}\mathbf{A}), \quad \mathbf{C}(\mathbf{B}u + v) + w\right)$$

利用矩阵乘法的结合律 $\mathbf{C}(\mathbf{B}\mathbf{A}) = (\mathbf{C}\mathbf{B})\mathbf{A} = \mathbf{C}\mathbf{B}\mathbf{A}$，以及矩阵对向量乘法的分配律 $\mathbf{C}(\mathbf{B}u + v) = \mathbf{C}\mathbf{B}u + \mathbf{C}v$，得到：

$$\text{右边} = \left(\mathbf{C}\mathbf{B}\mathbf{A}, \quad \mathbf{C}\mathbf{B}u + \mathbf{C}v + w\right)$$

**结论。** 左边与右边相等，都等于 $\left(\mathbf{C}\mathbf{B}\mathbf{A}, \quad \mathbf{C}\mathbf{B}u + \mathbf{C}v + w\right)$，命题成立。

这个结合律成立的核心原因有两个：矩阵乘法本身满足结合律；矩阵对向量的乘法满足分配律。

### 4. 树形归约：把深度从 $O(L)$ 降到 $O(\log L)$

既然 $\otimes$ 满足结合律，就可以使用并行扫描算法（例如 Blelloch Scan）。

假设序列长度 $L = 8$。朴素的串行做法是 $((((((h_0 \otimes s_1) \otimes s_2) \otimes s_3) \dots) \otimes s_8)$，需要 7 步串行。

并行扫描则按树形结构组织：第一层同时计算 $(s_1 \otimes s_2)$、$(s_3 \otimes s_4)$、$(s_5 \otimes s_6)$、$(s_7 \otimes s_8)$；第二层把第一层的结果两两合并；第三层得到最终的全局前缀和。每一层内部的所有计算都是彼此独立的，可以在 GPU 上同时执行。

由此带来复杂度的变化：串行递推的计算深度（也就是延迟）是 $O(L)$；并行扫描的计算深度降到 $O(\log L)$。

通过定义满足结合律的 $\otimes$ 运算符，Mamba 把 LTV 系统的串行递推转化成了 GPU 上高度并行的树形归约操作。这从数学上弥补了失去 FFT 的遗憾。


## 三、工程优化：GPU 内存层次与瓶颈

有了并行扫描，Mamba 在理论上已经很快了。但在真实 GPU 上跑起来，作者发现速度依然不理想，原因是遇到了内存带宽瓶颈（Memory-bound）。要理解这一点，先要了解 GPU 的内存层次结构。

### 1. HBM 与 SRAM

**HBM（高带宽内存，也就是显存）** 容量大（几十 GB），读写速度相对较慢。

**SRAM（片上缓存）** 容量极小（几 MB），读写速度极快（比 HBM 快几十倍），而且计算单元（ALU）直接连着它。

### 2. 标准实现的灾难

在并行扫描中，我们需要保存中间状态 $h_t$。对于批大小 $B$、序列长度 $L$、状态维度 $N$，状态张量的大小是 $B \times L \times N$。

如果在计算过程中不断把 $h_t$ 从 SRAM 写回 HBM，下一步计算时再从 HBM 读回 SRAM，那么 GPU 大部分时间都花在等待数据搬运（IO）上，真正用于矩阵乘法（Compute）的时间很少。这就是内存带宽瓶颈。

## 四、Mamba 的两个工程策略

### 1. 策略 A：算子融合（Kernel Fusion）

在 PyTorch 中，`Conv1D`、`SSM`、`SiLU`、`Linear` 是彼此独立的算子。每执行一个算子，数据都要在 HBM 和 SRAM 之间进出一次。

Mamba 作者手写了一个很大的 CUDA Kernel（底层算子），把上述所有操作缝合在一起：数据一旦从 HBM 读入 SRAM，就一直留在 SRAM 中；在 SRAM 内依次完成卷积、SSM 并行扫描、门控相乘、线性投影；只有最终的输出结果才写回 HBM。

这样做的结果是，IO 次数从 5 次降到 2 次（一进一出），速度大幅提升。

### 2. 策略 B：不保存状态（重计算 / Recomputation）

在标准的并行扫描中，为了做反向传播求梯度，必须在前向传播时把每一个中间状态 $h_t$ 都保存在 HBM 中，这会带来 Activation Memory 爆炸。

Mamba 做了一个反直觉的决定：前向传播时根本不把 $h_t$ 写入 HBM。

前向传播在 SRAM 中算完 $h_t$ 后，直接算出输出 $y_t$，然后立刻丢弃 $h_t$，只保存输入 $x$ 和参数。反向传播需要 $h_t$ 来算梯度时，重新算一遍。因为 SSM 的计算很快，利用 SRAM 重新跑一次前向扫描的代价，远远小于在 HBM 中读写庞大 $h_t$ 的 IO 代价。

### 3. 重计算的具体做法：按块重算

一个容易产生的误解是：反向传播时是不是要把 $h_1$ 到 $h_L$ 全部算完存起来，再挑出需要的？实际情况不是这样。如果那样做，内存占用和直接保存没有区别，重计算也就失去了意义。

真实的做法是分块重计算。前向传播时，每一块只保留块边界处的状态（该块的起始状态或结束状态），块内部的中间状态在计算完该块的输出后全部丢弃。反向传播时，当需要第 $k$ 块内部某个 $h_t$ 来计算梯度，就读取该块的起始状态，在 SRAM 中从这个起始状态开始重新递推，算出块内所有时间步的 $h_t$，用它算完梯度后再丢弃。

举一个具体的例子。设序列长度 $L = 12$，块大小为 4，分成 3 块。

前向传播时，块 1 从 $h_0$ 递推到 $h_4$，保存 $h_4$ 这个块边界状态，丢弃 $h_1, h_2, h_3$；块 2 从 $h_4$ 递推到 $h_8$，保存 $h_8$，丢弃 $h_5, h_6, h_7$；块 3 从 $h_8$ 递推到 $h_{12}$，保存 $h_{12}$，丢弃 $h_9, h_{10}, h_{11}$。最终 HBM 中实际保存的只有 $h_0, h_4, h_8, h_{12}$ 四个向量；如果把每个时间步的状态都保存下来，则是 $h_0$ 到 $h_{12}$ 共 13 个向量。

反向传播处理块 2 时（需要 $h_5, h_6, h_7$ 的梯度）：先读取块 2 的起始状态 $h_4$；在 SRAM 中重新递推，依次算出 $h_5 = \bar{A}_5 h_4 + \bar{B}_5 x_5$、$h_6 = \bar{A}_6 h_5 + \bar{B}_6 x_6$、$h_7 = \bar{A}_7 h_6 + \bar{B}_7 x_7$；用这三个状态计算梯度；算完后丢弃。

分块重计算带来两方面的收益。内存方面，HBM 中不需要保存全部 $h_t$（大小为 $B \times L \times N$），只需要保存输入 $x$ 和参数（大小为 $B \times L \times ED$）。计算方面，重算一次前向扫描的额外计算量，远远小于在 HBM 和 SRAM 之间搬运 $h_t$ 的 IO 时间，因为前者是 compute-bound，后者是 memory-bound。

一句话概括：每一块保留的是块边界的起始状态（一个向量），块内部的中间状态全部不保存；反向传播时从块边界状态重新递推，把块内的 $h_t$ 重新算一遍。得到 $h_t$ 的方式始终是重新计算，内存里保存的只有块边界状态。

## 五、关于算子融合的两个补充问题

### 1. 它是不是只是一个常数级优化

算子融合带来的不是常数级的加速。在内存带宽受限（memory-bound）的场景下，它带来的是数量级的提升。

原因在于：如果不融合，每个算子（如 `Conv1D`、`SSM`、`SiLU`）执行时都要从 HBM 读入数据、计算、再写回 HBM；GPU 的 HBM 带宽（例如 A100 的 2 TB/s）远低于计算单元的算力；当计算量小于数据搬运量时，GPU 大部分时间在等数据，真正用于计算的时间很少。算子融合消除了中间结果的 HBM 读写，让数据留在 SRAM 中流动，直接把瓶颈从内存带宽转移到计算能力。

实测效果是：在长序列上，融合后的选择性扫描算子比朴素实现快 20 到 40 倍。这属于数量级的收益，远超百分之五、百分之十这种常数级的微调。

### 2. 能否无缝迁移到别的模型

不能无缝迁移，原因有三点。

**开发成本极高。** 算子融合需要手写 CUDA 代码（或使用 Triton 等底层框架），并针对特定的计算图（DAG）设计。Mamba 的融合算子是专门为“扩展、SSM、门控、投影”这一流程设计的。

**架构依赖性强。** Transformer 的 FlashAttention 是另一个算子融合的例子，它专门针对 Attention 的 $QK^\top V$ 计算模式。Mamba 的融合算子无法直接用在 Transformer 上。

**硬件依赖性强。** 融合策略需要根据 GPU 的 SRAM 大小、带宽、计算单元数量来调整，不同型号的 GPU（如 A100 与 H100）可能需要不同的融合策略。

不过，“减少中间结果的内存读写，把多个算子合并成一个”这个思想是通用的，可以应用到任何深度学习模型中。典型的例子有：Transformer 方向的 FlashAttention，融合了 $QK^\top$、Softmax 与 $V$ 乘法；Mamba 方向融合了 Conv1D、SSM、SiLU、Gating、Linear；一个普通的 MLP 也可以融合 Linear、ReLU、Linear。

所以准确的结论是：算子融合是解决内存瓶颈的关键手段，它带来的收益远超常数级优化；同时它的具体实现高度依赖架构与硬件，可以借鉴思想，很难直接照搬。

## 六、Mamba 的完整形态

把本篇的两步优化与前面的内容合起来，可以对照三大架构的核心指标。

**核心数学：** Transformer(Attention) 使用全局矩阵乘法 $QK^\top V$；S4(LTI SSM) 使用全局卷积（FFT）；Mamba(Selective SSM) 使用并行扫描（Parallel Scan）。

**计算复杂度：** Transformer 是 $O(L^2)$；S4 是 $O(L \log L)$（FFT 卷积）；Mamba 的总计算量与序列长度成正比，也就是 $O(L)$，并行扫描带来的额外代价只是 $O(\log L)$ 的计算深度。

**内容感知：** Transformer 强（Softmax）；S4 无（LTI）；Mamba 强（$\Delta$、$B$、$C$ 输入依赖）。

**推理显存：** Transformer 是 $O(L)$（KV Cache 爆炸）；S4 是 $O(1)$；Mamba 是 $O(1)$。

**工程优化手段：** Transformer 用 FlashAttention；S4 依赖复杂的 Cauchy 算法；Mamba 用算子融合加重计算（硬件感知）。

一句话总结：Mamba 通过发现线性递推的结合律，利用并行扫描算法在数学上恢复了并行训练能力；又通过算子融合与重计算，在工程上消除了 GPU 的内存 IO 瓶颈，最终实现了训练快、推理省、带选择性的完整形态。

## 七、小结

- LTV 系统的递推 $h_t = \bar{\mathbf{A}}_t h_{t-1} + \bar{\mathbf{B}}_t x_t$ 可以写成仿射变换，并用二元组 $(\mathbf{A}_t, u_t)$ 表示。
- 定义 $\otimes$ 运算并证明它满足结合律之后，串行递推可以转化为树形归约，计算深度从 $O(L)$ 降到 $O(\log L)$。
- 并行扫描与硬件感知优化都只用于训练。推理阶段使用逐 token 的串行递推，每步计算量与显存都是 $O(1)$。
- GPU 上的瓶颈往往来自 HBM 与 SRAM 之间的搬运。算子融合把多个算子缝合成一个 CUDA Kernel，让数据留在 SRAM 中，IO 次数从 5 次降到 2 次。
- 重计算在前向传播时不保存中间状态 $h_t$，反向前按块从块边界状态重新递推，用少量额外计算换取大幅显存节省。
- 算子融合带来的是数量级的加速（长序列上比朴素实现快 20 到 40 倍），但它的实现高度依赖具体架构与硬件，思想通用、代码无法直接照搬。
