---
title: "【LLM】从零开始学大语言模型 | SSM篇 | 3.5 | 数学知识前置讲解"
date: 2026-09-06T12:00:00+08:00
draft: false
categories: ["LLM"]
tags: ["LLM","SSM","数学"]
---
# 数学知识前置讲解：后续优化篇中一定会用到的数学工具

在接下来的 SSM 优化系列文章中，我们会频繁用到一些超出本科线性代数和高等数学范畴的数学知识和算法思想。本文集中讲解这些前置知识，让读者在后续阅读中不会因为数学工具的陌生而阻碍对核心思想的理解。

本文假设读者已具备线性代数基础（矩阵运算、特征值、矩阵指数）、高等数学基础（微分方程、积分）以及生成函数的基本概念。在此基础上，本文重点讲解两个读者可能尚未接触的领域：**矩阵运算的计算复杂度分析**和**快速傅里叶变换（FFT）**。同时也会简要介绍 Cauchy 矩阵和递推与卷积的联系。


## 一、矩阵运算的计算复杂度

SSM 优化的核心目标之一，就是把 $O(N^3)$ 的密集矩阵运算降到 $O(N)$。要理解为什么这是可能的，我们需要先理解：不同矩阵运算的复杂度分别是多少，以及为什么是这个数量级。

### 1. 为什么需要关心复杂度

在深度学习中，"一个算法能不能用"不仅取决于它在数学上是否正确，还取决于它在工程上是否可行。HiPPO 给出了理论上最优的 $\mathbf{A}$ 矩阵，但由于密集矩阵运算的复杂度过高，直到 S4 通过 DPLR 结构将其优化到可接受的范围，这个理论优势才真正落地。

这里还有一个常被跳过的小票据：S4 在效果上的表现，最早是靠 Long Range Arena（LRA）这类长序列基准里的显著结果让大家信服的。比如在 LRA 六项任务里，S4 的算术平均约为 81.84%，而标准 Transformer 约为 45.32%；到了最长的 Path-X（长度 16000）任务上，Performer、BigBird 这些高效注意力方法也和标准 Transformer 一样崩溃至 0%，S4 仍然达到 86.73%。有的总结材料里会把 S4 的平均写成约 86.1%，这个数和从六项明细算出的 81.84% 并不完全一致，可能来源于不同报告版本或平均方式的差异，所以引用时最好还是以逐任务明细为准。

### 2. 矩阵-向量乘法：$O(N)$ 与 $O(N^2)$ 的区别

**一般矩阵乘向量：** 设 $\mathbf{A} \in \mathbb{R}^{N \times N}$，$\mathbf{h} \in \mathbb{R}^N$，计算 $\mathbf{y} = \mathbf{A}\mathbf{h}$。矩阵 $\mathbf{A}$ 有 $N^2$ 个元素，每个输出分量 $y_i = \sum_{j=1}^{N} A_{ij} h_j$ 需要 $N$ 次乘法和 $N-1$ 次加法。总共 $N$ 个分量，总计算量为 $N \times N = N^2$ 次乘法，即 $O(N^2)$。

**对角矩阵乘向量：** 如果 $\mathbf{A}$ 是对角矩阵，即 $A_{ij} = 0$（$i \neq j$），那么 $y_i = A_{ii} h_i$，每个分量只需 1 次乘法。总计算量为 $N$ 次乘法，即 $O(N)$。

这个从 $O(N^2)$ 到 $O(N)$ 的差距，正是 DPLR 结构的动机之一：通过将矩阵参数化为对角部分加低秩修正，大部分运算退化为对角矩阵操作。

### 3. 矩阵-矩阵乘法：为什么是 $O(N^3)$

设 $\mathbf{A}, \mathbf{B} \in \mathbb{R}^{N \times N}$，计算 $\mathbf{C} = \mathbf{A}\mathbf{B}$。标准算法的实现是三重循环：

$$C_{ij} = \sum_{k=1}^{N} A_{ik} B_{kj}$$

- 外层循环：$i$ 从 1 到 $N$（$N$ 次）。
- 中层循环：$j$ 从 1 到 $N$（$N$ 次）。
- 内层循环：$k$ 从 1 到 $N$（$N$ 次乘加）。

总计算量为 $N \times N \times N = N^3$，即 $O(N^3)$。

**对角矩阵相乘：** 如果 $\mathbf{A}$ 和 $\mathbf{B}$ 都是对角矩阵，$C_{ij} = 0$（$i \neq j$），且 $C_{ii} = A_{ii} B_{ii}$。只需 $N$ 次乘法，$O(N)$。

### 4. 矩阵求逆与特征分解：为什么是 $O(N^3)$

一般 $N \times N$ 矩阵的求逆和特征分解的标准算法（如 LU 分解、QR 分解）都需要 $O(N^3)$ 次运算。这是因为在分解过程中，需要对矩阵的每一行/列进行消元操作，每一步消元涉及 $O(N^2)$ 次运算，总共需要 $O(N)$ 步。

**对角矩阵求逆：** $\mathbf{\Lambda}^{-1} = \text{diag}(1/\lambda_0, 1/\lambda_1, \ldots, 1/\lambda_{N-1})$，只需 $N$ 次除法，$O(N)$。

**对角矩阵的幂：** $\mathbf{\Lambda}^k = \text{diag}(\lambda_0^k, \lambda_1^k, \ldots, \lambda_{N-1}^k)$，只需 $N$ 次幂运算，$O(N)$。

**对角矩阵的指数：** $e^{\mathbf{\Lambda}t} = \text{diag}(e^{\lambda_0 t}, e^{\lambda_1 t}, \ldots, e^{\lambda_{N-1} t})$，只需 $N$ 次指数运算，$O(N)$。

### 5. 秩 1 修正求逆：Sherman-Morrison 如何将 $O(N^3)$ 降到 $O(N)$

一般矩阵求逆是 $O(N^3)$。但如果矩阵具有"对角矩阵加秩 1 修正"的特殊结构 $\mathbf{M} + \mathbf{u}\mathbf{v}^\top$，且 $\mathbf{M}$ 是对角矩阵，Sherman-Morrison 公式允许我们利用已知的 $\mathbf{M}^{-1}$（$O(N)$ 可算）来快速计算 $(\mathbf{M} + \mathbf{u}\mathbf{v}^\top)^{-1}$：

$$(\mathbf{M} + \mathbf{u}\mathbf{v}^\top)^{-1} = \mathbf{M}^{-1} - \frac{\mathbf{M}^{-1}\mathbf{u}\mathbf{v}^\top\mathbf{M}^{-1}}{1 + \mathbf{v}^\top\mathbf{M}^{-1}\mathbf{u}}$$

公式右侧的运算分解：
- $\mathbf{M}^{-1}\mathbf{u}$：对角矩阵乘向量，$O(N)$。
- $\mathbf{v}^\top(\mathbf{M}^{-1}\mathbf{u})$：向量内积，$O(N)$。
- 标量除法：$O(1)$。
- 向量乘标量：$O(N)$。

总复杂度 $O(N)$，远低于直接对 $\mathbf{M} + \mathbf{u}\mathbf{v}^\top$ 求逆的 $O(N^3)$。S4 的 DPLR 结构（$\mathbf{A} = \mathbf{\Lambda} - \mathbf{p}\mathbf{q}^\top$）恰好是这种形式。

### 6. 卷积：$O(L^2)$ 与 $O(L \log L)$ 的区别

**直接卷积：** 给定两个长度为 $L$ 的序列 $a$ 和 $b$，循环卷积 $(a * b)_k = \sum_{n=0}^{L-1} a_n b_{(k-n) \bmod L}$ 需要对每个 $k$（$L$ 个）和每个 $n$（$L$ 个）做乘加，总复杂度 $O(L^2)$。

**FFT 卷积：** 利用卷积定理（时域卷积 = 频域逐元素乘法），先做 FFT（$O(L \log L)$），频域相乘（$O(L)$），再做 IFFT（$O(L \log L)$），总复杂度 $O(L \log L)$。

FFT 如何将 $O(L^2)$ 降到 $O(L \log L)$ 的具体算法，将在下一节详细讲解。

### 7. SSM 优化中的复杂度总结

以下表格总结了 SSM 优化中涉及的关键运算及其复杂度：

- **密集矩阵 $\mathbf{A}$ 的矩阵指数 $e^{\mathbf{A}\Delta}$：** $O(N^3)$（需要特征分解）。S4 用双线性变换避免了矩阵指数。
- **密集矩阵 $\mathbf{A}$ 的幂 $\mathbf{A}^k$：** $O(N^3)$ 每次。S4 用 DPLR 结构避免了直接求幂。
- **对角矩阵 $\mathbf{\Lambda}$ 的幂 $\mathbf{\Lambda}^k$：** $O(N)$。
- **DPLR 矩阵的求逆 $(z\mathbf{I} - \bar{\mathbf{A}})^{-1}$：** $O(N)$（Sherman-Morrison）。
- **卷积核序列的逐项计算：** $O(LN^2)$（每项 $O(N^2)$，共 $L$ 项）。S4 用 Z 变换避免了逐项计算。
- **Z 变换 + Cauchy Matvec + FFT：** $O(L \log L + N)$。


## 二、快速傅里叶变换（FFT）

### 1. 问题的起点：离散傅里叶变换（DFT）

假设我们有一个长度为 $L$ 的离散序列 $\{x_0, x_1, x_2, \ldots, x_{L-1}\}$。例如，这是 SSM 训练时的一个 token 序列在某个特征维度上的数值。

**离散傅里叶变换（DFT）** 将这个"时域"序列转化为一个"频域"序列 $\{X_0, X_1, \ldots, X_{L-1}\}$，定义为：

$$X_k = \sum_{n=0}^{L-1} x_n \cdot e^{-2\pi i \cdot kn / L}, \quad k = 0, 1, \ldots, L-1$$

其中 $i = \sqrt{-1}$ 是虚数单位。

**这个公式在计算什么？** 每个 $X_k$ 是序列 $\{x_n\}$ 与一组复指数 $e^{-2\pi i \cdot kn/L}$ 的内积。直观地说，$X_k$ 度量了原始序列中"频率为 $k$"的成分有多强。$X_0$ 是直流分量（所有元素的和），$X_1$ 是最低频率的交流分量，$X_{L/2}$ 是最高频率的分量。

**直接计算的复杂度：** 对每个 $k$（共 $L$ 个），需要对 $n$ 求和（共 $L$ 项），每项做一次复数乘法和加法。总计算量为 $L \times L = L^2$ 次复数运算，即 $O(L^2)$。

当 $L = 16384$ 时，$L^2 \approx 2.7 \times 10^8$，这在实际训练中是不可接受的。

### 2. FFT 的核心思想：分治

快速傅里叶变换（FFT）是计算 DFT 的高效算法，由 Cooley 和 Tukey 于 1965 年提出。它的核心思想是**分治**：将一个长度为 $L$ 的 DFT 分解为两个长度为 $L/2$ 的 DFT，递归求解。

**关键观察：** $L$ 次单位根 $W_L = e^{-2\pi i / L}$ 具有以下性质：

$$W_L^{L/2} = e^{-2\pi i \cdot (L/2) / L} = e^{-\pi i} = -1$$

这意味着 $W_L^{k + L/2} = -W_L^k$，即后半部分的旋转因子是前半部分的相反数。

**分治步骤（假设 $L$ 是 2 的幂）：**

**第一步：奇偶分组。** 将输入序列按索引的奇偶分成两组：

$$\text{偶数项：} \{x_0, x_2, x_4, \ldots, x_{L-2}\}$$
$$\text{奇数项：} \{x_1, x_3, x_5, \ldots, x_{L-1}\}$$

**第二步：递归计算。** 分别对偶数项和奇数项做长度为 $L/2$ 的 DFT，得到 $E_k$（偶数项的 DFT）和 $O_k$（奇数项的 DFT），$k = 0, 1, \ldots, L/2 - 1$。

**第三步：合并。** 利用 $W_L^{k+L/2} = -W_L^k$ 的性质，原始 DFT 可以写成：

$$X_k = E_k + W_L^k \cdot O_k, \quad k = 0, 1, \ldots, L/2 - 1$$

$$X_{k+L/2} = E_k - W_L^k \cdot O_k, \quad k = 0, 1, \ldots, L/2 - 1$$

这一步只需要 $O(L)$ 次复数乘法和加法（每个 $k$ 对应一次乘法和两次加法）。

**复杂度分析：** 设 $T(L)$ 是计算长度为 $L$ 的 DFT 的复杂度，则：

$$T(L) = 2 \cdot T(L/2) + O(L)$$

展开递推：$T(L) = 2(2T(L/4) + O(L/2)) + O(L) = 4T(L/4) + O(L) + O(L) = \ldots$

递推 $\log_2 L$ 层后，$T(L) = L \cdot T(1) + O(L \log L) = O(L \log L)$。

**从 $O(L^2)$ 到 $O(L \log L)$ 的加速：** 当 $L = 16384$ 时：

- DFT：$L^2 = 2.7 \times 10^8$ 次运算。
- FFT：$L \log_2 L = 16384 \times 14 \approx 2.3 \times 10^5$ 次运算。

加速约 1000 倍。

### 3. 逆 FFT（IFFT）

逆离散傅里叶变换（IDFT）将频域序列 $\{X_k\}$ 还原为时域序列 $\{x_n\}$：

$$x_n = \frac{1}{L} \sum_{k=0}^{L-1} X_k \cdot e^{2\pi i \cdot kn / L}$$

IDFT 的公式与 DFT 几乎相同，只是指数符号相反（$+$ 代替 $-$）且多了一个 $\frac{1}{L}$ 的归一化因子。因此，IDFT 可以用完全相同的 FFT 算法来计算，复杂度同样是 $O(L \log L)$。

### 4. FFT 与卷积的关系

FFT 最重要的应用之一是加速卷积运算。给定两个长度为 $L$ 的序列 $a$ 和 $b$，它们的（循环）卷积定义为：

$$(a * b)_k = \sum_{n=0}^{L-1} a_n \cdot b_{(k-n) \bmod L}$$

直接计算卷积的复杂度是 $O(L^2)$。但有一个极其重要的定理：

> **卷积定理：** 时域中的卷积等价于频域中的逐元素乘法。

即：$\text{FFT}(a * b) = \text{FFT}(a) \cdot \text{FFT}(b)$（逐元素相乘）。

因此，计算卷积的高效方法是：

1. 对 $a$ 做 FFT：$O(L \log L)$。
2. 对 $b$ 做 FFT：$O(L \log L)$。
3. 频域逐元素相乘：$O(L)$。
4. 对结果做 IFFT：$O(L \log L)$。

总复杂度：$O(L \log L)$，对比直接卷积的 $O(L^2)$，在 $L$ 较大时加速显著。

这个技巧正是 SSM 训练时高效计算卷积 $Y = X * \bar{K}$ 的基础。


## 三、Cauchy 矩阵

### 1. 定义

给定两组复数 $\{x_i\}_{i=0}^{L-1}$ 和 $\{y_j\}_{j=0}^{N-1}$（要求 $x_i \neq y_j$），Cauchy 矩阵 $\mathbf{M} \in \mathbb{C}^{L \times N}$ 定义为：

$$M_{ij} = \frac{1}{x_i - y_j}$$

即矩阵的第 $(i, j)$ 个元素是两组复数中对应元素之差的倒数。

### 2. Cauchy 矩阵-向量乘（Cauchy Matvec）

给定一个 $N$ 维向量 $\mathbf{w}$，Cauchy Matvec 要计算 $\mathbf{y} = \mathbf{M}\mathbf{w}$，即：

$$y_i = \sum_{j=0}^{N-1} \frac{w_j}{x_i - y_j}, \quad i = 0, 1, \ldots, L-1$$

**朴素算法：** 对每个 $i$（$L$ 个）和每个 $j$（$N$ 个），做一次除法和加法，总复杂度 $O(LN)$。

**快速算法：** Cauchy 矩阵有特殊的代数结构，可以利用快速算法加速。

当 $x_i$ 是均匀分布的单位根（$x_k = e^{2\pi ik/L}$）时——这恰好是 S4 中的情况——可以利用单位根的代数性质，将 Cauchy Matvec 转化为循环卷积，从而用 FFT 在 $O(L \log L)$ 内完成。

更一般的情况下（$x_i$ 非均匀分布），可以用快速多极子方法（Fast Multipole Method, FMM）在 $O((L+N)\log(L+N))$ 内完成。FMM 的核心思想是将远距离的"源-观测点"交互用低阶多项式近似，避免逐对计算。FMM 属于计算数学的研究生级别内容，本文不展开其完整推导。

### 3. 在 SSM 中的出现

SSM 的训练需要计算卷积核序列。通过 Z 变换，这个计算被归结为在 $L$ 个单位根上求值一个有理函数。当 $\mathbf{A}$ 是对角矩阵时，求值的核心运算恰好是 Cauchy Matvec：

$$\hat{K}(z_k) = \sum_{n=0}^{N-1} \frac{C_n \bar{B}_n}{z_k - \bar{\lambda}_n}$$

其中 $z_k$ 是单位根，$\bar{\lambda}_n$ 是 $\bar{\mathbf{A}}$ 的对角元素。这个求和就是 Cauchy Matvec 的一个实例，可以利用 FFT 在 $O(L \log L)$ 内完成。


## 四、其他数学工具的简要说明

以下数学工具在后续优化篇中会用到，读者应已具备相关基础。这里仅说明它们在 SSM 中的角色，不展开讲解。

### 1. 矩阵指数

矩阵指数 $e^{\mathbf{A}t}$ 在 ZOH 离散化中出现，给出 $\bar{\mathbf{A}} = e^{\mathbf{A}\Delta}$。对于一般密集矩阵，计算 $e^{\mathbf{A}t}$ 需要 $O(N^3)$（通过特征分解）。对于对角矩阵，只需 $O(N)$（对角线元素各自取指数）。这个复杂度差异正是 DPLR 结构优化的动机之一。

### 2. Sherman-Morrison 公式与 Woodbury 恒等式

Sherman-Morrison 公式给出秩 1 修正矩阵的逆：

$$(\mathbf{M} + \mathbf{u}\mathbf{v}^\top)^{-1} = \mathbf{M}^{-1} - \frac{\mathbf{M}^{-1}\mathbf{u}\mathbf{v}^\top\mathbf{M}^{-1}}{1 + \mathbf{v}^\top\mathbf{M}^{-1}\mathbf{u}}$$

当 $\mathbf{M}$ 是对角矩阵时，公式右侧所有运算的复杂度均为 $O(N)$，远低于直接求逆的 $O(N^3)$。S4 的 DPLR 结构（$\mathbf{A} = \mathbf{\Lambda} - \mathbf{p}\mathbf{q}^\top$）恰好是秩 1 修正，因此所有涉及 $(z\mathbf{I} - \bar{\mathbf{A}})^{-1}$ 的计算都可以用此公式在 $O(N)$ 内完成。

Woodbury 恒等式是 Sherman-Morrison 的推广，处理秩 $r$ 修正：$(\mathbf{M} + \mathbf{U}\mathbf{V}^\top)^{-1}$，当 $r \ll N$ 时复杂度为 $O(Nr^2)$。

### 3. Z 变换与生成函数

Z 变换将离散序列 $\{a_i\}$ 转化为复变量函数 $\hat{A}(z) = \sum_i a_i z^{-i}$。在 SSM 中，卷积核序列 $\bar{K}_i = \mathbf{C}\bar{\mathbf{A}}^i\bar{\mathbf{B}}$ 的 Z 变换为：

$$\hat{K}(z) = \mathbf{C}(z\mathbf{I} - \bar{\mathbf{A}})^{-1}\bar{\mathbf{B}}$$

这个封闭形式将"逐项计算 $\bar{\mathbf{A}}^i$"的问题转化为"计算矩阵 $(z\mathbf{I} - \bar{\mathbf{A}})^{-1}$"的问题。当 $\bar{\mathbf{A}}$ 是 DPLR 结构时，矩阵求逆可用 Sherman-Morrison 在 $O(N)$ 内完成。


## 五、递推关系与卷积

SSM 离散化后的递推公式 $h_k = \bar{\mathbf{A}}h_{k-1} + \bar{\mathbf{B}}x_k$ 是一个线性常系数递推。从 $h_0 = 0$ 开始逐步展开，输出 $y_k = \mathbf{C}h_k$ 是过去所有输入的线性组合：

$$y_k = \sum_{i=0}^{k-1} \bar{K}_i x_{k-i}, \quad \text{其中 } \bar{K}_i = \mathbf{C}\bar{\mathbf{A}}^i\bar{\mathbf{B}}$$

这正是卷积的定义。递推关系与卷积之间的这种等价联系，是 SSM 能够在训练时切换到卷积视角的数学基础。

这种等价转换成立需要两个前提条件：

- **线性性：** 递推关系必须是线性的。如果状态更新中嵌入了非线性操作（如 $h_k = \tanh(\ldots)$），叠加原理被破坏，无法展开为卷积。
- **时不变性：** 递推系数 $\bar{\mathbf{A}}$、$\bar{\mathbf{B}}$、$\mathbf{C}$ 不随时间步 $k$ 变化。如果每一步的参数不同，卷积核无法预计算，卷积视角失效。

这也是为什么 SSM 的核心状态演化必须保持纯线性的原因：一旦加入非线性或时变参数，整个并行训练的优势就不复存在。


## 六、总结

本文覆盖了后续优化篇中需要用到的数学工具：

- **矩阵运算的计算复杂度：** 从 $O(N^3)$ 的密集矩阵运算到 $O(N)$ 的对角矩阵运算，以及 Sherman-Morrison 如何利用 DPLR 结构将矩阵求逆从 $O(N^3)$ 降到 $O(N)$。这些复杂度差异正是 S4 所有优化技巧的动机。
- **FFT：** 从 DFT 的定义出发，讲解了 Cooley-Tukey 分治算法如何将复杂度从 $O(L^2)$ 降到 $O(L \log L)$，以及 FFT 如何加速卷积运算。
- **Cauchy 矩阵：** SSM 高效计算中出现的特殊矩阵结构，可利用 FFT 加速矩阵-向量乘法。
- **矩阵指数、Sherman-Morrison/Woodbury、Z 变换：** 这些工具在后续文章中会详细展开，此处仅说明它们在 SSM 中的角色。
- **递推与卷积的等价性：** SSM 双视角的数学基础，成立的前提是线性和时不变性。

掌握这些工具后，读者就可以无障碍地理解后续 SSM 优化篇中的所有数学推导了。
