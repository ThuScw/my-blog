---
title: "【量化】从零到LGBM | 6 | LGBM代码实现和调用"
date: 2026-08-20T12:29:00+08:00
draft: false
categories: ["量化"]
tags: [ "量化","算法"]
---

# LightGBM 代码实现

在上一篇文章中，我们从数学和工程设计的角度拆解了 LightGBM 的四大核心机制。本文主要来记录其代码实现。

本文分两部分回答这个问题。第一部分，我们用几百行伪代码，从零手搓一个"麻雀虽小、五脏俱全"的迷你LightGBM第二部分，我们带着手搓时拧过的每一颗"螺丝"，去学习如何正确使用 LightGBM 的 Python 官方接口。

---

## 第一部分：手搓 LightGBM

### 0. 总体设计：先看清整张图纸

在写代码之前，我们先明确这个迷你版本要包含哪些功能，以及它们的组装顺序：

1. **EFB 特征捆绑**：在数据进入模型之前，把互斥的稀疏特征合并降维（预处理阶段）；
2. **离散分箱**：把连续特征值映射为整数桶编号（直方图的地基）；
3. **GOSS 采样**：每轮迭代时，保留大梯度样本、采样小梯度样本；
4. **直方图构建与作差**：以 $O(N)$ 的代价统计梯度信息，分裂时用减法省一半计算；
5. **Leaf-wise 生长**：用优先队列实现"谁增益大谁分裂"的精英制，并用正则化参数套上缰绳；
6. **提升主循环**：每轮对"当前预测值"求一二阶导数，建一棵树，累加进模型。

这个顺序与 LightGBM 真实源码的数据流向是一致的。下面逐块实现。

### 1. 目标函数：先明确 $g$ 和 $h$ 从哪来

上一篇我们反复强调：一阶导 $g_i$ 和二阶导 $h_i$ 是**对当前模型预测值求导**的结果，下标 $i$ 表示第 $i$ 个样本。在代码里，这就是一个极其简单的函数。以平方损失 $L=\frac{1}{2}(\hat{y}_i-y_i)^2$ 为例：

```python
import numpy as np

def grad_hess(pred, y):
    """对预测值求导：g 是一阶导（误差推力），h 是二阶导（曲率）"""
    g = pred - y                 # dL/d(pred)
    h = np.ones_like(y)          # d2L/d(pred)2，平方损失下恒为 1
    return g, h
```

注意这个接口的**通用性**：如果换成分类任务，只需要把损失换成对数似然，$g$ 和 $h$ 换成对应的导数即可，后面所有的树构建代码**一行都不用改**。这正是 GBDT 框架"泛函梯度下降"的优雅之处——树只关心梯度，不关心具体任务。

### 2. EFB：互斥特征捆绑

EFB 发生在建树之前的数据准备阶段。它的任务是：检测哪些特征"几乎从不同时非零"，然后把它们打包成一个特征。

```python
def efb_bundle(X, max_conflict_rate=0.01):
    """检测互斥特征，返回捆绑方案（一个列表的列表）"""
    n, m = X.shape
    nonzero = (X != 0) & ~np.isnan(X)          # 非零掩码
    bundles = []
    for j in range(m):
        for b in bundles:
            # 冲突率：与捆内已有特征"同时非零"的样本比例
            conflict = np.mean(nonzero[:, j] & nonzero[:, b].any(axis=1))
            if conflict < max_conflict_rate:   # 几乎互斥，就塞进这个捆
                b.append(j)
                break
        else:
            bundles.append([j])                # 放不下，新开一捆
    return bundles

def efb_merge(X, bundles):
    """按捆绑方案把多列合并成一列，信息不丢失"""
    cols = []
    for b in bundles:
        if len(b) == 1:
            cols.append(X[:, b[0]])
            continue
        merged = np.zeros(X.shape[0])
        offset = 0.0
        for j in b:
            span = np.nanmax(np.abs(X[:, j])) + 1
            # 关键：给每个特征加不同的偏移量，使取值区间互不重叠。
            # 因为特征互斥（不会同时非零），模型能从合并值反推出是谁在生效
            merged = merged + np.where(X[:, j] != 0, X[:, j] + offset, 0.0)
            offset += span
        cols.append(merged)
    return np.column_stack(cols)
```

这段代码对应上一篇讲的"图着色"思想：把特征看作节点，冲突关系看作边，贪心地把不冲突的特征染成同一种颜色（放进同一个捆）。合并时通过**偏移量错开取值区间**，保证信息零损失。

### 3. 离散分箱

分箱的任务是把连续特征值变成整数桶编号。注意这里**不需要对全量数据排序**，我们只需要采样一小部分数据来估计分位数边界，成本几乎可以忽略：

```python
class BinMapper:
    def __init__(self, n_bins=255):
        self.n_bins = n_bins
        self.boundaries_ = []

    def fit(self, X):
        for j in range(X.shape[1]):
            col = X[:, j]
            col = col[~np.isnan(col)]
            # 采样估计分位数边界，而不是全量精确排序
            sample = np.random.choice(col, min(10000, len(col)), replace=False)
            qs = np.linspace(0, 1, self.n_bins + 1)[1:-1]
            self.boundaries_.append(np.unique(np.quantile(sample, qs)))
        return self

    def transform(self, X):
        binX = np.zeros(X.shape, dtype=np.int32)
        for j in range(X.shape[1]):
            miss = np.isnan(X[:, j])
            # 二分查找桶编号；缺失值单独占用一个特殊桶
            binX[~miss, j] = np.digitize(X[~miss, j], self.boundaries_[j])
            binX[miss, j] = len(self.boundaries_[j])
        return binX
```

分箱完成后，原始的 32 位浮点特征矩阵就被替换成了一个 8 位整数的桶编号矩阵——内存占用降到约 1/8，这正是上一篇提到的内存暴降的来源。

### 4. GOSS：单边梯度采样

GOSS 在每轮迭代开始时执行：保留梯度最大的前 $a\%$ 样本，再从剩下的样本里随机抽 $b\%$，并给被抽中的小梯度样本乘上放大系数 $\frac{1-a}{b}$，以补偿分布偏移。

```python
def goss_sampling(g, h, a=0.2, b=0.1):
    n = len(g)
    order = np.argsort(-np.abs(g))             # 按梯度绝对值从大到小
    keep = order[: int(a * n)]                 # 大梯度样本：全部保留
    rest = order[int(a * n):]
    sampled = np.random.choice(rest, int(b * n), replace=False)
    amplify = (1 - a) / b                      # 放大系数，补偿采样带来的分布改变
    g = g.copy(); h = h.copy()
    g[sampled] *= amplify
    h[sampled] *= amplify
    return np.concatenate([keep, sampled]), g, h
```

### 5. 直方图、作差与最佳分裂

直方图就是一个分块中的块：每个块里所有样本的 $g$ 之和、$h$ 之和、样本数。构建它只需要**一遍遍历、只加不排**，时间复杂度严格 $O(N)$。

```python
def build_hist(bin_col, g, h, rows, n_bins):
    """O(行数) 构建直方图：只累加，不排序"""
    G = np.zeros(n_bins); H = np.zeros(n_bins); C = np.zeros(n_bins)
    for i in rows:
        b = bin_col[i]
        G[b] += g[i]; H[b] += h[i]; C[b] += 1
    return G, H, C

def hist_sub(parent, child):
    """直方图作差：父 - 子 = 另一个子。利用加法可逆性，计算量砍半"""
    return parent[0] - child[0], parent[1] - child[1], parent[2] - child[2]
```

有了直方图，寻找最佳分裂点就变成了**对桶边界的一次前缀和扫描**。回顾上一篇的增益公式：

$$Gain = \frac{1}{2}\left[\frac{G_L^2}{H_L+\lambda} + \frac{G_R^2}{H_R+\lambda} - \frac{G^2}{H+\lambda}\right] - \gamma$$

```python
def find_best_split(G, H, C, lam, gamma, min_data):
    """扫描所有桶边界，返回 (最佳桶边界, 最大增益)"""
    GT, HT, CT = G.sum(), H.sum(), C.sum()
    best_bin, best_gain = -1, -np.inf
    GL = HL = CL = 0.0
    for b in range(len(G) - 1):            # 只在桶边界上尝试切分（提速的关键）
        GL += G[b]; HL += H[b]; CL += C[b] # 前缀和：左边三个统计量
        GR, HR, CR = GT - GL, HT - HL, CT - CL
        if CL < min_data or CR < min_data: # 正则化缰绳：叶子最小样本数
            continue
        gain = 0.5 * (GL*GL/(HL+lam) + GR*GR/(HR+lam) - GT*GT/(HT+lam)) - gamma
        if gain > best_gain:
            best_gain, best_bin = gain, b
    return best_bin, best_gain
```

请留意这个循环的规模：它最多只跑 255 次，与样本量 $N$ 完全无关。无论节点里有 1 万个样本还是 1000 万个样本，找分裂点的代价都是常数级的——这就是 LightGBM 快的数学本质。

### 6. Leaf-wise：用优先队列实现精英制

Leaf-wise 的代码化身是一个**以增益为优先级的最大堆**：每次只弹出增益最大的叶子进行分裂。同时，`max_depth`、`min_data`、`min_split_gain` 三根缰绳在分裂时逐一检查。

```python
import heapq, itertools

class Node:
    def __init__(self, rows, depth, hist):
        self.rows, self.depth, self.hist = rows, depth, hist
        self.left = self.right = None
        self.split_feature = self.split_bin = None
        self.weight = 0.0

def build_tree(binX, g, h, rows, p):
    n_feat = binX.shape[1]

    def all_hists(rows):
        return [build_hist(binX[:, j], g, h, rows, p['n_bins']) for j in range(n_feat)]

    def evaluate(node):
        """为节点找出它最好的分裂方案，记在 node.split 上"""
        best = None
        for j in range(n_feat):
            b, gain = find_best_split(*node.hist[j], p['lam'], p['gamma'], p['min_data'])
            if b >= 0 and (best is None or gain > best[2]):
                best = (j, b, gain)
        node.split = best
        return node

    root = evaluate(Node(rows, 0, all_hists(rows)))
    tie = itertools.count()                          # 堆的平局裁判
    pq = [(-root.split[2], next(tie), root)] if root.split else []
    n_leaves = 1

    while pq and n_leaves < p['num_leaves']:         # 叶子预算：num_leaves
        _, _, node = heapq.heappop(pq)               # 永远先分裂增益最大的叶子
        if node.split is None: continue
        j, b, gain = node.split
        if gain < p['min_split_gain']: continue      # 缰绳一：增益门槛
        if node.depth >= p['max_depth']: continue    # 缰绳二：深度上限

        mask = binX[node.rows, j] <= b               # 按桶边界把样本劈成两半
        l_rows, r_rows = node.rows[mask], node.rows[~mask]
        node.split_feature, node.split_bin = j, b

        # 直方图作差：只遍历较小的那一半，另一半用减法得到
        if len(l_rows) <= len(r_rows):
            l_hist = all_hists(l_rows)
            r_hist = [hist_sub(node.hist[k], l_hist[k]) for k in range(n_feat)]
        else:
            r_hist = all_hists(r_rows)
            l_hist = [hist_sub(node.hist[k], r_hist[k]) for k in range(n_feat)]

        node.left  = evaluate(Node(l_rows, node.depth + 1, l_hist))
        node.right = evaluate(Node(r_rows, node.depth + 1, r_hist))
        n_leaves += 1
        for child in (node.left, node.right):
            if child.split:
                heapq.heappush(pq, (-child.split[2], next(tie), child))

    # 生长结束：用最优权重公式给每个叶子"宣判"
    for leaf in collect_leaves(root, []):
        G = g[leaf.rows].sum(); H = h[leaf.rows].sum()
        leaf.weight = -G / (H + p['lam'])            # w* = -Σg / (Σh + λ)
    return root

def collect_leaves(node, out):
    if node.left is None:
        out.append(node)
    else:
        collect_leaves(node.left, out); collect_leaves(node.right, out)
    return out
```

其中，优先队列就是"精英培养制"；`max_depth` 和 `min_data` 就是防止树退化成"庸俗问题链"的缰绳；叶子权重公式 $w^*=-\frac{\sum g}{\sum h+\lambda}$ 就是正则化收缩的代码体现。

### 7. 主循环：把一切组装起来

最后是提升框架的主循环。注意一个细节：**每棵树的梯度，是对"加入之前所有树之后的当前预测值"重新计算的**。

```python
class MiniLightGBM:
    def __init__(self, n_estimators=200, lr=0.1, n_bins=255, num_leaves=31,
                 max_depth=6, min_data=50, min_split_gain=0.0,
                 lam=1e-3, gamma=0.0, use_goss=True, use_efb=True):
        self.p = dict(n_estimators=n_estimators, lr=lr, n_bins=n_bins,
                      num_leaves=num_leaves, max_depth=max_depth, min_data=min_data,
                      min_split_gain=min_split_gain, lam=lam, gamma=gamma,
                      use_goss=use_goss, use_efb=use_efb)

    def fit(self, X, y):
        p = self.p
        if p['use_efb']:                              # 先降维
            X = efb_merge(X, efb_bundle(X))
        mapper = BinMapper(p['n_bins']).fit(X)
        binX = mapper.transform(X)                    # 离散化

        F = np.full(len(y), y.mean())                 # 初始预测（base score）
        self.trees_ = []
        for t in range(p['n_estimators']):
            g, h = grad_hess(F, y)                    # 对当前预测值求导
            rows = np.arange(len(y))
            if p['use_goss']:                         # 采样
                rows, g, h = goss_sampling(g, h)
            tree = build_tree(binX, g, h, rows, p)    # 建树
            self.trees_.append(tree)
            F += p['lr'] * self._predict_tree(binX, tree)   # 累加进模型
        return self

    def _predict_tree(self, binX, tree):
        out = np.zeros(len(binX))
        for i in range(len(binX)):
            node = tree
            while node.left is not None:              # if-else 一路走到叶子
                node = node.left if binX[i, node.split_feature] <= node.split_bin \
                       else node.right
            out[i] = node.weight
        return out

    def predict(self, X):
        # （实际使用时需对 X 做与训练时相同的 EFB 合并与分箱，此处从略）
        ...
```

至此，一个包含全部核心功能的迷你 LightGBM 就完成了。不过，它与工业级实现之间还隔着一些工程距离，这是很显然的。

---

## 第二部分：正确使用 LightGBM 的 Python 接口

### 1. 手搓代码 ↔ 官方参数对照表

| 手搓代码中的变量 | 官方参数 | 说明 |
| :--- | :--- | :--- |
| `n_bins` | `max_bin` | 默认 255，桶的数量 |
| 分箱采样量 10000 | `bin_construct_sample_cnt` | 默认 200000，用多少样本估计桶边界 |
| `min_data` | `min_data_in_leaf` | 默认 20，叶子最小样本数（防单链的缰绳） |
| `gamma` | `min_split_gain` | 分裂的增益门槛（"手续费"） |
| `lam` | `lambda_l2` | 叶子权重 L2 收缩；另有 `lambda_l1` |
| `num_leaves` | `num_leaves` | 默认 31，Leaf-wise 的叶子预算 |
| `max_depth` | `max_depth` | 默认 -1（不限制！务必手动设置） |
| `lr` | `learning_rate` | 默认 0.1 |
| GOSS 的 `a`、`b` | `top_rate`、`other_rate` | 需配合 `data_sample_strategy='goss'` |
| EFB | 无直接参数 | 在 Dataset 构建阶段自动完成 |
| —— | `feature_fraction` / `bagging_fraction` | 列采样 / 行采样，我们手搓版未实现的正则手段 |。

### 2. 原生接口（`lgb.train`）：研究与实战的首选

LightGBM 提供两套接口：sklearn 风格（`LGBMRegressor`）和原生风格（`lgb.train`）。我们主要讲后者。

```python
import lightgbm as lgb
import pandas as pd

# ---- 第一步：数据准备。类别特征转成 category 类型，交给原生机制处理 ----
for col in ['industry', 'concept']:
    X_train[col] = X_train[col].astype('category')
    X_valid[col] = X_valid[col].astype('category')

# ---- 第二步：构建 Dataset。分箱、EFB 等都在这一步内部完成 ----
train_set = lgb.Dataset(X_train, label=y_train,
                        categorical_feature=['industry', 'concept'],
                        free_raw_data=False)
# 关键细节：验证集要 reference 训练集，保证两者使用同一套桶边界
valid_set = lgb.Dataset(X_valid, label=y_valid, reference=train_set)

# ---- 第三步：参数字典 ----
params = {
    'objective': 'regression',      # 平方损失
    'num_leaves': 31,               # 叶子预算
    'max_depth': 6,                 # 给 Leaf-wise 套上缰绳
    'min_data_in_leaf': 50,         # 叶子最小样本数
    'lambda_l2': 1e-3,              # 权重收缩
    'learning_rate': 0.05,
    'feature_fraction': 0.8,        # 列采样，防过拟合
    'verbose': -1,
}

# ---- 第四步：训练 + 早停 ----
model = lgb.train(
    params, train_set,
    num_boost_round=2000,
    valid_sets=[train_set, valid_set],
    valid_names=['train', 'valid'],
    callbacks=[lgb.early_stopping(stopping_rounds=100),
               lgb.log_evaluation(period=100)],
)

# ---- 第五步：预测与解释 ----
pred = model.predict(X_test, num_iteration=model.best_iteration)
imp = pd.Series(model.feature_importance(importance_type='gain'),
                index=model.feature_name()).sort_values(ascending=False)
```

这段代码里有几个值得展开的细节：

**`reference=train_set` 不可省略。** 如果验证集独立分箱，它的桶边界和训练集不一致，模型在验证集上的表现评估就会出现不必要的偏差。`reference` 让验证集复用训练集的分箱映射，这正对应手搓代码里"全模型共用一个 `BinMapper`"的设计。

**早停与 `best_iteration`。** 早停监控的是验证集指标；训练结束后，`model.best_iteration` 记录了验证集表现最好的轮数。预测时显式传入 `num_iteration=model.best_iteration`，避免把过拟合后期的树也累加进预测。

**`importance_type` 的两种口径。** `'split'` 统计特征被用来分裂的次数， `'gain'` 统计特征带来的累计增益。前者容易被高频琐碎特征刷高，**研究因子贡献时建议看 `gain`**。

### 3. 自定义目标函数：把"对预测值求导"亲手写进官方接口

手搓代码里 `grad_hess` 的接口，在官方 API 里对应自定义目标函数。注意签名是**预测值在前**，返回一阶导和二阶导两个数组——这与我们手搓版的约定完全一致：

```python
def custom_l2(pred, train_data):
    y = train_data.get_label()
    g = pred - y
    h = np.ones_like(y)
    return g, h

model = lgb.train(params, train_set, fobj=custom_l2, ...)
```

一旦理解了"目标函数只负责提供 $g$ 和 $h$"这一接口约定，你就会明白：换成分类、分位数回归、甚至自定义的金融损失函数，树构建部分的代码完全不用动。这是 GBDT 框架留给使用者最大的自由度。

### 4. 几个高频踩坑点

**`num_leaves` 才是油门，`max_depth` 是保险。** LightGBM 不会根据 `max_depth` 自动缩小 `num_leaves`。若只设 `num_leaves=255` 而不设深度，树依然可能长出极深的偏科结构。经验做法是两者同设，且 `num_leaves` 不超过 $2^{\text{max\_depth}}$。

**类别特征不要 One-Hot。** 上一篇讲过，One-Hot 会制造大量稀疏互斥列，虽然 EFB 能捆绑它们，但 LightGBM 对 `category` 类型特征有更强大的原生处理（在类别集合之间直接寻找最优"多对多"分裂）。把行业、概念这类特征转成 `category` 交给模型，通常比手动 One-Hot 效果更好、速度更快。

**可复现性要显式声明。** 多线程下浮点累加顺序不同会导致结果轻微漂移。需要严格复现时，固定 `seed`、`deterministic=True` 与 `num_threads`。

**行采样与 GOSS 二选一。** `bagging_fraction` 配合 `bagging_freq` 是传统的随机行采样；若启用 `data_sample_strategy='goss'`，则走梯度采样路线。两者同时设置时行为以 GOSS 为准，不要叠加误用。

**训练集指标只用于观察，不用于决策。** 训练集与验证集指标的差距，本身就是过拟合程度的体温计。这一点在金融数据上尤其致命，也是我们下一篇文章的主题。