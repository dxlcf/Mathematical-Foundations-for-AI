# 随机梯度下降与 Mini-batch（SGD & Mini-batch Gradient Descent）

> **核心问题：** 当数据量达到百万甚至亿级时，如何高效地进行梯度下降训练？

---

## 批量梯度下降的瓶颈

上一节介绍的梯度下降，每次更新参数时需要计算**全部训练样本**上的损失和梯度：

$$\nabla \mathcal{L}(\theta) = \frac{1}{N} \sum_{i=1}^{N} \nabla \mathcal{L}_i(\theta)$$

其中 $N$ 是训练集大小，$\mathcal{L}_i$ 是第 $i$ 个样本上的损失。

这在数据量小时没有问题，但当 $N = 10^6$ 甚至 $10^9$ 时：

- 每一步更新都需要扫描整个数据集
- 内存放不下全部数据
- 训练极其缓慢

**解决方案：不用全部数据，用一部分数据的梯度来近似全局梯度。**

---

## 随机梯度下降（Stochastic Gradient Descent，SGD）

最极端的近似：**每次只随机选 1 个样本**来计算梯度，然后立即更新参数。

$$\theta \leftarrow \theta - \eta \cdot \nabla \mathcal{L}_i(\theta)$$

其中 $i$ 是随机选取的样本索引。

### SGD 的直觉

批量梯度下降像是问了所有人的意见再做决定；SGD 像是随机问一个人，立即行动。

每次的决策都很"噪声"，方向不准确，但胜在**速度快**，一个 Epoch 内参数更新 $N$ 次，而不是 1 次。

### SGD 的损失曲线

```
损失曲线示意：

批量梯度下降          SGD
L ↑                  L ↑
  |*                   | * *
  | **                 |* * **
  |   ***              |    * * *
  |      ****          |       * * **
  +─────────→迭代      +──────────→迭代
  （平滑）              （震荡但总体下降）
```

SGD 的曲线抖动明显，但整体趋势仍在下降。

---

## Mini-batch 梯度下降（Mini-batch Gradient Descent）

实践中最常用的方法：**每次随机选取 $B$ 个样本**（称为一个 **batch**）计算梯度：

$$\nabla \mathcal{L}_{\text{batch}}(\theta) = \frac{1}{B} \sum_{i \in \mathcal{B}} \nabla \mathcal{L}_i(\theta)$$

$$\theta \leftarrow \theta - \eta \cdot \nabla \mathcal{L}_{\text{batch}}(\theta)$$

这是批量梯度下降（全量）和 SGD（单样本）之间的折中。

### 批大小的选取

典型的批大小 $B$ 取值：$16, 32, 64, 128, 256$（通常取 2 的幂次，便于 GPU 并行计算）。

| 批大小 $B$ | 梯度估计 | 更新频率 | 训练稳定性 | 内存占用 |
|-----------|---------|---------|-----------|---------|
| 1（纯 SGD） | 噪声大 | 最高 | 差 | 最低 |
| 32 ~ 128 | 较准确 | 中等 | 较好 | 中等 |
| $N$（全量） | 精确 | 最低 | 最好 | 最高 |

---

## 关键概念：Epoch、Iteration、Batch

这三个概念容易混淆，需要区分清楚：

**Batch（批）：** 一次前向/反向传播使用的数据子集，大小为 $B$。

**Iteration（迭代次）：** 完成一个 batch 的计算和参数更新，称为一次 iteration。

**Epoch（轮）：** 将全部训练数据过一遍所需的 iteration 数。

$$\text{每 Epoch 的 iterations} = \lceil N / B \rceil$$

**例：** 训练集 50,000 个样本，batch size = 100，则每个 Epoch 包含 500 次 iteration。

---

## 随机性的作用：噪声是敌人还是朋友？

SGD 和 Mini-batch 引入了**梯度噪声**，这看起来是缺点，但在某些情况下反而有益：

### 帮助逃离局部极小值和鞍点

损失函数在高维空间中有大量的局部极小值和鞍点（梯度为零但不是全局最小）。确定性的批量梯度下降一旦陷入这些位置就无法逃脱，而带噪声的 SGD 有机会"抖出去"。

```
          ↑ L
    局部极小  |    /\
      |      |   /  \___/ ←鞍点附近平坦区域
      ↓      |  /
             | /
             |/___________→ θ
```

### 正则化效果

梯度噪声在一定程度上起到了隐式正则化的作用，有助于模型泛化。

### 代价

收敛曲线震荡，最终解的精度稍低，需要配合**学习率衰减**来进一步降低损失。

---

## 学习率衰减（Learning Rate Decay）

随着训练进行，梯度噪声带来的好处逐渐减弱，震荡反而影响精度。常见的解决方案是随时间降低学习率：

**阶梯衰减（Step Decay）：** 每隔固定 Epoch 将学习率乘以衰减因子

$$\eta_t = \eta_0 \times \gamma^{\lfloor t / s \rfloor}$$

**余弦退火（Cosine Annealing）：** 学习率按余弦曲线从初始值降到接近 0

$$\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\frac{t\pi}{T}\right)$$

**指数衰减（Exponential Decay）：** $\eta_t = \eta_0 \cdot e^{-\lambda t}$

现代训练框架（PyTorch、TensorFlow）均内置了这些调度器（Scheduler）。

---

## 数据打乱（Shuffling）

使用 Mini-batch 时，通常在每个 Epoch 开始前**随机打乱数据顺序**，再切分成 batch。

这样做的目的：
- 避免相邻 batch 数据分布偏差造成梯度估计的系统性偏差
- 增加随机性，帮助训练

```python
# PyTorch 中的典型用法
DataLoader(dataset, batch_size=32, shuffle=True)
```

---

## SGD 与 Mini-batch 在深度学习中的实际地位

今天大多数深度学习框架中，"SGD"这个名字通常指的其实是 **Mini-batch SGD**，即用小批量数据计算随机梯度。

真正的单样本 SGD 在实践中几乎不使用（太不稳定），而全量批量梯度下降只在特定小规模场景下出现。

**Mini-batch SGD 是现代深度学习训练的基础**，在此基础上发展出了 Momentum、Adam 等更复杂的优化器（见下一节）。

---

## 本节小结

- 批量梯度下降在大数据下效率极低，引出随机梯度近似思想
- SGD 每次用 1 个样本更新参数，速度快但噪声大
- Mini-batch 梯度下降是折中：用小批量数据估计梯度，兼顾效率与稳定性
- 梯度噪声有时反而有助于逃离局部极小值，起到隐式正则化作用
- 实践中通常配合学习率衰减和数据打乱一起使用
- 深度学习框架中的"SGD"通常指 Mini-batch SGD
