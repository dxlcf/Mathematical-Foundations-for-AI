# 03 图算法与 GNN 直觉

## 1. 经典图算法概览

图算法是 AI 系统的重要组成部分，无论是传统推荐系统中的路径查询，还是 GNN 中消息传递的设计，都能在经典算法中找到对应的直觉。

本节重点介绍三类算法：**图遍历**、**最短路径**、**中心性度量**，并在最后建立通向 GNN 的桥梁。

---

## 2. 图遍历

### 2.1 广度优先搜索（BFS，Breadth-First Search）

**核心思想**：从起点出发，先访问所有距离为 1 的邻居，再访问距离为 2 的邻居，以此类推——按"层"展开。

**伪代码**：

```
BFS(G, s):
    队列 Q = {s}
    已访问集合 visited = {s}
    while Q 非空:
        v = Q.dequeue()
        for u in 邻居(v):
            if u not in visited:
                visited.add(u)
                Q.enqueue(u)
```

**时间复杂度**：$O(|V| + |E|)$

**在 AI 中的意义**：
* BFS 天然计算**最短路径**（在无权图中）
* GNN 的第 $k$ 层聚合，等价于将每个节点的感受野（Receptive Field）扩展到 $k$ 跳邻居——这正是 BFS 逐层展开的过程

### 2.2 深度优先搜索（DFS，Depth-First Search）

**核心思想**：从起点出发，沿一条路径尽可能深入，遇到死路再回溯。

**在 AI 中的意义**：
* 拓扑排序（Topological Sort）基于 DFS，用于确定计算图中的执行顺序
* 决策树的递归构建遵循 DFS 的模式

---

## 3. 最短路径算法

### 3.1 Dijkstra 算法

**适用场景**：有向/无向**加权图**，边权非负。

**核心思想**：贪心策略——始终选择当前距离起点最近的未访问节点，更新其邻居的距离估计。

设 $d[v]$ 为从起点 $s$ 到 $v$ 的当前最短距离估计：

$$d[u] = \min_{v \in \text{已确定}} \left(d[v] + w_{vu}\right)$$

**时间复杂度**：使用优先队列时为 $O((|V| + |E|) \log |V|)$

**AI 应用**：
* 知识图谱中的实体关联路径查找
* 机器人路径规划
* 网络延迟优化

### 3.2 随机游走（Random Walk）

**定义**：从节点 $v$ 出发，以均匀概率随机选择一个邻居移动，重复 $k$ 步。

**转移概率矩阵**（Transition Matrix）：

$$P = D^{-1} A$$

其中 $P_{ij} = \frac{A_{ij}}{d_i}$ 表示从节点 $i$ 跳到节点 $j$ 的概率。

**稳态分布**（Stationary Distribution）：当随机游走足够长时，到达各节点的概率趋于稳定：

$$\pi_i = \frac{d_i}{\sum_j d_j}$$

度越高的节点越容易被访问——这正是 **PageRank** 算法的核心思想。

**AI 应用**：
* **Node2Vec**：通过带偏置的随机游走生成节点序列，再用 Word2Vec 训练节点嵌入（Node Embedding）
* **DeepWalk**：纯随机游走 + Skip-gram，是最早的图嵌入方法之一
* **GraphSAGE** 采样策略的直觉也源于此

---

## 4. 节点中心性度量

节点并非都同等重要。**中心性**（Centrality）度量节点在图中的"重要程度"。

| 中心性 | 定义 | 含义 | AI 场景 |
|--------|------|------|---------|
| 度中心性（Degree） | $C_d(v) = \deg(v) / (n-1)$ | 直接连接多 | 影响力节点识别 |
| 接近中心性（Closeness） | $C_c(v) = (n-1) / \sum_u d(v,u)$ | 离其他节点近 | 信息传播效率 |
| 介数中心性（Betweenness） | 经过 $v$ 的最短路径比例 | 是否是"桥梁" | 瓶颈节点检测 |
| 特征向量中心性（Eigenvector） | 邻居重要则自身也重要 | 全局影响力 | PageRank 的前身 |

### PageRank

PageRank 是特征向量中心性的实用化版本，其迭代更新公式：

$$r(v) = \frac{1-d}{n} + d \sum_{u \in \text{In}(v)} \frac{r(u)}{\deg^+(u)}$$

其中 $d \approx 0.85$ 为阻尼因子（Damping Factor），防止游走陷入无穷循环。

矩阵形式：$\mathbf{r} = d \cdot P^\top \mathbf{r} + \frac{1-d}{n} \mathbf{1}$，即求解一个线性方程组。

---

## 5. 从图算法到 GNN

经典图算法是**手工设计**的，针对特定任务（最短路、排序等）有明确的规则。**图神经网络**（Graph Neural Network, GNN）的核心突破是：**让模型从数据中自动学习在图上的计算规则**。

### 5.1 消息传递框架（Message Passing Framework）

现代 GNN 大多遵循**消息传递神经网络**（Message Passing Neural Network, MPNN）框架（Gilmer et al., 2017）：

**第 $k$ 轮迭代，节点 $v$ 的更新分两步**：

**步骤一：聚合（Aggregate）**
$$\mathbf{m}_v^{(k)} = \text{AGGREGATE}^{(k)}\!\left(\left\{\mathbf{h}_u^{(k-1)} \mid u \in \mathcal{N}(v)\right\}\right)$$

**步骤二：更新（Update）**
$$\mathbf{h}_v^{(k)} = \text{UPDATE}^{(k)}\!\left(\mathbf{h}_v^{(k-1)},\, \mathbf{m}_v^{(k)}\right)$$

其中 $\mathbf{h}_v^{(0)} = \mathbf{x}_v$ 为节点 $v$ 的初始特征，$\mathcal{N}(v)$ 为邻居集合。

**与 BFS 的对比**：

| | BFS | GNN 消息传递 |
|-|-----|------------|
| 展开方式 | 按层逐步展开 | 按层逐步聚合 |
| 信息来源 | 邻居是否被访问 | 邻居的**特征向量** |
| 规则 | 人工定义 | **从数据中学习** |
| 层数含义 | 搜索深度 | 感受野半径 |

### 5.2 三种经典 GNN 变体

**GCN（Graph Convolutional Network）**：

$$\mathbf{h}_v^{(k)} = \sigma\!\left(W^{(k)} \cdot \text{MEAN}\left(\mathbf{h}_v^{(k-1)} \cup \left\{\mathbf{h}_u^{(k-1)} \mid u \in \mathcal{N}(v)\right\}\right)\right)$$

聚合方式：均值，简单高效。

**GraphSAGE（Hamilton et al., 2017）**：

$$\mathbf{h}_v^{(k)} = \sigma\!\left(W^{(k)} \cdot \left[\mathbf{h}_v^{(k-1)} \,\|\, \text{AGG}\left(\left\{\mathbf{h}_u^{(k-1)}\right\}_{u \in \mathcal{N}(v)}\right)\right]\right)$$

$[\cdot \| \cdot]$ 表示拼接（Concatenation）。支持邻居采样，可扩展到百万级节点的工业图。

**GAT（Graph Attention Network，Veličković et al., 2018）**：

$$\mathbf{h}_v^{(k)} = \sigma\!\left(\sum_{u \in \mathcal{N}(v) \cup \{v\}} \alpha_{vu}\, W^{(k)} \mathbf{h}_u^{(k-1)}\right)$$

其中注意力系数（Attention Coefficient）：

$$\alpha_{vu} = \frac{\exp\!\left(\text{LeakyReLU}\!\left(\mathbf{a}^\top \left[W\mathbf{h}_v \,\|\, W\mathbf{h}_u\right]\right)\right)}{\sum_{w \in \mathcal{N}(v)} \exp\!\left(\text{LeakyReLU}\!\left(\mathbf{a}^\top \left[W\mathbf{h}_v \,\|\, W\mathbf{h}_w\right]\right)\right)}$$

核心思想：不同邻居对中心节点的贡献权重**由模型自动学习**，而不是简单均值。

### 5.3 GNN 的表达能力限制

GNN 的表达能力与**Weisfeiler-Leman（WL）图同构测试**密切相关。

标准的消息传递 GNN 至多与 1-WL 测试等价：若两个节点的 $k$ 跳邻域结构完全相同，则 GNN 无法区分它们——即使它们在图中扮演不同角色。

这是推动更强 GNN（如高阶 GNN、图 Transformer）发展的重要理论动机。

---

## 6. GNN 的典型任务

| 任务类型 | 预测目标 | 示例 |
|---------|---------|------|
| 节点分类（Node Classification） | 每个节点的标签 | 用户欺诈检测、论文类别 |
| 边预测（Link Prediction） | 两节点间是否有边 | 推荐系统、知识图谱补全 |
| 图分类（Graph Classification） | 整图的标签 | 分子属性预测、蛋白质功能 |
| 图生成（Graph Generation） | 生成新的合法图 | 药物分子设计 |

**读出操作**（Readout / Global Pooling）：图分类任务需要将节点级特征汇聚为图级表示：

$$\mathbf{h}_G = \text{READOUT}\left(\left\{\mathbf{h}_v^{(K)} \mid v \in V\right\}\right)$$

常见方式：求和（SUM）、均值（MEAN）、最大值（MAX）或可学习的注意力池化（Attention Pooling）。

---

## 7. 图算法 vs. GNN：何时选哪个

| 情形 | 推荐方案 |
|------|---------|
| 明确的结构性任务（最短路、连通性） | 经典图算法，效率高且可解释 |
| 有节点/边特征，需要从数据中学习模式 | GNN |
| 图规模极大（亿级边），对实时性要求高 | 图算法 + 采样，或工业级 GNN（GraphSAGE） |
| 可解释性是硬需求 | 经典算法或配合注意力可视化的 GAT |

---

## 8. 本节小结

```
BFS / DFS
    ↓ 按层展开邻居的直觉
消息传递框架（MPNN）
    ↓ 不同的聚合函数
GCN（均值）→ GraphSAGE（采样+拼接）→ GAT（注意力权重）
    ↓ 任务适配
节点分类 / 边预测 / 图分类 / 图生成
```

> **核心直觉**：GNN 做的事，本质上是**让每个节点通过迭代地"听取"邻居的信息来更新自己的表示**。层数越多，节点能"看到"的图结构范围越大——但层数过多会导致**过度平滑**（Over-Smoothing）问题，所有节点的表示趋于一致。这是 GNN 设计中的核心张力之一。
