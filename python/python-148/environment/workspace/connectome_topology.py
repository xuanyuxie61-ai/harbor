"""
connectome_topology.py — 脑连接组拓扑分析与渗流动力学
=======================================================
融合 usa_matrix（稀疏邻接矩阵构建）与 percolation_simulation
（二维格点渗流模拟）两个项目的核心算法。

科学背景：
大脑皮层区域之间的连接可建模为加权无向图 G=(V,E,W)。
BCI 解码需要理解信息如何从感觉/运动皮层通过连接组传播。
渗流理论描述当连接强度超过临界阈值时，
大规模连通分量突然出现的相变现象，对应于神经网络中
信息全局传播的“临界点”。

核心数学：
---
**图拉普拉斯与谱划分：**
对邻接矩阵 A，度矩阵 D=diag(deg_i)，非归一化图拉普拉斯：

    L = D - A

归一化对称拉普拉斯：

    L_sym = I - D^{-1/2} A D^{-1/2}

Fiedler 值（第二小特征值 λ_2）反映图的连通性：
λ_2 > 0 当且仅当图连通。

---
**渗流阈值：**
对占据概率 p 的二维方格点渗流，临界阈值精确解为：

    p_c = 0.59274605...

当 p > p_c 时，出现跨越整个系统的渗流簇，
对应神经信息从局部编码到全局传播的转变。

---
**有效传导率：**
对连接图，两个区域 i,j 之间的有效电阻（ commute time 的倒数）
可通过图拉普拉斯伪逆计算：

    R_eff(i,j) = (e_i - e_j)^T L^+ (e_i - e_j)

其中 L^+ 为 Moore-Penrose 伪逆。
"""

import numpy as np
from utils import sparse_adjacency_to_laplacian


class BrainConnectomeGraph:
    """
    大脑连接组图模型，基于 usa_matrix 的稀疏矩阵思想构建。
    """

    def __init__(self, n_regions=50, connection_prob=0.15, weight_dist='lognormal',
                 random_state=None):
        """
        n_regions      : 脑区数量（模拟人类皮层约 50 个 Brodmann 区）
        connection_prob: 随机连接概率
        weight_dist    : 'lognormal'（对数正态，符合真实连接组统计）
                         或 'uniform'
        """
        if random_state is not None:
            np.random.seed(random_state)
        self.n_regions = n_regions
        self.connection_prob = connection_prob
        # 构建稀疏邻接矩阵
        self.adjacency = self._build_adjacency(weight_dist)
        self.laplacian = sparse_adjacency_to_laplacian(self.adjacency)
        self.degree_matrix = np.diag(np.sum(self.adjacency, axis=1))

    def _build_adjacency(self, weight_dist):
        """构建对称加权邻接矩阵。"""
        n = self.n_regions
        A = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(i + 1, n):
                if np.random.rand() < self.connection_prob:
                    if weight_dist == 'lognormal':
                        # 大脑连接权重近似对数正态分布
                        w = np.random.lognormal(mean=-1.0, sigma=1.0)
                    else:
                        w = np.random.uniform(0.1, 1.0)
                    A[i, j] = w
                    A[j, i] = w
        # 确保图连通：若存在孤立节点，随机连接一条边
        for i in range(n):
            if np.sum(A[i, :]) == 0:
                j = np.random.randint(0, n)
                if j == i:
                    j = (j + 1) % n
                w = 0.5
                A[i, j] = w
                A[j, i] = w
        return A

    def compute_fiedler_value(self):
        """计算 Fiedler 值（图拉普拉斯第二小特征值）。"""
        eigenvalues = np.linalg.eigvalsh(self.laplacian)
        # 排序后取第二个最小值
        eigenvalues = np.sort(eigenvalues)
        if len(eigenvalues) > 1:
            return float(eigenvalues[1])
        return 0.0

    def compute_effective_resistance(self, i, j):
        """
        计算节点 i,j 之间的有效电阻 R_eff(i,j)。
        使用图拉普拉斯伪逆：R_eff = L^+_ii + L^+_jj - 2*L^+_ij
        """
        L_pinv = np.linalg.pinv(self.laplacian)
        return float(L_pinv[i, i] + L_pinv[j, j] - 2.0 * L_pinv[i, j])

    def compute_communicability(self, t=1.0):
        """
        计算图可通讯性矩阵：C = exp(t * A)，
        其中矩阵指数 captures 所有长度路径的加权贡献。
        """
        from scipy.linalg import expm
        return expm(t * self.adjacency)

    def simulate_diffusion(self, initial_concentration, dt=0.01, n_steps=1000):
        """
        模拟图上的热扩散/信息传播过程：
            dc/dt = -L * c
        离散：c_{k+1} = c_k - dt * L * c_k
        """
        c = np.asarray(initial_concentration, dtype=float).copy()
        history = np.zeros((n_steps + 1, self.n_regions), dtype=float)
        history[0] = c.copy()
        for k in range(n_steps):
            c = c - dt * (self.laplacian @ c)
            history[k + 1] = c.copy()
        return history


class NeuralPercolationAnalyzer:
    """
    神经网络渗流分析器，基于 percolation_simulation 项目。
    模拟脑区激活的渗流过程，分析信息传播的临界行为。
    """

    def __init__(self, shape=(64, 64)):
        self.shape = shape
        self.M, self.N = shape

    def simulate_site_percolation(self, p, random_state=None):
        """
        二维格点点渗流模拟。
        返回占据矩阵、连通分量标签矩阵、各分量大小。
        """
        if random_state is not None:
            np.random.seed(random_state)
        occupied = np.random.rand(self.M, self.N) < p
        labels = np.zeros((self.M, self.N), dtype=int)
        label_id = 0
        sizes = []
        # 使用栈实现的洪水填充（flood-fill/BFS）标记连通分量
        for i in range(self.M):
            for j in range(self.N):
                if occupied[i, j] and labels[i, j] == 0:
                    label_id += 1
                    stack = [(i, j)]
                    labels[i, j] = label_id
                    count = 0
                    while stack:
                        ci, cj = stack.pop()
                        count += 1
                        # 4-邻域
                        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            ni, nj = ci + di, cj + dj
                            if 0 <= ni < self.M and 0 <= nj < self.N:
                                if occupied[ni, nj] and labels[ni, nj] == 0:
                                    labels[ni, nj] = label_id
                                    stack.append((ni, nj))
                    sizes.append(count)
        return occupied, labels, sizes

    def find_spanning_cluster(self, labels):
        """
        检测是否存在从左到右或从上到下的跨越簇。
        返回布尔值和跨越簇的标号集合。
        """
        M, N = labels.shape
        spanning = False
        spanning_labels = set()
        # 水平跨越：存在某个标签同时出现在第一列和最后一列
        left_labels = set(labels[:, 0])
        right_labels = set(labels[:, -1])
        horiz = left_labels & right_labels
        horiz.discard(0)
        if horiz:
            spanning = True
            spanning_labels.update(horiz)
        # 垂直跨越
        top_labels = set(labels[0, :])
        bottom_labels = set(labels[-1, :])
        vert = top_labels & bottom_labels
        vert.discard(0)
        if vert:
            spanning = True
            spanning_labels.update(vert)
        return spanning, spanning_labels

    def estimate_critical_threshold(self, n_samples=20, p_values=None):
        """
        通过多次采样估计渗流临界阈值 p_c。
        对每个 p 值计算存在跨越簇的概率，然后插值找到 P_span=0.5 处的 p。
        """
        if p_values is None:
            p_values = np.linspace(0.4, 0.7, 16)
        span_probs = []
        for p in p_values:
            spans = 0
            for _ in range(n_samples):
                _, labels, _ = self.simulate_site_percolation(p)
                is_spanning, _ = self.find_spanning_cluster(labels)
                if is_spanning:
                    spans += 1
            span_probs.append(spans / n_samples)
        span_probs = np.array(span_probs)
        # 线性插值找到 P=0.5 处的 p
        above_half = np.where(span_probs >= 0.5)[0]
        below_half = np.where(span_probs < 0.5)[0]
        if len(above_half) == 0 or len(below_half) == 0:
            p_c = p_values[np.argmin(np.abs(span_probs - 0.5))]
        else:
            i_low = below_half[-1]
            i_high = above_half[0]
            p_low, p_high = p_values[i_low], p_values[i_high]
            prob_low, prob_high = span_probs[i_low], span_probs[i_high]
            if abs(prob_high - prob_low) < 1e-10:
                p_c = 0.5 * (p_low + p_high)
            else:
                p_c = p_low + (0.5 - prob_low) * (p_high - p_low) / (prob_high - prob_low)
        return p_c, p_values, span_probs

    def compute_fractal_dimension(self, labels, target_label):
        """
        计算特定渗流簇的分形维数（盒计数法）。
        对分形维数 D，盒子数 N(ε) ~ ε^{-D}。
        """
        mask = (labels == target_label)
        coords = np.argwhere(mask)
        if len(coords) < 10:
            return 0.0
        max_box = min(self.M, self.N) // 2
        if max_box < 2:
            max_box = 2
        box_sizes = 2 ** np.arange(1, int(np.log2(max_box)) + 1)
        counts = []
        for box_size in box_sizes:
            # 统计覆盖所有点所需的最少 box 数
            bins_i = coords[:, 0] // box_size
            bins_j = coords[:, 1] // box_size
            unique_boxes = set(zip(bins_i, bins_j))
            counts.append(len(unique_boxes))
        counts = np.array(counts, dtype=float)
        box_sizes = np.array(box_sizes, dtype=float)
        valid = counts > 0
        if np.sum(valid) < 2:
            return 0.0
        # 线性回归 log(N) = -D * log(ε) + C
        log_boxes = np.log(box_sizes[valid])
        log_counts = np.log(counts[valid])
        coeffs = np.polyfit(log_boxes, log_counts, 1)
        D = -coeffs[0]
        return float(D)


class ConnectomePercolationBridge:
    """
    连接组图与渗流分析的桥梁：
    将大脑连接组的激活模式映射到二维渗流格点，
    分析 BCI 指令在不同脑区传播时的临界行为。
    """

    def __init__(self, connectome, grid_shape=(32, 32)):
        self.connectome = connectome
        self.grid_shape = grid_shape
        self.percolation = NeuralPercolationAnalyzer(grid_shape)

    def map_connectome_to_grid(self, active_regions, activation_strength):
        """
        将连接组中的激活脑区映射到二维渗流格点。
        使用随机投影方法：每个脑区对应格点上的一个随机区域，
        激活概率与 activation_strength 成正比。
        """
        M, N = self.grid_shape
        grid = np.zeros((M, N), dtype=float)
        n_regions = self.connectome.n_regions
        for r in range(n_regions):
            # 每个脑区占据随机子区域
            ci = np.random.randint(0, M)
            cj = np.random.randint(0, N)
            radius = max(1, int(np.sqrt(M * N / n_regions) * 0.5))
            for i in range(max(0, ci - radius), min(M, ci + radius + 1)):
                for j in range(max(0, cj - radius), min(N, cj + radius + 1)):
                    if (i - ci) ** 2 + (j - cj) ** 2 <= radius ** 2:
                        if r in active_regions:
                            grid[i, j] = activation_strength[r]
        return grid

    def analyze_information_spread(self, seed_region, steps=50):
        """
        模拟从 seed_region 开始的信息在连接组中的传播，
        并分析其渗流特性。
        返回：传播历史、最大连通分量大小、是否发生渗流。
        """
        n = self.connectome.n_regions
        activation = np.zeros(n, dtype=float)
        activation[seed_region] = 1.0
        history = [activation.copy()]
        # 使用图扩散模型模拟信息传播
        A_norm = self.connectome.adjacency.copy()
        row_sums = np.sum(A_norm, axis=1)
        row_sums[row_sums == 0] = 1.0
        A_norm = A_norm / row_sums[:, None]
        for _ in range(steps):
            activation = 0.8 * activation + 0.2 * (A_norm @ activation)
            activation = np.clip(activation, 0.0, 1.0)
            history.append(activation.copy())
        # 将最终激活映射到渗流格点
        active_regions = [i for i, a in enumerate(activation) if a > 0.3]
        grid = self.map_connectome_to_grid(active_regions, activation)
        # 二值化进行渗流分析
        occupied = grid > np.mean(grid)
        labels = np.zeros(grid.shape, dtype=int)
        label_id = 0
        sizes = []
        M, N = grid.shape
        for i in range(M):
            for j in range(N):
                if occupied[i, j] and labels[i, j] == 0:
                    label_id += 1
                    stack = [(i, j)]
                    labels[i, j] = label_id
                    count = 0
                    while stack:
                        ci, cj = stack.pop()
                        count += 1
                        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            ni, nj = ci + di, cj + dj
                            if 0 <= ni < M and 0 <= nj < N:
                                if occupied[ni, nj] and labels[ni, nj] == 0:
                                    labels[ni, nj] = label_id
                                    stack.append((ni, nj))
                    sizes.append(count)
        is_spanning, _ = self.percolation.find_spanning_cluster(labels)
        max_size = max(sizes) if sizes else 0
        return {
            'activation_history': np.array(history),
            'final_activation': activation,
            'max_cluster_size': max_size,
            'is_spanning': is_spanning,
            'grid_labels': labels
        }
