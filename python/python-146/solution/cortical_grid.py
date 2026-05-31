"""
cortical_grid.py
皮层编码拓扑与距离模块

融合 square_grid (二维正方形网格生成)
与 square_surface_distance (表面距离统计)。

核心科学模型：
  皮层空间编码网格：
    在二维皮层区域 [xL, xR] x [yL, yR] 上生成逻辑矩形网格。
    每个网格点对应一个神经元的感受野中心。

    网格生成公式 (square_grid 思想):
      x_i = a + (i - 1) * (b - a) / (nx - 1),   i = 1..nx
      y_j = c + (j - 1) * (d - c) / (ny - 1),   j = 1..ny
      总节点数 N = nx * ny

  连接概率与皮层表面距离：
    神经元 i 与 j 之间的连接概率由皮层表面测地距离决定：
      P_{conn}(i,j) = p_0 * exp( -d_{ij}^2 / (2 sigma^2) )

    其中 d_{ij} 可以是欧氏距离或考虑皮层折叠的测地距离。
    对于平坦二维皮层：
      d_{ij} = sqrt( (x_i - x_j)^2 + (y_i - y_j)^2 )

  距离统计 (square_surface_distance 思想)：
    随机采样两个皮层位置，计算其距离的期望和方差。
    对于单位正方形 [0,1]^2：
      E[d] ≈ 0.5214
      Var[d] ≈ 0.0615

  编码容量：
    二维网格的独立编码单元数 N = nx * ny。
    若每个单元采用脉冲时序编码 (M 个时间 bin)，
    总编码状态数 ≈ (2^M)^N = 2^{M N} (不考虑相关性)。
"""

import numpy as np


class CorticalGrid:
    """
    皮层编码网格，融合 square_grid 的网格生成算法。
    """

    def __init__(self, nx, ny, xlim=(-1.0, 1.0), ylim=(-1.0, 1.0)):
        if nx < 1 or ny < 1:
            raise ValueError("nx and ny must be positive.")
        self.nx = nx
        self.ny = ny
        self.xlim = xlim
        self.ylim = ylim
        self.N = nx * ny
        self._generate_grid()

    def _generate_grid(self):
        """生成二维网格点。"""
        x = np.linspace(self.xlim[0], self.xlim[1], self.nx)
        y = np.linspace(self.ylim[0], self.ylim[1], self.ny)
        self.X_grid, self.Y_grid = np.meshgrid(x, y)
        self.positions = np.column_stack([
            self.X_grid.ravel(), self.Y_grid.ravel()
        ])

    def euclidean_distance(self, i, j):
        """计算两个网格位置间的欧氏距离。"""
        if not (0 <= i < self.N and 0 <= j < self.N):
            raise IndexError("Index out of bounds.")
        dx = self.positions[i, 0] - self.positions[j, 0]
        dy = self.positions[i, 1] - self.positions[j, 1]
        return np.sqrt(dx ** 2 + dy ** 2)

    def connection_probability(self, i, j, p0=0.3, sigma=0.5):
        """
        基于距离的高斯连接概率。
        p0: 最大连接概率
        sigma: 空间尺度
        """
        if i == j:
            return 0.0
        d = self.euclidean_distance(i, j)
        return p0 * np.exp(-d ** 2 / (2.0 * sigma ** 2))

    def build_connectivity_matrix(self, p0=0.3, sigma=0.5):
        """构建整个网格的连接概率矩阵。"""
        W = np.zeros((self.N, self.N))
        for i in range(self.N):
            for j in range(self.N):
                if i != j:
                    W[i, j] = self.connection_probability(i, j, p0, sigma)
        return W

    def distance_statistics(self, n_samples=10000):
        """
        皮层表面距离的统计量。
        融合 square_surface_distance 的随机采样思想。
        """
        rng = np.random.default_rng(seed=11)
        idx1 = rng.integers(0, self.N, size=n_samples)
        idx2 = rng.integers(0, self.N, size=n_samples)
        distances = np.zeros(n_samples)
        for k in range(n_samples):
            distances[k] = self.euclidean_distance(idx1[k], idx2[k])
        dmu = np.mean(distances)
        dvar = np.var(distances)
        return dmu, dvar, distances

    def spatial_receptive_field(self, i, sigma_rf=0.2):
        """
        计算位置 i 的高斯感受野权重分布。
        w_j = exp( -||pos_j - pos_i||^2 / (2 sigma_rf^2) )
        """
        center = self.positions[i]
        diff = self.positions - center
        dist_sq = np.sum(diff ** 2, axis=1)
        weights = np.exp(-dist_sq / (2.0 * sigma_rf ** 2))
        weights[i] = 0.0  # 自身无连接
        return weights


def demo_grid_encoding():
    """皮层网格编码 demo。"""
    grid = CorticalGrid(nx=5, ny=5, xlim=(0.0, 1.0), ylim=(0.0, 1.0))
    W = grid.build_connectivity_matrix(p0=0.4, sigma=0.3)
    dmu, dvar, _ = grid.distance_statistics(n_samples=2000)
    rf = grid.spatial_receptive_field(i=12, sigma_rf=0.25)
    return W, dmu, dvar, rf


def demo_distance_stats():
    """距离统计 demo。"""
    grid = CorticalGrid(nx=10, ny=10, xlim=(0.0, 1.0), ylim=(0.0, 1.0))
    dmu, dvar, distances = grid.distance_statistics(n_samples=5000)
    return dmu, dvar
