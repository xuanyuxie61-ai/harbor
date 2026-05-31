"""
mesh_generator.py
=================
声学域网格生成器。

融合种子项目：
  - 1146_square_hex_grid : 六边形网格点生成
  - 250_cvt_3d_sampling  : 3D CVT Lloyd算法
  - 239_cvt_1_movie      : CVT迭代（find_closest, cvt_iterate）

科学应用：
  为声学冲击波传播模拟生成高质量的计算网格。
  使用六边形网格覆盖二维声学域，并通过CVT优化使网格
  在冲击波前沿区域自适应加密。
"""

import numpy as np


def hex_grid_points(nodes_per_layer, layers, box):
    """
    生成坐标框内的六边形网格点。

    原始算法来自 1146_square_hex_grid/hex_grid_points.m。
    六边形网格在 y 方向的间距为 hy = hx * sqrt(3) / 2，
    偶数行相对于奇数行错开 hx/2。

    Parameters
    ----------
    nodes_per_layer : int
        第一层（奇数层）的节点数。
    layers : int
        水平层数。
    box : np.ndarray, shape (2, 2)
        box[0] = [x_min, x_max], box[1] = [y_min, y_max]。

    Returns
    -------
    np.ndarray, shape (N, 2)
        网格点坐标。
    """
    nodes_per_layer = int(nodes_per_layer)
    layers = int(layers)
    if nodes_per_layer < 1:
        return np.zeros((0, 2))
    if nodes_per_layer == 1:
        pt = (box[:, 0] + box[:, 1]) / 2.0
        return pt.reshape(1, 2)

    hx = (box[0, 1] - box[0, 0]) / (nodes_per_layer - 1)
    hy = hx * np.sqrt(3.0) / 2.0

    points = []
    for j in range(layers):
        y = box[1, 0] + hy * j
        jmod = j % 2
        if jmod == 0:
            for i in range(nodes_per_layer):
                x = box[0, 0] + (box[0, 1] - box[0, 0]) * i / (nodes_per_layer - 1)
                points.append([x, y])
        else:
            for i in range(nodes_per_layer - 1):
                x = box[0, 0] + (box[0, 1] - box[0, 0]) * (2 * i + 1) / (2 * nodes_per_layer - 2)
                points.append([x, y])

    return np.array(points, dtype=float)


def hex_grid_approximate_n(nodes_per_layer, layers):
    """
    估算六边形网格总节点数。
    """
    if nodes_per_layer < 1:
        return 0
    if nodes_per_layer == 1:
        return 1
    n_odd = ((layers + 1) // 2) * nodes_per_layer
    n_even = (layers // 2) * (nodes_per_layer - 1)
    return n_odd + n_even


def find_closest(ndim, n_generators, n_samples, samples, generators):
    """
    对每个样本点找到最近的生成器索引。

    原始算法来自 239_cvt_1_movie/find_closest.m。
    使用向量化距离计算提高效率。

    Parameters
    ----------
    ndim : int
        空间维度。
    n_generators : int
        生成器数量。
    n_samples : int
        样本点数量。
    samples : np.ndarray, shape (ndim, n_samples)
        样本点。
    generators : np.ndarray, shape (ndim, n_generators)
        生成器。

    Returns
    -------
    np.ndarray, shape (n_samples,)
        每个样本点最近生成器的索引。
    """
    # 向量化计算距离矩阵
    # distances[i,j] = ||sample_i - generator_j||^2
    # 使用广播: (ndim, n_samples, 1) - (ndim, 1, n_generators)
    s = samples.reshape(ndim, n_samples, 1)
    g = generators.reshape(ndim, 1, n_generators)
    dists = np.sum((s - g) ** 2, axis=0)  # shape (n_samples, n_generators)
    nearest = np.argmin(dists, axis=1)
    return nearest


def cvt_iterate_2d(generators, ratio, region_box):
    """
    执行一次2D CVT (Centroidal Voronoi Tessellation) Lloyd迭代。

    原始算法来自 239_cvt_1_movie/cvt_iterate.m 与 250_cvt_3d_sampling。

    Parameters
    ----------
    generators : np.ndarray, shape (2, n)
        当前生成器位置。
    ratio : int
        每个生成器对应的样本点数。
    region_box : np.ndarray, shape (2, 2)
        区域边界 [xmin,xmax; ymin,ymax]。

    Returns
    -------
    generators_new : np.ndarray, shape (2, n)
        更新后的生成器位置（Voronoi cell centroids近似）。
    diff : float
        生成器移动量的L2和。
    energy : float
        CVT能量函数值。
    """
    ndim = 2
    n = generators.shape[1]
    sample_num = ratio * n

    # 在区域内均匀随机采样
    samples = np.zeros((ndim, sample_num), dtype=float)
    samples[0, :] = region_box[0, 0] + np.random.rand(sample_num) * (region_box[0, 1] - region_box[0, 0])
    samples[1, :] = region_box[1, 0] + np.random.rand(sample_num) * (region_box[1, 1] - region_box[1, 0])

    nearest = find_closest(ndim, n, sample_num, samples, generators)

    generators_new = np.zeros_like(generators)
    counts = np.zeros(n, dtype=int)
    energy = 0.0

    for j in range(sample_num):
        idx = nearest[j]
        generators_new[:, idx] += samples[:, j]
        energy += np.sum((generators[:, idx] - samples[:, j]) ** 2)
        counts[idx] += 1

    # 避免除零
    for j in range(n):
        if counts[j] > 0:
            generators_new[:, j] /= counts[j]
        else:
            generators_new[:, j] = generators[:, j]

    energy /= sample_num
    diff = np.sum(np.sqrt(np.sum((generators_new - generators) ** 2, axis=0)))

    return generators_new, diff, energy


def cvt_optimize_2d(initial_points, region_box, it_max=50, ratio=1000, tol=1e-5):
    """
    通过多次CVT迭代优化网格点分布。

    Parameters
    ----------
    initial_points : np.ndarray, shape (n, 2)
        初始网格点。
    region_box : np.ndarray, shape (2, 2)
        区域边界。
    it_max : int
        最大迭代次数。
    ratio : int
        每个生成器的样本点数。
    tol : float
        收敛容差。

    Returns
    -------
    np.ndarray, shape (n, 2)
        优化后的网格点。
    """
    generators = initial_points.T.copy()
    n = generators.shape[1]
    if n == 0:
        return initial_points

    for it in range(it_max):
        generators_new, diff, energy = cvt_iterate_2d(generators, ratio, region_box)
        generators = generators_new
        if diff < tol * n:
            break

    return generators.T.copy()


def adaptive_density_function(x, y, shock_center, shock_width, base_density=1.0,
                              peak_density=10.0):
    """
    自适应密度函数：在冲击波前沿附近提高点密度。

    .. math::
        \rho(x,y) = \rho_{base} + (\rho_{peak} - \rho_{base})
        \exp\left(-\frac{(x - x_c)^2 + (y - y_c)^2}{2 \sigma^2}\right)

    Parameters
    ----------
    x, y : float or np.ndarray
        坐标。
    shock_center : tuple(float, float)
        冲击波中心位置。
    shock_width : float
        冲击波特征宽度。
    base_density : float
        基础密度。
    peak_density : float
        峰值密度。

    Returns
    -------
    float or np.ndarray
        密度值。
    """
    dx = x - shock_center[0]
    dy = y - shock_center[1]
    r2 = dx ** 2 + dy ** 2
    sigma2 = 2.0 * shock_width ** 2
    if sigma2 <= 0.0:
        return np.full_like(x, base_density) if isinstance(x, np.ndarray) else base_density
    rho = base_density + (peak_density - base_density) * np.exp(-r2 / sigma2)
    return rho


def rejection_sampling_adaptive(n_points, region_box, shock_center, shock_width,
                                base_density=1.0, peak_density=10.0, max_trials=1000000):
    """
    使用拒绝采样根据自适应密度函数生成网格点。

    Parameters
    ----------
    n_points : int
        目标点数。
    region_box : np.ndarray, shape (2, 2)
        区域边界。
    shock_center : tuple
        冲击波中心。
    shock_width : float
        冲击波宽度。
    base_density : float
        基础密度。
    peak_density : float
        峰值密度。
    max_trials : int
        最大尝试次数。

    Returns
    -------
    np.ndarray, shape (n_points, 2)
        生成的点集。
    """
    points = []
    trials = 0
    while len(points) < n_points and trials < max_trials:
        x = region_box[0, 0] + np.random.rand() * (region_box[0, 1] - region_box[0, 0])
        y = region_box[1, 0] + np.random.rand() * (region_box[1, 1] - region_box[1, 0])
        rho_val = adaptive_density_function(x, y, shock_center, shock_width,
                                            base_density, peak_density)
        # 拒绝采样：接受概率 = rho_val / peak_density
        if np.random.rand() < (rho_val / peak_density):
            points.append([x, y])
        trials += 1

    if len(points) < n_points:
        # 未达目标，用随机点补足
        remaining = n_points - len(points)
        x_rand = region_box[0, 0] + np.random.rand(remaining) * (region_box[0, 1] - region_box[0, 0])
        y_rand = region_box[1, 0] + np.random.rand(remaining) * (region_box[1, 1] - region_box[1, 0])
        extra = np.column_stack((x_rand, y_rand))
        if len(points) > 0:
            return np.vstack((np.array(points), extra))
        return extra

    return np.array(points, dtype=float)


class AcousticMesh:
    """
    声学计算域网格管理器。
    """

    def __init__(self, box, method='hex', nodes_per_layer=20, layers=20,
                 cvt_iters=30, adaptive=False, shock_center=None, shock_width=None):
        """
        Parameters
        ----------
        box : np.ndarray, shape (2, 2)
            计算域 [xmin,xmax; ymin,ymax]。
        method : str
            'hex' 或 'cvt' 或 'adaptive_cvt'。
        nodes_per_layer : int
            六边形网格每行节点数。
        layers : int
            六边形网格层数。
        cvt_iters : int
            CVT优化迭代次数。
        adaptive : bool
            是否使用自适应密度。
        shock_center : tuple or None
            冲击波中心坐标。
        shock_width : float or None
            冲击波特征宽度。
        """
        self.box = np.asarray(box, dtype=float)
        if self.box.shape != (2, 2):
            raise ValueError("box must have shape (2, 2)")
        if np.any(self.box[:, 1] <= self.box[:, 0]):
            raise ValueError("box max must be greater than min in each dimension.")

        self.method = method
        self.nodes_per_layer = int(nodes_per_layer)
        self.layers = int(layers)
        self.cvt_iters = int(cvt_iters)

        if method == 'hex':
            self.points = hex_grid_points(nodes_per_layer, layers, self.box)
        elif method == 'cvt':
            n = hex_grid_approximate_n(nodes_per_layer, layers)
            init = np.random.rand(n, 2)
            init[:, 0] = self.box[0, 0] + init[:, 0] * (self.box[0, 1] - self.box[0, 0])
            init[:, 1] = self.box[1, 0] + init[:, 1] * (self.box[1, 1] - self.box[1, 0])
            self.points = cvt_optimize_2d(init, self.box, it_max=cvt_iters)
        elif method == 'adaptive_cvt':
            if shock_center is None or shock_width is None:
                raise ValueError("adaptive_cvt requires shock_center and shock_width.")
            n = hex_grid_approximate_n(nodes_per_layer, layers)
            init = rejection_sampling_adaptive(n, self.box, shock_center, shock_width)
            self.points = cvt_optimize_2d(init, self.box, it_max=cvt_iters)
        else:
            raise ValueError(f"Unknown mesh method: {method}")

        self.n_points = self.points.shape[0]
        if self.n_points == 0:
            raise ValueError("Mesh generation produced zero points.")

    def compute_element_size(self):
        """
        估算网格特征尺寸（最近邻平均距离）。

        Returns
        -------
        float
            特征网格尺寸。
        """
        if self.n_points < 2:
            return 0.0
        # 使用向量化计算最近邻距离
        pts = self.points
        # 为避免 O(N^2) 内存爆炸，对大数据集使用子采样
        if self.n_points > 2000:
            idx = np.random.choice(self.n_points, 2000, replace=False)
            pts_sub = pts[idx]
        else:
            idx = np.arange(self.n_points)
            pts_sub = pts

        # 计算子集内最近邻
        diff = pts_sub[:, np.newaxis, :] - pts_sub[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_dists = np.min(dists, axis=1)
        return float(np.mean(min_dists))

    def compute_mesh_quality(self):
        """
        基于CVT能量评估网格质量。

        Returns
        -------
        float
            质量指标（越低越好）。
        """
        if self.n_points == 0:
            return np.inf
        ratio = max(10, min(1000, 50000 // self.n_points))
        sample_num = ratio * self.n_points
        samples = np.zeros((2, sample_num), dtype=float)
        samples[0, :] = self.box[0, 0] + np.random.rand(sample_num) * (self.box[0, 1] - self.box[0, 0])
        samples[1, :] = self.box[1, 0] + np.random.rand(sample_num) * (self.box[1, 1] - self.box[1, 0])

        generators = self.points.T
        nearest = find_closest(2, self.n_points, sample_num, samples, generators)
        energy = 0.0
        for j in range(sample_num):
            energy += np.sum((generators[:, nearest[j]] - samples[:, j]) ** 2)
        energy /= sample_num
        return float(energy)
