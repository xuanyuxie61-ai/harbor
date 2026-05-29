"""
sensor_optimizer.py
===================
声学传感器阵列布局优化与路径规划。

融合种子项目：
  - 1365_tsp_greedy : 贪心算法求解旅行商问题
  - 239_cvt_1_movie : CVT 迭代优化点分布
  - 250_cvt_3d_sampling : 3D CVT Lloyd 算法

科学应用：
  在非线性声学实验中，传感器阵列的最优布局直接影响冲击波
  场重构精度。本模块结合 CVT 空间覆盖优化与 TSP 路径规划，
  实现传感器位置优化与测量路径最小化。

  核心优化目标：
  .. math::
      \min_{\{s_i\}} \int_{\Omega} \min_i \| x - s_i \|^2 \rho(x) dx
      + \lambda \cdot \mathrm{TSP}(\{s_i\})
"""

import numpy as np


def path_cost(n, distance, p):
    """
    计算 TSP 路径总成本。

    原始算法来自 1365_tsp_greedy/path_cost.m。

    Parameters
    ----------
    n : int
        城市数。
    distance : np.ndarray, shape (n, n)
        距离矩阵。
    p : np.ndarray, shape (n,)
        路径（排列）。

    Returns
    -------
    float
        总路径长度。
    """
    cost = 0.0
    i1 = n - 1
    for i2 in range(n):
        cost += distance[p[i1], p[i2]]
        i1 = i2
    return cost


def path_greedy(n, distance, start):
    """
    贪心算法构造 TSP 路径。

    原始算法来自 1365_tsp_greedy/path_greedy.m。

    Parameters
    ----------
    n : int
        城市数。
    distance : np.ndarray, shape (n, n)
        距离矩阵。
    start : int
        起始城市索引 (0-based)。

    Returns
    -------
    np.ndarray, shape (n,)
        贪心路径。
    """
    p = np.zeros(n, dtype=int)
    p[0] = start
    d = distance.copy()
    d[:, start] = np.inf
    np.fill_diagonal(d, np.inf)

    from_city = start
    for j in range(1, n):
        to_city = int(np.argmin(d[from_city, :]))
        p[j] = to_city
        d[:, to_city] = np.inf
        from_city = to_city
    return p


def tsp_greedy_solver(coordinates):
    """
    对给定坐标点集求解 TSP（贪心策略，多起点）。

    原始算法来自 1365_tsp_greedy/tsp_greedy.m。

    Parameters
    ----------
    coordinates : np.ndarray, shape (n, dim)
        点坐标。

    Returns
    -------
    np.ndarray, shape (n,)
        最优贪心路径的索引排列。
    float
        路径总长度。
    """
    coordinates = np.asarray(coordinates, dtype=float)
    n = coordinates.shape[0]
    if n < 4:
        # 小情况直接返回顺序
        return np.arange(n), 0.0

    # 构造欧氏距离矩阵
    diff = coordinates[:, np.newaxis, :] - coordinates[np.newaxis, :, :]
    distance = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(distance, 0.0)

    best_cost = np.inf
    best_p = np.arange(n)

    for start in range(min(n, 20)):  # 限制起始点数以控制复杂度
        p = path_greedy(n, distance, start)
        cost = path_cost(n, distance, p)
        if cost < best_cost:
            best_cost = cost
            best_p = p

    return best_p, best_cost


def cvt_sensor_iterate(sensors, region_box, n_samples_per_sensor=500,
                       density_func=None):
    """
    对传感器位置执行一次 CVT 迭代。

    融合 239_cvt_1_movie/cvt_iterate.m 与 250_cvt_3d_sampling。

    Parameters
    ----------
    sensors : np.ndarray, shape (n, dim)
        当前传感器位置。
    region_box : np.ndarray, shape (dim, 2)
        区域边界。
    n_samples_per_sensor : int
        每个传感器的样本数。
    density_func : callable or None
        密度函数 density_func(x) -> float。

    Returns
    -------
    np.ndarray
        更新后的传感器位置。
    float
        平均移动距离。
    """
    sensors = np.asarray(sensors, dtype=float)
    n, dim = sensors.shape
    sample_num = n_samples_per_sensor * n

    samples = np.zeros((sample_num, dim), dtype=float)
    for d in range(dim):
        samples[:, d] = region_box[d, 0] + np.random.rand(sample_num) * (
            region_box[d, 1] - region_box[d, 0])

    # 加权采样（若提供密度函数）
    if density_func is not None:
        weights = np.array([density_func(samples[i, :]) for i in range(sample_num)])
        weights = np.clip(weights, 0.0, None)
        if np.sum(weights) > 0.0:
            # 拒绝采样变体：按权重重采样
            probs = weights / np.sum(weights)
            indices = np.random.choice(sample_num, size=sample_num, p=probs)
            samples = samples[indices, :]

    # 找到每个样本最近的传感器
    diff = samples[:, np.newaxis, :] - sensors[np.newaxis, :, :]
    dists = np.sum(diff ** 2, axis=2)
    nearest = np.argmin(dists, axis=1)

    sensors_new = np.zeros_like(sensors)
    counts = np.zeros(n, dtype=int)

    for j in range(sample_num):
        idx = nearest[j]
        sensors_new[idx, :] += samples[j, :]
        counts[idx] += 1

    for j in range(n):
        if counts[j] > 0:
            sensors_new[j, :] /= counts[j]
        else:
            sensors_new[j, :] = sensors[j, :]

    avg_move = np.mean(np.sqrt(np.sum((sensors_new - sensors) ** 2, axis=1)))
    return sensors_new, avg_move


def optimize_sensor_array(n_sensors, region_box, it_max=100, tol=1e-5,
                          density_func=None, return_tsp=True):
    """
    综合优化传感器阵列：CVT 空间分布 + TSP 路径规划。

    Parameters
    ----------
    n_sensors : int
        传感器数量。
    region_box : np.ndarray, shape (dim, 2)
        区域边界。
    it_max : int
        CVT 最大迭代次数。
    tol : float
        CVT 收敛容差。
    density_func : callable or None
        自适应密度函数。
    return_tsp : bool
        是否同时计算 TSP 路径。

    Returns
    -------
    dict
        {'sensors': ..., 'tsp_path': ..., 'tsp_cost': ..., 'cvt_energy': ...}
    """
    dim = region_box.shape[0]
    # 初始化：均匀随机分布
    sensors = np.zeros((n_sensors, dim), dtype=float)
    for d in range(dim):
        sensors[:, d] = region_box[d, 0] + np.random.rand(n_sensors) * (
            region_box[d, 1] - region_box[d, 0])

    # CVT 优化
    for it in range(it_max):
        sensors_new, avg_move = cvt_sensor_iterate(
            sensors, region_box, n_samples_per_sensor=500,
            density_func=density_func)
        sensors = sensors_new
        if avg_move < tol:
            break

    # 计算 CVT 能量
    sample_num = 5000
    samples = np.zeros((sample_num, dim), dtype=float)
    for d in range(dim):
        samples[:, d] = region_box[d, 0] + np.random.rand(sample_num) * (
            region_box[d, 1] - region_box[d, 0])

    diff = samples[:, np.newaxis, :] - sensors[np.newaxis, :, :]
    dists = np.sum(diff ** 2, axis=2)
    min_dists = np.min(dists, axis=1)
    cvt_energy = float(np.mean(min_dists))

    result = {
        'sensors': sensors,
        'cvt_energy': cvt_energy,
        'iterations': it + 1
    }

    if return_tsp:
        tsp_path, tsp_cost = tsp_greedy_solver(sensors)
        result['tsp_path'] = tsp_path
        result['tsp_cost'] = tsp_cost

    return result


class SensorArray:
    """
    传感器阵列管理器，用于冲击波场测量与重构。
    """

    def __init__(self, positions, sensitivity=1.0, noise_level=0.01):
        """
        Parameters
        ----------
        positions : np.ndarray, shape (n, dim)
            传感器空间位置。
        sensitivity : float
            灵敏度系数。
        noise_level : float
            噪声水平（相对标准差）。
        """
        self.positions = np.asarray(positions, dtype=float)
        self.n_sensors = self.positions.shape[0]
        self.dim = self.positions.shape[1]
        self.sensitivity = float(sensitivity)
        self.noise_level = float(noise_level)

    def measure(self, true_field_func):
        """
        模拟传感器测量。

        Parameters
        ----------
        true_field_func : callable
            真实场函数 func(x) -> float。

        Returns
        -------
        np.ndarray, shape (n_sensors,)
            带噪声的测量值。
        """
        measurements = np.zeros(self.n_sensors, dtype=float)
        for i in range(self.n_sensors):
            val = true_field_func(self.positions[i, :])
            if not np.isfinite(val):
                val = 0.0
            noise = np.random.randn() * self.noise_level * max(abs(val), 1.0)
            measurements[i] = self.sensitivity * val + noise
        return measurements

    def reconstruction_mse(self, true_field_func, reconstructed_field_func):
        """
        计算传感器位置处的重构均方误差。

        Parameters
        ----------
        true_field_func : callable
        reconstructed_field_func : callable

        Returns
        -------
        float
            MSE。
        """
        mse = 0.0
        for i in range(self.n_sensors):
            true_val = true_field_func(self.positions[i, :])
            recon_val = reconstructed_field_func(self.positions[i, :])
            if np.isfinite(true_val) and np.isfinite(recon_val):
                mse += (true_val - recon_val) ** 2
        return mse / self.n_sensors
