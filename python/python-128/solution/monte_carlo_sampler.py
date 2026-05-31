"""
monte_carlo_sampler.py
======================
椭球形细胞域上的蒙特卡洛采样与体积估计

融合原始项目：
  - 334_ellipsoid_monte_carlo：椭球内均匀采样、Cholesky 分解、体积计算

数学物理模型：
  1. 椭球定义：
       E = { x ∈ ℝ³ | (x - v)^T A (x - v) ≤ r² }
     其中 A 为正定对称矩阵，v 为中心，r 为等效半径。

  2. 体积公式：
       Vol(E) = r³ · Vol(unit_sphere_3D) / √(det A)
              = (4/3) π r³ / √(det A)

  3. 采样算法：
       a) 计算 A 的 Cholesky 分解：A = U^T U
       b) 在单位球内生成均匀随机点 Y
       c) 通过线性变换得到椭球内点：X = v + U^{-1} Y · r

  4. Monte Carlo 积分：
       对定义在椭球上的函数 f，有
       ∫_E f(x) dx ≈ Vol(E) · (1/N) Σ_{i=1}^N f(X_i)
"""

import numpy as np


def uniform_in_sphere3d(n: int):
    """
    在单位球 B(0,1) ⊂ ℝ³ 内生成 n 个均匀分布的随机点。

    算法：
      1. 生成标准正态分布向量 g ~ N(0, I₃)
      2. 归一化到球面：u = g / ||g||
      3. 生成径向坐标：r = U^{1/3}，其中 U ~ Uniform(0,1)
      4. 点坐标：X = r · u
    """
    n = max(1, int(n))
    g = np.random.normal(0.0, 1.0, size=(n, 3))
    norms = np.linalg.norm(g, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    u = g / norms
    r = np.random.uniform(0.0, 1.0, size=(n, 1)) ** (1.0 / 3.0)
    return r * u


def cholesky_upper(A: np.ndarray):
    """
    计算正定矩阵 A 的上三角 Cholesky 因子 U，满足 A = U^T U。

    算法（平方根法）：
        U_{ii} = √( A_{ii} - Σ_{k=1}^{i-1} U_{ki}² )
        U_{ij} = ( A_{ij} - Σ_{k=1}^{i-1} U_{ki} U_{kj} ) / U_{ii},  j > i
    """
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("cholesky_upper: A 必须为方阵")
    n = A.shape[0]
    U = np.zeros((n, n), dtype=float)
    for i in range(n):
        s = A[i, i] - np.sum(U[:i, i] ** 2)
        if s <= 1e-15:
            raise ValueError("cholesky_upper: 矩阵非正定或接近奇异 (i=%d, s=%g)" % (i, s))
        U[i, i] = np.sqrt(s)
        for j in range(i + 1, n):
            U[i, j] = (A[i, j] - np.dot(U[:i, i], U[:i, j])) / U[i, i]
    return U


def solve_upper_triangular(U: np.ndarray, b: np.ndarray):
    """
    求解上三角系统 U x = b（回代法）。
    """
    n = U.shape[0]
    x = b.copy().astype(float)
    for i in range(n - 1, -1, -1):
        if abs(U[i, i]) < 1e-15:
            raise ValueError("solve_upper_triangular: 零对角元")
        x[i] = (x[i] - np.dot(U[i, i + 1:], x[i + 1:])) / U[i, i]
    return x


def ellipsoid_sample(m: int, n: int, A: np.ndarray, v: np.ndarray, r: float):
    """
    在 m 维椭球内生成 n 个均匀随机样本点。

    参数
    ----
    m : int
        空间维度
    n : int
        样本数
    A : np.ndarray, shape (m, m)
        正定对称矩阵定义椭球形状
    v : np.ndarray, shape (m,)
        中心点
    r : float
        等效半径

    返回
    ----
    X : np.ndarray, shape (m, n)
        样本点（每列一个点）
    """
    A = np.asarray(A, dtype=float)
    v = np.asarray(v, dtype=float)
    if A.shape != (m, m):
        raise ValueError("ellipsoid_sample: A 维度不匹配")
    if v.size != m:
        raise ValueError("ellipsoid_sample: v 维度不匹配")

    U = cholesky_upper(A)
    # 生成单位球内点 Y，形状 (m, n)
    if m == 3:
        Y = uniform_in_sphere3d(n).T  # (3, n)
    else:
        # 通用维度：使用拒绝采样
        Y = np.zeros((m, n), dtype=float)
        accepted = 0
        batch = max(n, 1000)
        while accepted < n:
            g = np.random.normal(0.0, 1.0, size=(m, batch))
            norms = np.linalg.norm(g, axis=0)
            g = g / np.maximum(norms, 1e-15)
            rad = np.random.uniform(0.0, 1.0, size=batch) ** (1.0 / m)
            candidates = g * rad
            for j in range(batch):
                if accepted >= n:
                    break
                Y[:, accepted] = candidates[:, j]
                accepted += 1

    Y *= r
    X = np.zeros((m, n), dtype=float)
    for j in range(n):
        X[:, j] = solve_upper_triangular(U, Y[:, j]) + v
    return X


def ellipsoid_volume_mc(A: np.ndarray, r: float, m: int = 3):
    """
    通过 Monte Carlo 估计椭球体积。

    公式：
        Vol = r^m · V_unit_sphere(m) / √(det A)
    其中 V_unit_sphere(3) = 4π/3。
    """
    A = np.asarray(A, dtype=float)
    U = cholesky_upper(A)
    sqrt_det = np.prod(np.diag(U))
    if m == 3:
        V_unit = 4.0 * np.pi / 3.0
    else:
        # 高维单位球体积
        from math import gamma
        V_unit = np.pi ** (m / 2.0) / gamma(m / 2.0 + 1.0)
    return (r ** m) * V_unit / sqrt_det


class CellMonteCarloSampler:
    """
    基于 Monte Carlo 的细胞状态采样器。

    用于估计：
      - 细胞在局部微环境中的占据体积
      - 受体-配体结合事件的期望值
      - 力学相互作用积分
    """

    def __init__(self, cell_agent, n_samples: int = 500):
        from cell_dynamics import CellAgent
        if not isinstance(cell_agent, CellAgent):
            raise TypeError("CellMonteCarloSampler: 需要 CellAgent 对象")
        self.cell = cell_agent
        self.n_samples = max(10, int(n_samples))

    def sample_cell_body(self):
        """
        在细胞椭球体内均匀采样点。

        返回
        ----
        points : np.ndarray, shape (3, n_samples)
        """
        a, b, c = self.cell.shape
        # 等效椭球矩阵：diag(1/a², 1/b², 1/c²)
        A = np.diag([1.0 / (a * a), 1.0 / (b * b), 1.0 / (c * c)])
        v = self.cell.position
        return ellipsoid_sample(3, self.n_samples, A, v, 1.0)

    def estimate_receptor_binding(self, concentration_func):
        """
        估计细胞表面受体-配体结合事件的期望值。

        假设受体在细胞表面均匀分布，结合概率与局部浓度成正比：
            ⟨B⟩ ≈ (1/N) Σ_{x∈surface} c(x) / (K_d + c(x))

        这里简化为在体积采样点上计算平均浓度（作为代理）。
        """
        pts = self.sample_cell_body()
        Kd = 0.1  # 解离常数
        vals = np.zeros(self.n_samples)
        for j in range(self.n_samples):
            c_loc = concentration_func(pts[:, j])
            vals[j] = c_loc / (Kd + c_loc)
        return float(np.mean(vals))

    def estimate_local_volume(self):
        """
        计算细胞占据体积。
        """
        a, b, c = self.cell.shape
        return ellipsoid_volume_mc(np.diag([1.0 / (a * a), 1.0 / (b * b), 1.0 / (c * c)]), 1.0, 3)
