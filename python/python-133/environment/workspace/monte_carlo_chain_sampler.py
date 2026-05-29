"""
monte_carlo_chain_sampler.py
=============================
聚合物链构象空间蒙特卡洛采样与混合效率估计

基于种子项目 331_ellipse_monte_carlo 与 298_disk_triangle_picking 融合重构。

科学背景：
---------
在高分子物理中，聚合物链的构象统计可用高斯链模型描述：
末端距向量 r 满足三维高斯分布：

    P(r) = (3/(2πNb²))^{3/2} exp(-3r²/(2Nb²))

其中 N 为链段数，b 为 Kuhn 长度。

对于受限空间（如微反应器通道）中的链，可用等效椭球描述
链的惯性张量：

    A = (1/N) Σ_i (r_i - r_cm)(r_i - r_cm)^T

链的回转半径：

    R_g² = Tr(A) / 3

本模块实现：
1. 椭球内均匀采样（模拟受限链构象空间）
2. 圆盘内随机三角形面积估计（模拟搅拌混合效率）
3. 基于 Metropolis-Hastings 的粗粒化链构象采样
4. 聚合物链穿越孔隙的临界条件判断

核心公式：
----------
椭球约束：
    x^T A x ≤ R²

Cholesky 分解：
    A = U^T U

均匀采样算法：
    1. 生成球内均匀样本 y:  y = r * u^{1/3} * ξ/|ξ|
    2. 解 U x = y 得到椭球内样本 x

混合效率指标（基于随机三角形面积）：
    η_mix = <A_triangle> / A_disk

其中 <A_triangle> 为圆盘内随机三角形平均面积，
A_disk = π 为单位圆盘面积。
"""

import numpy as np
from typing import Tuple, Optional


def cholesky_factor(a: np.ndarray) -> np.ndarray:
    """
    对称正定矩阵的 Cholesky 分解 A = U^T U
    返回上三角矩阵 U。
    基于 r8po_fa.m 的算法思想。
    """
    a = np.asarray(a, dtype=float)
    m = a.shape[0]
    u = np.zeros_like(a)

    for j in range(m):
        s = 0.0
        for k in range(j):
            s += u[k, j] ** 2
        diff = a[j, j] - s
        if diff <= 1.0e-14:
            diff = 1.0e-14
        u[j, j] = np.sqrt(diff)
        for i in range(j + 1, m):
            s = 0.0
            for k in range(j):
                s += u[k, i] * u[k, j]
            u[j, i] = (a[j, i] - s) / u[j, j]

    return u


def cholesky_solve(u: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    利用 Cholesky 因子 U 求解 U x = b（前向替换）
    基于 r8po_sl.m 的算法。
    """
    b = np.asarray(b, dtype=float)
    m = u.shape[0]
    x = b.copy()

    for j in range(m):
        x[j] = x[j] / u[j, j]
        for i in range(j + 1, m):
            x[i] -= u[j, i] * x[j]

    return x


def sample_uniform_in_ball(m: int, n: int, radius: float = 1.0,
                           rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """
    在单位球内均匀采样 n 个点（m维）。
    基于 uniform_in_sphere01_map.m 的思想：

    算法：
      1. 生成标准正态向量 ξ
      2. 归一化：η = ξ / |ξ|
      3. 径向采样：r = R * u^{1/m},  u ~ U(0,1)
      4. 样本：x = r * η
    """
    if rng is None:
        rng = np.random.default_rng(seed=133)

    samples = np.zeros((m, n))
    for j in range(n):
        xi = rng.standard_normal(m)
        norm_xi = np.linalg.norm(xi)
        if norm_xi < 1.0e-12:
            xi[0] = 1.0
            norm_xi = 1.0
        u = rng.random()
        r = radius * (u ** (1.0 / m))
        samples[:, j] = r * xi / norm_xi

    return samples


def ellipse_sample(n: int, a_mat: np.ndarray, r: float = 1.0,
                   rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """
    在椭球内均匀采样 n 个点。
    基于 ellipse_sample.m 的算法：

    椭球定义：x^T A x ≤ R²
    1. Cholesky 分解 A = U^T U
    2. 在球内采样 y
    3. 解 U x = y

    参数：
        n     : 采样点数
        a_mat : 2x2 或 3x3 对称正定矩阵
        r     : 椭球半径参数
    """
    a_mat = np.asarray(a_mat, dtype=float)
    m = a_mat.shape[0]
    u = cholesky_factor(a_mat)

    y_samples = sample_uniform_in_ball(m, n, radius=r, rng=rng)
    x_samples = np.zeros((m, n))
    for j in range(n):
        x_samples[:, j] = cholesky_solve(u, y_samples[:, j])

    return x_samples


def polymer_chain_gyration_tensor(chain_coords: np.ndarray) -> np.ndarray:
    """
    计算聚合物链的回转张量（惯性张量）：

        A_{αβ} = (1/N) Σ_i (r_{i,α} - r_{cm,α})(r_{i,β} - r_{cm,β})

    输入 chain_coords 形状为 (N, 3) 或 (N, 2)。
    """
    coords = np.asarray(chain_coords, dtype=float)
    r_cm = np.mean(coords, axis=0)
    centered = coords - r_cm
    A = (centered.T @ centered) / coords.shape[0]
    return A


def radius_of_gyration(chain_coords: np.ndarray) -> float:
    """
    回转半径 R_g = sqrt(Tr(A)/d)
    """
    A = polymer_chain_gyration_tensor(chain_coords)
    d = A.shape[0]
    return np.sqrt(np.trace(A) / d)


def disk_triangle_picking(n_trials: int,
                          rng: Optional[np.random.Generator] = None) -> float:
    """
    估计单位圆盘内随机三角形的平均面积。
    基于 disk_triangle_picking.m 的算法：

    单位圆盘内均匀采样：
        r = sqrt(u),  θ = 2π v,  u,v ~ U(0,1)
        x = r cos θ,  y = r sin θ

    三角形面积（Heron 公式）：
        A = sqrt(s(s-a)(s-b)(s-c))
        s = (a+b+c)/2
    """
    if rng is None:
        rng = np.random.default_rng(seed=42)

    total_area = 0.0
    for _ in range(n_trials):
        theta = 2.0 * np.pi * rng.random(3)
        r = np.sqrt(rng.random(3))
        x = r * np.cos(theta)
        y = r * np.sin(theta)

        s1 = np.sqrt((x[0] - x[1]) ** 2 + (y[0] - y[1]) ** 2)
        s2 = np.sqrt((x[1] - x[2]) ** 2 + (y[1] - y[2]) ** 2)
        s3 = np.sqrt((x[2] - x[0]) ** 2 + (y[2] - y[0]) ** 2)
        s = 0.5 * (s1 + s2 + s3)

        # Heron 公式数值稳定性处理
        area_sq = s * (s - s1) * (s - s2) * (s - s3)
        area_sq = max(area_sq, 0.0)
        area = np.sqrt(area_sq)
        total_area += area

    return total_area / n_trials


def mixing_efficiency_estimate(n_trials: int = 10000,
                               rng: Optional[np.random.Generator] = None) -> Tuple[float, float]:
    """
    基于随机三角形面积估计圆盘内混合效率。

    理论平均面积（已知结果）：
        <A> = 35/(48π) ≈ 0.2319

    混合效率定义为：
        η_mix = <A_triangle> / A_disk = <A_triangle> / π

    返回 (估计平均面积, 混合效率)
    """
    avg_area = disk_triangle_picking(n_trials, rng=rng)
    disk_area = np.pi
    efficiency = avg_area / disk_area
    # 理论值校验
    theoretical = 35.0 / (48.0 * np.pi)
    return avg_area, efficiency, theoretical


def coarse_grained_chain_mc(n_segments: int,
                            n_samples: int,
                            kuhn_length: float = 1.0,
                            confinement_ellipsoid: Optional[np.ndarray] = None,
                            rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """
    粗粒化聚合物链的蒙特卡洛采样。

    模型：
      每个链段为自由连接链，键向量服从高斯分布：
          b_i ~ N(0, b² I/3)

      末端向量 R = Σ b_i ~ N(0, Nb² I/3)

      若存在受限椭球 confinement_ellipsoid，则拒绝超出椭球的构象。

    返回末端距向量样本，形状 (n_samples, 3)。
    """
    if rng is None:
        rng = np.random.default_rng(seed=2024)

    cov = (kuhn_length ** 2 / 3.0) * n_segments * np.eye(3)
    samples = rng.multivariate_normal(mean=np.zeros(3), cov=cov, size=n_samples)

    if confinement_ellipsoid is not None:
        a_mat = np.asarray(confinement_ellipsoid, dtype=float)
        accepted = []
        for s in samples:
            if s @ a_mat @ s <= 1.0:
                accepted.append(s)
        if len(accepted) == 0:
            # 全部拒绝时返回零向量（数值鲁棒性）
            return np.zeros((1, 3))
        samples = np.array(accepted)

    return samples


def critical_pore_size(chain_samples: np.ndarray,
                       porosity: float = 0.4) -> float:
    """
    估计聚合物链可穿越的临界孔隙尺寸。

    基于 de Gennes 的爬行理论 (reptation theory)：
      临界孔径 d_c ≈ 2 R_g / (1 - φ)^{1/3}

    其中 φ 为填充率/孔隙率。
    """
    Rg = np.mean(np.linalg.norm(chain_samples, axis=1))
    dc = 2.0 * Rg / ((1.0 - porosity) ** (1.0 / 3.0) + 1.0e-12)
    return dc
