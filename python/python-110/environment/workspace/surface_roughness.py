"""
surface_roughness.py - 量子点界面粗糙度与分形扰动模块

融合原项目 446_fractal_coastline（分形海岸线扰动）的核心算法，
用于模拟量子点与周围基质界面的原子尺度粗糙度对光学性质的影响。

物理模型：
    - 界面粗糙度导致局域势场涨落，引起非辐射复合与谱线展宽
    - 分形维数 D_f 表征粗糙度空间关联特性：
        D_f = 2 - H，其中 H 为 Hurst 指数
    - 粗糙界面引起的能级位移（Stark 效应等效）：
        Delta_E ~ (hbar^2 / 2m*) (Delta_S / S_0) / R_dot^2
    - 谱线非均匀展宽：
        Gamma_inhom ~ sqrt(<Delta_V^2>) / hbar
"""

import numpy as np
from typing import Tuple
from utils import validate_array_1d, validate_array_2d


def coastline_perturb(
    boundary_points: np.ndarray,
    mu: float,
    n_iter: int = 1,
) -> np.ndarray:
    """
    对闭合曲线进行分形扰动（源自 coastline_perturb）。
    
    每次迭代在相邻顶点间插入中点并进行随机扰动：
        q_{2i}   = p_i
        q_{2i+1} = 0.5(p_i + p_{i+1}) + w_i * (p_i + p_{i+1}) - w_i * (p_{i-1} + p_{i+2})
    
    其中 w_i ~ N(mu, mu^2)。
    
    参数:
        boundary_points: 2 x N 的边界顶点坐标
        mu: 扰动强度控制参数 (0 <= mu <= 0.25)
        n_iter: 迭代次数（每次点数翻倍）
    
    返回:
        扰动后的边界顶点
    """
    boundary_points = validate_array_2d(boundary_points, "boundary_points")
    if boundary_points.shape[0] != 2:
        raise ValueError("boundary_points must be 2 x N")
    if not (0.0 <= mu <= 0.5):
        mu = np.clip(mu, 0.0, 0.5)
    p = boundary_points.T.copy()  # N x 2
    for _ in range(n_iter):
        n = p.shape[0]
        sig = mu ** 2
        w = mu + sig * np.random.randn(n)
        # 计算扰动后的中点
        p_next = np.roll(p, -1, axis=0)
        p_prev = np.roll(p, 1, axis=0)
        p_next2 = np.roll(p, -2, axis=0)
        perturb = (
            0.5 * (p + p_next)
            + w[:, None] * (p + p_next)
            - w[:, None] * (p_prev + p_next2)
        )
        perturb = np.roll(perturb, -1, axis=0)
        q = np.zeros((2 * n, 2), dtype=float)
        q[0:2 * n:2, :] = p
        q[1:2 * n:2, :] = perturb
        p = q
    return p.T  # 返回 2 x (2^n_iter * N)


def fractal_dimension_box_counting(
    curve: np.ndarray,
    n_scales: int = 10,
) -> float:
    """
    使用盒计数法估算分形维数。
    
        N(epsilon) ~ epsilon^{-D_f}
        D_f = - lim_{epsilon->0} log N(epsilon) / log epsilon
    """
    curve = validate_array_2d(curve, "curve")
    x = curve[0, :]
    y = curve[1, :]
    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)
    L_max = max(x_max - x_min, y_max - y_min)
    if L_max < 1e-15:
        return 0.0
    epsilons = L_max / (2.0 ** np.arange(1, n_scales + 1))
    counts = []
    for eps in epsilons:
        nx = int(np.ceil((x_max - x_min) / eps))
        ny = int(np.ceil((y_max - y_min) / eps))
        if nx < 1:
            nx = 1
        if ny < 1:
            ny = 1
        occupied = np.zeros((nx, ny), dtype=bool)
        for i in range(curve.shape[1]):
            ix = int(np.floor((x[i] - x_min) / eps))
            iy = int(np.floor((y[i] - y_min) / eps))
            ix = np.clip(ix, 0, nx - 1)
            iy = np.clip(iy, 0, ny - 1)
            occupied[ix, iy] = True
        counts.append(float(np.sum(occupied)))
    counts = np.array(counts, dtype=float)
    epsilons = np.array(epsilons, dtype=float)
    # 线性拟合 log(N) vs log(1/epsilon)
    valid = counts > 0
    if np.sum(valid) < 2:
        return 1.0
    log_eps = np.log(1.0 / epsilons[valid])
    log_N = np.log(counts[valid])
    # 最小二乘拟合
    A = np.vstack([log_eps, np.ones_like(log_eps)]).T
    D_f, _ = np.linalg.lstsq(A, log_N, rcond=None)[0]
    return float(D_f)


def roughness_induced_broadening(
    rms_roughness_nm: float,
    dot_radius_nm: float,
    m_star_ratio: float = 0.023,
) -> float:
    """
    估算界面粗糙度引起的非均匀展宽（简谐振子近似）：
    
        Gamma_inhom / hbar ~ (hbar / (2 m* R_dot^2)) * (delta_r / R_dot)
    
    其中 delta_r 为 RMS 粗糙度。
    返回角频率展宽 (rad/s)。
    """
    if rms_roughness_nm <= 0 or dot_radius_nm <= 0:
        raise ValueError("Roughness and radius must be positive")
    H_BAR = 1.054571817e-34
    M_E = 9.10938356e-31
    m_star = m_star_ratio * M_E
    R_dot = dot_radius_nm * 1e-9
    delta_r = rms_roughness_nm * 1e-9
    # 特征能量尺度
    E_conf = H_BAR ** 2 / (2.0 * m_star * R_dot ** 2)
    # 粗糙度引起的相对扰动
    Delta_E = E_conf * (delta_r / R_dot)
    Gamma_ang = Delta_E / H_BAR
    return float(Gamma_ang)


def generate_rough_quantum_dot_boundary(
    R_nominal: float,
    n_vertices: int = 64,
    mu_perturb: float = 0.02,
    n_iter: int = 3,
) -> Tuple[np.ndarray, float]:
    """
    生成分形扰动后的量子点圆形边界。
    
    参数:
        R_nominal: 标称半径 (m)
        n_vertices: 初始顶点数
        mu_perturb: 分形扰动强度
        n_iter: 分形迭代次数
    
    返回:
        boundary: 2 x N 的扰动后边界坐标
        D_f: 估算的分形维数
    """
    if R_nominal <= 0:
        raise ValueError("Radius must be positive")
    theta = np.linspace(0.0, 2.0 * np.pi, n_vertices, endpoint=False)
    x = R_nominal * np.cos(theta)
    y = R_nominal * np.sin(theta)
    boundary = np.vstack([x, y])
    rough_boundary = coastline_perturb(boundary, mu_perturb, n_iter)
    D_f = fractal_dimension_box_counting(rough_boundary)
    return rough_boundary, D_f


def effective_potential_perturbation(
    r_grid: np.ndarray,
    R_nominal: float,
    rms_roughness: float,
    barrier_height_eV: float = 0.5,
) -> np.ndarray:
    """
    由界面粗糙度导致的有效势场局域涨落。
    
    模型：在边界附近（|r - R| < delta_r）引入随机势垒起伏：
        V_pert(r) = V0 * exp( - (r - R_rough(r))^2 / (2 delta_r^2) )
    
    此处简化为径向高斯型涨落叠加。
    """
    r_grid = validate_array_1d(r_grid, "r_grid")
    if R_nominal <= 0 or rms_roughness < 0:
        raise ValueError("Invalid geometric parameters")
    EV_TO_J = 1.602176634e-19
    V0 = barrier_height_eV * EV_TO_J
    delta = rms_roughness
    if delta < 1e-12:
        return np.zeros_like(r_grid)
    # 随机涨落振幅
    amp = 0.1 * V0 * (2.0 * np.random.rand() - 1.0)
    V_pert = amp * np.exp(-0.5 * ((r_grid - R_nominal) / delta) ** 2)
    return V_pert
