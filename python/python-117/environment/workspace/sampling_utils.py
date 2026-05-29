"""
sampling_utils.py
=================
概率分布采样与几何采样模块（融合 seed 541_histogram_pdf_sample 与 seed 1192_svd_sphere）

在粗粒化分子动力学与蒙特卡洛模拟中，高效的统计采样是核心需求。本模块提供：

1. 直方图/CDF 逆变换采样（源自 seed 541_histogram_pdf_sample）：
   将离散的概率密度直方图转化为累积分布函数（CDF），再通过逆变换法生成服从
   该分布的随机样本。数学上，若 CDF 为 F(x)，则样本 X = F^{-1}(U)，其中
   U ~ Uniform(0,1)。

2. 球面均匀采样与 SVD 变形分析（源自 seed 1192_svd_sphere）：
   - Marsaglia 方法生成单位球面上的均匀随机点；
   - SVD 分解分析膜变形的本征模式（主成分）。

关键公式：
    逆变换采样：
        Given CDF values c_y at breakpoints c_x,
        draw u ~ Uniform(0,1), find i s.t. c_y[i] <= u < c_y[i+1],
        interpolate: x = c_x[i] + (u - c_y[i])/(c_y[i+1]-c_y[i]) * (c_x[i+1]-c_x[i])

    Marsaglia 球面采样：
        Sample v ~ N(0, I_3), then normalize: x = v / ||v||_2

    SVD 变形模式分析：
        For membrane vertex displacement matrix Delta (3 x n_v),
        compute SVD: Delta = U * Sigma * V^T
        The left singular vectors U[:,k] are the principal deformation modes.
"""

import numpy as np
from typing import Tuple


# ---------------------------------------------------------------------------
# 1. 直方图 / CDF 采样（源自 seed 541_histogram_pdf_sample）
# ---------------------------------------------------------------------------

def pdf_to_histogram(pdf_func: callable,
                     n_bins: int = 64,
                     x_min: float = -1.0,
                     x_max: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    将连续 PDF 离散化为直方图。

    Parameters
    ----------
    pdf_func : callable
        一维概率密度函数 p(x)。
    n_bins : int
        直方图箱数。
    x_min, x_max : float
        采样区间。

    Returns
    -------
    b_p : ndarray
        每箱的概率密度值（中点处评估）。
    b_l : ndarray
        左边界。
    b_r : ndarray
        右边界。
    """
    b_l = np.linspace(x_min, x_max, n_bins + 1)[:-1]
    b_r = np.linspace(x_min, x_max, n_bins + 1)[1:]
    b_m = 0.5 * (b_l + b_r)
    b_p = np.array([pdf_func(xm) for xm in b_m], dtype=np.float64)
    # 保证非负
    b_p = np.clip(b_p, 0.0, None)
    return b_p, b_l, b_r


def histogram_to_cdf(b_p: np.ndarray, b_l: np.ndarray, b_r: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    将直方图转化为离散 CDF（源自 seed 541_histogram_pdf_sample 核心算法）。

    归一化：
        p_norm[i] = b_p[i] * (b_r[i] - b_l[i])
        P_total = sum(p_norm)
        c_y[0] = 0
        c_y[i+1] = c_y[i] + p_norm[i] / P_total

    Returns
    -------
    c_x : ndarray, shape (n_bins+1,)
        CDF 定义点（即直方图边界）。
    c_y : ndarray, shape (n_bins+1,)
        累积概率，范围 [0, 1]。
    """
    widths = b_r - b_l
    mass = b_p * widths
    total = np.sum(mass)
    if total <= 0:
        total = 1.0
    c_x = np.concatenate(([b_l[0]], b_r))
    c_y = np.zeros(len(b_p) + 1, dtype=np.float64)
    for i in range(len(b_p)):
        c_y[i + 1] = c_y[i] + mass[i] / total
    c_y[-1] = 1.0
    return c_x, c_y


def cdf_to_sample(c_x: np.ndarray, c_y: np.ndarray, n_samples: int) -> np.ndarray:
    """
    逆 CDF 采样（源自 seed 541_histogram_pdf_sample 核心算法）。

    算法：
        1. 生成 u ~ Uniform(0,1)；
        2. 对每個 u，二分查找满足 c_y[left] <= u < c_y[left+1] 的 left；
        3. 线性插值：
               x = c_x[left] + (u - c_y[left])/(c_y[left+1]-c_y[left]) * (c_x[left+1]-c_x[left])

    Parameters
    ----------
    c_x, c_y : ndarray
        CDF 节点与累积概率。
    n_samples : int
        所需样本数。

    Returns
    -------
    samples : ndarray
        服从原始 PDF 分布的样本。
    """
    u = np.random.rand(n_samples)
    samples = np.empty(n_samples, dtype=np.float64)
    n = len(c_y)
    for k in range(n_samples):
        uk = u[k]
        # 二分查找
        lo, hi = 0, n - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if c_y[mid] <= uk:
                lo = mid
            else:
                hi = mid
        left = lo
        # 避免除零
        dy = c_y[left + 1] - c_y[left]
        if abs(dy) < 1e-30:
            samples[k] = c_x[left]
        else:
            frac = (uk - c_y[left]) / dy
            samples[k] = c_x[left] + frac * (c_x[left + 1] - c_x[left])
    return samples


# ---------------------------------------------------------------------------
# 2. 球面采样与 SVD 变形分析（源自 seed 1192_svd_sphere）
# ---------------------------------------------------------------------------

def sphere_sample_marsaglia(n: int) -> np.ndarray:
    """
    Marsaglia 方法在单位球面 S^2 上均匀采样 n 个点。

    数学原理：若 v ~ N(0, I_3)，则 v/||v||_2 在 S^2 上服从均匀分布。
    这是因为多元高斯分布的等概率密度面为球面，且径向与角度独立。

    Parameters
    ----------
    n : int
        采样点数。

    Returns
    -------
    points : ndarray, shape (3, n)
        球面上的点，每列为一个单位向量。
    """
    v = np.random.randn(3, n)
    norms = np.linalg.norm(v, axis=0)
    norms[norms == 0] = 1.0
    points = v / norms
    return points


def svd_deformation_modes(displacement_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    对膜顶点位移矩阵进行 SVD，提取主变形模式（源自 seed 1192_svd_sphere 核心思想）。

    设 Delta 为 3 x n_v 的位移矩阵（3 个空间分量，n_v 个顶点），则：

        Delta = U * Sigma * V^T

    其中：
        - U 的列向量 u_k 为空间变形主方向（3x3 旋转矩阵）；
        - Sigma 的对角元 sigma_k 为各模式的变形幅度；
        - V 的列向量 v_k 为顶点权重分布。

    变形能按模式分解：
        E_k = 0.5 * kappa * sigma_k^2

    Parameters
    ----------
    displacement_matrix : ndarray, shape (3, n_v)
        顶点相对于平衡位置的位移。

    Returns
    -------
    U : ndarray, shape (3, 3)
        左奇异向量（空间模式）。
    S : ndarray, shape (3,)
        奇异值。
    Vt : ndarray, shape (3, n_v)
        右奇异向量的转置。
    """
    U, S, Vt = np.linalg.svd(displacement_matrix, full_matrices=False)
    return U, S, Vt


def sample_random_orientation() -> np.ndarray:
    """
    采样一个随机的三维旋转矩阵（均匀分布于 SO(3)）。

    算法：先生成单位四元数，再转化为旋转矩阵。
    这里采用简化的 SVD 方法：对随机矩阵做 SVD，令 R = U * V^T，
    必要时修正行列式为 +1。
    """
    M = np.random.randn(3, 3)
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


def boltzmann_acceptance(delta_E: float, T: float = 300.0,
                         k_B: float = 8.314e-3) -> bool:
    """
    Metropolis 蒙特卡洛接受准则：

        若 delta_E < 0，必定接受；
        若 delta_E >= 0，以概率 exp(-delta_E/(k_B*T)) 接受。

    Parameters
    ----------
    delta_E : float
        能量变化（kJ/mol）。
    T : float
        温度（K）。
    k_B : float
        玻尔兹曼常数（kJ/(mol*K)）。

    Returns
    -------
    accept : bool
        是否接受该构象变化。
    """
    if delta_E < 0:
        return True
    p = np.exp(-delta_E / (k_B * T))
    return np.random.rand() < p
