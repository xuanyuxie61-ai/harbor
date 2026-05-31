"""
adaptive_grid.py
自适应采样点优化与光谱边界检测

融合原项目:
  - 238_cvt: Centroidal Voronoi Tessellation 迭代优化
  - 585_image_sample: 采样点选取与边界坐标提取思想

科学背景:
  超连续谱产生过程中，光谱从初始窄带迅速展宽至倍频程甚至更高。
  固定均匀网格在光谱边缘分辨率不足，而在中心区域过度采样。
  利用 CVT（Centroidal Voronoi Tessellation）算法，可将采样点
  密度自适应地匹配光谱功率密度分布，实现计算资源的最优配置。
  CVT 的数学本质是寻找生成点集 {z_i} 使得能量泛函
      E(z_1,...,z_n) = sum_i integral_{V_i} rho(x) ||x - z_i||^2 dx
  最小化，其中 rho(x) 为密度函数（此处取光谱功率密度），
  V_i 为 Voronoi 单元。Lloyd 算法通过交替执行:
      (1) 构建 Voronoi 剖分
      (2) 将生成点移至各单元的质心
  实现能量下降。
"""

import numpy as np
from typing import Callable, Tuple


def cvt_energy(dim_num: int, n: int, r: np.ndarray, sample_num: int,
               sample_points: np.ndarray) -> float:
    """
    计算离散 CVT 能量泛函。

    公式:
        E = (1/N_s) * sum_{k=1}^{N_s} ||x_k - z_{c(k)}||^2
    其中 z_{c(k)} 为距离 x_k 最近的生成点。

    Parameters
    ----------
    dim_num : int
        空间维度。
    n : int
        生成点数量。
    r : np.ndarray
        生成点坐标，形状 (dim_num, n)。
    sample_num : int
        采样点数量。
    sample_points : np.ndarray
        采样点坐标，形状 (dim_num, sample_num)。

    Returns
    -------
    float
        归一化 CVT 能量。
    """
    if r.shape != (dim_num, n):
        raise ValueError("cvt_energy: r shape mismatch")
    if sample_points.shape != (dim_num, sample_num):
        raise ValueError("cvt_energy: sample_points shape mismatch")
    total = 0.0
    for k in range(sample_num):
        diff = r - sample_points[:, k:k + 1]
        dist2 = np.sum(diff ** 2, axis=0)
        total += np.min(dist2)
    return total / sample_num


def find_closest(dim_num: int, n: int, r: np.ndarray, sample_points: np.ndarray) -> np.ndarray:
    """
    对每个采样点，找到距离最近的生成点索引。

    Parameters
    ----------
    dim_num : int
        空间维度。
    n : int
        生成点数量。
    r : np.ndarray
        生成点，形状 (dim_num, n)。
    sample_points : np.ndarray
        采样点，形状 (dim_num, m)。

    Returns
    -------
    np.ndarray
        最近生成点索引，形状 (m,)，整数。
    """
    m = sample_points.shape[1]
    closest = np.zeros(m, dtype=int)
    for k in range(m):
        diff = r - sample_points[:, k:k + 1]
        dist2 = np.sum(diff ** 2, axis=0)
        closest[k] = int(np.argmin(dist2))
    return closest


def cvt_iterate(dim_num: int, n: int, sample_points: np.ndarray,
                r: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """
    执行一次 Lloyd CVT 迭代。

    步骤:
        1. 对每个采样点，找到最近的生成点（Voronoi 单元归属）。
        2. 将每个生成点更新为其 Voronoi 单元内采样点的质心。
        3. 计算移动距离（L2范数）和能量。

    Parameters
    ----------
    dim_num : int
        空间维度。
    n : int
        生成点数量。
    sample_points : np.ndarray
        采样点，形状 (dim_num, m)。
    r : np.ndarray
        当前生成点，形状 (dim_num, n)。

    Returns
    -------
    tuple
        (r_new, it_diff, energy)
    """
    m = sample_points.shape[1]
    r_new = np.zeros_like(r)
    counts = np.zeros(n, dtype=int)
    closest = find_closest(dim_num, n, r, sample_points)
    for k in range(m):
        idx = closest[k]
        r_new[:, idx] += sample_points[:, k]
        counts[idx] += 1
    # 处理空单元：保留原位置
    for i in range(n):
        if counts[i] > 0:
            r_new[:, i] /= counts[i]
        else:
            r_new[:, i] = r[:, i]
    it_diff = np.linalg.norm(r_new - r, 'fro')
    energy = cvt_energy(dim_num, n, r_new, m, sample_points)
    return r_new, it_diff, energy


def cvt_sample_uniform(dim_num: int, n_samples: int, bounds: Tuple[float, float]) -> np.ndarray:
    """
    在指定区间内均匀随机采样。

    Parameters
    ----------
    dim_num : int
        维度。
    n_samples : int
        采样点数量。
    bounds : tuple
        (low, high)。

    Returns
    -------
    np.ndarray
        形状 (dim_num, n_samples)。
    """
    low, high = bounds
    return np.random.uniform(low, high, (dim_num, n_samples))


def adaptive_cvt_grid(n_points: int, density_func: Callable,
                      domain: Tuple[float, float],
                      n_samples: int = 5000,
                      it_max: int = 30,
                      tol: float = 1e-5) -> np.ndarray:
    """
    基于密度函数的自适应 CVT 采样点生成。

    算法:
        1. 在 domain 内按密度函数密度生成大量随机采样点（拒绝采样或加权采样）。
        2. 初始化 n_points 个生成点（均匀分布）。
        3. 执行 Lloyd 迭代直至收敛。
        4. 返回优化后的生成点（排序后）。

    在超连续谱仿真中，density_func 可取光谱功率密度 S(omega)，
    使得采样点在光谱功率高的区域更密集。

    Parameters
    ----------
    n_points : int
        目标采样点数量。
    density_func : callable
        密度函数 f(x) >= 0。
    domain : tuple
        (x_min, x_max)。
    n_samples : int
        每次迭代的采样点数量。
    it_max : int
        最大迭代次数。
    tol : float
        收敛阈值（生成点移动量）。

    Returns
    -------
    np.ndarray
        优化后的采样点，形状 (n_points,)。
    """
    if n_points < 2:
        raise ValueError("adaptive_cvt_grid: n_points must be >= 2")
    x_min, x_max = domain
    dim_num = 1
    # 初始化生成点
    r = np.linspace(x_min, x_max, n_points).reshape(1, n_points)
    # 生成带权重的采样点（重要性采样）
    # 先生成大量均匀点，然后按密度函数加权重采样
    uniform = np.random.uniform(x_min, x_max, n_samples * 2)
    weights = np.maximum(density_func(uniform), 1e-12)
    weights /= np.sum(weights)
    # 按权重重采样
    sample_points = np.random.choice(uniform, size=n_samples, p=weights, replace=True)
    sample_points = sample_points.reshape(1, n_samples)
    for it in range(it_max):
        r_new, it_diff, _ = cvt_iterate(dim_num, n_points, sample_points, r)
        r = r_new
        if it_diff < tol:
            break
    points = r.flatten()
    points = np.sort(points)
    # 限制在 domain 内
    points = np.clip(points, x_min, x_max)
    return points


def spectral_boundary_detect(power_spectrum: np.ndarray,
                              omega: np.ndarray,
                              threshold_db: float = -30.0) -> Tuple[float, float]:
    """
    检测光谱的有效边界（-30 dB 带宽）。

    基于 image_sample 的边界采样思想：将光谱视为"图像"，
    通过阈值提取有效功率区域边界。

    算法:
        1. 将功率谱转换为 dB: P_dB = 10*log10(P / max(P))。
        2. 找到 P_dB > threshold_db 的左右边界索引。
        3. 返回对应的 omega 边界。

    Parameters
    ----------
    power_spectrum : np.ndarray
        功率谱密度（线性尺度）。
    omega : np.ndarray
        对应的角频率数组。
    threshold_db : float
        阈值（dB），默认 -30。

    Returns
    -------
    tuple
        (omega_left, omega_right) 有效光谱边界。
    """
    if len(power_spectrum) != len(omega):
        raise ValueError("spectral_boundary_detect: length mismatch")
    p_max = np.max(power_spectrum)
    if p_max <= 0.0:
        return float(omega[0]), float(omega[-1])
    p_db = 10.0 * np.log10(power_spectrum / p_max + 1e-20)
    mask = p_db > threshold_db
    indices = np.where(mask)[0]
    if len(indices) == 0:
        return float(omega[0]), float(omega[-1])
    idx_left = int(indices[0])
    idx_right = int(indices[-1])
    # 边界插值
    if idx_left > 0:
        t = (threshold_db - p_db[idx_left - 1]) / (p_db[idx_left] - p_db[idx_left - 1] + 1e-20)
        omega_left = omega[idx_left - 1] + t * (omega[idx_left] - omega[idx_left - 1])
    else:
        omega_left = float(omega[idx_left])
    if idx_right < len(omega) - 1:
        t = (threshold_db - p_db[idx_right]) / (p_db[idx_right + 1] - p_db[idx_right] + 1e-20)
        omega_right = omega[idx_right] + t * (omega[idx_right + 1] - omega[idx_right])
    else:
        omega_right = float(omega[idx_right])
    return float(omega_left), float(omega_right)


def log_spaced_grid(omega_min: float, omega_max: float, n_points: int) -> np.ndarray:
    """
    在对数尺度上生成等间距频率网格。

    对于超连续谱，对数网格在短波长（高频率）端提供更高的分辨率，
    这与物理上色散和非线性效应的频率依赖性相匹配。

    Parameters
    ----------
    omega_min : float
        最小角频率。
    omega_max : float
        最大角频率。
    n_points : int
        点数。

    Returns
    -------
    np.ndarray
        对数间距网格点。
    """
    if omega_min <= 0.0 or omega_max <= omega_min:
        raise ValueError("log_spaced_grid: invalid frequency range")
    log_min = np.log10(omega_min)
    log_max = np.log10(omega_max)
    log_grid = np.linspace(log_min, log_max, n_points)
    return 10.0 ** log_grid
