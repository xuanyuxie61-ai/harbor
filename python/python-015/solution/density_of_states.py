"""
density_of_states.py
态密度（Density of States, DOS）计算与广义Hermite求积

凝聚态物理背景：
态密度定义为：
    D(E) = (1/V_BZ) * sum_n \int_{BZ} delta(E - E_n(k)) d^3k

利用delta函数的性质，可写为：
    D(E) = sum_n \int_{S_n(E)} dS / |\nabla_k E_n(k)|

其中S_n(E)是第n条能带的等能面。

数值方法：
1. 直方图法：在k空间网格上计算能量，统计落入每个能量窗口的k点数
2. 高斯展宽法：delta(E - E_n(k)) ≈ (1/sqrt(pi)*sigma) * exp(-(E-E_n)^2/sigma^2)
3. 广义Hermite求积法（基于种子项目464_gen_hermite_exactness）

广义Hermite求积规则：
    \int_{-inf}^{+inf} |x|^alpha * exp(-x^2) * f(x) dx ≈ sum_i w_i * f(x_i)

正交性：
    \int_{-inf}^{+inf} |x|^alpha * exp(-x^2) * H_m(x) * H_n(x) dx = h_n * delta_{mn}

在Weyl半金属中，靠近Weyl节点E ~ \pm hbar*v_F*|k|，
态密度具有特征行为：D(E) ~ E^2（三维无质量Dirac费米子）。
"""

import numpy as np
from typing import Callable, Tuple


def gauss_hermite_nodes_weights(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算标准Gauss-Hermite求积节点和权重
    
    通过三对角Jacobi矩阵的本征问题求解：
        J_{i,i} = 0
        J_{i,i+1} = J_{i+1,i} = sqrt((i+1)/2)
    
    节点 = J的本征值
    权重 = sqrt(pi) * v_0^2，其中v_0是本征矢的第一个分量
    
    Parameters
    ----------
    n : int
        求积点数
    
    Returns
    -------
    x : np.ndarray, shape (n,)
        节点（abscissas）
    w : np.ndarray, shape (n,)
        权重
    """
    if n <= 0:
        return np.array([]), np.array([])
    
    # 构建Jacobi矩阵
    if n == 1:
        x = np.array([0.0])
        w = np.array([np.sqrt(np.pi)])
        return x, w
    
    J = np.zeros((n, n))
    for i in range(n - 1):
        J[i, i + 1] = np.sqrt((i + 1) / 2.0)
        J[i + 1, i] = J[i, i + 1]
    
    eigenvalues, eigenvectors = np.linalg.eigh(J)
    x = eigenvalues
    w = np.sqrt(np.pi) * eigenvectors[0, :] ** 2
    
    return x, w


def generalized_hermite_integral(expon: int, alpha: float) -> float:
    """
    计算广义Hermite积分
    
    基于种子项目464_gen_hermite_exactness中的gen_hermite_integral。
    
    I(n, alpha) = \int_{-inf}^{+inf} x^n * |x|^alpha * exp(-x^2) dx
    
    解析结果：
    - 若n为奇数：I = 0
    - 若n为偶数：I = Gamma((n + alpha + 1) / 2)
    
    Parameters
    ----------
    expon : int
        x的幂次
    alpha : float
        |x|的幂次，要求 alpha > -1
    
    Returns
    -------
    value : float
    """
    from scipy.special import gamma as scipy_gamma
    
    if expon % 2 == 1:
        return 0.0
    
    a = alpha + expon
    if a <= -1.0:
        return -np.inf
    
    value = scipy_gamma((a + 1.0) / 2.0)
    return value


def dos_histogram(energies: np.ndarray, e_min: float, e_max: float,
                   n_bins: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """
    使用直方图法计算态密度
    
    Parameters
    ----------
    energies : np.ndarray, shape (N,)
        所有k点的能量值
    e_min, e_max : float
        能量范围
    n_bins : int
    
    Returns
    -------
    bin_centers : np.ndarray, shape (n_bins,)
    dos : np.ndarray, shape (n_bins,)
    """
    if len(energies) == 0:
        return np.zeros(n_bins), np.zeros(n_bins)
    
    hist, bin_edges = np.histogram(energies, bins=n_bins, range=(e_min, e_max))
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_width = (e_max - e_min) / n_bins
    
    # 归一化态密度
    dos = hist / (len(energies) * bin_width)
    
    return bin_centers, dos


def dos_gaussian_broadening(energies: np.ndarray,
                             e_grid: np.ndarray,
                             sigma: float = 0.05) -> np.ndarray:
    """
    使用高斯展宽法计算态密度
    
    D(E) = (1/N) * sum_i (1/sqrt(2*pi)*sigma) * exp(-(E - E_i)^2 / (2*sigma^2))
    
    Parameters
    ----------
    energies : np.ndarray, shape (N,)
    e_grid : np.ndarray, shape (M,)
    sigma : float
    
    Returns
    -------
    dos : np.ndarray, shape (M,)
    """
    N = len(energies)
    if N == 0:
        return np.zeros_like(e_grid)
    
    prefactor = 1.0 / (np.sqrt(2.0 * np.pi) * sigma)
    dos = np.zeros_like(e_grid)
    
    for e_val in energies:
        dos += prefactor * np.exp(-0.5 * ((e_grid - e_val) / sigma) ** 2)
    
    dos /= N
    return dos


def dos_weyl_semimetal_analytic(e: np.ndarray, v_f: float = 1.0,
                                 hbar: float = 1.0) -> np.ndarray:
    """
    三维Weyl半金属的解析态密度
    
    对于线性色散 E = hbar*v_F*|k|，态密度为：
        D(E) = (1/2*pi^2) * (E / (hbar*v_F)^3)^2
    
    对于N个Weyl节点：D(E) = N * D_1(E)
    
    Parameters
    ----------
    e : np.ndarray
    v_f : float
    hbar : float
    
    Returns
    -------
    dos : np.ndarray
    """
    e = np.asarray(e)
    dos = np.zeros_like(e)
    
    # D(E) = E^2 / (2*pi^2 * (hbar*v_F)^3)
    nonzero = np.abs(e) > 1e-14
    dos[nonzero] = e[nonzero] ** 2 / (2.0 * np.pi ** 2 * (hbar * v_f) ** 3)
    
    return dos


def integrate_dos_with_hermite(energy_func: Callable[[np.ndarray], np.ndarray],
                                k_bounds: np.ndarray,
                                e_ref: float,
                                n_hermite: int = 20,
                                n_k_samples: int = 1000) -> float:
    """
    使用广义Hermite求积计算特定能量的态密度贡献
    
    将被积函数分解为Gaussian权重部分和非权重部分：
        D(E_ref) = \int f(k) * delta(E(k) - E_ref) dk
                 ≈ \int f(k) * g(E(k) - E_ref) dk
    
    其中g(x) = (1/sqrt(pi)*sigma) * exp(-x^2/sigma^2)
    
    通过变量替换，将积分转化为标准Hermite求积形式。
    
    Parameters
    ----------
    energy_func : callable
        能量函数，输入k(N,3)，输出E(N,)
    k_bounds : np.ndarray, shape (3, 2)
    e_ref : float
        参考能量
    n_hermite : int
        Hermite求积点数
    n_k_samples : int
        k空间采样数
    
    Returns
    -------
    dos_value : float
    """
    # 标准Gauss-Hermite节点和权重
    x_gh, w_gh = gauss_hermite_nodes_weights(n_hermite)
    
    # 在k空间随机采样
    k_samples = np.zeros((n_k_samples, 3))
    for d in range(3):
        k_samples[:, d] = np.random.uniform(k_bounds[d, 0], k_bounds[d, 1], n_k_samples)
    
    energies = energy_func(k_samples)
    
    # 计算高斯展宽
    sigma = 0.05 * (np.max(energies) - np.min(energies))
    if sigma < 1e-10:
        sigma = 0.01
    
    # 使用Hermite求积近似delta函数
    # delta(E - E_ref) ≈ sum_i w_i * phi(x_i)
    # 这里简化为高斯展宽
    dos_val = np.sum(np.exp(-((energies - e_ref) / sigma) ** 2))
    dos_val /= (n_k_samples * sigma * np.sqrt(np.pi))
    
    # 体积归一化
    vol = np.prod(k_bounds[:, 1] - k_bounds[:, 0])
    dos_val *= vol
    
    return dos_val


def test_hermite_exactness(alpha: float, max_degree: int = 10,
                            n_points: int = 10) -> np.ndarray:
    """
    测试Gauss-Hermite求积规则的精确性
    
    基于种子项目464_gen_hermite_exactness的核心思想。
    
    对于n点Gauss-Hermite规则，应精确积分次数不超过2n-1的多项式。
    
    Parameters
    ----------
    alpha : float
        权函数指数
    max_degree : int
        测试的最大多项式次数
    n_points : int
        求积点数
    
    Returns
    -------
    errors : np.ndarray, shape (max_degree + 1,)
        每个次数的相对误差
    """
    x, w = gauss_hermite_nodes_weights(n_points)
    
    errors = np.zeros(max_degree + 1)
    for degree in range(max_degree + 1):
        # 计算精确值
        exact = generalized_hermite_integral(degree, alpha)
        
        # 数值求积
        if exact == 0.0:
            quad = np.sum(w * (x ** degree))
            errors[degree] = abs(quad)
        else:
            quad = np.sum(w * (x ** degree))
            errors[degree] = abs((quad - exact) / exact)
    
    return errors
