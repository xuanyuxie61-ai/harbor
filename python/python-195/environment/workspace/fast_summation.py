"""
fast_summation.py
快速求和与采样模块

实现基于 Toeplitz 矩阵结构的高效矩阵-向量乘法，
以及用于粒子初始化的各向同性球面/球体采样。

核心数学：
    - Toeplitz 矩阵:
        T_{ij} = a_{i-j},   即每条对角线元素相同
        
        N x N Toeplitz 矩阵只有 2N-1 个独立元素：
            第一行: a_0, a_1, ..., a_{N-1}
            第一列: a_0, a_N, a_{N+1}, ..., a_{2N-2}
        
        存储格式 R8TO:
            A = [a_0, a_1, ..., a_{N-1}, a_N, ..., a_{2N-2}]
    
    - Toeplitz 矩阵-向量乘法:
        朴素算法: O(N^2)
        FFT 加速: O(N log N)（利用 Toeplitz 矩阵可嵌入循环矩阵的性质）
      
        这里实现朴素版本（保证数值鲁棒性），并嵌入到粒子快速求和框架中。
    
    - 球内均匀采样（Muller 方法）:
        1) 生成三维标准正态向量: v ~ N(0, I_3)
        2) 归一化到单位球面: u = v / ||v||
        3) 生成径向坐标: r = U^{1/3},  U ~ Uniform[0,1]
        4) 最终点: x = r * u
      
        三维球内均匀分布的径向概率密度:
            f(r) = 3 * r^2,   0 <= r <= 1
        因此 r = U^{1/3} 正确。
    
    - 快速多极子展开（简化版）:
        对远距离粒子组，用多极子展开近似相互作用:
            Phi(x) = sum_j q_j / |x - x_j|
            approx Q / |x - x_c| + P . (x - x_c) / |x - x_c|^3 + ...
        其中 Q = sum_j q_j, P = sum_j q_j * (x_j - x_c)
    
    - 负载均衡的快速度量:
        使用前缀和（Prefix Sum）在 O(N) 内计算任意区域的粒子数，
        辅助 ORB 切分决策。
"""

import numpy as np
from typing import Tuple, Optional
from utils import EPSILON_MACHINE


def toeplitz_mv(n: int, a: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Toeplitz 矩阵与向量相乘。
    
    矩阵 A 由 2N-1 个元素定义:
        a[0:N]     : 第一行（从左到右）
        a[N:2N-1]  : 第一列（从第二个元素开始，从上到下）
    
    矩阵元素:
        A[i, j] = a[j - i]      若 j >= i
        A[i, j] = a[n + i - j]  若 j < i
    
    数学验证:
        (A*x)_i = sum_{j=0}^{N-1} A[i,j] * x_j
    
    Parameters
    ----------
    n : int
        矩阵维数
    a : np.ndarray, shape (2*n-1,)
        Toeplitz 数据
    x : np.ndarray, shape (n,)
        输入向量
    
    Returns
    -------
    np.ndarray, shape (n,)
        结果向量
    """
    a = np.asarray(a, dtype=float)
    x = np.asarray(x, dtype=float)
    if a.size < 2 * n - 1:
        raise ValueError(f"a too short: need {2*n-1}, got {a.size}")
    if x.size < n:
        raise ValueError(f"x too short: need {n}, got {x.size}")

    b = np.zeros(n, dtype=float)
    for i in range(n):
        # j >= i: 使用第一行部分
        for j in range(i, n):
            b[i] += a[j - i] * x[j]
        # j < i: 使用第一列部分
        for j in range(i):
            b[i] += a[n + i - j - 1] * x[j]
    return b


def toeplitz_embedded_fft_mv(n: int, a: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    使用 FFT 加速的 Toeplitz 矩阵-向量乘法（可选）。
    
    将 Toeplitz 矩阵嵌入到 (2N-1) x (2N-1) 循环矩阵中，
    利用 FFT 实现 O(N log N) 复杂度。
    
    嵌入向量 c:
        c = [a_0, a_1, ..., a_{N-1}, a_{N-1}, a_{N-2}, ..., a_1]
    
    然后 A*x = first N elements of IFFT(FFT(c) * FFT(x_padded))
    
    Parameters
    ----------
    n : int
        矩阵维数
    a : np.ndarray, shape (2*n-1,)
    x : np.ndarray, shape (n,)
    
    Returns
    -------
    np.ndarray, shape (n,)
    """
    a = np.asarray(a, dtype=float)
    x = np.asarray(x, dtype=float)
    m = 2 * n - 1
    # 构造循环矩阵的第一列
    # 标准嵌入: [first_col; reverse(first_row[1:])]
    # first_row = a[0:n], first_col = [a[0], a[n], a[n+1], ..., a[2n-2]]
    c = np.zeros(m, dtype=complex)
    c[0] = a[0]
    if n > 1:
        c[1:n] = a[n:2*n-1]
        c[n:] = a[n-1:0:-1]

    x_padded = np.zeros(m, dtype=complex)
    x_padded[:n] = x[:n]

    y = np.fft.ifft(np.fft.fft(c) * np.fft.fft(x_padded))
    return np.real(y[:n])


def sample_unit_ball_positive(n_samples: int) -> np.ndarray:
    """
    在单位正八分球内均匀采样。
    
    实际上采样整个单位球，然后取绝对值得到正八分球。
    
    数学:
        v ~ N(0, I_3)
        u = |v| / ||v||   (正八分球方向)
        r = U^{1/3}, U ~ Uniform[0,1]
        x = r * u
    
    Parameters
    ----------
    n_samples : int
        采样点数
    
    Returns
    -------
    np.ndarray, shape (n_samples, 3)
        采样点
    """
    if n_samples <= 0:
        return np.empty((0, 3))
    v = np.random.randn(n_samples, 3)
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms = np.maximum(norms, EPSILON_MACHINE)
    u = np.abs(v) / norms
    r = np.random.rand(n_samples, 1) ** (1.0 / 3.0)
    return r * u


def sample_unit_sphere_surface(n_samples: int, dim: int = 3) -> np.ndarray:
    """
    在单位球面上均匀采样。
    
    使用正态分布归一化法（Muller 方法）:
        v ~ N(0, I_d)
        x = v / ||v||
    
    Parameters
    ----------
    n_samples : int
        采样点数
    dim : int
        维度
    
    Returns
    -------
    np.ndarray, shape (n_samples, dim)
    """
    if n_samples <= 0:
        return np.empty((0, dim))
    v = np.random.randn(n_samples, dim)
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms = np.maximum(norms, EPSILON_MACHINE)
    return v / norms


def compute_prefix_sum_2d(particles: np.ndarray,
                          domain: Tuple[float, float, float, float],
                          nx: int, ny: int) -> np.ndarray:
    """
    计算二维前缀和数组，用于 O(1) 区域粒子数查询。
    
    前缀和定义:
        S[i,j] = sum_{0<=x<i, 0<=y<j} grid[x,y]
    
    区域 [x1,x2) x [y1,y2) 的粒子数:
        N = S[x2,y2] - S[x1,y2] - S[x2,y1] + S[x1,y1]
    
    Parameters
    ----------
    particles : np.ndarray, shape (n, 2)
    domain : tuple
    nx, ny : int
    
    Returns
    -------
    np.ndarray, shape (nx+1, ny+1)
        前缀和数组
    """
    particles = np.asarray(particles, dtype=float)
    xmin, xmax, ymin, ymax = domain
    dx = (xmax - xmin) / nx
    dy = (ymax - ymin) / ny

    grid = np.zeros((nx, ny), dtype=int)
    for p in range(particles.shape[0]):
        ix = int((particles[p, 0] - xmin) / dx)
        iy = int((particles[p, 1] - ymin) / dy)
        ix = max(0, min(nx - 1, ix))
        iy = max(0, min(ny - 1, iy))
        grid[ix, iy] += 1

    prefix = np.zeros((nx + 1, ny + 1), dtype=int)
    for i in range(nx):
        for j in range(ny):
            prefix[i + 1, j + 1] = (
                grid[i, j]
                + prefix[i, j + 1]
                + prefix[i + 1, j]
                - prefix[i, j]
            )
    return prefix


def query_region_count(prefix: np.ndarray,
                       ix1: int, ix2: int,
                       iy1: int, iy2: int) -> int:
    """
    使用前缀和查询矩形区域内的粒子数。
    
    Parameters
    ----------
    prefix : np.ndarray
    ix1, ix2 : int
        x方向索引范围 [ix1, ix2)
    iy1, iy2 : int
        y方向索引范围 [iy1, iy2)
    
    Returns
    -------
    int
        粒子数
    """
    ix1 = max(0, ix1)
    iy1 = max(0, iy1)
    ix2 = min(prefix.shape[0] - 1, ix2)
    iy2 = min(prefix.shape[1] - 1, iy2)
    return (
        prefix[ix2, iy2]
        - prefix[ix1, iy2]
        - prefix[ix2, iy1]
        + prefix[ix1, iy1]
    )


def multipole_expansion(particles: np.ndarray, charges: np.ndarray,
                        center: np.ndarray, max_order: int = 2) -> np.ndarray:
    """
    计算粒子组的多极子展开系数。
    
    展开式:
        Phi(x) = sum_{l=0}^{L} sum_{m=-l}^{l} M_{lm} / |x - x_c|^{l+1} * Y_{lm}(theta, phi)
    
    简化版（笛卡尔坐标）:
        M_0 = sum_j q_j                (单极子 / 总电荷)
        M_1 = sum_j q_j * (x_j - x_c)  (偶极子)
        M_2 = sum_j q_j * (x_j - x_c) (x_j - x_c)^T  (四极子)
    
    Parameters
    ----------
    particles : np.ndarray, shape (n, d)
        粒子位置
    charges : np.ndarray, shape (n,)
        粒子电荷/权重
    center : np.ndarray, shape (d,)
        展开中心
    max_order : int
        最高阶数（0=单极子, 1=偶极子, 2=四极子）
    
    Returns
    -------
    dict
        多极子系数
    """
    particles = np.asarray(particles, dtype=float)
    charges = np.asarray(charges, dtype=float)
    center = np.asarray(center, dtype=float)
    d = particles.shape[1]

    result = {}
    # 单极子
    result['monopole'] = np.sum(charges)

    if max_order >= 1:
        # 偶极子
        dipole = np.zeros(d)
        for j in range(len(charges)):
            dipole += charges[j] * (particles[j] - center)
        result['dipole'] = dipole

    if max_order >= 2:
        # 四极子张量
        quadrupole = np.zeros((d, d))
        for j in range(len(charges)):
            r = particles[j] - center
            quadrupole += charges[j] * np.outer(r, r)
        result['quadrupole'] = quadrupole

    return result


def build_interaction_matrix_toeplitz(n: int, kernel_func: callable,
                                      h: float = 1.0) -> np.ndarray:
    """
    构建一维相互作用核的 Toeplitz 矩阵。
    
    用于快速计算粒子间相互作用:
        A[i,j] = K(|i-j| * h)
    
    典型核函数:
        - 引力/库仑: K(r) = 1/r
        - 高斯: K(r) = exp(-r^2 / sigma^2)
        - 样条: K(r) = ...
    
    Parameters
    ----------
    n : int
        维数
    kernel_func : callable
        核函数 K(r)
    h : float
        网格间距
    
    Returns
    -------
    np.ndarray, shape (2*n-1,)
        Toeplitz 数据
    """
    a = np.zeros(2 * n - 1, dtype=float)
    # 第一行: j - i = 0, 1, ..., n-1
    for k in range(n):
        r = k * h
        a[k] = kernel_func(r) if r > 1e-14 else kernel_func(h * 1e-10)
    # 第一列: i - j = 1, ..., n-1
    for k in range(1, n):
        r = k * h
        a[n + k - 1] = kernel_func(r)
    return a
