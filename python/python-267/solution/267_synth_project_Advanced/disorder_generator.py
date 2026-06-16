"""
disorder_generator.py — 关联无序势生成器
==========================================

本模块为拓扑绝缘体表面生成具有空间关联的随机无序势,
模拟实际材料中杂质、缺陷等对边界态的影响。

核心方法 (借鉴 220_correlation 的高斯随机场采样):

1. **Cholesky 分解法**: K = LLᵀ, V = Lz, z ~ N(0,I)
   精确但 O(N³), 仅适合小规模系统。

2. **特征值分解法**: K = VDVᵀ, V = V√D z
   可截断小特征值以保证正定性。

3. **循环嵌入 + FFT 法** (Dietrich-Newsam):
   将 Toeplitz 关联矩阵嵌入循环矩阵, 用 FFT 加速。
   O(N log N), 适合大系统。

关联函数 (来自凝聚态物理中的无序模型):
    C(r) = exp(-r²/(2ξ²))           — Gauss 关联
    C(r) = exp(-r/ξ)                 — 指数关联
    C(r) = (r/ξ)^ν K_ν(r/ξ) / 2^{ν-1}Γ(ν) — Matérn 关联
    C(r) = max(0, 1-r/ξ)            — 球型关联

其中 ξ 为关联长度, 控制无序的空间光滑度。

物理背景
--------
拓扑绝缘体中的无序效应:
- 弱无序: 边界态不受背散射保护 (时间反演对称性), 电导保持量子化
- 强无序: 可能发生拓扑 Anderson 绝缘体相变
- 关联无序: 影响边界态的局域化长度和能谱统计

来源映射
--------
- 220_correlation: Cholesky/Eigen/FFT 采样方法
- 1373_uniform: 随机数生成 (LCG 用于可重复性)
"""

import numpy as np
from typing import Tuple, Dict, Optional


def gaussian_correlation(r: np.ndarray, xi: float) -> np.ndarray:
    """
    Gauss 关联函数: C(r) = exp(-r²/(2ξ²))

    对应光滑的随机势, 模拟长程杂质势。
    功率谱密度: S(k) = ξ√(2π) exp(-k²ξ²/2)

    Parameters
    ----------
    r : ndarray
        距离数组
    xi : float
        关联长度 (nm)

    Returns
    -------
    C : ndarray
    """
    return np.exp(-r ** 2 / (2.0 * xi ** 2))


def exponential_correlation(r: np.ndarray, xi: float) -> np.ndarray:
    """
    指数关联函数: C(r) = exp(-r/ξ)

    对应短程关联, 模拟点状杂质。
    功率谱密度: S(k) = 2ξ/(1 + k²ξ²) (Lorentzian)

    Parameters
    ----------
    r : ndarray
    xi : float

    Returns
    -------
    C : ndarray
    """
    return np.exp(-r / xi)


def matern_correlation(r: np.ndarray, xi: float, nu: float = 1.5
                       ) -> np.ndarray:
    """
    Matérn 关联函数:
        C(r) = (r/ξ)^ν K_ν(r/ξ) / (2^{ν-1} Γ(ν))

    其中 K_ν 是修正 Bessel 函数 (第二类)。

    ν → ∞ 时趋向 Gauss 关联。
    ν = 0.5 时等价于指数关联。
    ν = 1.5 时: C(r) = (1 + r/ξ) exp(-r/ξ)

    Parameters
    ----------
    r : ndarray
    xi : float
    nu : float
        光滑度参数

    Returns
    -------
    C : ndarray
    """
    x = r / xi
    x_safe = np.where(x < 1e-15, 1e-15, x)

    if abs(nu - 0.5) < 1e-10:
        return np.exp(-x)
    elif abs(nu - 1.5) < 1e-10:
        return (1.0 + x) * np.exp(-x)
    elif abs(nu - 2.5) < 1e-10:
        return (1.0 + x + x ** 2 / 3.0) * np.exp(-x)
    else:
        # 一般情况: 使用 scipy
        from scipy.special import kv, gamma
        norm = 2.0 ** (nu - 1.0) * gamma(nu)
        C = (x_safe ** nu) * kv(nu, x_safe) / norm
        C = np.where(r < 1e-15, 1.0, C)
        return C


def spherical_correlation(r: np.ndarray, xi: float) -> np.ndarray:
    """
    球型关联函数:
        C(r) = max(0, 1 - 3r/(2ξ) + r³/(2ξ³))  for r ≤ 2ξ
        C(r) = 0                                    for r > 2ξ

    有限支撑的关联函数。

    Parameters
    ----------
    r : ndarray
    xi : float

    Returns
    -------
    C : ndarray
    """
    x = r / (2.0 * xi)
    C = np.where(x <= 1.0,
                 1.0 - 1.5 * x + 0.5 * x ** 3,
                 0.0)
    return C


CORRELATION_FUNCTIONS = {
    'gaussian': gaussian_correlation,
    'exponential': exponential_correlation,
    'matern': matern_correlation,
    'spherical': spherical_correlation,
}


def build_correlation_matrix_2d(Nx: int, Ny: int, hx: float, hy: float,
                                xi: float,
                                corr_type: str = 'gaussian'
                                ) -> np.ndarray:
    """
    构建二维网格上的关联矩阵 K。

    K_{ij} = W² × C(|r_i - r_j| / ξ)

    其中 W 是无序强度, C 是归一化关联函数。

    Parameters
    ----------
    Nx, Ny : int
        网格尺寸
    hx, hy : float
        格距
    xi : float
        关联长度 (nm)
    corr_type : str
        关联函数类型

    Returns
    -------
    K : ndarray, shape (Nx*Ny, Nx*Ny)
        关联矩阵 (未乘 W²)
    """
    if corr_type not in CORRELATION_FUNCTIONS:
        raise ValueError(f"未知关联函数: {corr_type}, "
                         f"可选: {list(CORRELATION_FUNCTIONS.keys())}")

    corr_func = CORRELATION_FUNCTIONS[corr_type]
    N_total = Nx * Ny

    # 坐标
    x = np.arange(Nx) * hx
    y = np.arange(Ny) * hy
    xx, yy = np.meshgrid(x, y, indexing='ij')
    coords = np.column_stack([xx.ravel(), yy.ravel()])

    # 距离矩阵
    # |r_i - r_j|² = (x_i-x_j)² + (y_i-y_j)²
    dx = coords[:, 0:1] - coords[:, 0:1].T  # (N, N)
    dy = coords[:, 1:2] - coords[:, 1:2].T
    dist = np.sqrt(dx ** 2 + dy ** 2)

    K = corr_func(dist, xi)

    return K


def sample_disorder_cholesky(K: np.ndarray, W: float,
                             rng: np.random.RandomState
                             ) -> np.ndarray:
    """
    Cholesky 分解法采样关联无序势。

    K = LLᵀ (下三角 Cholesky 分解)
    V = W · L · z,  z ~ N(0, I)

    精确方法, 复杂度 O(N³)。

    Parameters
    ----------
    K : ndarray, shape (N, N)
        关联矩阵
    W : float
        无序强度 (eV)
    rng : RandomState
        随机数生成器

    Returns
    -------
    V : ndarray, shape (N,)
        无序势
    """
    N = K.shape[0]

    # 添加小量保证正定性
    K_reg = K + 1e-12 * np.eye(N)

    try:
        L = np.linalg.cholesky(K_reg)
    except np.linalg.LinAlgError:
        # Cholesky 失败, 使用特征值修复
        eigvals, eigvecs = np.linalg.eigh(K_reg)
        eigvals = np.maximum(eigvals, 1e-10)
        K_fixed = eigvecs @ np.diag(eigvals) @ eigvecs.T
        L = np.linalg.cholesky(K_fixed)

    z = rng.randn(N)
    V = W * L @ z

    return V


def sample_disorder_eigen(K: np.ndarray, W: float,
                          rng: np.random.RandomState,
                          threshold: float = 1e-10
                          ) -> np.ndarray:
    """
    特征值分解法采样关联无序势。

    K = VDVᵀ
    K^{1/2} = V √D Vᵀ  (截断负特征值)
    V_sample = W · K^{1/2} · z

    可控制有效维度 (截断小特征值)。

    Parameters
    ----------
    K : ndarray, shape (N, N)
    W : float
    rng : RandomState
    threshold : float
        特征值截断阈值

    Returns
    -------
    V : ndarray, shape (N,)
    """
    eigvals, eigvecs = np.linalg.eigh(K)

    # 截断负特征值和极小特征值
    eigvals_clipped = np.maximum(eigvals, threshold)
    sqrt_D = np.diag(np.sqrt(eigvals_clipped))

    K_half = eigvecs @ sqrt_D @ eigvecs.T

    z = rng.randn(K.shape[0])
    V = W * K_half @ z

    return V


def sample_disorder_fft(Nx: int, Ny: int, hx: float, hy: float,
                        xi: float, W: float,
                        corr_type: str = 'gaussian',
                        rng: np.random.RandomState = None
                        ) -> np.ndarray:
    """
    FFT 循环嵌入法采样关联无序势 (Dietrich-Newsam 方法)。

    将 Toeplitz 关联矩阵嵌入 (2Nx)×(2Ny) 的循环矩阵,
    利用循环矩阵可被 FFT 对角化的性质:

    1. 计算功率谱密度 S(k) = FFT[C(r)]
    2. 生成复高斯噪声 z̃(k)
    3. V = Re[IFFT(√S(k) · z̃(k))]

    复杂度 O(N log N), 远优于 Cholesky 的 O(N³)。

    Parameters
    ----------
    Nx, Ny : int
    hx, hy : float
    xi : float
    W : float
    corr_type : str
    rng : RandomState

    Returns
    -------
    V : ndarray, shape (Nx*Ny,)
    """
    if rng is None:
        rng = np.random.RandomState(42)

    if corr_type not in CORRELATION_FUNCTIONS:
        raise ValueError(f"未知关联函数: {corr_type}")
    corr_func = CORRELATION_FUNCTIONS[corr_type]

    # 扩展网格 (循环嵌入)
    Nx2 = 2 * Nx
    Ny2 = 2 * Ny

    x = np.arange(Nx2) * hx
    y = np.arange(Ny2) * hy

    # 处理循环距离
    x = np.where(x >= Nx * hx, x - Nx2 * hx, x)
    y = np.where(y >= Ny * hy, y - Ny2 * hy, y)

    xx, yy = np.meshgrid(x, y, indexing='ij')
    dist = np.sqrt(xx ** 2 + yy ** 2)

    # 关联函数在循环网格上的值
    C_circ = corr_func(dist, xi)

    # 2D FFT 得到功率谱
    S_k = np.fft.fft2(C_circ)

    # 确保非负 (数值误差可能导致微小负值)
    S_k = np.maximum(S_k.real, 0) + 0j

    # 生成复高斯噪声
    z_real = rng.randn(Nx2, Ny2)
    z_imag = rng.randn(Nx2, Ny2)
    z_tilde = (z_real + 1j * z_imag) / np.sqrt(2.0)

    # 采样
    V_full = W * np.real(np.fft.ifft2(np.sqrt(S_k + 0j) * z_tilde))

    # 提取原始区域
    V = V_full[:Nx, :Ny].ravel()

    # 归一化: 确保方差正确
    actual_var = np.var(V)
    target_var = W ** 2
    if actual_var > 1e-30:
        V *= np.sqrt(target_var / actual_var)

    return V


def generate_disorder_potential(Nx: int, Ny: int, hx: float, hy: float,
                                W: float, xi: float,
                                method: str = 'fft',
                                corr_type: str = 'gaussian',
                                seed: int = 42
                                ) -> Dict:
    """
    生成拓扑绝缘体表面的关联无序势 (统一接口)。

    Parameters
    ----------
    Nx, Ny : int
    hx, hy : float
    W : float
        无序强度 (eV)
    xi : float
        关联长度 (nm)
    method : str
        采样方法: 'cholesky', 'eigen', 'fft'
    corr_type : str
        关联函数类型
    seed : int
        随机种子

    Returns
    -------
    result : dict
        'potential': ndarray (Nx*Ny,)
        'mean': float
        'variance': float
        'method': str
    """
    rng = np.random.RandomState(seed)
    N_total = Nx * Ny

    if method == 'fft':
        V = sample_disorder_fft(Nx, Ny, hx, hy, xi, W, corr_type, rng)
    elif method in ('cholesky', 'eigen'):
        K = build_correlation_matrix_2d(Nx, Ny, hx, hy, xi, corr_type)
        K *= W ** 2  # 乘无序强度
        if method == 'cholesky':
            V = sample_disorder_cholesky(K, 1.0, rng)
        else:
            V = sample_disorder_eigen(K, 1.0, rng)
    else:
        raise ValueError(f"未知方法: {method}")

    return {
        'potential': V,
        'mean': float(np.mean(V)),
        'variance': float(np.var(V)),
        'max': float(np.max(V)),
        'min': float(np.min(V)),
        'method': method,
        'corr_type': corr_type,
        'W': W,
        'xi': xi,
        'Nx': Nx,
        'Ny': Ny,
        'seed': seed
    }


def disorder_statistics_report(result: Dict) -> str:
    """生成无序势的统计分析报告。"""
    lines = []
    lines.append("=" * 60)
    lines.append("无序势统计分析报告")
    lines.append("=" * 60)
    lines.append(f"生成方法: {result['method']}")
    lines.append(f"关联函数: {result['corr_type']}")
    lines.append(f"无序强度 W = {result['W']:.6f} eV")
    lines.append(f"关联长度 ξ = {result['xi']:.4f} nm")
    lines.append(f"网格: {result['Nx']} × {result['Ny']}")
    lines.append(f"随机种子: {result['seed']}")
    lines.append(f"\n统计量:")
    lines.append(f"  均值: {result['mean']:.6e} eV")
    lines.append(f"  方差: {result['variance']:.6e} eV²")
    lines.append(f"  最大值: {result['max']:.6e} eV")
    lines.append(f"  最小值: {result['min']:.6e} eV")
    lines.append(f"  标准差: {np.sqrt(result['variance']):.6e} eV")

    # 关联验证
    V = result['potential']
    Nx, Ny = result['Nx'], result['Ny']
    V_2d = V.reshape(Nx, Ny)

    # 计算数值关联函数 (沿 x 方向)
    if Nx > 2:
        corr_num = np.correlate(V_2d[:, Ny//2] - np.mean(V),
                                V_2d[:, Ny//2] - np.mean(V),
                                mode='full')
        corr_num = corr_num[Nx-1:] / corr_num[Nx-1]
        lines.append(f"\n数值自关联 (x方向, 归一化):")
        n_show = min(5, len(corr_num))
        for i in range(n_show):
            lines.append(f"  C({i}) = {corr_num[i]:.6f}")

    return "\n".join(lines)
