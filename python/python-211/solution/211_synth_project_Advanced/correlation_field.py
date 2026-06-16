"""
correlation_field.py
====================
相关函数与高斯随机场生成 —— 随机优化地景构造

融合种子项目:
  - 220_correlation: 球面/线性/Bessel/幂律相关函数, FFT/Cholesky 采样

核心公式:
  1. 球面相关: C(ρ) = 1 - 1.5|ρ̂| + 0.5|ρ̂|³, |ρ̂| = min(|ρ|/ρ₀, 1)
  2. 线性相关: C(ρ) = max(1 - |ρ|/ρ₀, 0)
  3. 指数相关: C(ρ) = exp(-|ρ|/ρ₀)
  4. 高斯相关: C(ρ) = exp(-ρ²/(2ρ₀²))
  5. Matérn 相关: C(ρ) = 2^{1-ν}/Γ(ν) (√(2ν)ρ/ρ₀)^ν K_ν(√(2ν)ρ/ρ₀)
  6. Bessel 相关: C(ρ) = J₀(ρ/ρ₀) (零阶 Bessel 函数)
  7. 幂律相关: C(ρ) = (1 + ρ²/ρ₀²)^{-α}
  8. 谱密度 (Wiener-Khinchin): S(k) = ℱ{C(ρ)}
"""

import numpy as np
from typing import Callable, Tuple, Optional
import math


# ---------------------------------------------------------------------------
# 1. 相关函数族 (源自 220_correlation)
# ---------------------------------------------------------------------------

def correlation_spherical(rho: np.ndarray, rho0: float) -> np.ndarray:
    """球面相关函数 (源自 220_correlation/correlation_spherical).
    基于两球体重叠体积:
        C(ρ) = 1 - 1.5·ρ̂ + 0.5·ρ̂³,  其中 ρ̂ = min(|ρ|/ρ₀, 1)

    紧支集: C(ρ) = 0 当 |ρ| > ρ₀.
    连续可微一次, 在随机场建模中产生粗糙但连续的样本.
    """
    rho = np.asarray(rho, dtype=float)
    rhohat = np.minimum(np.abs(rho) / rho0, 1.0)
    return 1.0 - 1.5 * rhohat + 0.5 * rhohat ** 3


def correlation_linear(rho: np.ndarray, rho0: float) -> np.ndarray:
    """线性相关函数 (源自 220_correlation/correlation_linear).
    C(ρ) = max(1 - |ρ|/ρ₀, 0)
    三角形相关, 紧支集, Lipschitz 连续.
    """
    rho = np.asarray(rho, dtype=float)
    return np.maximum(1.0 - np.abs(rho) / rho0, 0.0)


def correlation_exponential(rho: np.ndarray, rho0: float) -> np.ndarray:
    """指数相关函数.
    C(ρ) = exp(-|ρ|/ρ₀)
    Ornstein-Uhlenbeck 过程的协方差. 连续但不可微.
    """
    rho = np.asarray(rho, dtype=float)
    return np.exp(-np.abs(rho) / rho0)


def correlation_gaussian(rho: np.ndarray, rho0: float) -> np.ndarray:
    """高斯相关函数.
    C(ρ) = exp(-ρ²/(2ρ₀²))
    无穷次可微, 产生非常光滑的随机场样本.
    """
    rho = np.asarray(rho, dtype=float)
    return np.exp(-rho ** 2 / (2.0 * rho0 ** 2))


def correlation_matern(rho: np.ndarray, rho0: float, nu: float = 1.5) -> np.ndarray:
    """Matérn 相关函数.
    C(ρ) = 2^{1-ν}/Γ(ν) · (√(2ν)|ρ|/ρ₀)^ν · K_ν(√(2ν)|ρ|/ρ₀)

    参数 ν 控制光滑度:
      ν → ∞: 趋近高斯相关
      ν = 0.5: 等价指数相关
      ν = 1.5: 一次可微
      ν = 2.5: 二次可微

    K_ν 是修正 Bessel 函数 (第二类).
    对半整数 ν 有闭式:
      K_{1/2}(x) = √(π/(2x)) e^{-x}
      K_{3/2}(x) = √(π/(2x)) e^{-x} (1 + 1/x)
      K_{5/2}(x) = √(π/(2x)) e^{-x} (1 + 3/x + 3/x²)
    """
    rho = np.asarray(rho, dtype=float)
    r = np.abs(rho) / rho0

    # 处理原点
    result = np.ones_like(r)
    mask = r > 1e-10

    if abs(nu - 0.5) < 1e-10:
        # ν = 0.5: 指数相关
        result[mask] = np.exp(-np.sqrt(1.0) * r[mask])
    elif abs(nu - 1.5) < 1e-10:
        # ν = 1.5
        sr3 = np.sqrt(3.0) * r[mask]
        result[mask] = (1.0 + sr3) * np.exp(-sr3)
    elif abs(nu - 2.5) < 1e-10:
        # ν = 2.5
        sr5 = np.sqrt(5.0) * r[mask]
        result[mask] = (1.0 + sr5 + sr5 ** 2 / 3.0) * np.exp(-sr5)
    else:
        # 通用情形: 级数展开
        for i in np.where(mask)[0]:
            ri = r[i]
            x = np.sqrt(2.0 * nu) * ri
            # 小参数展开 (前 20 项)
            s = 0.0
            for k in range(30):
                # K_ν(x) 的级数展开 (简化)
                term = ((-1) ** k / math.factorial(k)
                        * (x / 2.0) ** (2 * k)
                        / max(abs(math.gamma(nu + k + 1)), 1e-300))
                s += term
            # 近似: 使用指数衰减
            result[i] = (x ** nu) * np.exp(-x) * max(abs(s), 1e-300)

    return np.clip(result, -1.0, 1.0)


def correlation_power(rho: np.ndarray, rho0: float, alpha: float = 2.0) -> np.ndarray:
    """幂律相关函数 (源自 220_correlation/correlation_power).
    C(ρ) = (1 + ρ²/ρ₀²)^{-α}
    重尾相关, 在 α>d/2 时正定 (d 为空间维数).
    """
    rho = np.asarray(rho, dtype=float)
    return (1.0 + rho ** 2 / rho0 ** 2) ** (-alpha)


def correlation_bessel(rho: np.ndarray, rho0: float) -> np.ndarray:
    """Bessel 相关函数 (源自 220_correlation/correlation_besselj).
    C(ρ) = J₀(|ρ|/ρ₀) (零阶第一类 Bessel 函数)

    J₀(x) = Σ_{k=0}^∞ (-1)^k (x/2)^{2k} / (k!)²

    注意: 此相关函数非正定, 但可用于某些谱方法.
    """
    rho = np.asarray(rho, dtype=float).ravel()
    x = np.abs(rho) / rho0

    # 级数展开
    result = np.zeros_like(rho)
    for idx in range(len(rho)):
        xi = x[idx]
        s = 0.0
        term = 1.0
        for k in range(30):
            s += term
            term *= -(xi / 2.0) ** 2 / ((k + 1) ** 2)
            if abs(term) < 1e-15 * abs(s) and k > 5:
                break
        result[idx] = s
    return result.reshape(np.asarray(rho).shape) if np.asarray(rho).shape else result[0]


# ---------------------------------------------------------------------------
# 2. Toeplitz 矩阵构造
# ---------------------------------------------------------------------------

def build_toeplitz_correlation(n: int, rho_max: float, rho0: float,
                               correlation_func: Callable) -> np.ndarray:
    """构造对称 Toeplitz 相关矩阵.
    R[i,j] = C(|i-j| · Δρ), Δρ = ρ_max/(n-1).

    Toeplitz 结构: 仅依赖 |i-j|, 因此 O(n) 存储.
    正定性: 由 Bochner 定理保证 (对正定相关函数).
    """
    drho = rho_max / max(n - 1, 1)
    rho_vec = np.arange(n) * drho
    cor_vec = correlation_func(rho_vec, rho0)

    R = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            R[i, j] = cor_vec[abs(i - j)]
    return R


# ---------------------------------------------------------------------------
# 3. Cholesky 采样法 (源自 220_correlation/sample_paths2_cholesky)
# ---------------------------------------------------------------------------

def sample_paths_cholesky(n: int, n_paths: int, rho_max: float, rho0: float,
                          correlation_func: Callable,
                          rng: Optional[np.random.Generator] = None) -> Tuple[np.ndarray, np.ndarray]:
    """使用 Cholesky 分解生成高斯随机场的样本路径.
    源自 220_correlation/sample_paths2_cholesky.

    算法:
      1. 构造相关矩阵 R
      2. Cholesky 分解: R = L L^T
      3. 采样: x = L z, z ~ N(0, I)

    复杂度: O(n³) (Cholesky), O(n²·n_paths) (采样).

    返回 (rho_vec, X), X 形状 (n, n_paths).
    """
    if rng is None:
        rng = np.random.default_rng(42)

    R = build_toeplitz_correlation(n, rho_max, rho0, correlation_func)

    # 数值正则化: 确保正定
    R += 1e-10 * np.eye(n)

    try:
        L = np.linalg.cholesky(R)
    except np.linalg.LinAlgError:
        # 若仍不正定, 使用特征值修正
        eigvals, eigvecs = np.linalg.eigh(R)
        eigvals = np.maximum(eigvals, 1e-8)
        R_reg = eigvecs @ np.diag(eigvals) @ eigvecs.T
        L = np.linalg.cholesky(R_reg)

    z = rng.standard_normal((n, n_paths))
    X = L @ z

    drho = rho_max / max(n - 1, 1)
    rho_vec = np.arange(n) * drho
    return rho_vec, X


# ---------------------------------------------------------------------------
# 4. FFT 采样法 (源自 220_correlation/sample_paths_fft)
# ---------------------------------------------------------------------------

def sample_paths_fft(n: int, n_paths: int, rho_max: float, rho0: float,
                     correlation_func: Callable,
                     rng: Optional[np.random.Generator] = None) -> Tuple[np.ndarray, np.ndarray]:
    """使用 FFT circulant 嵌入生成平稳相关函数的样本路径.
    源自 220_correlation/sample_paths_fft.

    算法 (Dietrich & Newsam, 1997):
      1. 构造 2n 长向量 c (循环协方差序列)
      2. FFT: λ = FFT(c) (谱密度)
      3. 采样: z = IFFT(√λ · w), w ~ CN(0,I)
      4. 取前 n 个分量

    复杂度: O(n log n).
    """
    if rng is None:
        rng = np.random.default_rng(42)

    drho = rho_max / max(n - 1, 1)
    rho_vec_half = np.arange(n) * drho
    cor_half = correlation_func(rho_vec_half, rho0)

    # Circulant 嵌入: 构造 2n 长序列
    # 标准 Toeplitz→circulant 嵌入:
    # c = [c(0), c(1), ..., c(n-1), 0, c(n-1), c(n-2), ..., c(1)]
    # 长度 2n, 确保 circulant 矩阵的第一列给出原 Toeplitz 矩阵
    c = np.zeros(2 * n)
    c[:n] = cor_half                    # [c(0), c(1), ..., c(n-1)]
    c[n] = 0.0                          # 中间填 0
    if n > 1:
        # c[n+1:2n] = [c(n-1), c(n-2), ..., c(1)] 共 n-1 个元素
        c[n + 1:2 * n] = cor_half[n - 1:0:-1]

    # 谱分解
    lam = np.real(np.fft.fft(c))
    lam = np.maximum(lam, 0.0)  # 截断负特征值

    sqrt_lam = np.sqrt(lam)

    # 生成复高斯随机数
    w = rng.standard_normal((2 * n, n_paths)) + 1j * rng.standard_normal((2 * n, n_paths))
    z_freq = sqrt_lam[:, np.newaxis] * w
    z = np.real(np.fft.ifft(z_freq, axis=0))

    X = z[:n, :] * np.sqrt(2 * n)  # 归一化

    return rho_vec_half, X


# ---------------------------------------------------------------------------
# 5. 随机场插值 (用于优化目标函数)
# ---------------------------------------------------------------------------

class CorrelatedRandomField:
    """基于相关函数构造的随机场, 可用于生成随机优化测试问题.

    场表示为: f(x) = Σ_{i=1}^{N} w_i · C(|x - x_i|/ρ₀)
    其中 w_i 是随机权重, x_i 是随机中心.

    这种表示确保场的光滑度由相关函数控制.
    """

    def __init__(self, dim: int, n_centers: int, rho0: float,
                 correlation_func: Callable = None,
                 rng: Optional[np.random.Generator] = None):
        if rng is None:
            rng = np.random.default_rng(42)
        self.dim = dim
        self.n_centers = n_centers
        self.rho0 = rho0
        self.correlation_func = correlation_func or correlation_gaussian
        self.centers = rng.uniform(-2, 2, (n_centers, dim))
        self.weights = rng.standard_normal(n_centers)
        self.rng = rng

    def evaluate(self, x: np.ndarray) -> float:
        """求值随机场 f(x).
        x: (dim,) 或 (n_points, dim)
        """
        x = np.atleast_2d(x)
        result = np.zeros(x.shape[0])
        for i in range(self.n_centers):
            dist = np.sqrt(np.sum((x - self.centers[i]) ** 2, axis=1))
            c_vals = self.correlation_func(dist, self.rho0)
            result += self.weights[i] * c_vals
        return float(result[0]) if result.size == 1 else result

    def gradient(self, x: np.ndarray) -> np.ndarray:
        """随机场的梯度 (解析).
        ∇f(x) = Σ w_i · C'(|x-x_i|/ρ₀) · (x-x_i)/(|x-x_i|·ρ₀)
        """
        x = np.atleast_1d(x)
        grad = np.zeros(self.dim)
        eps = 1e-8
        for d in range(self.dim):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[d] += eps
            x_minus[d] -= eps
            grad[d] = (self.evaluate(x_plus) - self.evaluate(x_minus)) / (2 * eps)
        return grad
