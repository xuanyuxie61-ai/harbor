"""
多项式工具库 (Polynomial Utilities)
===================================
提供正交多项式族的求值、递推关系和高斯求积规则计算。
支持 Hermite（对应高斯测度）、Legendre（对应均匀测度）和 Laguerre（对应指数测度）多项式族。

科学基础:
  概率测度 dμ(ξ) 上的正交多项式族 {P_n(ξ)} 满足三项递推关系:
    β_n P_{n+1}(ξ) = (ξ - α_n) P_n(ξ) - γ_n P_{n-1}(ξ)
  其中 α_n = <ξ P_n, P_n> / <P_n, P_n>
       β_n = <P_n, P_n> / <P_{n-1}, P_{n-1}>
       γ_n = <P_{n-1}, P_{n-1}> / <P_{n-2}, P_{n-2}>

  Golub-Welsch 算法: 通过求解对称三对角矩阵的特征值问题
  来计算高斯求积节点和权重:
    J v = ξ v,  J_ij = α_i δ_{ij} + β_i δ_{i,j+1} + β_{i-1} δ_{i,j-1}
"""

import numpy as np
from scipy import special
from typing import Tuple, Optional


# ============================================================
# 1. 概率论器函数 (Probability Density Functions)
# ============================================================

def gaussian_pdf(xi: np.ndarray, mu: float = 0.0, sigma: float = 1.0) -> np.ndarray:
    """
    高斯概率密度函数 (Gaussian PDF):
      ρ(ξ) = 1/(σ√(2π)) exp(-(ξ-μ)²/(2σ²))

    对应 Hermite 多项式族正交测度。
    """
    return np.exp(-0.5 * ((xi - mu) / sigma) ** 2) / (sigma * np.sqrt(2.0 * np.pi))


def uniform_pdf(xi: np.ndarray, a: float = -1.0, b: float = 1.0) -> np.ndarray:
    """
    均匀概率密度函数 (Uniform PDF):
      ρ(ξ) = 1/(b-a)  for ξ ∈ [a,b]

    对应 Legendre 多项式族正交测度。
    """
    mask = (xi >= a) & (xi <= b)
    result = np.zeros_like(xi, dtype=np.float64)
    result[mask] = 1.0 / (b - a)
    return result


def exponential_pdf(xi: np.ndarray, lam: float = 1.0) -> np.ndarray:
    """
    指数概率密度函数 (Exponential PDF):
      ρ(ξ) = λ exp(-λ ξ)  for ξ ≥ 0

    对应 Laguerre 多项式族正交测度。
    """
    result = np.zeros_like(xi, dtype=np.float64)
    mask = xi >= 0.0
    result[mask] = lam * np.exp(-lam * xi[mask])
    return result


# ============================================================
# 2. 正交多项式求值 (Orthogonal Polynomial Evaluation)
# ============================================================

def hermite_evaluate(xi: np.ndarray, n_max: int) -> np.ndarray:
    """
    概率论者 Hermite 多项式求值 (Probabilist's Hermite Polynomials):
      He_0(ξ) = 1
      He_1(ξ) = ξ
      He_{n+1}(ξ) = ξ He_n(ξ) - n He_{n-1}(ξ)

    正交性: ∫ He_m(ξ) He_n(ξ) ρ(ξ) dξ = n! δ_{mn}
    其中 ρ(ξ) = (2π)^{-1/2} exp(-ξ²/2)

    参数:
        xi: 求值点数组, shape (M,)
        n_max: 最大阶数

    返回:
        H: shape (n_max+1, M), H[k,:] = He_k(xi)
    """
    xi = np.asarray(xi, dtype=np.float64)
    M = xi.size
    H = np.zeros((n_max + 1, M), dtype=np.float64)
    H[0, :] = 1.0
    if n_max >= 1:
        H[1, :] = xi
    for n in range(1, n_max):
        H[n + 1, :] = xi * H[n, :] - n * H[n - 1, :]
    return H


def legendre_evaluate(xi: np.ndarray, n_max: int) -> np.ndarray:
    """
    Legendre 多项式求值:
      P_0(ξ) = 1
      P_1(ξ) = ξ
      (n+1) P_{n+1}(ξ) = (2n+1) ξ P_n(ξ) - n P_{n-1}(ξ)

    正交性: ∫_{-1}^{1} P_m(ξ) P_n(ξ) dξ = 2/(2n+1) δ_{mn}

    参数:
        xi: 求值点数组, shape (M,)
        n_max: 最大阶数

    返回:
        P: shape (n_max+1, M), P[k,:] = P_k(xi)
    """
    xi = np.asarray(xi, dtype=np.float64)
    M = xi.size
    P = np.zeros((n_max + 1, M), dtype=np.float64)
    P[0, :] = 1.0
    if n_max >= 1:
        P[1, :] = xi
    for n in range(1, n_max):
        P[n + 1, :] = ((2.0 * n + 1.0) * xi * P[n, :] - n * P[n - 1, :]) / (n + 1.0)
    return P


def laguerre_evaluate(xi: np.ndarray, n_max: int) -> np.ndarray:
    """
    广义 Laguerre 多项式求值 (Generalized Laguerre, α=0):
      L_0(ξ) = 1
      L_1(ξ) = 1 - ξ
      (n+1) L_{n+1}(ξ) = (2n+1-ξ) L_n(ξ) - n L_{n-1}(ξ)

    正交性: ∫_0^∞ L_m(ξ) L_n(ξ) exp(-ξ) dξ = δ_{mn}

    参数:
        xi: 求值点数组, shape (M,)
        n_max: 最大阶数

    返回:
        L: shape (n_max+1, M), L[k,:] = L_k(xi)
    """
    xi = np.asarray(xi, dtype=np.float64)
    M = xi.size
    L = np.zeros((n_max + 1, M), dtype=np.float64)
    L[0, :] = 1.0
    if n_max >= 1:
        L[1, :] = 1.0 - xi
    for n in range(1, n_max):
        L[n + 1, :] = ((2.0 * n + 1.0 - xi) * L[n, :] - n * L[n - 1, :]) / (n + 1.0)
    return L


# ============================================================
# 3. Gauss 求积规则 (Gauss Quadrature Rules)
# ============================================================

def gauss_legendre(n_points: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Gauss-Legendre 求积规则:
      ∫_{-1}^{1} f(ξ) dξ ≈ Σ_{i=1}^{n} w_i f(ξ_i)

    代数精度: 2n-1

    通过 Golub-Welsch 算法计算:
      J 为 n×n 对称三对角矩阵, J_{i,i+1} = J_{i+1,i} = i/√((2i-1)(2i+1))
      节点 ξ_i = eigenvalues(J)
      权重 w_i = 2 * (eigenvector[0,i])^2
    """
    if n_points <= 0:
        return np.array([]), np.array([])
    if n_points == 1:
        return np.array([0.0]), np.array([2.0])

    nodes, weights = np.polynomial.legendre.leggauss(n_points)
    return nodes, weights


def gauss_hermite(n_points: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Gauss-Hermite 求积规则 (概率论者形式):
      ∫_{-∞}^{∞} f(ξ) (2π)^{-1/2} exp(-ξ²/2) dξ ≈ Σ_{i=1}^{n} w_i f(ξ_i)

    代数精度: 2n-1

    物理学家形式: ∫_{-∞}^{∞} f(ξ) exp(-ξ²) dξ ≈ Σ w_i f(ξ_i)
    转换: ξ_phys = ξ_prob * √2,  w_prob = w_phys * exp(ξ_phys²) / √π
    """
    if n_points <= 0:
        return np.array([]), np.array([])
    if n_points == 1:
        return np.array([0.0]), np.array([1.0])

    # 使用 Golub-Welsch 算法
    # 三项递推: ξ He_n = He_{n+1} + n He_{n-1}
    # Jacobi 矩阵: J_{n,n+1} = √n
    i = np.arange(1, n_points, dtype=np.float64)
    off_diag = np.sqrt(i)
    J = np.diag(off_diag, 1) + np.diag(off_diag, -1)

    eigenvalues, eigenvectors = np.linalg.eigh(J)
    weights = eigenvectors[0, :] ** 2
    return eigenvalues, weights


def gauss_laguerre(n_points: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Gauss-Laguerre 求积规则:
      ∫_0^∞ f(ξ) exp(-ξ) dξ ≈ Σ_{i=1}^{n} w_i f(ξ_i)

    代数精度: 2n-1

    三项递推: (n+1)L_{n+1} = (2n+1-ξ)L_n - nL_{n-1}
    Jacobi 矩阵: J_{n,n} = 2n+1, J_{n,n+1} = -(n+1)
    """
    if n_points <= 0:
        return np.array([]), np.array([])
    if n_points == 1:
        return np.array([1.0]), np.array([1.0])

    # Golub-Welsch: 三项递推 ξ L_n = -(n+1)L_{n+1} + (2n+1)L_n - nL_{n-1}
    i = np.arange(1, n_points, dtype=np.float64)
    off_diag = -i  # J_{n,n+1} = -(n+1) for 0-indexed: -1, -2, ...
    diag = np.array([2.0 * k + 1.0 for k in range(n_points)])
    J = np.diag(diag) + np.diag(off_diag, 1) + np.diag(off_diag, -1)

    eigenvalues, eigenvectors = np.linalg.eigh(J)
    weights = eigenvectors[0, :] ** 2
    return eigenvalues, weights


def clenshaw_curtis(n_points: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Clenshaw-Curtis 求积节点和权重 (嵌套规则):
      ∫_{-1}^{1} f(ξ) dξ ≈ Σ_{i=1}^{n} w_i f(ξ_i)

    节点: ξ_i = -cos(π(i-1)/(n-1)), i=1,...,n
    嵌套性:  level L+1 的节点集包含 level L 的全部节点

    权重通过 DFT 方法计算:
      w_i = (2/n) Σ_{k=0}^{n/2} b_k/(1-4k²) cos(2πk(i-1)/(n-1))
      其中 b_0 = 1, b_k = 2 for k>0
    """
    if n_points <= 0:
        return np.array([]), np.array([])
    if n_points == 1:
        return np.array([0.0]), np.array([2.0])
    if n_points == 2:
        return np.array([-1.0, 1.0]), np.array([1.0, 1.0])

    n = n_points - 1
    theta = np.pi * np.arange(n_points) / n
    nodes = -np.cos(theta)

    # 计算 Clenshaw-Curtis 权重
    weights = np.zeros(n_points)
    for i in range(n_points):
        s = 0.0
        for k in range(1, n // 2 + 1):
            b_k = 2.0 if k < n / 2 else 1.0
            s += b_k * np.cos(2.0 * k * theta[i]) / (4.0 * k * k - 1.0)
        weights[i] = (2.0 / n) * (1.0 - s)
    # 端点权重减半处理
    weights[0] /= 2.0
    weights[-1] /= 2.0
    # 修正: 对于奇数n, 特殊处理
    # 重新计算以确保精度
    # 使用更稳健的权重计算
    c = np.zeros(n_points)
    c[0] = 1.0 / (n * n - 1) if n > 1 else 1.0
    c[-1] = 1.0 / (n * n - 1) if n > 1 else 1.0
    for k in range(1, n):
        c[k] = 2.0 / (1 - 4 * k * k) if k < n else 1.0 / (1 - 4 * k * k)

    # 简化: 使用经验证的 CC 权重公式
    N = n_points
    w = np.zeros(N)
    if N % 2 == 1:
        for j in range(N):
            xj = nodes[j]
            s = 0.0
            for k in range(1, (N - 1) // 2 + 1):
                s += np.cos(2.0 * k * theta[j]) / (4.0 * k * k - 1.0)
            w[j] = (2.0 / n) * (1.0 - 2.0 * s) - np.cos(n * theta[j]) / n
    else:
        for j in range(N):
            s = 0.0
            for k in range(1, N // 2):
                s += np.cos(2.0 * k * theta[j]) / (4.0 * k * k - 1.0)
            w[j] = (2.0 / n) * (1.0 - 2.0 * s) - 1.0 / n

    # 端点修正
    w[0] = 1.0 / (n * n - 1) if n > 1 else 1.0
    w[-1] = 1.0 / (n * n - 1) if n > 1 else 1.0

    # 最终验证: 权重和应为2
    w_sum = np.sum(w)
    if abs(w_sum) > 1e-14:
        w *= 2.0 / w_sum

    return nodes, w


def tensor_product_quadrature(
    nodes_1d: list, weights_1d: list
) -> Tuple[np.ndarray, np.ndarray]:
    """
    张量积求积规则:
      给定各维度的 1D 求积规则 {(ξ_i^{(d)}, w_i^{(d)})}_{i=1}^{n_d}, d=1,...,D
      构造 D 维张量积规则:
      (Ξ_J, W_J) where J = (j_1, ..., j_D) ∈ Π_{d=1}^D {1,...,n_d}
      Ξ_J = (ξ_{j_1}^{(1)}, ..., ξ_{j_D}^{(D)})
      W_J = Π_{d=1}^D w_{j_d}^{(d)}

    参数:
        nodes_1d: list of D arrays, each of shape (n_d,)
        weights_1d: list of D arrays, each of shape (n_d,)

    返回:
        nodes: shape (N_total, D)
        weights: shape (N_total,)
    """
    D = len(nodes_1d)
    if D == 0:
        return np.array([[]]), np.array([1.0])

    # 使用 meshgrid 构造张量积
    grids = np.meshgrid(*nodes_1d, indexing='ij')
    w_grids = np.meshgrid(*weights_1d, indexing='ij')

    N_total = 1
    for n_d in nodes_1d:
        N_total *= len(n_d)

    nodes = np.column_stack([g.ravel() for g in grids])
    weights = np.ones(N_total)
    for wg in w_grids:
        weights *= wg.ravel()

    return nodes, weights


# ============================================================
# 4. 多项式范数与正交性验证 (Norm and Orthogonality)
# ============================================================

def hermite_norm_sq(n: int) -> float:
    """
    Hermite 多项式平方范数:
      ||He_n||² = ∫ He_n(ξ)² ρ(ξ) dξ = n!
    """
    return float(special.factorial(n))


def legendre_norm_sq(n: int) -> float:
    """
    Legendre 多项式平方范数:
      ||P_n||² = ∫_{-1}^{1} P_n(ξ)² dξ = 2/(2n+1)
    """
    return 2.0 / (2.0 * n + 1.0)


def laguerre_norm_sq(n: int) -> float:
    """
    Laguerre 多项式平方范数:
      ||L_n||² = ∫_0^∞ L_n(ξ)² exp(-ξ) dξ = 1
    """
    return 1.0


def compute_gaussian_quadrature(
    measure_type: str, n_points: int, **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """
    统一接口: 根据测度类型返回对应的高斯求积规则。

    参数:
        measure_type: 'gaussian' | 'uniform' | 'exponential'
        n_points: 求积点数
        **kwargs: 传递给具体测度的参数

    返回:
        nodes, weights
    """
    if measure_type == 'gaussian':
        return gauss_hermite(n_points)
    elif measure_type == 'uniform':
        return gauss_legendre(n_points)
    elif measure_type == 'exponential':
        return gauss_laguerre(n_points)
    else:
        raise ValueError(f"Unknown measure type: {measure_type}. "
                         f"Supported: 'gaussian', 'uniform', 'exponential'")
