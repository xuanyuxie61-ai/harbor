"""
quadrature_validation.py
========================
基于 hypercube_exactness (557_hypercube_exactness) 与 hermite_exactness (519_hermite_exactness) 的
求积规则精确度验证框架，用于 ENSO 模式中高维参数积分的数值误差控制。

科学背景
--------
ENSO 预测的不确定性量化需要在高维参数空间上进行积分：
- 海气耦合系数的不确定性（β 分布或均匀分布）；
- 初始条件的不确定性（高斯分布）；
- 随机强迫的谱分布（白噪声、红噪声）。

高维积分（维度 d > 5）的精确计算需要验证求积规则的精确度，
确保单项式积分直至指定阶数均无误差。本模块提供两类验证：
1. 超立方体 [0,1]^d 上的张量积 Gauss-Legendre 规则；
2. 全空间 (-∞, +∞)^d 上的 Gauss-Hermite 规则（用于随机初值积分）。

核心公式
--------
1. 超立方体上的单项式精确积分：
   
   I(α) = ∫_{[0,1]^d} Π_{j=1}^{d} x_j^{α_j} dx
        = Π_{j=1}^{d} 1 / (α_j + 1)

2. Gauss-Hermite（物理学家权重）单项式积分：
   
   I(α) = ∫_{-∞}^{+∞} x^{α} exp(-x²) dx

   当 α 为偶数时：I(α) = (α-1)!! * √π / 2^{α/2}
   当 α 为奇数时：I(α) = 0

3. 求积规则误差：
   
   E(α) = |Q(α) - I(α)| / |I(α)|

   其中 Q(α) = Σ_{k=1}^{N} w_k * Π_j x_{k,j}^{α_j}

4. 多维组合规则（Smolyak 稀疏网格）：
   对于 d 维积分，全张量积需要 N^d 个点，稀疏网格将点数降至
   O(N * (log N)^{d-1})，同时保持多项式精确度。
"""

import numpy as np
from typing import Tuple, List
import itertools


def hypercube_monomial_integral(exponents: Tuple[int, ...]) -> float:
    """
    计算 [0,1]^d 上单项式的精确积分。

    公式：I(α) = Π_{j=1}^{d} 1/(α_j + 1)
    """
    result = 1.0
    for alpha in exponents:
        if alpha < 0:
            raise ValueError("Exponents must be non-negative")
        result /= (alpha + 1.0)
    return result


def hermite_monomial_integral_1d(alpha: int, weight_type: str = "physicist") -> float:
    """
    计算一维 Gauss-Hermite 权函数下的单项式积分。

    物理学家权重：w(x) = exp(-x²)
    概率学家权重：w(x) = exp(-x²/2)

    公式：
    ∫_{-∞}^{+∞} x^α exp(-x²) dx =
        0                          , α 为奇数
        (α-1)!! * √π / 2^{α/2}     , α 为偶数

    对于概率学家权重，结果乘以 2^{(α+1)/2}。
    """
    if alpha < 0:
        raise ValueError("Exponent must be non-negative")

    if alpha % 2 == 1:
        return 0.0

    import math
    double_fact = 1.0
    for k in range(alpha - 1, 0, -2):
        double_fact *= k

    result = double_fact * np.sqrt(np.pi) / (2.0 ** (alpha / 2.0))

    if weight_type == "probabilist":
        result *= 2.0 ** ((alpha + 1.0) / 2.0)

    return result


def gauss_legendre_points_weights_1d(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算 [0, 1] 上 n 点 Gauss-Legendre 求积节点和权重。

    通过 numpy.polynomial.legendre.leggauss 获取 [-1,1] 规则，
    再线性变换到 [0,1]。
    """
    if n < 1:
        raise ValueError("n must be positive")
    x, w = np.polynomial.legendre.leggauss(n)
    # 变换到 [0, 1]
    x = 0.5 * (x + 1.0)
    w = 0.5 * w
    return x, w


def gauss_hermite_points_weights_1d(n: int,
                                     weight_type: str = "physicist") -> Tuple[np.ndarray, np.ndarray]:
    """
    计算 n 点 Gauss-Hermite 求积节点和权重。

    物理学家权重：∫ f(x) exp(-x²) dx ≈ Σ w_k f(x_k)
    概率学家权重：∫ f(x) exp(-x²/2) dx ≈ Σ w_k f(x_k)
    """
    if n < 1:
        raise ValueError("n must be positive")
    x, w = np.polynomial.hermite.hermgauss(n)

    if weight_type == "probabilist":
        x *= np.sqrt(2.0)
        w *= np.sqrt(2.0)

    return x, w


def tensor_product_quadrature_1d_to_nd(points_1d: List[np.ndarray],
                                        weights_1d: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    从一维求积规则构造 d 维张量积规则。

    参数
    ----
    points_1d : List[np.ndarray]
        各维度的节点数组。
    weights_1d : List[np.ndarray]
        各维度的权重数组。

    返回
    ----
    points_nd : np.ndarray, shape (N, d)
        d 维节点。
    weights_nd : np.ndarray, shape (N,)
        d 维权重。
    """
    d = len(points_1d)
    grids = np.meshgrid(*points_1d, indexing='ij')
    points_nd = np.stack([g.ravel() for g in grids], axis=-1)

    w_grids = np.meshgrid(*weights_1d, indexing='ij')
    weights_nd = np.prod(np.stack([g.ravel() for g in w_grids], axis=-1), axis=-1)

    return points_nd, weights_nd


def validate_hypercube_quadrature(dim: int, n_points: int,
                                  degree_max: int = 5) -> dict:
    """
    验证 [0,1]^d 上 n 点 Gauss-Legendre 张量积规则的精确度。

    参数
    ----
    dim : int
        空间维度。
    n_points : int
        每维节点数。
    degree_max : int
        测试的最大单项式总阶数。

    返回
    ----
    result : dict
        各阶单项式的精确积分、数值积分、相对误差。
    """
    x_1d, w_1d = gauss_legendre_points_weights_1d(n_points)
    points_1d = [x_1d.copy() for _ in range(dim)]
    weights_1d = [w_1d.copy() for _ in range(dim)]
    pts, wts = tensor_product_quadrature_1d_to_nd(points_1d, weights_1d)

    errors = []
    max_degree_passed = -1

    for total_degree in range(degree_max + 1):
        # 生成所有非负整数解 α_1 + ... + α_d = total_degree
        for exponents in itertools.combinations_with_replacement(range(total_degree + 1), dim):
            if sum(exponents) != total_degree:
                continue
            # 生成所有排列
            for alpha in set(itertools.permutations(exponents)):
                exact = hypercube_monomial_integral(alpha)
                numerical = np.sum(wts * np.prod(pts ** np.array(alpha), axis=1))
                if abs(exact) > 1e-14:
                    rel_err = abs(numerical - exact) / abs(exact)
                else:
                    rel_err = abs(numerical)
                errors.append({
                    "exponents": alpha,
                    "degree": total_degree,
                    "exact": exact,
                    "numerical": numerical,
                    "relative_error": rel_err,
                })

        # 检查该阶是否全部通过
        degree_errors = [e for e in errors if e["degree"] == total_degree]
        if all(e["relative_error"] < 1e-12 for e in degree_errors):
            max_degree_passed = total_degree
        else:
            break

    return {
        "dim": dim,
        "n_points_per_dim": n_points,
        "total_points": pts.shape[0],
        "max_degree_passed": max_degree_passed,
        "errors": errors,
    }


def validate_hermite_quadrature_1d(n_points: int,
                                   degree_max: int = 10,
                                   weight_type: str = "physicist") -> dict:
    """
    验证一维 Gauss-Hermite 求积规则的精确度。

    参数
    ----
    n_points : int
        节点数。
    degree_max : int
        测试的最大单项式阶数。
    weight_type : str
        "physicist" 或 "probabilist"。

    返回
    ----
    result : dict
        精确度验证结果。
    """
    x, w = gauss_hermite_points_weights_1d(n_points, weight_type)
    max_degree_passed = -1
    errors = []

    for alpha in range(degree_max + 1):
        exact = hermite_monomial_integral_1d(alpha, weight_type)
        numerical = np.sum(w * (x ** alpha))
        if abs(exact) > 1e-14:
            rel_err = abs(numerical - exact) / abs(exact)
        else:
            rel_err = abs(numerical)
        errors.append({
            "degree": alpha,
            "exact": exact,
            "numerical": numerical,
            "relative_error": rel_err,
        })
        if rel_err < 1e-12:
            max_degree_passed = alpha
        else:
            break

    return {
        "n_points": n_points,
        "weight_type": weight_type,
        "max_degree_passed": max_degree_passed,
        "errors": errors,
    }


def smolyak_sparse_grid_1d_to_nd(level: int, dim: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    构造一维 Clenshaw-Curtis 节点的 Smolyak 稀疏网格（简化实现）。

    对于 level L，一维节点数为 n = 2^{L-1} + 1（L≥1），n=1（L=0）。
    本实现使用 Gauss-Legendre 节点替代，点数略多但更简单。
    """
    if level < 0:
        raise ValueError("level must be non-negative")
    if dim < 1:
        raise ValueError("dim must be positive")

    # 简化：使用 level 对应的全张量积（对于低维低阶，稀疏网格=全张量积）
    n = max(1, level + 1)
    x, w = gauss_legendre_points_weights_1d(n)
    pts_1d = [x for _ in range(dim)]
    wts_1d = [w for _ in range(dim)]
    return tensor_product_quadrature_1d_to_nd(pts_1d, wts_1d)


def ensemble_mean_integral(ensemble_values: np.ndarray,
                           quadrature_points: np.ndarray,
                           quadrature_weights: np.ndarray) -> Tuple[float, float]:
    """
    使用求积规则计算集合预测的均值和方差。

    公式：
    μ = Σ_k w_k * <ensemble>_k
    σ² = Σ_k w_k * (<ensemble>_k - μ)²
    """
    mean_vals = np.mean(ensemble_values, axis=0)  # 每个求积点上的集合均值
    mu = np.sum(quadrature_weights * mean_vals)
    sigma_sq = np.sum(quadrature_weights * (mean_vals - mu) ** 2)
    return float(mu), float(sigma_sq)
