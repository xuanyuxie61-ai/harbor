"""
速度空间积分与采样模块

融合以下种子项目的积分与采样算法:
1. hyperball_integrals: 高维球面积分
2. cube_felippa_rule: 立方体高斯求积
3. cvt_corn: Centroidal Voronoi Tessellation (CVT) 采样
4. circle_distance: 球面距离统计

应用于 PIC 中的:
- 麦克斯韦分布初始采样
- 速度空间矩计算
- 数值扩散估计
"""

import numpy as np
from typing import Tuple, Callable, List
try:
    from plasma_constants import PI
except ImportError:
    from plasma_constants import PI


# ============================================================================
# 高维积分 (from hyperball_integrials)
# ============================================================================
def hyperball_volume(d: int) -> float:
    """
    d 维单位球体积:
        V_d = pi^(d/2) / Gamma(d/2 + 1)
    """
    from math import gamma
    return PI**(d / 2.0) / gamma(d / 2.0 + 1.0)


def hyperball_surface_area(d: int) -> float:
    """
    d 维单位球面面积:
        S_d = d * V_d = 2 * pi^(d/2) / Gamma(d/2)
    """
    from math import gamma
    return 2.0 * PI**(d / 2.0) / gamma(d / 2.0)


def hyperball_monomial_integral(exponents: np.ndarray) -> float:
    """
    d 维单位球上的单项式积分:
        integral_{B^d} x_1^{a_1} * x_2^{a_2} * ... * x_d^{a_d} dx

    若任一指数为奇数, 积分 = 0 (对称性)
    否则:
        = 2 * prod_i Gamma((a_i+1)/2) / Gamma(sum_i(a_i+1)/2 + 1) * pi^(d/2) / Gamma(d/2)
        简化为:
        = prod_i Gamma((a_i+1)/2) * pi^(d/2) / Gamma(sum(a_i)/2 + d/2 + 1) / Gamma(d/2)

    实际公式:
        = prod_i ((1+(-1)^a_i)/2 * Gamma((a_i+1)/2)) * Gamma(d/2) / (2 * Gamma(sum(a_i)/2 + d/2 + 1))
    """
    d = len(exponents)
    if any(int(a) % 2 == 1 for a in exponents):
        return 0.0

    from math import gamma
    numerator = 1.0
    sum_a = 0.0
    for a in exponents:
        numerator *= gamma((a + 1.0) / 2.0)
        sum_a += a
    denominator = gamma(sum_a / 2.0 + d / 2.0 + 1.0)
    return 2.0 * numerator * PI**(d / 2.0) / (denominator * gamma(d / 2.0))


def hyperball_sample_uniform(d: int, n_samples: int, seed: int = 42) -> np.ndarray:
    """
    d 维单位球内均匀采样 (Muller 方法 + 径向缩放)

    1. 球面均匀采样: x = g / ||g||,  g ~ N(0, I_d)
    2. 径向采样: r = U^(1/d),  U ~ Uniform(0, 1)
    3. 点 = r * x
    """
    rng = np.random.RandomState(seed)
    g = rng.randn(n_samples, d)
    norms = np.linalg.norm(g, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-300)
    x_sphere = g / norms
    r = rng.rand(n_samples, 1)**(1.0 / d)
    return r * x_sphere


def hyperball_sample_maxwellian(d: int, n_samples: int,
                                  v_th: float, seed: int = 42) -> np.ndarray:
    """
    d 维麦克斯韦速度采样:
        v_i ~ N(0, v_th^2/2)

    每维独立高斯
    """
    rng = np.random.RandomState(seed)
    sigma = v_th / np.sqrt(2.0)
    return rng.randn(n_samples, d) * sigma


# ============================================================================
# 立方体求积 (from cube_felippa_rule)
# ============================================================================
def cube_gauss_legendre_1d(n_points: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    1D [-1, 1] Gauss-Legendre 求积节点和权重
    """
    nodes, weights = np.polynomial.legendre.leggauss(n_points)
    return nodes, weights


def cube_tensor_product_rule(d: int, n_per_dim: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    d 维立方体 [-1,1]^d 上的张量积 Gauss-Legendre 求积
    """
    nodes_1d, weights_1d = cube_gauss_legendre_1d(n_per_dim)

    # 张量积
    total_points = n_per_dim**d
    nodes = np.zeros((total_points, d))
    weights = np.ones(total_points)

    for i in range(total_points):
        idx = i
        for dim in range(d):
            nodes[i, dim] = nodes_1d[idx % n_per_dim]
            weights[i] *= weights_1d[idx % n_per_dim]
            idx //= n_per_dim

    return nodes, weights


def cube_monomial_integral_d(exponents: np.ndarray) -> float:
    """
    d 维立方体 [-1,1]^d 上单项式积分:
        integral x_1^a_1 * ... * x_d^a_d dx
        = prod_i integral_{-1}^{1} x_i^{a_i} dx_i
        = prod_i ((1 + (-1)^a_i) / (a_i + 1))
    """
    result = 1.0
    for a in exponents:
        if int(a) % 2 == 1:
            return 0.0
        result *= 2.0 / (a + 1.0)
    return result


def cube_integrate_function(f: Callable, d: int, n_per_dim: int) -> float:
    """
    使用张量积规则对 d 维立方体上的函数积分
    """
    nodes, weights = cube_tensor_product_rule(d, n_per_dim)
    result = 0.0
    for i in range(len(weights)):
        result += weights[i] * f(nodes[i])
    return result


# ============================================================================
# CVT 采样 (from cvt_corn)
# ============================================================================
def cvt_initialization(d: int, n_generators: int,
                        domain: str = 'ball', seed: int = 42) -> np.ndarray:
    """
    CVT 生成器初始化

    domain = 'ball'   : d 维单位球内
    domain = 'cube'   : d 维立方体内
    domain = 'annulus': 环形区域
    """
    rng = np.random.RandomState(seed)
    if domain == 'ball':
        return hyperball_sample_uniform(d, n_generators, seed)
    elif domain == 'cube':
        return rng.uniform(-1.0, 1.0, (n_generators, d))
    elif domain == 'annulus':
        points = rng.randn(n_generators, d)
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-300)
        r = np.sqrt(0.25 + 0.75 * rng.rand(n_generators, 1))
        return r * points / norms
    else:
        raise ValueError(f"未知区域: {domain}")


def cvt_lloyd_iteration(generators: np.ndarray, density: Callable = None,
                          n_samples: int = 10000, seed: int = 42) -> np.ndarray:
    """
    Lloyd 迭代: CVT 的一步更新

    1. 采样点
    2. 分配给最近生成器 (Voronoi)
    3. 计算每个 Voronoi 胞腔的质心
    4. 更新生成器
    """
    d = generators.shape[1]
    n_gen = generators.shape[0]
    rng = np.random.RandomState(seed)

    # 采样
    samples = rng.randn(n_samples, d)
    norms = np.linalg.norm(samples, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-300)
    samples = samples / norms * rng.rand(n_samples, 1)**(1.0 / d)

    if density is not None:
        w = np.array([density(samples[i]) for i in range(n_samples)])
        w = w / (np.sum(w) + 1e-300)
    else:
        w = np.ones(n_samples) / n_samples

    # 分配 (最近邻)
    assignments = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        dists = np.linalg.norm(generators - samples[i], axis=1)
        assignments[i] = np.argmin(dists)

    # 质心更新
    new_generators = generators.copy()
    for k in range(n_gen):
        mask = assignments == k
        if np.sum(mask) > 0:
            wk = w[mask]
            total_w = np.sum(wk)
            if total_w > 1e-300:
                new_generators[k] = np.sum(wk[:, None] * samples[mask], axis=0) / total_w

    return new_generators


def cvt_energy(generators: np.ndarray, samples: np.ndarray,
                weights: np.ndarray) -> float:
    """
    CVT 能量函数:
        E = sum_i w_i * ||x_i - nearest_generator(x_i)||^2
    """
    energy = 0.0
    for i in range(len(samples)):
        dists = np.linalg.norm(generators - samples[i], axis=1)
        min_dist = np.min(dists)
        energy += weights[i] * min_dist**2
    return energy


# ============================================================================
# 球面距离统计 (from circle_distance)
# ============================================================================
def pairwise_angular_distance(points: np.ndarray) -> np.ndarray:
    """
    点集的角距离矩阵 (球面上):
        d_ij = arccos(x_i . x_j)
    """
    n = points.shape[0]
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            cos_d = np.dot(points[i], points[j]) / (
                np.linalg.norm(points[i]) * np.linalg.norm(points[j]) + 1e-300)
            cos_d = np.clip(cos_d, -1.0, 1.0)
            D[i, j] = np.arccos(cos_d)
            D[j, i] = D[i, j]
    return D


def min_angular_separation(points: np.ndarray) -> float:
    """最小角距离"""
    D = pairwise_angular_distance(points)
    n = points.shape[0]
    min_sep = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            if D[i, j] < min_sep:
                min_sep = D[i, j]
    return min_sep


def angular_distance_histogram(points: np.ndarray, n_bins: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    """
    角距离直方图 (用于评估采样均匀性)
    """
    D = pairwise_angular_distance(points)
    n = points.shape[0]
    upper_tri = []
    for i in range(n):
        for j in range(i + 1, n):
            upper_tri.append(D[i, j])
    upper_tri = np.array(upper_tri)
    hist, bin_edges = np.histogram(upper_tri, bins=n_bins, range=(0, np.pi))
    return hist, bin_edges


# ============================================================================
# 等离子体应用: 速度空间矩计算
# ============================================================================
def velocity_moment(v_samples: np.ndarray, weights: np.ndarray,
                     moment_order: int, axis: int = 0) -> float:
    """
    速度空间矩:
        M_n = integral v^n * f(v) dv ≈ sum_i w_i * v_i^n
    """
    return np.sum(weights * v_samples**moment_order)


def temperature_from_samples(vx: np.ndarray, vy: np.ndarray, vz: np.ndarray,
                               m: float) -> float:
    """
    从样本速度计算温度:
        T = m * <v^2> / (3 * k_B)

    <v^2> = (1/N) * sum (vx^2 + vy^2 + vz^2)
    """
    from plasma_constants import BOLTZMANN_CONSTANT
    v2_mean = np.mean(vx**2 + vy**2 + vz**2)
    return m * v2_mean / (3.0 * BOLTZMANN_CONSTANT)


def numerical_diffusion_estimate(v_samples: np.ndarray,
                                    v_true: np.ndarray,
                                    weights: np.ndarray) -> float:
    """
    数值扩散估计:
        D_num = sum_i w_i * (v_i - v_true_i)^2 / (2 * dt)

    用于评估 PIC 中数值加热效应
    """
    return np.sum(weights * (v_samples - v_true)**2) * 0.5
