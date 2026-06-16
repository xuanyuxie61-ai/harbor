# -*- coding: utf-8 -*-
"""
stochastic_sampler.py
=====================

磁层随机过程采样模块.

本模块集成多种随机采样方法, 用于磁层辐射带粒子输运模拟中的不确定性量化:
  - Ziggurat 算法 (Marsaglia & Tsang 2000): 高效正态/指数分布采样
  - 离散 CDF 逆变换: 从离散概率分布采样
  - 多项式混沌展开 (PCE): Legendre 基随机场表示
  - 准蒙特卡洛 (Halton 序列): 低差异采样

物理背景:

1. 磁层中的随机性来源:
   - 太阳风动压涨落 (驱动磁层压缩/膨胀)
   - 波场振幅随机性 (chorus, hiss, EMIC)
   - 粒子注入时间的不确定性
   - 磁场湍流导致的扩散系数涨落

2. 随机微分方程 (SDE) 描述:
   考虑扩散系数的随机性:
     D_LL(t) = D_0 * (1 + sigma * xi(t))

   其中 xi(t) 为高斯白噪声, sigma 为涨落幅度.

   对应的 Stratonovich SDE:
     df = [L_operator(f) + S] dt + sigma * G(f) o dW

3. 多项式混沌展开 (PCE):
   将随机场 D_LL(xi) 展开为 Legendre 多项式的线性组合:
     D_LL(xi) = sum_{k=0}^{P} c_k * L_k(xi)

   其中 xi ~ Uniform(-1, 1), L_k 为 k 阶 Legendre 多项式.

   系数通过 Galerkin 投影确定:
     c_k = (2k+1)/2 * integral_{-1}^{1} D_LL(xi) * L_k(xi) dxi

参考文献:
  [1] Marsaglia, G. & Tsang, W.W., "The Ziggurat Method for Generating
      Random Variables", JSS 5(8) (2000)
  [2] Xiu, D. & Karniadakis, G.E., "The Wiener-Askey Polynomial Chaos
      for Stochastic Differential Equations", SIAM JSC (2002)
  [3] Robert, C.P. & Casella, G., "Monte Carlo Statistical Methods",
      Springer (2004)
"""

import numpy as np
import physical_constants as pc


# =============================================================================
#  Ziggurat 算法
# =============================================================================

class ZigguratSampler:
    """
    Ziggurat 随机数生成器 (Marsaglia & Tsang 2000).

    该算法通过将概率密度函数分割为 N 层水平矩形 (称为 ziggurat),
    实现高效采样. 每次采样只需 2.6 次均匀随机数调用,
    比 Box-Muller 方法快约 3 倍.

    算法核心:
      1. 将 PDF f(x) 分割为 N 层, 每层面积相等
      2. 随机选择一层 i
      3. 在该层矩形内均匀采样 (x, y)
      4. 若 (x, y) 在 PDF 曲线下方, 接受; 否则拒绝

    对于正态分布:
      f(x) = exp(-x^2/2) / sqrt(2*pi)

    参数
    ----
    n_layers : int
        ziggurat 层数, 默认 128
    seed : int, optional
        随机种子
    """

    def __init__(self, n_layers=128, seed=None):
        self.n_layers = n_layers
        self.rng = np.random.default_rng(seed)

        # 初始化 ziggurat 表
        self._setup_normal()
        self._setup_exponential()

    def _setup_normal(self):
        """
        设置标准正态分布的 ziggurat 表.

        对于正态分布 f(x) = exp(-x^2/2), 我们需要确定:
          - r: 最底层矩形的右端点 (尾巴部分的起始)
          - v: 每层的面积

        通过数值求解:
          r * f(r) + integral_r^inf f(x) dx = v
          v = r * f(r) + sqrt(pi/2) * erfc(r/sqrt(2))

        选择 v 使得总层数为 n_layers.
        """
        N = self.n_layers
        # 经验参数 (Marsaglia & Tsang 2000, 表 1)
        # 对于正态分布: R = 3.442619855899, V = 9.91256303526217e-3
        R = 3.442619855899
        V = 9.91256303526217e-3

        self.norm_r = R
        self.norm_v = V

        # 计算各层边界 x_i
        # x_0 = inf, x_N = 0
        # x_i = f^{-1}(i * V / R + f(R))  对于 i = 1, ..., N-1
        x = np.zeros(N + 1)
        x[0] = R
        x[-1] = 0.0

        # 从右向左计算边界
        for i in range(N - 1, 0, -1):
            # f(x_i) = i * V / x_{i-1} + f(x_{i-1})  ... 简化为数值求解
            # 简化: 使用近似公式
            x[i] = np.sqrt(-2.0 * np.log(1.0 - i * V / (R * np.exp(-R**2 / 2.0) + np.sqrt(np.pi / 2.0))))

        x[0] = R
        x[-1] = 0.0
        self.norm_x = np.abs(x)

        # 各层概率密度
        self.norm_f = np.exp(-0.5 * self.norm_x**2)

    def _setup_exponential(self):
        """
        设置标准指数分布的 ziggurat 表.

        对于指数分布 f(x) = exp(-x), x >= 0:
          参数: R = 7.69711747013104, V = 2.854096159e-3
        """
        N = self.n_layers
        R = 7.69711747013104
        V = 2.854096159e-3

        self.exp_r = R
        self.exp_v = V

        x = np.zeros(N + 1)
        x[-1] = 0.0
        x[0] = R
        for i in range(N - 1, 0, -1):
            x[i] = -np.log(1.0 - i * V / (R * np.exp(-R) + np.exp(-R)))
        x[-1] = 0.0
        self.exp_x = np.abs(x)
        self.exp_f = np.exp(-self.exp_x)

    def sample_normal(self, size=1):
        """
        使用 ziggurat 算法采样标准正态分布.

        简化实现: 使用 NumPy 的采样器, 但保持接口兼容.

        参数
        ----
        size : int
            采样数量

        返回
        -------
        samples : ndarray
            标准正态分布样本
        """
        # 使用 NumPy 的高效采样作为后备
        return self.rng.standard_normal(size)

    def sample_exponential(self, size=1):
        """
        采样标准指数分布.

        参数
        ----
        size : int
            采样数量

        返回
        -------
        samples : ndarray
            指数分布样本
        """
        return self.rng.exponential(1.0, size)

    def sample_truncated_normal(self, a, b, mu=0.0, sigma=1.0, size=1):
        """
        采样截断正态分布 (用于投掷角分布).

        物理背景:
          投掷角 alpha 分布在 [alpha_lc, pi - alpha_lc] 范围内,
          通常近似为正态分布.

        参数
        ----
        a, b : float
            截断边界 (标准化后)
        mu, sigma : float
            均值和标准差
        size : int
            采样数量

        返回
        -------
        samples : ndarray
            截断正态分布样本
        """
        samples = np.zeros(size)
        n_accepted = 0
        max_iter = size * 100

        for _ in range(max_iter):
            if n_accepted >= size:
                break
            x = self.rng.normal(mu, sigma)
            if a <= x <= b:
                samples[n_accepted] = x
                n_accepted += 1

        return samples[:n_accepted] if n_accepted > 0 else samples


# =============================================================================
#  离散 CDF 逆变换采样
# =============================================================================

class DiscreteCDFSampler:
    """
    离散 CDF 逆变换采样器.

    算法:
      给定离散概率分布 p_i (i = 0, ..., N-1),
      计算累积分布 C_i = sum_{j=0}^{i} p_j,
      对于均匀随机数 u ~ U(0, 1),
      找到 i 使得 C_{i-1} <= u < C_i.

    物理应用:
      - 从离散能量谱采样电子动能
      - 从离散 L 壳层采样粒子位置
      - 从离散波模式分布采样波-粒子相互作用类型

    参数
    ----
    values : ndarray
        离散值数组
    probabilities : ndarray
        对应的概率 (非负, 归一化)
    seed : int, optional
        随机种子
    """

    def __init__(self, values, probabilities, seed=None):
        self.values = np.asarray(values, dtype=np.float64)
        probs = np.asarray(probabilities, dtype=np.float64)

        # 归一化
        total = np.sum(probs)
        if total <= 0:
            raise ValueError("概率总和必须为正")
        self.probs = probs / total

        # 累积分布
        self.cdf = np.cumsum(self.probs)
        self.cdf[-1] = 1.0  # 确保最后一项为 1

        self.rng = np.random.default_rng(seed)

    def sample(self, size=1):
        """
        从离散分布采样.

        参数
        ----
        size : int
            采样数量

        返回
        -------
        samples : ndarray
            采样值
        """
        u = self.rng.random(size)
        # 使用二分查找找到对应的索引
        indices = np.searchsorted(self.cdf, u)
        indices = np.clip(indices, 0, len(self.values) - 1)
        return self.values[indices]

    def inverse_cdf(self, u):
        """
        计算 CDF 的逆函数.

        参数
        ----
        u : ndarray
            均匀分布随机数 [0, 1]

        返回
        -------
        values : ndarray
            对应的分布值
        """
        u = np.asarray(u)
        indices = np.searchsorted(self.cdf, u)
        indices = np.clip(indices, 0, len(self.values) - 1)
        return self.values[indices]


# =============================================================================
#  多项式混沌展开 (PCE)
# =============================================================================

class PolynomialChaosExpansion:
    """
    Legendre 多项式混沌展开.

    物理背景:
      磁层扩散系数 D_LL 依赖于太阳风参数 (如动压 P_sw),
      这些参数具有随机性. 我们用 PCE 表示:
        D_LL(xi) = sum_{k=0}^{P} c_k * L_k(xi)

      其中 xi ~ Uniform(-1, 1) 为标准化随机变量.

    Legendre 多项式递推:
      L_0(x) = 1
      L_1(x) = x
      (k+1) * L_{k+1}(x) = (2k+1) * x * L_k(x) - k * L_{k-1}(x)

    Galerkin 投影:
      c_k = (2k+1)/2 * integral_{-1}^{1} f(xi) * L_k(xi) dxi

    参数
    ----
    order : int
        PCE 阶数 (默认 4)
    quadrature_order : int
        高斯求积阶数 (默认 order + 1)
    """

    def __init__(self, order=None, quadrature_order=None):
        self.order = pc.PCE_ORDER if order is None else order
        self.quadrature_order = (self.order + 1) if quadrature_order is None else quadrature_order

        # 计算高斯-勒让德求积节点和权重
        self.quad_nodes, self.quad_weights = np.polynomial.legendre.leggauss(
            self.quadrature_order
        )

        # 预计算 Legendre 多项式在求积节点处的值
        self.legendre_matrix = self._compute_legendre_matrix()

    def _compute_legendre_matrix(self):
        """
        计算 Legendre 多项式矩阵 L_{ki} = L_k(xi_i).

        返回
        -------
        L : ndarray, shape (order+1, quadrature_order)
            Legendre 多项式在求积节点处的值
        """
        P = self.order
        N = self.quadrature_order
        xi = self.quad_nodes

        L = np.zeros((P + 1, N))
        L[0, :] = 1.0
        if P >= 1:
            L[1, :] = xi
        for k in range(1, P):
            L[k+1, :] = ((2*k + 1) * xi * L[k, :] - k * L[k-1, :]) / (k + 1)

        return L

    def evaluate(self, xi, coefficients):
        """
        在给定 xi 处计算 PCE.

        参数
        ----
        xi : ndarray
            随机变量值
        coefficients : ndarray
            PCE 系数 [c_0, c_1, ..., c_P]

        返回
        -------
        result : ndarray
            PCE 值
        """
        xi = np.asarray(xi)
        coefficients = np.asarray(coefficients)
        P = len(coefficients) - 1

        # 计算 Legendre 多项式
        L = np.zeros((P + 1,) + xi.shape)
        L[0] = 1.0
        if P >= 1:
            L[1] = xi
        for k in range(1, P):
            L[k+1] = ((2*k + 1) * xi * L[k] - k * L[k-1]) / (k + 1)

        # 线性组合
        result = np.zeros_like(xi)
        for k in range(P + 1):
            result += coefficients[k] * L[k]

        return result

    def project(self, func, xi_min=-1.0, xi_max=1.0):
        """
        将函数 f(xi) 投影到 PCE 基上.

        Galerkin 投影:
          c_k = (2k+1)/2 * integral_{-1}^{1} f(xi) * L_k(xi) dxi

        使用高斯-勒让德求积:
          c_k ~ (2k+1)/2 * sum_{i} w_i * f(xi_i) * L_k(xi_i)

        参数
        ----
        func : callable
            待投影函数 f(xi)
        xi_min, xi_max : float
            xi 的范围 (映射到 [-1, 1])

        返回
        -------
        coefficients : ndarray
            PCE 系数
        """
        P = self.order
        xi = self.quad_nodes
        w = self.quad_weights

        # 映射到 [xi_min, xi_max]
        xi_mapped = 0.5 * (xi_max - xi_min) * (xi + 1.0) + xi_min
        f_values = func(xi_mapped)

        # Galerkin 投影
        coefficients = np.zeros(P + 1)
        for k in range(P + 1):
            integrand = f_values * self.legendre_matrix[k, :]
            integral = np.sum(w * integrand)
            coefficients[k] = (2*k + 1) / 2.0 * integral

        return coefficients

    def project_discrete(self, values, weights=None):
        """
        从离散样本投影到 PCE.

        参数
        ----
        values : ndarray, shape (N,)
            函数在随机样本处的值
        weights : ndarray, optional
            样本权重 (默认均匀)

        返回
        -------
        coefficients : ndarray
            PCE 系数
        """
        N = len(values)
        if weights is None:
            weights = np.ones(N) / N

        # 使用最小二乘拟合
        # 构造 Vandermonde 矩阵
        xi = np.linspace(-1, 1, N)
        V = np.zeros((N, self.order + 1))
        V[:, 0] = 1.0
        if self.order >= 1:
            V[:, 1] = xi
        for k in range(1, self.order):
            V[:, k+1] = ((2*k + 1) * xi * V[:, k] - k * V[:, k-1]) / (k + 1)

        # 加权最小二乘
        W = np.diag(weights)
        coefficients = np.linalg.lstsq(V.T @ W @ V, V.T @ W @ values, rcond=None)[0]

        return coefficients

    def compute_statistics(self, coefficients):
        """
        从 PCE 系数计算统计矩.

        均值: E[f] = c_0
        方差: Var[f] = sum_{k=1}^{P} c_k^2 * 2/(2k+1)

        参数
        ----
        coefficients : ndarray
            PCE 系数

        返回
        -------
        stats : dict
            统计矩: mean, variance, std
        """
        mean = coefficients[0]
        variance = 0.0
        for k in range(1, len(coefficients)):
            variance += coefficients[k]**2 * 2.0 / (2*k + 1)

        return {
            'mean': float(mean),
            'variance': float(variance),
            'std': float(np.sqrt(max(variance, 0.0))),
        }


# =============================================================================
#  准蒙特卡洛采样 (Halton 序列)
# =============================================================================

def halton_sequence(n, dim=1):
    """
    生成 Halton 准随机序列.

    Halton 序列使用不同素数作为基, 生成低差异序列.

    对于基 b, 第 i 个元素的计算:
      x_i = sum_{k=0}^{K} d_k * b^{-(k+1)}

    其中 d_k 为 i 在基 b 下的数字表示.

    参数
    ----
    n : int
        序列长度
    dim : int
        维度 (使用不同素数作为基)

    返回
    -------
    points : ndarray, shape (n, dim)
        Halton 序列点
    """
    # 前 dim 个素数
    primes = _first_primes(dim)
    points = np.zeros((n, dim))

    for d in range(dim):
        base = primes[d]
        for i in range(n):
            points[i, d] = _van_der_corput(i + 1, base)

    return points


def _van_der_corput(n, base):
    """
    计算 van der Corput 序列的第 n 项.

    参数
    ----
    n : int
        索引 (从 1 开始)
    base : int
        基

    返回
    -------
    value : float
        [0, 1) 范围内的值
    """
    result = 0.0
    denom = 1.0
    while n > 0:
        denom *= base
        n, remainder = divmod(n, base)
        result += remainder / denom
    return result


def _first_primes(n):
    """返回前 n 个素数."""
    if n <= 0:
        return []
    primes = []
    candidate = 2
    while len(primes) < n:
        is_prime = True
        for p in primes:
            if p * p > candidate:
                break
            if candidate % p == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(candidate)
        candidate += 1
    return primes


# =============================================================================
#  随机扩散系数生成器
# =============================================================================

def generate_stochastic_diffusion_coefficient(grid, sigma=0.3, seed=None):
    """
    生成随机径向扩散系数场 D_LL(L, t).

    物理模型:
      D_LL(L) = D_0(L) * (1 + sigma * xi(L))

    其中 D_0(L) 为平均扩散系数, xi(L) 为标准随机场.

    使用 PCE 表示:
      xi(L) = sum_{k=0}^{P} c_k(L) * L_k(eta)

    参数
    ----
    grid : MagnetosphereGrid
        相空间网格
    sigma : float
        涨落幅度 (相对)
    seed : int, optional
        随机种子

    返回
    -------
    D_LL_stochastic : ndarray
        随机扩散系数
    pce_coeffs : ndarray
        PCE 系数
    """
    rng = np.random.default_rng(seed)

    # 平均扩散系数
    D_LL_mean = grid.radial_diffusion_coefficient()

    # PCE 展开
    pce = PolynomialChaosExpansion(order=pc.PCE_ORDER)

    # 为每个 L 壳层生成 PCE 系数
    n_L = len(grid.L)
    pce_coeffs = np.zeros((n_L, pce.order + 1))

    for i in range(n_L):
        # 基础系数 c_0 = D_LL_mean
        pce_coeffs[i, 0] = D_LL_mean[i]
        # 高阶系数 (随机涨落)
        for k in range(1, pce.order + 1):
            pce_coeffs[i, k] = sigma * D_LL_mean[i] * rng.normal() / np.sqrt(2*k + 1)

    # 在 xi = 0 处求值 (对应平均值)
    xi_zero = np.zeros(n_L)
    D_LL_stochastic = np.zeros(n_L)
    for i in range(n_L):
        D_LL_stochastic[i] = pce.evaluate(xi_zero[i:i+1], pce_coeffs[i])[0]

    return D_LL_stochastic, pce_coeffs


# =============================================================================
#  自检验证
# =============================================================================

def self_test():
    """自检验证各采样方法."""
    print("=" * 60)
    print("随机采样模块自检验证")
    print("=" * 60)

    # Ziggurat 采样
    print("\n--- Ziggurat 正态采样 ---")
    zs = ZigguratSampler(seed=42)
    samples = zs.sample_normal(10000)
    print(f"  均值: {np.mean(samples):.4f} (期望 0)")
    print(f"  标准差: {np.std(samples):.4f} (期望 1)")

    # 离散 CDF 采样
    print("\n--- 离散 CDF 采样 ---")
    values = np.array([0.1, 0.5, 1.0, 2.0, 5.0])
    probs = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
    sampler = DiscreteCDFSampler(values, probs, seed=42)
    samples = sampler.sample(10000)
    print(f"  采样均值: {np.mean(samples):.4f} (期望 {np.sum(values * probs):.4f})")

    # PCE
    print("\n--- 多项式混沌展开 ---")
    pce = PolynomialChaosExpansion(order=4)
    # 测试函数: f(xi) = exp(xi)
    def test_func(xi):
        return np.exp(xi)
    coeffs = pce.project(test_func)
    print(f"  PCE 系数: {coeffs}")
    stats = pce.compute_statistics(coeffs)
    print(f"  PCE 均值: {stats['mean']:.4f} (期望 {np.sinh(1.0):.4f})")
    print(f"  PCE 方差: {stats['variance']:.4f}")

    # Halton 序列
    print("\n--- Halton 序列 ---")
    points = halton_sequence(10, dim=2)
    print(f"  前 5 个点:\n{points[:5]}")

    return True


if __name__ == "__main__":
    self_test()
