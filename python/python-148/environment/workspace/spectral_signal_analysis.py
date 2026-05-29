"""
spectral_signal_analysis.py — 神经信号谱分析与最优采样
==========================================================
融合 clausen（Chebyshev 级数/Clenshaw 递推）、padua（Padua 最优点）、
legendre_fast_rule（Gauss-Legendre 快速求积）三个项目的核心算法。

功能：
1. 使用 Chebyshev 谱方法对神经信号（LFP/EEG）进行时频分解
2. 通过 Padua 点构造二维最优采样网格用于空间信号插值
3. Gauss-Legendre 高斯求积计算信号能量泛函与特征
4. Clausen 函数用于周期性神经节律的相位分析

核心数学：
---
**Chebyshev 谱表示：**
对定义在 [-1,1] 上的信号 f(x)，其 Chebyshev 展开为

    f(x) ≈ sum_{k=0}^{N} c_k * T_k(x)

其中 T_k(x) = cos(k * arccos(x)) 为第一类 Chebyshev 多项式。
系数由离散 Chebyshev 变换（DCT）获得。

信号在 Chebyshev 基下的能量：

    E_cheb = sum_{k=0}^{N} |c_k|^2 * (π/2) * (2 - δ_{k0})

---
**Padua 点（二维最优插值节点）：**
Padua 点是正方形 [-1,1]^2 上的第一个代数最优插值节点集，
具有最小 Lebesgue 常数增长 O((log n)^2)。

定义：对整数 n，Padua 点由两个 Chebyshev 子网格的交集给出：

    P_n = (C_{n+1}^{ext} × C_n^{ext})  ∪  (C_n^{ext} × C_{n+1}^{ext})

其中 C_m^{ext} = { cos(kπ/m) : k=0,...,m } 为扩展 Chebyshev 节点。
总点数为 (n+1)(n+2)/2。

---
**Clausen 函数 Cl_2(θ)：**

    Cl_2(θ) = -∫_0^θ log|2 sin(t/2)| dt = sum_{k=1}^{∞} sin(kθ) / k^2

用于分析神经振荡的相位延迟特性。
"""

import numpy as np
from utils import clenshaw_chebyshev_eval, gauss_legendre_nodes_weights


# 预计算的 Clausen 函数 Chebyshev 系数（对 |x|<π/2 和 π/2<x<3π/2）
_CLAUSEN_COEFFS_SMALL = np.array([
    1.3888888888888889e-02, 0.0, -2.7777777777777778e-04,
    0.0, 7.936507936507937e-06, 0.0, -2.505210838544172e-07,
    0.0, 8.417724804700504e-09, 0.0, -2.946634356703308e-10,
    0.0, 1.064193259978150e-11, 0.0, -3.932350536369160e-13,
    0.0, 1.480725065921570e-14
], dtype=float)

_CLAUSEN_COEFFS_LARGE = np.array([
    -1.3888888888888889e-02, 0.0, 2.7777777777777778e-04,
    0.0, -7.936507936507937e-06, 0.0, 2.505210838544172e-07,
    0.0, -8.417724804700504e-09, 0.0, 2.946634356703308e-10,
    0.0, -1.064193259978150e-11, 0.0, 3.932350536369160e-13,
    0.0, -1.480725065921570e-14
], dtype=float)


def clausen_function(x):
    """
    计算 Clausen 函数 Cl_2(x)。
    使用分段 Chebyshev 展开：
      - 对 |x| <= π/2：在 [0, π/2] 上展开
      - 对 π/2 < |x| <= π：利用对称性 Cl_2(π - x) = Cl_2(x)
      - 对 |x| > π：利用周期性 Cl_2(x + 2π) = Cl_2(x)
    """
    x = np.asarray(x, dtype=float)
    result = np.zeros_like(x)
    # 归约到 [-π, π]
    x_red = np.mod(x + np.pi, 2.0 * np.pi) - np.pi
    # 利用奇函数性质
    sign_flip = np.sign(x_red)
    x_abs = np.abs(x_red)
    # 利用 Cl_2(π - x) = Cl_2(x)
    mask_large = x_abs > 0.5 * np.pi
    x_eval = x_abs.copy()
    x_eval[mask_large] = np.pi - x_abs[mask_large]
    # 映射到 [0,1] 用于 Chebyshev 展开：t = 2x/π - 1
    t = 2.0 * x_eval / np.pi - 1.0
    # 对 t 使用 Clenshaw 递推
    for i in range(len(t)):
        result.flat[i] = clenshaw_chebyshev_eval(t.flat[i], _CLAUSEN_COEFFS_SMALL)
    result *= sign_flip
    return result


def generate_padua_points(n):
    """
    生成 n 次 Padua 点集。
    返回数组 shape (N, 2)，N = (n+1)(n+2)/2。
    算法：
      子网格1: C_{n+1}^{ext} × C_n^{odd/even}
      子网格2: C_n^{odd/even} × C_{n+1}^{ext}
    其中奇偶选择保证无重复。
    """
    if n < 0:
        return np.zeros((0, 2))
    # 扩展 Chebyshev 节点
    k1 = np.arange(n + 2)
    C_n1 = np.cos(np.pi * k1 / (n + 1))
    k2 = np.arange(n + 1)
    C_n = np.cos(np.pi * k2 / n) if n > 0 else np.array([1.0])
    points = []
    # 子网格 1: C_{n+1} × C_n，取 C_{n+1} 的偶数索引与 C_n 的奇数索引配对
    for i, xi in enumerate(C_n1):
        for j, yj in enumerate(C_n):
            if (i % 2 == 0 and j % 2 == 1) or (i % 2 == 1 and j % 2 == 0):
                if n % 2 == 0:
                    if i % 2 == 0:
                        points.append([xi, yj])
                else:
                    if i % 2 == 1:
                        points.append([xi, yj])
    # 子网格 2: C_n × C_{n+1}
    for i, xi in enumerate(C_n):
        for j, yj in enumerate(C_n1):
            if (i % 2 == 0 and j % 2 == 0) or (i % 2 == 1 and j % 2 == 1):
                if n % 2 == 0:
                    if i % 2 == 1:
                        points.append([xi, yj])
                else:
                    if i % 2 == 0:
                        points.append([xi, yj])
    # 上面的逻辑有点复杂，改用更标准的直接生成法：
    # Padua 点标准定义：
    #   (cos(iπ/n), cos(jπ/(n+1)))  当 i+j 为偶数
    #   加上
    #   (cos(iπ/(n+1)), cos(jπ/n))  当 i+j 为奇数
    points = []
    for i in range(n + 1):
        for j in range(n + 2):
            if (i + j) % 2 == 0:
                points.append([np.cos(i * np.pi / n), np.cos(j * np.pi / (n + 1))])
    for i in range(n + 2):
        for j in range(n + 1):
            if (i + j) % 2 == 1:
                points.append([np.cos(i * np.pi / (n + 1)), np.cos(j * np.pi / n)])
    pts = np.array(points, dtype=float)
    # 去重
    pts = np.unique(np.round(pts, 14), axis=0)
    return pts


def padua_weights(n):
    """
    计算 Padua 点的代数精确 cubature 权重（简化版）。
    对精确权重，需解线性方程组。这里采用基于 Chebyshev 矩的近似权重。
    对 n 次 Padua 点，权重总和应为 4（[-1,1]^2 面积）。
    """
    pts = generate_padua_points(n)
    N = len(pts)
    # 简化的均匀权重近似（实际应用中可解矩方程）
    # 更精确地，使用基于 Lagrange 基函数的权重
    w = np.ones(N, dtype=float) * (4.0 / N)
    return w


class ChebyshevSpectrumAnalyzer:
    """
    基于 Chebyshev 谱的信号时频分析器。
    """

    def __init__(self, n_modes=64):
        self.n_modes = n_modes

    def _dct_transform(self, signal, t_min=-1.0, t_max=1.0):
        """
        将信号从物理时间映射到 [-1,1] 后做离散 Chebyshev 变换。
        使用 Chebyshev-Gauss-Lobatto 节点与 FFT-based DCT。
        """
        signal = np.asarray(signal, dtype=float)
        N = len(signal)
        if N == 0:
            return np.zeros(self.n_modes + 1)
        # 重采样到 n_modes+1 个 Chebyshev-Gauss-Lobatto 节点
        n = self.n_modes
        j = np.arange(n + 1)
        x_nodes = np.cos(np.pi * j / n)
        # 线性插值到节点
        t_nodes = 0.5 * (t_max - t_min) * x_nodes + 0.5 * (t_max + t_min)
        f_nodes = np.interp(t_nodes, np.linspace(t_min, t_max, N), signal)
        # DCT-I
        from scipy.fft import dct
        coeffs = dct(f_nodes, type=1)
        coeffs[0] *= 0.5
        coeffs[n] *= 0.5
        coeffs *= (2.0 / n)
        return coeffs

    def analyze(self, signal, t_min=0.0, t_max=1.0):
        """
        分析信号，返回频谱系数、能量、主频率模式。
        """
        coeffs = self._dct_transform(signal, t_min, t_max)
        # Chebyshev 能量
        energy = np.sum(coeffs[1:] ** 2) * 0.5 * np.pi
        energy += coeffs[0] ** 2 * 0.5 * np.pi
        # 主模式（忽略直流分量后取最大）
        if len(coeffs) > 1:
            dominant_mode = np.argmax(np.abs(coeffs[1:])) + 1
        else:
            dominant_mode = 0
        return {
            'coefficients': coeffs,
            'energy': energy,
            'dominant_mode': dominant_mode,
            'dc_component': coeffs[0]
        }

    def reconstruct(self, coeffs, n_eval=512):
        """
        从 Chebyshev 系数重建信号。
        """
        x = np.linspace(-1, 1, n_eval)
        vals = np.array([clenshaw_chebyshev_eval(xi, coeffs) for xi in x])
        return x, vals


class GaussLegendreSignalIntegrator:
    """
    使用 Gauss-Legendre 数值积分计算信号的能量泛函与统计矩。
    """

    def __init__(self, n_points=64):
        self.n_points = n_points
        self.xi, self.wi = gauss_legendre_nodes_weights(n_points)

    def integrate_function(self, f, a=-1.0, b=1.0):
        """
        计算 ∫_a^b f(x) dx 的 Gauss-Legendre 近似：
            ∫_a^b f(x) dx ≈ (b-a)/2 * sum_i w_i * f( (b-a)/2 * xi + (a+b)/2 )
        """
        x_mapped = 0.5 * (b - a) * self.xi + 0.5 * (a + b)
        fx = np.array([f(x) for x in x_mapped], dtype=float)
        return 0.5 * (b - a) * np.sum(self.wi * fx)

    def signal_moments(self, signal, t):
        """
        计算信号的前四阶统计矩（均值、方差、偏度、峰度），
        通过 Gauss-Legendre 积分提高精度。
        """
        t = np.asarray(t, dtype=float)
        signal = np.asarray(signal, dtype=float)
        t_min, t_max = t[0], t[-1]
        # 构建插值函数
        def f_interp(x):
            return np.interp(x, t, signal)
        length = t_max - t_min
        mu1 = self.integrate_function(f_interp, t_min, t_max) / length
        mu2 = self.integrate_function(lambda x: (f_interp(x) - mu1) ** 2, t_min, t_max) / length
        mu3 = self.integrate_function(lambda x: (f_interp(x) - mu1) ** 3, t_min, t_max) / length
        mu4 = self.integrate_function(lambda x: (f_interp(x) - mu1) ** 4, t_min, t_max) / length
        variance = mu2
        std = np.sqrt(variance) if variance > 1e-15 else 1e-15
        skewness = mu3 / (std ** 3)
        kurtosis = mu4 / (std ** 4) - 3.0
        return {
            'mean': mu1,
            'variance': variance,
            'std': std,
            'skewness': skewness,
            'kurtosis': kurtosis
        }

    def signal_energy_functional(self, signal, t, alpha=2.0):
        """
        计算信号能量泛函：
            E[signal] = ∫ |signal(t)|^α dt
        α=2 时为标准 L2 能量。
        """
        t_min, t_max = t[0], t[-1]
        def f_abs_power(x):
            s = np.interp(x, t, signal)
            return np.abs(s) ** alpha
        return self.integrate_function(f_abs_power, t_min, t_max)


class SpatialPaduaSampler:
    """
    基于 Padua 点的二维空间最优采样器，用于皮层表面信号的空间插值。
    """

    def __init__(self, n_degree=16):
        self.n_degree = n_degree
        self.points = generate_padua_points(n_degree)
        self.weights = padua_weights(n_degree)

    def sample_field(self, field_func):
        """
        在 Padua 点处采样标量场 field_func(x,y)。
        返回采样值数组。
        """
        vals = np.array([field_func(p[0], p[1]) for p in self.points], dtype=float)
        return vals

    def integrate_field(self, field_func):
        """
        使用 Padua 点-权重对近似计算 ∫_{[-1,1]^2} field(x,y) dx dy。
        """
        vals = self.sample_field(field_func)
        return np.sum(self.weights * vals)

    def interpolate_to_grid(self, sampled_values, X, Y):
        """
        使用径向基函数 (RBF) 从 Padua 点采样值插值到规则网格 (X,Y)。
        这里采用 thin-plate spline RBF：φ(r) = r^2 * log(r)
        """
        from scipy.interpolate import RBFInterpolator
        pts = self.points
        # 展平网格
        grid_points = np.column_stack([X.ravel(), Y.ravel()])
        rbf = RBFInterpolator(pts, sampled_values, kernel='thin_plate_spline')
        vals_grid = rbf(grid_points)
        return vals_grid.reshape(X.shape)
