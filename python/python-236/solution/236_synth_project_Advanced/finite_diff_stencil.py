"""
finite_diff_stencil.py — 高阶有限差分算子与色散关系改善
=========================================================
融合种子项目:
  [916_prism_jaskowiec_rule] : 高精度求积节点 → 最优差分模板系数
  [945_quad_trapezoid] : 梯形求积 → 一阶差分 → 高阶推广

物理背景:
  格点 QCD 中标准 Wilson 费米子使用一阶有限差分近似导数:
  nabla_mu f(x) = (f(x+mu) - f(x-mu)) / (2a)
  这导致 O(a^2) 离散化误差. 高阶差分模板可将误差降至 O(a^{2k}).

  色散关系:  对于连续理论 E^2 = p^2 + m^2
  格点修正:  hat{p}_mu = (1/a) sin(p_mu * a)  (一阶)
  改善后:    hat{p}_mu = sum_{k=1}^{K} c_k * sin(k*p_mu*a) / a

  Symanzik 改善程序: 通过匹配连续-格点色散关系到 O(a^{2K})
  确定最优系数 c_k.

核心公式:
  2K+1 点中心差分模板:
  f'(x) ≈ (1/h) * sum_{k=-K}^{K} w_k * f(x + k*h)
  其中 w_k 由 Taylor 展开匹配到 O(h^{2K}).

  对 K=1 (3 点): w = [-1/2, 0, 1/2]
  对 K=2 (5 点): w = [1/12, -2/3, 0, 2/3, -1/12]
  对 K=3 (7 点): w = [-1/60, 3/20, -3/4, 0, 3/4, -3/20, 1/60]
  对 K=4 (9 点): w = [1/280, -4/105, 1/5, -4/5, 0, 4/5, -1/5, 4/105, -1/280]

  改善的格点 Laplacian:
  Delta_imp f(x) = sum_{k=1}^{K} b_k * sum_{+-mu} f(x +- k*mu) / (k*a)^2
"""

import numpy as np
from typing import Tuple, List
from fractions import Fraction


def finite_diff_weights(derivative_order: int,
                        accuracy_order: int) -> np.ndarray:
    """计算有限差分模板权重 (Fornberg 算法).

    对于 f^{(m)}(x) 的近似, 使用 N 个非等距点:
    f^{(m)}(x) ≈ sum_j w_j * f(x_j)

    这里使用等距点 x_j = j*h, j = -K, ..., K
    其中 K = (N-1)/2, N = 2*K+1.

    算法 (Fornberg, 1988):
    递推关系:
    c^{(m)}_{j,k} = alpha * c^{(m)}_{j-1,k} + ...
    其中 alpha 由匹配 Taylor 系数确定.

    参数
    ----
    derivative_order : int
        导数阶数 m (1=一阶导, 2=二阶导).
    accuracy_order : int
        精度阶数 (偶数), 实际截断误差 O(h^p).

    返回
    ----
    weights : ndarray
        模板权重数组, 长度 N = accuracy_order + 1.
    """
    if derivative_order < 0:
        raise ValueError(f"导数阶数 {derivative_order} 不能为负")
    if accuracy_order < 0:
        raise ValueError(f"精度阶数 {accuracy_order} 不能为负")

    # 等距点
    N = accuracy_order + 1
    x = np.arange(N, dtype=np.float64)

    # Fornberg 算法
    # c[j, k, m] = 在第 j 个点, 使用前 k+1 个点, 求 m 阶导的权重
    c = np.zeros((N, N, derivative_order + 1))
    c[0, 0, 0] = 1.0

    c1 = 1.0
    for i in range(1, N):
        mn = min(i, derivative_order)
        c2 = 1.0
        for j in range(i):
            c3 = x[i] - x[j]
            c2 *= c3
            for m in range(mn, 0, -1):
                c[i, j, m] = (x[i] * c[i - 1, j, m]
                               - m * c[i - 1, j, m - 1]) / c3
            c[i, j, 0] = x[i] * c[i - 1, j, 0] / c3

        for m in range(mn, 0, -1):
            c[i, i, m] = (c1 / c2) * (m * c[i - 1, i - 1, m - 1]
                                        - x[i - 1] * c[i - 1, i - 1, m])
        c[i, i, 0] = -(c1 / c2) * x[i - 1] * c[i - 1, i - 1, 0]
        c1 = c2

    return c[N - 1, :, derivative_order]


def central_diff_weights(derivative_order: int,
                         half_width: int) -> np.ndarray:
    """中心差分模板权重.

    使用对称点 x_j = j*h, j = -K, ..., K
    其中 K = half_width.

    参数
    ----
    derivative_order : int
        导数阶数.
    half_width : int
        半宽度 K. 模板总宽度 = 2*K+1.

    返回
    ----
    weights : ndarray, shape (2*K+1,)
        权重 w[-K], ..., w[0], ..., w[K].
    """
    N = 2 * half_width + 1
    # 使用 Fornberg 算法在 0, 1, ..., N-1 上计算
    w = finite_diff_weights(derivative_order, N - 1)
    # 重新排列为中心差分格式: w[0] 对应 x=-K, w[K] 对应 x=0
    return w


def improved_lattice_derivative(half_width: int = 2) -> np.ndarray:
    """改善的格点一阶导数模板.

    hat{p}_mu = sum_{k=1}^{K} c_k * [f(x+k*mu) - f(x-k*mu)] / (2*k*a)

    等价于中心差分模板:
    f'(x) ≈ (1/a) * sum_{k=-K}^{K} w_k * f(x + k*a)

    参数
    ----
    half_width : int
        模板半宽度 K (1, 2, 3, 4).

    返回
    ----
    coeffs : ndarray, shape (K,)
        c_k 系数, 满足 sum c_k = 1 (归一化条件).
    """
    if half_width < 1 or half_width > 8:
        raise ValueError(f"半宽度 {half_width} 超出 [1, 8]")

    w = central_diff_weights(1, half_width)
    # 提取正半部分系数: c_k = k * w[K+k] for k = 1, ..., K
    K = half_width
    coeffs = np.zeros(K)
    for k in range(1, K + 1):
        coeffs[k - 1] = k * w[K + k]

    # 归一化检查
    total = np.sum(coeffs)
    if abs(total) > 1e-15:
        coeffs /= total

    return coeffs


def dispersion_relation(p_continuous: np.ndarray,
                        half_width: int,
                        a: float = 1.0) -> np.ndarray:
    """计算改善格点导数的色散关系.

    连续:   p^2
    格点:   hat{p}^2 = (1/a^2) * [sum_{k=1}^K c_k * sin(k*p*a)]^2

    改善质量: delta(p) = hat{p}^2/p^2 - 1

    参数
    ----
    p_continuous : ndarray
        连续动量值.
    half_width : int
        模板半宽度.
    a : float
        晶格间距 (默认 a=1).

    返回
    ----
    p_hat_sq : ndarray
        改善的 hat{p}^2.
    """
    coeffs = improved_lattice_derivative(half_width)
    K = half_width
    p_hat = np.zeros_like(p_continuous, dtype=np.float64)
    for k in range(1, K + 1):
        p_hat += coeffs[k - 1] * np.sin(k * p_continuous * a) / a

    return p_hat ** 2


def symanzik_improvement_coefficients(max_order: int = 4
                                      ) -> List[Tuple[int, np.ndarray]]:
    """计算 Symanzik 改善系数 (匹配连续-格点色散关系).

    对每个半宽度 K, 求解线性方程组:
    sum_{k=1}^{K} c_k * k^{2j-1} = delta_{j,1}  for j = 1, ..., K

    这确保 hat{p} = p + O(p^{2K+1}).

    返回
    ----
    results : list of (K, coefficients)
    """
    results = []
    for K in range(1, max_order + 1):
        # 构建线性系统: A * c = b
        # A[j, k] = k^{2j-1} for j=0,...,K-1, k=1,...,K
        A = np.zeros((K, K))
        b = np.zeros(K)
        b[0] = 1.0  # 最低阶匹配

        for j in range(K):
            for k in range(1, K + 1):
                A[j, k - 1] = k ** (2 * j + 1)

        try:
            coeffs = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            coeffs = np.linalg.lstsq(A, b, rcond=None)[0]

        results.append((K, coeffs))

    return results


def lattice_laplacian_improved(f: np.ndarray,
                               half_width: int = 2,
                               a: float = 1.0) -> np.ndarray:
    """改善的格点 Laplacian.

    Delta_imp f(x) = (1/a^2) * sum_{mu} sum_{k=1}^K b_k
                     * [f(x+k*mu) + f(x-k*mu) - 2*f(x)]

    其中 b_k 由 Taylor 展开匹配: sum b_k * k^2 = 1 (最低阶)
    实际上 b_k = -w^{(2)}_k / k^2, 其中 w^{(2)} 是二阶导模板.

    参数
    ----
    f : ndarray, shape (L_0, L_1, L_2, L_3)
        格点标量场.
    half_width : int
        模板半宽度.
    a : float
        晶格间距.

    返回
    ----
    lap_f : ndarray
        Laplacian f.
    """
    shape = f.shape
    ndim = len(shape)
    lap_f = np.zeros_like(f)

    # 获取二阶导模板权重
    w2 = central_diff_weights(2, half_width)

    for mu in range(ndim):
        for k in range(-half_width, half_width + 1):
            if k == 0:
                lap_f += w2[k + half_width] * f / (a * a)
            else:
                f_shifted = np.roll(f, -k, axis=mu)
                lap_f += w2[k + half_width] * f_shifted / (a * a)

    return lap_f


def tadpole_improvement_factor(plaquette_avg: float, Nc: int = 3) -> float:
    """Tadpole 改善因子 u_0.

    u_0 = <P>^{1/4}  (平均 plaquette 的 1/4 次方)

    改善的链接: U_mu -> U_mu / u_0
    改善的耦合: g_0 -> g_0 / u_0^2

    参数
    ----
    plaquette_avg : float
        平均 plaquette 值.
    Nc : int
        色数.
    """
    if plaquette_avg <= 0:
        return 1.0
    u0 = plaquette_avg ** 0.25
    return max(u0, 1e-10)


def compute_discretization_error(masses_extracted: np.ndarray,
                                 a_values: np.ndarray,
                                 true_mass: float,
                                 order: int = 2) -> float:
    """计算离散化误差的幂次.

    m(a) = m_cont + c * a^p + O(a^{p+2})

    通过不同 a 值的外推确定 p.

    参数
    ----
    masses_extracted : ndarray
        不同格距下的质量.
    a_values : ndarray
        对应的格距.
    true_mass : float
        连续极限质量 (若已知).
    order : int
        期望的误差阶数.

    返回
    ----
    estimated_order : float
        估计的误差阶数 p.
    """
    if len(a_values) < 3:
        return float(order)

    # log-log 拟合: log|m(a) - m_cont| = log(c) + p * log(a)
    errors = np.abs(masses_extracted - true_mass)
    valid = errors > 1e-15
    if np.sum(valid) < 2:
        return float(order)

    log_a = np.log(a_values[valid])
    log_err = np.log(errors[valid])

    # 最小二乘拟合
    A = np.column_stack([np.ones_like(log_a), log_a])
    result = np.linalg.lstsq(A, log_err, rcond=None)
    p_est = result[0][1]

    return p_est
