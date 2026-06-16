"""
special_barrier.py - 特殊函数与障碍函数分析模块
=================================================

融合种子项目:
  - 187_clausen: Chebyshev 展开求特殊函数
  - 809_nonlin_regula: Regula Falsi 非线性求根

核心数学: 障碍函数的特殊计算

在Interior Point Method中, 对数障碍函数 phi(x) = -sum ln(s_i) 的
计算需要高精度特殊函数. 本模块实现:

1. Clausen函数 Cl_2(x) = -integral_0^x ln|2 sin(t/2)| dt
   用于计算周期性障碍函数的解析修正项.

   Cl_2(x) = sum_{k=1}^{inf} sin(k*x) / k^2

   Chebyshev展开系数由 Koelbig (1995) 给出.

2. 障碍参数 mu 的自适应更新: 使用 Regula Falsi 求解
   complementarity gap 方程:
       mu * sigma = x^T s / n
   其中 sigma in (0, 1) 是中心参数.

3. Mizuno-Todd-Predator 障碍函数的修正项:
   psi_MT(x, s, mu) = X S e - mu e
   其中 X = diag(x), S = diag(s).

公式:
  障碍问题: min f(x) - mu * sum_{i=1}^m ln(s_i)
            s.t. h(x) = 0, x + s = b, s > 0

  KKT条件 (障碍问题):
    nabla f(x) + A^T y - z = 0    (平稳性)
    h(x) = 0                        (可行性)
    X Z e = mu e                    (扰动互补松弛)

  其中 z 是对偶松弛变量, X = diag(x), Z = diag(z).

  Mehrotra predictor-corrector:
    预测步: 求解牛顿方程得到仿射方向 (x_aff, y_aff, z_aff)
    中心参数: sigma = (mu_aff / mu)^3
              mu_aff = (x + alpha_aff * dx_aff)^T (z + alpha_aff * dz_aff) / n
    修正步: 加入二阶修正项 dx_aff * dz_aff
"""

import numpy as np
from typing import Tuple, Optional


def clausen_chebyshev(x: float) -> float:
    """
    使用 Chebyshev 展开计算 Clausen 函数 Cl_2(x).

    Cl_2(x) = -integral_0^x ln|2 sin(t/2)| dt = sum_{k=1}^{inf} sin(kx)/k^2

    该函数出现在障碍函数的高阶修正项中, 特别是当障碍参数 mu 沿
    中心路径变化时, 修正项涉及 Clausen 函数的周期展开.

    公式 (Koelbig, 1995):
      对 -pi/2 < x < pi/2:
        Cl_2(x) = x - x*ln|x| + 0.5*x^3 * sum_{k=0}^{N} c1_k T_k(2x/pi)
      对 pi/2 < x < 3pi/2:
        Cl_2(x) = sum_{k=0}^{N} c2_k T_k(2x/pi - 2)

    其中 T_k 是 k 阶 Chebyshev 多项式.

    Parameters
    ----------
    x : float
        自变量

    Returns
    -------
    float
        Cl_2(x) 的值
    """
    # Chebyshev 系数 (区间 -pi/2 < x < pi/2), 来自 Koelbig (1995)
    c1 = np.array([
        0.05590566394715132269, 0.0, 0.00017630887438981157, 0.0,
        0.00000126627414611565, 0.0, 0.00000001171718181344, 0.0,
        0.00000000012300641288, 0.0, 0.00000000000139527290, 0.0,
        0.00000000000001669078, 0.0, 0.00000000000000020761, 0.0,
        0.00000000000000000266, 0.0, 0.00000000000000000003
    ])
    n1 = len(c1)

    # Chebyshev 系数 (区间 pi/2 < x < 3pi/2)
    c2 = np.array([
        0.0, -0.96070972149008358753, 0.0, 0.04393661151911392781,
        0.0, 0.00078014905905217505, 0.0, 0.00002621984893260601,
        0.0, 0.00000109292497472610, 0.0, 0.00000005122618343931,
        0.0, 0.00000000258863512670, 0.0, 0.00000000013787545462,
        0.0, 0.00000000000763448721, 0.0, 0.00000000000043556938,
        0.0, 0.00000000000002544696, 0.0, 0.00000000000000151561,
        0.0, 0.00000000000000009172, 0.0, 0.00000000000000000563,
        0.0, 0.00000000000000000035, 0.0, 0.00000000000000000002
    ])
    n2 = len(c2)

    # 将 x 折叠到 [-pi/2, 3pi/2]
    xa = -0.5 * np.pi
    xb = 0.5 * np.pi
    xc = 1.5 * np.pi

    x2 = float(x)
    two_pi = 2.0 * np.pi
    # 周期性折叠
    if x2 != 0.0:
        n_periods = int(np.floor((x2 - xa) / two_pi))
        x2 = x2 - n_periods * two_pi
        # 确保在范围内
        while x2 < xa:
            x2 += two_pi
        while x2 >= xc:
            x2 -= two_pi

    # 计算函数值
    if abs(x2) < 1.0e-15:
        return 0.0
    elif x2 < xb:
        x3 = 2.0 * x2 / np.pi
        cheb_val = _chebyshev_evaluate(x3, c1, n1)
        value = x2 - x2 * np.log(abs(x2)) + 0.5 * x2**3 * cheb_val
    else:
        x3 = 2.0 * x2 / np.pi - 2.0
        value = _chebyshev_evaluate(x3, c2, n2)

    return value


def _chebyshev_evaluate(x: float, c: np.ndarray, n: int) -> float:
    """
    Clenshaw 递推计算 Chebyshev 展开.

    sum_{k=0}^{n-1} c_k T_k(x) = c_0/2 + sum_{k=1}^{n-1} c_k T_k(x)

    Clenshaw 递推:
      d_{n+1} = d_n = 0
      d_k = 2*x*d_{k+1} - d_{k+2} + c_k,  k = n-1, ..., 1
      result = x*d_1 - d_2 + c_0

    Parameters
    ----------
    x : float
        自变量 (在 [-1, 1] 内)
    c : ndarray
        Chebyshev 系数
    n : int
        系数个数

    Returns
    -------
    float
        Chebyshev 展开的值
    """
    if n <= 0:
        return 0.0
    if n == 1:
        return c[0]

    d2 = 0.0
    d1 = 0.0
    for k in range(n - 1, 0, -1):
        d0 = 2.0 * x * d1 - d2 + c[k]
        d2 = d1
        d1 = d0

    return x * d1 - d2 + c[0]


def regula_falsi_barrier(f, a: float, b: float, tol: float = 1e-12,
                         max_iter: int = 200) -> Tuple[float, float, int]:
    """
    改进的 Regula Falsi 方法求解障碍参数方程.

    在 Interior Point Method 中, 需要求解中心参数 sigma 的方程:
        g(sigma) = mu_aff(sigma) / mu - sigma = 0

    其中 mu_aff 是仿射步后的互补间隙.

    Illinois 改进避免了一端停滞的问题:
        若连续两次在同一端更新, 则将另一端的函数值减半.

    公式:
        c = (a * f(b) - b * f(a)) / (f(b) - f(a))
        若 f(c) 与 f(a) 同号:
            a = c, f(a) = f(c)   ( Illinois: f(b) /= 2 )
        否则:
            b = c, f(b) = f(c)   ( Illinois: f(a) /= 2 )

    Parameters
    ----------
    f : callable
        目标函数
    a, b : float
        初始变号区间端点
    tol : float
        区间宽度容差
    max_iter : int
        最大迭代次数

    Returns
    -------
    root : float
        近似根
    f_root : float
        f(root)
    it : int
        迭代次数
    """
    fa = f(a)
    fb = f(b)

    # 验证变号条件
    if fa * fb > 0:
        # 使用更宽的区间
        a_try = a - 0.5 * abs(b - a)
        b_try = b + 0.5 * abs(b - a)
        fa_try = f(a_try)
        fb_try = f(b_try)
        if fa_try * fb_try < 0:
            a, b = a_try, b_try
            fa, fb = fa_try, fb_try
        else:
            # 返回中点作为最坏情况的估计
            mid = 0.5 * (a + b)
            return mid, f(mid), 0

    it = 0
    side = 0  # 记录上次更新的是哪一端

    while abs(b - a) > tol:
        if it >= max_iter:
            break
        it += 1

        # Regula falsi 插值
        denom = fb - fa
        if abs(denom) < 1.0e-30:
            break
        c = (a * fb - b * fa) / denom
        fc = f(c)

        if abs(fc) < 1.0e-15:
            return c, fc, it

        if fc * fa < 0:
            b = c
            fb = fc
            if side == -1:
                # Illinois 修正
                fa *= 0.5
            side = -1
        else:
            a = c
            fa = fc
            if side == 1:
                # Illinois 修正
                fb *= 0.5
            side = 1

    root = 0.5 * (a + b)
    f_root = f(root)
    return root, f_root, it


def compute_centering_parameter(mu_current: float,
                                 x: np.ndarray, s: np.ndarray,
                                 dx_aff: np.ndarray, ds_aff: np.ndarray,
                                 n: int) -> float:
    """
    Mehrotra 自适应中心参数计算.

    sigma = (mu_aff / mu)^3

    其中:
      mu = x^T s / n           (当前互补间隙)
      mu_aff = (x+alpha_p*dx_aff)^T (s+alpha_d*ds_aff) / n  (仿射步后间隙)
      alpha_p, alpha_d 是最大可行步长

    sigma 的物理意义:
      sigma -> 0: 偏向仿射方向 (快速减少互补间隙)
      sigma -> 1: 偏向中心方向 (保持在中心路径邻域内)

    三次方关系确保当仿射步已经大幅减少互补间隙时,
    修正步主要提供中心性修正.

    Parameters
    ----------
    mu_current : float
        当前互补间隙 mu = x^T s / n
    x, s : ndarray
        当前原始-对偶变量
    dx_aff, ds_aff : ndarray
        仿射步方向
    n : int
        变量维度

    Returns
    -------
    float
        中心参数 sigma in [0, 1]
    """
    # 计算最大可行步长
    alpha_p = _max_step_length(x, dx_aff)
    alpha_d = _max_step_length(s, ds_aff)

    # 仿射步后的互补间隙
    x_aff = x + alpha_p * dx_aff
    s_aff = s + alpha_d * ds_aff

    # 保证正性
    x_aff = np.maximum(x_aff, 1.0e-15)
    s_aff = np.maximum(s_aff, 1.0e-15)

    mu_aff = np.dot(x_aff, s_aff) / n

    # 中心参数
    if mu_current < 1.0e-30:
        return 0.1

    sigma = (mu_aff / mu_current) ** 3
    sigma = np.clip(sigma, 1.0e-6, 0.9999)

    return sigma


def _max_step_length(x: np.ndarray, dx: np.ndarray,
                      gamma: float = 0.995) -> float:
    """
    计算保持正性的最大步长.

    alpha_max = max { alpha in (0, 1] : x + alpha * dx >= 0 }

    使用 gamma 缩减以保证严格正性 (gamma < 1).

    公式:
      若 dx_i >= 0: 该分量不会违反正性
      若 dx_i < 0:  alpha_i = -gamma * x_i / dx_i
      alpha_max = min(1, min_i alpha_i)

    Parameters
    ----------
    x : ndarray
        当前点 (正值)
    dx : ndarray
        搜索方向
    gamma : float
        缩减因子 (默认 0.995)

    Returns
    -------
    float
        最大可行步长
    """
    alpha = 1.0
    neg_mask = dx < -1.0e-15

    if np.any(neg_mask):
        ratios = -gamma * x[neg_mask] / dx[neg_mask]
        alpha = min(alpha, float(np.min(ratios)))

    return max(alpha, 1.0e-10)


def log_barrier_gradient(x: np.ndarray) -> np.ndarray:
    """
    计算对数障碍函数 -sum ln(x_i) 的梯度.

    nabla phi(x) = -1/x_i  (逐分量)

    障碍函数将不等式约束 x > 0 转化为目标中的惩罚项,
    当 x_i -> 0+ 时, -ln(x_i) -> +inf, 阻止迭代越界.

    Parameters
    ----------
    x : ndarray
        正值向量

    Returns
    -------
    ndarray
        障碍梯度
    """
    x_safe = np.maximum(x, 1.0e-30)
    return -1.0 / x_safe


def log_barrier_hessian(x: np.ndarray) -> np.ndarray:
    """
    计算对数障碍函数 -sum ln(x_i) 的 Hessian.

    nabla^2 phi(x) = diag(1/x_i^2)

    这是一个对角矩阵, 其对角元素提供自适应缩放:
    当 x_i 接近 0 时, 1/x_i^2 很大, 提供强推力远离边界.

    Parameters
    ----------
    x : ndarray
        正值向量

    Returns
    -------
    ndarray
        障碍 Hessian (对角矩阵)
    """
    x_safe = np.maximum(x, 1.0e-30)
    return np.diag(1.0 / x_safe**2)


def complementarity_gap(x: np.ndarray, s: np.ndarray) -> float:
    """
    计算互补间隙 (duality gap).

    mu = x^T s / n = sum(x_i * s_i) / n

    互补间隙是 Interior Point Method 的核心收敛指标:
      mu -> 0 意味着 x_i * s_i -> 0 (互补松弛)
      结合可行性, 即趋向 KKT 点.

    中心路径定义为:
      C(mu) = { (x, y, s) : x_i * s_i = mu, for all i }

    算法沿中心路径邻域追踪, 逐步减小 mu -> 0.

    Parameters
    ----------
    x, s : ndarray
        原始-对偶变量

    Returns
    -------
    float
        互补间隙 mu
    """
    n = len(x)
    gap = np.dot(x, s) / n
    return max(gap, 1.0e-30)


def barrier_parameter_update(mu: float, sigma: float,
                              reduction_rate: float = 0.2) -> float:
    """
    障碍参数更新策略.

    混合策略:
      mu_new = sigma * mu   (Mehrotra)
    同时确保充分减小:
      mu_new >= (1 - reduction_rate) * mu  (防止过小)

    全局收敛性保证:
      mu_{k+1} <= (1 - delta/n) * mu_k
    其中 delta 是某个正常数, n 是维度.

    Parameters
    ----------
    mu : float
        当前障碍参数
    sigma : float
        中心参数
    reduction_rate : float
        最小缩减率

    Returns
    -------
    float
        新的障碍参数
    """
    mu_new = sigma * mu
    mu_min = (1.0 - reduction_rate) * mu * 0.01  # 极小值保护
    return max(mu_new, mu_min)
