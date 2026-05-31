"""
rootfinder.py
Laguerre根查找模块（对应种子项目 1430_zero_laguerre）

在光纤模式中，需要求解特征方程以确定传播常数β。
对于阶跃型光纤，标量波动方程导出如下特征方程：

  对于HE_{lm}模式:
    [J'_{l}(u)/u J_{l}(u)] + [K'_{l}(w)/w K_{l}(w)] = 0

  其中 u = a √(k₀² n_core² - β²)
       w = a √(β² - k₀² n_clad²)
       V = a k₀ √(n_core² - n_clad²) 为归一化频率

  利用关系 u² + w² = V²，可构造单变量函数 f(u) = 0。

Laguerre方法（高阶牛顿法）:
  对于多项式根查找，利用函数值及一阶、二阶导数：
    z = (f')² - (β+1) f f''
    dx = -(β+1) f / [β f' + sign(f') √z]
  其中 β = 1/(degree-1)。

  对于一般非线性方程，同样适用（假设局部近似为多项式）。
"""

import numpy as np
from scipy.special import jv, jvp, kv, kvp


def zero_laguerre(x0, degree, abserr, kmax, f):
    """
    Laguerre根查找方法。
    （对应种子项目 1430_zero_laguerre）

    参数:
        x0: float, 初始猜测
        degree: int, 预期多项式次数（用于缩放参数）
        abserr: float, 绝对误差容限
        kmax: int, 最大迭代次数
        f: callable, f(x, ider) 返回函数值(ider=0), 一阶导(ider=1), 二阶导(ider=2)

    返回:
        x: float, 根估计
        ierror: int, 0表示成功
        k: int, 迭代次数
    """
    if degree < 2:
        degree = 2

    x = x0
    ierror = 0
    k = 0
    beta = 1.0 / (degree - 1.0)

    while True:
        fx = f(x, 0)
        if abs(fx) <= abserr:
            break

        dfx = f(x, 1)
        d2fx = f(x, 2)

        k += 1
        if k > kmax:
            ierror = 2
            return x, ierror, k

        z = dfx ** 2 - (beta + 1.0) * fx * d2fx
        z = max(z, 0.0)

        bot = beta * dfx + np.sqrt(z)
        if abs(bot) < 1e-30:
            ierror = 3
            return x, ierror, k

        dx = -(beta + 1.0) * fx / bot
        x = x + dx

        if not np.isfinite(x):
            ierror = 4
            return x, ierror, k

    return x, ierror, k


def fiber_mode_characteristic_eq(u, V, l_mode, n_core, n_clad):
    """
    构造阶跃型光纤模式特征方程及其导数。

    参数:
        u: float, 横向相位参数
        V: float, 归一化频率
        l_mode: int, 方位角模数
        n_core, n_clad: float, 折射率

    返回:
        f_val: float
        f_prime: float
        f_double_prime: float
    """
    if u <= 0 or u >= V:
        return 1e10, 0.0, 0.0

    w = np.sqrt(max(V ** 2 - u ** 2, 0.0))
    if w < 1e-15:
        return 1e10, 0.0, 0.0

    # 使用Bessel函数
    Ju = jv(l_mode, u)
    Jpu = jvp(l_mode, u, 1)
    Kw = kv(l_mode, w)
    Kpw = kvp(l_mode, w, 1)

    if abs(Ju) < 1e-30 or abs(Kw) < 1e-30:
        return 1e10, 0.0, 0.0

    # 标量近似特征方程（弱导近似）:
    # u J_{l+1}(u) / J_l(u) = w K_{l+1}(w) / K_l(w)
    # 利用 J'_l(u) = -J_{l+1}(u) + (l/u) J_l(u)
    # 和 K'_l(w) = -K_{l+1}(w) + (l/w) K_l(w)
    # 等价于: u Jpu / J_l(u) - l = w Kpw / K_l(w) - l
    # 即: u Jpu / Ju = w Kpw / Kw
    # 因此特征方程为: u * Jpu / Ju - w * Kpw / Kw = 0
    term1 = u * Jpu / Ju
    term2 = w * Kpw / Kw
    f_val = term1 - term2

    # 数值微分求导数（避免解析导数过于复杂）
    du = max(1e-8 * u, 1e-10)
    fu_plus = fiber_mode_characteristic_eq_scalar(u + du, V, l_mode)
    fu_minus = fiber_mode_characteristic_eq_scalar(u - du, V, l_mode)
    fu_plus2 = fiber_mode_characteristic_eq_scalar(u + 2 * du, V, l_mode)
    fu_minus2 = fiber_mode_characteristic_eq_scalar(u - 2 * du, V, l_mode)

    f_prime = (fu_plus - fu_minus) / (2.0 * du)
    f_double_prime = (fu_plus2 - 2.0 * f_val + fu_minus2) / (du ** 2)

    return f_val, f_prime, f_double_prime


def fiber_mode_characteristic_eq_scalar(u, V, l_mode):
    """纯函数值版本，用于数值微分。"""
    if u <= 0 or u >= V:
        return 1e10
    w = np.sqrt(max(V ** 2 - u ** 2, 0.0))
    if w < 1e-15:
        return 1e10
    Ju = jv(l_mode, u)
    Jpu = jvp(l_mode, u, 1)
    Kw = kv(l_mode, w)
    Kpw = kvp(l_mode, w, 1)
    if abs(Ju) < 1e-30 or abs(Kw) < 1e-30:
        return 1e10
    return u * Jpu / Ju - w * Kpw / Kw


def find_fiber_mode_roots(V, l_mode, n_core, n_clad, n_roots=5):
    """
    查找阶跃型光纤的模式根u。

    返回:
        roots: list of float, 找到的u值
        betas: list of float, 对应的传播常数
    """
    roots = []
    betas = []
    k0 = 2.0 * np.pi / 1550e-9  # 假设1550nm

    # 在(0, V)区间内搜索
    n_scan = max(200, int(V * 50))
    u_scan = np.linspace(0.01 * V, 0.99 * V, n_scan)
    f_scan = np.array([fiber_mode_characteristic_eq_scalar(u, V, l_mode) for u in u_scan])

    # 检测符号变化
    for i in range(n_scan - 1):
        if np.isfinite(f_scan[i]) and np.isfinite(f_scan[i + 1]):
            if f_scan[i] * f_scan[i + 1] < 0:
                x0 = 0.5 * (u_scan[i] + u_scan[i + 1])

                # 使用二分法（更稳健）
                u_left = u_scan[i]
                u_right = u_scan[i + 1]
                f_left = f_scan[i]
                f_right = f_scan[i + 1]
                root = None
                for _ in range(60):  # 二分迭代
                    u_mid = 0.5 * (u_left + u_right)
                    f_mid = fiber_mode_characteristic_eq_scalar(u_mid, V, l_mode)
                    if f_mid == 0:
                        root = u_mid
                        break
                    if f_left * f_mid < 0:
                        u_right = u_mid
                        f_right = f_mid
                    else:
                        u_left = u_mid
                        f_left = f_mid
                if root is None:
                    root = 0.5 * (u_left + u_right)

                if 0 < root < V:
                    # 去重
                    is_new = True
                    for r in roots:
                        if abs(r - root) < 1e-6:
                            is_new = False
                            break
                    if is_new:
                        roots.append(root)
                        # 计算β
                        a = 4e-6  # 假设芯径4um
                        w = np.sqrt(V ** 2 - root ** 2)
                        beta = np.sqrt((k0 * n_core) ** 2 - (root / a) ** 2)
                        betas.append(beta)

        if len(roots) >= n_roots:
            break

    return roots, betas


def degree_estimation(V):
    """粗略估计特征方程的有效多项式次数。"""
    return min(max(int(V), 2), 20)
