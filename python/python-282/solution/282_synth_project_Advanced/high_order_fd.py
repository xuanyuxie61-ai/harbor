"""
high_order_fd.py
================

高阶有限差分算子构造模块（2阶、4阶中心差分 + Gram 多项式重构）。

融合种子项目：
    360_fd1d_heat_explicit：二阶中心差分模板
    479_gram_polynomial：Gram 正交多项式递推与求值
    1402_wave_pde：周期边界 Laplacian

数学基础：
    二阶导数 2 阶精度：
        f''(x_i) ≈ (f_{i-1} - 2 f_i + f_{i+1}) / dx^2   O(dx^2)

    二阶导数 4 阶精度：
        f''(x_i) ≈ (-f_{i-2} + 16 f_{i-1} - 30 f_i + 16 f_{i+1} - f_{i+2})
                    / (12 dx^2)                            O(dx^4)

    Gram 多项式递推（离散正交）：
        P_0(x) = 1
        P_1(x) = x
        P_{k+1}(x) = x P_k(x) - beta_k P_{k-1}(x)
    其中 beta_k = <x P_k, P_k> / <P_k, P_k>

作者: DA-Synthesis
"""

import math
try:
    from . import sei_parameters as P
except ImportError:
    import sei_parameters as P


# ============================================================
#  基本有限差分算子
# ============================================================

def laplacian_2nd_order(u, dx, periodic=False):
    """
    二阶精度中心差分 Laplacian（参考 360_fd1d_heat_explicit）。

    数学形式：
        (L u)_i = (u_{i-1} - 2 u_i + u_{i+1}) / dx^2

    边界处理：
        - 非周期：边界使用一阶单侧差分
        - 周期：  u_0 = u_N（参考 1402_wave_pde laplacian_pp）

    Parameters
    ----------
    u : list[float]
        场变量在网格节点上的值。
    dx : float
        空间步长。
    periodic : bool
        是否使用周期边界。

    Returns
    -------
    lu : list[float]
        Laplacian 在各节点的值。
    """
    n = len(u)
    lu = [0.0] * n

    if periodic:
        # 周期边界（参考 1402_wave_pde laplacian_pp.m）
        lu[0] = (u[n - 2] - 2.0 * u[0] + u[1]) / (dx * dx)
        for i in range(1, n - 1):
            lu[i] = (u[i - 1] - 2.0 * u[i] + u[i + 1]) / (dx * dx)
        lu[n - 1] = (u[n - 2] - 2.0 * u[n - 1] + u[1]) / (dx * dx)
    else:
        # Dirichlet 边界（边界值保持 0）
        for i in range(1, n - 1):
            lu[i] = (u[i - 1] - 2.0 * u[i] + u[i + 1]) / (dx * dx)
        lu[0] = 0.0
        lu[n - 1] = 0.0

    return lu


def laplacian_4th_order(u, dx):
    """
    四阶精度中心差分 Laplacian。

    数学形式（5 点模板）：
        f''(x_i) ≈ (-f_{i-2} + 16 f_{i-1} - 30 f_i + 16 f_{i+1} - f_{i+2})
                    / (12 dx^2)

    截断误差：O(dx^4)

    边界处理：
        对 i < 2 和 i > N-3 的节点，退化到二阶模板。

    Parameters
    ----------
    u : list[float]
        场变量值。
    dx : float
        空间步长。

    Returns
    -------
    lu : list[float]
        四阶 Laplacian 值。
    """
    n = len(u)
    lu = [0.0] * n
    inv_dx2 = 1.0 / (dx * dx)

    for i in range(n):
        if i < 2 or i > n - 3:
            # 边界附近退化到二阶
            if 1 <= i <= n - 2:
                lu[i] = (u[i - 1] - 2.0 * u[i] + u[i + 1]) * inv_dx2
            else:
                lu[i] = 0.0
        else:
            # 5 点四阶模板
            lu[i] = (-u[i - 2] + 16.0 * u[i - 1] - 30.0 * u[i]
                     + 16.0 * u[i + 1] - u[i + 2]) / (12.0 * dx * dx)

    return lu


def first_derivative_4th_order(u, dx):
    """
    四阶精度中心差分之一阶导数。

    数学形式：
        f'(x_i) ≈ (f_{i-2} - 8 f_{i-1} + 8 f_{i+1} - f_{i+2}) / (12 dx)

    Parameters
    ----------
    u : list[float]
        场变量值。
    dx : float
        空间步长。

    Returns
    -------
    du : list[float]
        一阶导数值。
    """
    n = len(u)
    du = [0.0] * n
    inv_12dx = 1.0 / (12.0 * dx)

    for i in range(n):
        if i < 2 or i > n - 3:
            # 退化到二阶中心差分
            if 1 <= i <= n - 2:
                du[i] = (u[i + 1] - u[i - 1]) / (2.0 * dx)
            else:
                du[i] = 0.0
        else:
            du[i] = (u[i - 2] - 8.0 * u[i - 1]
                     + 8.0 * u[i + 1] - u[i + 2]) * inv_12dx
    return du


# ============================================================
#  Gram 正交多项式（参考 479_gram_polynomial）
# ============================================================

def gram_beta(k, m):
    """
    Gram 多项式递推系数 beta_k（参考 479_gram_polynomial/gram_beta.m）。

    数学形式：
        beta_k = (k^2 * (m^2 - k^2)) / (4 * (4 k^2 - 1))

    Parameters
    ----------
    k : int
        多项式阶数。
    m : int
        离散点总数。

    Returns
    -------
    float
        beta_k 系数。
    """
    if k <= 0:
        return 0.0
    k2 = k * k
    m2 = m * m
    return (k2 * (m2 - k2)) / (4.0 * (4.0 * k2 - 1.0))


def gram_polynomial_evaluate(n_order, m_points, x):
    """
    计算 Gram 正交多项式 P_n(x) 在点 x 处的值（参考 479）。

    使用三项递推：
        P_0(x) = 1
        P_1(x) = x
        P_{k+1}(x) = x * P_k(x) - beta_k * P_{k-1}(x)

    Parameters
    ----------
    n_order : int
        多项式阶数 n < m。
    m_points : int
        离散点总数。
    x : list[float]
        求值点（通常在 [-1, 1] 上）。

    Returns
    -------
    g : list[float]
        多项式在 x 各点的值。
    """
    k_pts = len(x)
    pnm1 = [0.0] * k_pts
    pn = [1.0] * k_pts
    pnp1 = [xi for xi in x]

    for step in range(n_order):
        if step == 0:
            pn = [0.0] * k_pts
            pnp1 = [1.0] * k_pts
        else:
            pnm1 = list(pn)
            pn = list(pnp1)
            bk = gram_beta(step - 1, m_points)
            pnp1 = [x[j] * pn[j] - bk * pnm1[j] for j in range(k_pts)]

    return list(pnp1)


def gram_inner_product(f, g):
    """
    离散 Gram 内积 <f, g> = sum_i f_i * g_i。

    Parameters
    ----------
    f, g : list[float]
        两个离散函数。

    Returns
    -------
    float
        内积值。
    """
    return sum(fi * gi for fi, gi in zip(f, g))


def gram_projection(f_values, j_order, m_points, abscissas):
    """
    函数 f 到第 j 个 Gram 多项式上的投影系数（参考 479 gram_projection）。

    数学形式：
        c_j = <f, P_j> / <P_j, P_j>

    Parameters
    ----------
    f_values : list[float]
        函数在离散点上的值。
    j_order : int
        投影多项式阶数。
    m_points : int
        离散点总数。
    abscissas : list[float]
        离散求值点。

    Returns
    -------
    float
        投影系数 c_j。
    """
    gj = gram_polynomial_evaluate(j_order, m_points, abscissas)
    numerator = gram_inner_product(f_values, gj)
    denominator = gram_inner_product(gj, gj)
    if abs(denominator) < 1.0e-30:
        return 0.0
    return numerator / denominator


def reconstruct_with_gram(u, dx, gram_order):
    """
    使用 Gram 多项式重构场变量的高阶表示，然后计算二阶导数。

    思想：在局部窗口内将 u(x) 投影到 Gram 多项式基上，
    利用解析导数提高精度。对于偶数阶多项式 P_{2k}，
    在中心点 x=0 处 P_{2k}''(0) = (2k)(2k-1) * (leading coefficient)。

    Parameters
    ----------
    u : list[float]
        场变量值。
    dx : float
        空间步长。
    gram_order : int
        Gram 多项式阶数。

    Returns
    -------
    lu_gram : list[float]
        通过 Gram 重构得到的 Laplacian。
    """
    n = len(u)
    lu_gram = [0.0] * n
    half_win = gram_order + 1

    for i in range(half_win, n - half_win):
        # 局部窗口
        window_size = 2 * half_win + 1
        x_local = [(j - half_win) * dx for j in range(window_size)]
        # 归一化到 [-1, 1]
        scale = half_win * dx
        x_norm = [xi / scale for xi in x_local]
        u_local = [u[i - half_win + j] for j in range(window_size)]

        # 投影到 Gram 基
        coeffs = []
        for k in range(gram_order + 1):
            ck = gram_projection(u_local, k, window_size, x_norm)
            coeffs.append(ck)

        # 在中心点 x = 0 处的二阶导数
        # 通过有限差分近似 P_k''(0)
        h = 1.0e-4
        d2u_center = 0.0
        for k in range(gram_order + 1):
            p_plus = gram_polynomial_evaluate(k, window_size, [h])[0]
            p_zero = gram_polynomial_evaluate(k, window_size, [0.0])[0]
            p_minus = gram_polynomial_evaluate(k, window_size, [-h])[0]
            pk_d2 = (p_plus - 2.0 * p_zero + p_minus) / (h * h)
            d2u_center += coeffs[k] * pk_d2

        # 从归一化坐标变换回物理坐标
        d2u_center = d2u_center / (scale * scale)
        lu_gram[i] = d2u_center

    # 边界退化到二阶中心差分
    for i in range(1, half_win):
        lu_gram[i] = (u[i - 1] - 2.0 * u[i] + u[i + 1]) / (dx * dx)
    for i in range(n - half_win, n - 1):
        lu_gram[i] = (u[i - 1] - 2.0 * u[i] + u[i + 1]) / (dx * dx)
    lu_gram[0] = 0.0
    lu_gram[n - 1] = 0.0

    return lu_gram


# ============================================================
#  统一接口
# ============================================================

def compute_laplacian(u, dx, scheme="4th_order"):
    """
    统一 Laplacian 计算接口。

    Parameters
    ----------
    u : list[float]
        场变量。
    dx : float
        空间步长。
    scheme : str
        差分格式: '2nd_order', '4th_order', 'gram'。

    Returns
    -------
    list[float]
        Laplacian 值。
    """
    if scheme == "2nd_order":
        return laplacian_2nd_order(u, dx)
    elif scheme == "4th_order":
        return laplacian_4th_order(u, dx)
    elif scheme == "gram":
        return reconstruct_with_gram(u, dx, P.GRAM_ORDER)
    else:
        raise ValueError(f"未知差分格式: {scheme}")


if __name__ == "__main__":
    # 简单验证：对已知函数 f(x) = sin(2*pi*x/L)，f'' = -(2*pi/L)^2 sin
    L = P.L_DOMAIN
    x = [i * P.DX for i in range(P.N_GRID)]
    k_wave = 2.0 * math.pi / L
    u_exact = [math.sin(k_wave * xi) for xi in x]
    lu_exact = [-(k_wave ** 2) * math.sin(k_wave * xi) for xi in x]

    lu_2 = laplacian_2nd_order(u_exact, P.DX)
    lu_4 = laplacian_4th_order(u_exact, P.DX)

    err_2 = max(abs(lu_2[i] - lu_exact[i]) for i in range(P.N_GRID))
    err_4 = max(abs(lu_4[i] - lu_exact[i]) for i in range(P.N_GRID))

    print(f"[high_order_fd] 验证 f(x) = sin(2πx/L) 的二阶导数")
    print(f"  二阶精度误差: {err_2:.6e}")
    print(f"  四阶精度误差: {err_4:.6e}")
    print(f"  精度提升比: {err_2 / max(err_4, 1e-30):.1f}")
