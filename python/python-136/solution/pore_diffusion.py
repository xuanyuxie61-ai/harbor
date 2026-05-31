r"""
pore_diffusion.py
=================
催化剂孔道与颗粒内的扩散-反应方程数值求解。

基于种子项目 358_fd1d_bvp 与 386_fem1d_bvp_linear 重构：
- fd1d_bvp 使用有限差分法（FDM）求解一维边值问题；
- fem1d_bvp_linear 使用有限元法（FEM，分段线性基函数）求解同类问题。

在本系统中求解的核心方程为：

    球形颗粒内的稳态扩散-反应方程：
        \frac{D_e}{r^2} \frac{d}{dr}\left(r^2 \frac{dC}{dr}\right)
        - R(C, T) = 0,    0 < r < R

    等价展开形式：
        D_e \left( \frac{d^2C}{dr^2} + \frac{2}{r} \frac{dC}{dr} \right)
        - R(C, T) = 0

    边界条件：
        \frac{dC}{dr}\bigg|_{r=0} = 0   （对称性）
        C(R) = C_{bulk}                （表面浓度）

    有限元弱形式（乘以测试函数 v 并分部积分）：
        \int_0^R D_e \frac{dC}{dr} \frac{dv}{dr} r^2 dr
        + \int_0^R R(C,T) v r^2 dr = 0

    有限差分离散（非均匀网格）：
        D_e \left[
            \frac{C_{i+1} - 2C_i + C_{i-1}}{\Delta r^2}
            + \frac{2}{r_i} \frac{C_{i+1} - C_{i-1}}{2\Delta r}
        \right] - R_i = 0

所有求解器均包含奇异点 r=0 的边界处理与数值稳定性保障。
"""

import numpy as np
from linear_solvers import solve_tridiagonal, solve_sparse_system


class PoreDiffusionError(Exception):
    """扩散-反应求解异常。"""
    pass


def _symmetry_bc_fd(C0, C1, dr):
    r"""
    在 r=0 处使用对称边界条件的有限差分近似。

    由洛必达法则：
        \lim_{r\to 0} \frac{1}{r^2} \frac{d}{dr}\left(r^2 \frac{dC}{dr}\right)
        = 3 \frac{d^2C}{dr^2}\bigg|_{r=0}

    因此 r=0 处的离散方程为：
        3 D_e \frac{C_1 - 2C_0 + C_{-1}}{dr^2} - R_0 = 0
    结合对称性 C_{-1} = C_1，得：
        6 D_e \frac{C_1 - C_0}{dr^2} - R_0 = 0
    """
    return 6.0 * (C1 - C0) / (dr ** 2)


def solve_diffusion_reaction_fd(r_nodes, D_e, reaction_func,
                                C_surface, max_iter=100, tol=1e-10):
    """
    使用有限差分法求解非线性扩散-反应方程。

    Parameters
    ----------
    r_nodes : ndarray
        径向节点，必须包含 0 和 R，且已排序。
    D_e : float
        有效扩散系数 [m²/s]。
    reaction_func : callable
        函数签名 f(C, r) -> float，返回局部反应消耗速率 [mol/(m³·s)]。
    C_surface : float
        表面浓度 [mol/m³]。
    max_iter : int
        非线性迭代最大次数。
    tol : float
        收敛容忍。

    Returns
    -------
    C : ndarray
        径向浓度分布 [mol/m³]。
    info : dict
        包含迭代次数与最终残差。
    """
    n = r_nodes.size
    if n < 3:
        raise PoreDiffusionError("节点数至少为 3")
    if r_nodes[0] != 0.0:
        raise PoreDiffusionError("第一个节点必须为 r=0")

    C = np.linspace(C_surface * 0.5, C_surface, n)
    C[-1] = C_surface

    for it in range(max_iter):
        # 构造三对角系统
        a_diag = np.zeros(n)
        b_sub = np.zeros(n - 1)
        c_sup = np.zeros(n - 1)
        rhs = np.zeros(n)

        # 内部节点
        for i in range(1, n - 1):
            rm = r_nodes[i]
            dr_p = r_nodes[i + 1] - r_nodes[i]
            dr_m = r_nodes[i] - r_nodes[i - 1]
            dr_avg = 0.5 * (dr_p + dr_m)

            # 系数 a(r) = D_e * r^2
            am = D_e * (rm ** 2)
            ap = D_e * (rm ** 2)

            # 中心差分：-d/dr(a dC/dr) 的离散
            # 非均匀网格公式
            coeff_m = -2.0 * am / (dr_m * (dr_p + dr_m))
            coeff_p = -2.0 * ap / (dr_p * (dr_p + dr_m))
            coeff_c = 2.0 * am / (dr_m * dr_p)

            # 实际方程为 D_e * Laplacian(C) - R = 0
            # 在球坐标：D_e * (d²C/dr² + 2/r dC/dr)
            # 使用守恒形式更稳定：1/r² d/dr(r² D_e dC/dr)
            # 采用有限体积法离散
            r_plus = 0.5 * (r_nodes[i] + r_nodes[i + 1])
            r_minus = 0.5 * (r_nodes[i] + r_nodes[i - 1])
            flux_plus = r_plus ** 2 * D_e * (C[i + 1] - C[i]) / dr_p
            flux_minus = r_minus ** 2 * D_e * (C[i] - C[i - 1]) / dr_m

            vol = (r_nodes[i] ** 2) * dr_avg  # 近似体积元

            a_diag[i] = (r_plus ** 2 * D_e / dr_p + r_minus ** 2 * D_e / dr_m) / vol
            c_sup[i] = -(r_plus ** 2 * D_e / dr_p) / vol
            b_sub[i - 1] = -(r_minus ** 2 * D_e / dr_m) / vol
            rhs[i] = -reaction_func(C[i], rm)

        # r=0 边界：对称性
        # 近似：3 D_e (C_1 - C_0) / (dr^2) * 2 = -reaction_0
        dr0 = r_nodes[1] - r_nodes[0]
        a_diag[0] = 6.0 * D_e / (dr0 ** 2)
        c_sup[0] = -6.0 * D_e / (dr0 ** 2)
        rhs[0] = -reaction_func(C[0], 0.0)

        # r=R 边界：Dirichlet
        a_diag[-1] = 1.0
        b_sub[-1] = 0.0
        rhs[-1] = C_surface

        # 解线性系统
        C_new = solve_tridiagonal(a_diag, b_sub, c_sup, rhs)

        # 阻尼牛顿/不动点迭代
        relax = 0.7
        C = relax * C_new + (1.0 - relax) * C
        C[-1] = C_surface  # 强制边界

        # 检查收敛
        change = np.linalg.norm(C_new - C) / max(np.linalg.norm(C), 1e-12)
        if change < tol:
            return C, {"iter": it + 1, "resid": change}

    return C, {"iter": max_iter, "resid": change}


def solve_diffusion_reaction_fem(r_nodes, D_e, reaction_func, C_surface):
    r"""
    使用有限元法（分段线性基函数）求解扩散-反应方程。

    弱形式（球坐标）：
        \int_0^R D_e \frac{dC}{dr} \frac{dv}{dr} r^2 dr
        + \int_0^R R(C) v r^2 dr = 0

    单元刚度矩阵（局部坐标 ξ ∈ [-1, 1]）：
        K_{ij}^{(e)} = \int_{-1}^{1} D_e \frac{dN_i}{d\xi} \frac{dN_j}{d\xi}
                       \left(\frac{d\xi}{dr}\right)^2 r(\xi)^2 \frac{dr}{d\xi} d\xi

    采用两点 Gauss-Legendre 数值积分。

    Parameters
    ----------
    r_nodes : ndarray
        径向节点。
    D_e : float
    reaction_func : callable
        f(C, r) -> float。
    C_surface : float

    Returns
    -------
    C : ndarray
    info : dict
    """
    n = r_nodes.size
    if n < 3:
        raise PoreDiffusionError("节点数至少为 3")

    # 两点 Gauss-Legendre
    xi_q = np.array([-1.0, 1.0]) / np.sqrt(3.0)
    w_q = np.array([1.0, 1.0])

    A = np.zeros((n, n))
    b_vec = np.zeros(n)

    e_num = n - 1
    for e in range(e_num):
        l = e
        r = e + 1
        xl = r_nodes[l]
        xr = r_nodes[r]
        h = xr - xl

        for q in range(2):
            xi = xi_q[q]
            rq = 0.5 * ((1.0 - xi) * xl + (1.0 + xi) * xr)
            w = w_q[q] * h / 2.0

            # 形函数及其导数
            Nl = 0.5 * (1.0 - xi)
            Nr = 0.5 * (1.0 + xi)
            dNldr = -1.0 / h
            dNrdr = 1.0 / h

            # 刚度矩阵项
            A[l, l] += w * D_e * dNldr * dNldr * (rq ** 2)
            A[l, r] += w * D_e * dNldr * dNrdr * (rq ** 2)
            A[r, l] += w * D_e * dNrdr * dNldr * (rq ** 2)
            A[r, r] += w * D_e * dNrdr * dNrdr * (rq ** 2)

            # 右端项（反应源项）
            Cq = Nl * C_surface * 0.5 + Nr * C_surface * 0.5  # 初始猜测
            Rq = reaction_func(Cq, rq)
            # 弱形式推导：∫ D dC/dr dv/dr r² dr = -∫ R v r² dr
            b_vec[l] -= w * Rq * Nl * (rq ** 2)
            b_vec[r] -= w * Rq * Nr * (rq ** 2)

    # 边界条件
    # r=0：自然边界条件（对称性）在弱形式中自动满足，无需强制
    # 但为避免奇点，保持第一行作为通式参与
    # 这里采用软约束：C[0] ≈ C[1]（近似零梯度）
    A[0, :] = 0.0
    A[0, 0] = 1.0
    A[0, 1] = -1.0
    b_vec[0] = 0.0

    A[-1, :] = 0.0
    A[-1, -1] = 1.0
    b_vec[-1] = C_surface

    C = solve_sparse_system(A, b_vec)
    return C, {"method": "FEM"}


def diffusion_flux_at_surface(C, r_nodes, D_e):
    r"""
    计算颗粒表面的扩散通量 [mol/(m²·s)]。

        J = -D_e \left.\frac{dC}{dr}\right|_{r=R}

    Parameters
    ----------
    C : ndarray
    r_nodes : ndarray
    D_e : float

    Returns
    -------
    flux : float
    """
    n = r_nodes.size
    dr = r_nodes[-1] - r_nodes[-2]
    dCdr = (C[-1] - C[-2]) / dr
    flux = -D_e * dCdr
    return flux


def effectiveness_factor_from_profile(C, r_nodes, R, reaction_func):
    r"""
    从浓度剖面计算内部效率因子。

        \eta = \frac{\int_0^R R(C(r)) 4\pi r^2 dr}
                    {R(C_{surf}) \cdot \frac{4}{3}\pi R^3}

    Parameters
    ----------
    C : ndarray
    r_nodes : ndarray
    R : float
    reaction_func : callable
        f(C, r) -> float。

    Returns
    -------
    eta : float
    """
    rates = np.array([reaction_func(Ci, ri) for Ci, ri in zip(C, r_nodes)])
    vol_int = np.trapezoid(rates * 4.0 * np.pi * r_nodes ** 2, r_nodes)
    bulk_rate = reaction_func(C[-1], R)
    if abs(bulk_rate) < np.finfo(float).eps:
        return 1.0
    denom = bulk_rate * (4.0 / 3.0) * np.pi * R ** 3
    eta = vol_int / denom
    return float(max(eta, 0.0))
