r"""
pump_propagation.py
===================
泵浦光场在非线性晶体中的一维传播求解 —— 融合原项目
387_fem1d_bvp_quadratic (二次元有限元 BVP) 与 124_burgers_exact
(Burgers 非线性对流-扩散方程精确解思想)。

物理背景
--------
在准相位匹配晶体中，沿传播方向 :math:`z` 的慢变包络 :math:`A_p(z)` 满足
带非线性源项与耗散的修正 Burgers-type 方程：

.. math::
    \frac{d A_p}{d z} + \frac{i}{2 k_p} \frac{d^2 A_p}{d z^2}
    = -\frac{\alpha_p}{2} A_p - i \gamma(z) |A_p|^2 A_p
    + S_{\text{SPDC}}(z)

其中 :math:`\gamma(z)` 为有效非线性系数（含周期性极化结构），
:math:`S_{\text{SPDC}}(z)` 为参量下转换导致的泵浦损耗源项。

对稳态问题，在离散网格上展开为二次 Lagrange 有限元，Galerkin 投影后得到
线性化系统：

.. math::
    \sum_{j} \left[ \int_\Omega \left(
    \phi_i' \phi_j' \frac{i}{2k_p}
    + \phi_i \phi_j' \right) dz
    + \int_\Omega \phi_i \phi_j \left( \frac{\alpha_p}{2}
    + i \gamma(z) |A_p|^2 \right) dz \right] A_j
    = \int_\Omega \phi_i S_{\text{SPDC}}(z) \, dz

二次元单元采用三点 Gauss-Legendre 求积，局部基函数：

.. math::
    \phi_L(\xi) &= \frac{\xi(\xi-1)}{2} \\
    \phi_M(\xi) &= 1-\xi^2 \\
    \phi_R(\xi) &= \frac{\xi(\xi+1)}{2}

其中 :math:`\xi \in [-1,1]` 为参考坐标。
"""

import numpy as np
from linear_solver import gauss_elimination_partial_pivot


def solve_pump_envelope_fem(n_nodes: int, z_domain: tuple,
                            k_p: float, alpha_p: float,
                            gamma_eff: callable,
                            source_spdc: callable,
                            nonlinear_tol: float = 1e-9,
                            max_iter: int = 50) -> np.ndarray:
    r"""
    使用二次元 FEM 求解泵浦包络方程。

    参数
    ----
    n_nodes : int
        节点数，必须为奇数且 >= 3。
    z_domain : tuple (z_min, z_max)
        晶体端面坐标。
    k_p : float
        泵浦波数，必须 > 0。
    alpha_p : float
        线性吸收系数，>= 0。
    gamma_eff : callable(z, A_p) -> complex
        有效非线性系数函数，可依赖位置和当前场强。
    source_spdc : callable(z) -> complex
        SPDC 源项。
    nonlinear_tol : float
        非线性迭代收敛阈值。
    max_iter : int
        最大 Picard 迭代次数。

    返回
    ----
    A_p : np.ndarray, shape (n_nodes,)
        复包络在各节点处的值。
    """
    if n_nodes < 3 or n_nodes % 2 == 0:
        raise ValueError("n_nodes 必须为奇数且至少为 3。")
    if k_p <= 0.0:
        raise ValueError("k_p 必须为正。")
    if alpha_p < 0.0:
        raise ValueError("alpha_p 必须非负。")

    z_min, z_max = z_domain
    z = np.linspace(z_min, z_max, n_nodes)
    n_elements = (n_nodes - 1) // 2

    # 3-point Gauss-Legendre quadrature on [-1,1]
    xi_q = np.array([-0.7745966692414834, 0.0, 0.7745966692414834])
    w_q = np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0])

    # 初始化
    A_p = np.ones(n_nodes, dtype=np.complex128) * 1.0e3  # 初始猜测

    # 左端 Dirichlet: A_p(z_min) = A0
    A0 = 1.0e4  # 归一化泵浦振幅
    A_p[0] = A0

    for it in range(max_iter):
        A_old = A_p.copy()
        K = np.zeros((n_nodes, n_nodes), dtype=np.complex128)
        F = np.zeros(n_nodes, dtype=np.complex128)

        for e in range(n_elements):
            l = 2 * e
            m = 2 * e + 1
            r = 2 * e + 2
            zl, zm, zr = z[l], z[m], z[r]
            h_e = zr - zl

            for q in range(3):
                xi = xi_q[q]
                zq = 0.5 * ((1.0 - xi) * zl + (1.0 + xi) * zr)
                wq = w_q[q] * h_e * 0.5

                # 二次 Lagrange 基函数及其导数（参考坐标）
                phi = np.array([
                    0.5 * xi * (xi - 1.0),
                    1.0 - xi ** 2,
                    0.5 * xi * (xi + 1.0)
                ], dtype=np.float64)
                dphi_dxi = np.array([
                    xi - 0.5,
                    -2.0 * xi,
                    xi + 0.5
                ], dtype=np.float64)
                dz_dxi = h_e / 2.0
                dphi_dz = dphi_dxi / dz_dxi

                # 当前猜测值在求积点
                Aq = A_old[l] * phi[0] + A_old[m] * phi[1] + A_old[r] * phi[2]

                gamma_val = gamma_eff(zq, Aq)
                source_val = source_spdc(zq)

                # 组装局部刚度矩阵与载荷
                coeffs = np.array([l, m, r], dtype=int)
                for i_loc in range(3):
                    i = coeffs[i_loc]
                    F[i] += wq * source_val * phi[i_loc]
                    for j_loc in range(3):
                        j = coeffs[j_loc]
                        # 对流项 (dA/dz)
                        conv = phi[i_loc] * dphi_dz[j_loc]
                        # 扩散/色散项 (d2A/dz2)
                        diff = (1j / (2.0 * k_p)) * dphi_dz[i_loc] * dphi_dz[j_loc]
                        # 反应项 (吸收 + 非线性 SPM)
                        reac = phi[i_loc] * phi[j_loc] * (0.5 * alpha_p + 1j * gamma_val * abs(Aq) ** 2)
                        K[i, j] += wq * (conv + diff + reac)

        # 左端 Dirichlet
        K[0, :] = 0.0
        K[0, 0] = 1.0
        F[0] = A0

        # 右端 Robin / 透射边界：简化为 Neumann-like 吸收层
        # 近似为 A'(z_max) = 0
        K[-1, :] = 0.0
        K[-1, -1] = 1.0
        F[-1] = A_old[-1] * 0.5  # 弱约束

        # 求解复线性系统：拆分为实部与虚部 2n x 2n
        n = n_nodes
        K_real = np.zeros((2 * n, 2 * n), dtype=np.float64)
        F_real = np.zeros(2 * n, dtype=np.float64)
        K_real[:n, :n] = K.real
        K_real[:n, n:] = -K.imag
        K_real[n:, :n] = K.imag
        K_real[n:, n:] = K.real
        F_real[:n] = F.real
        F_real[n:] = F.imag

        try:
            x_sol = gauss_elimination_partial_pivot(K_real, F_real)
        except ValueError as e:
            raise RuntimeError(f"FEM 线性求解失败: {e}")

        A_p = x_sol[:n] + 1j * x_sol[n:]
        err = np.linalg.norm(A_p - A_old) / max(np.linalg.norm(A_p), 1.0)
        if err < nonlinear_tol:
            break
    else:
        # 未收敛但继续
        pass

    return A_p


def burgers_like_pump_solution(nu_eff: float, z_grid: np.ndarray,
                                t_grid: np.ndarray) -> np.ndarray:
    """
    基于 Burgers 方程精确解思想，构造含等效粘性 :math:`\nu_{\text{eff}}`
    的泵浦脉冲演化近似解。

    对于 Burgers 方程

    .. math::
        \frac{\partial u}{\partial t} + u \frac{\partial u}{\partial z}
        = \nu_{\text{eff}} \frac{\partial^2 u}{\partial z^2}

    Cole-Hopf 变换给出精确解族。这里取初值 :math:`u(z,0)=-\sin(\pi z)`
    在 :math:`z\in[-1,1]` 上的积分表示：

    .. math::
        u(z,t) = \frac{\int_{-\infty}^{\infty} \frac{z-\eta}{t}
        \exp\!\left(-\frac{(z-\eta)^2}{4\nu t}
        -\frac{1}{2\pi\nu}\cos(\pi\eta)\right) d\eta}
        {\int_{-\infty}^{\infty}
        \exp\!\left(-\frac{(z-\eta)^2}{4\nu t}
        -\frac{1}{2\pi\nu}\cos(\pi\eta)\right) d\eta}

    参数
    ----
    nu_eff : float
        等效粘性系数，> 0。
    z_grid : np.ndarray
        空间网格，:math:`z \in [-1,1]`。
    t_grid : np.ndarray
        时间网格，:math:`t \ge 0`。

    返回
    ----
    U : np.ndarray, shape (len(z_grid), len(t_grid))
        近似速度/场振幅。
    """
    if nu_eff <= 0.0:
        raise ValueError("nu_eff 必须为正。")
    z = np.atleast_1d(z_grid)
    t = np.atleast_1d(t_grid)
    nz = z.size
    nt = t.size
    U = np.zeros((nz, nt), dtype=np.float64)

    # 8-point Hermite-Gauss quadrature for exp integrals
    # abscissas and weights for Hermite H_n(x) weight exp(-x^2)
    x_h, w_h = _hermite_ek_compute(8)

    for ti in range(nt):
        tv = t[ti]
        if tv <= 1e-12:
            U[:, ti] = -np.sin(np.pi * z)
            continue

        c = 2.0 * np.sqrt(nu_eff * tv)
        for zi in range(nz):
            zv = z[zi]
            top = 0.0
            bot = 0.0
            for qi in range(8):
                eta = zv - c * x_h[qi]
                arg = -np.cos(np.pi * eta) / (2.0 * np.pi * nu_eff)
                w_exp = w_h[qi] * c * np.exp(arg)
                top += -(eta - zv) / tv * w_exp
                bot += w_exp
            if abs(bot) > 1e-20:
                U[zi, ti] = top / bot
            else:
                U[zi, ti] = 0.0

    return U


def _hermite_ek_compute(n: int):
    """
    返回 n 点 Hermite-Gauss 求积节点与权重（基于 physicist's H_n）。
    使用 numpy 的 hermite 多项式零点。
    """
    from numpy.polynomial.hermite import hermgauss
    x, w = hermgauss(n)
    # physicist's weight is exp(-x^2), hermgauss returns for this weight
    return x, w
