"""
membrane_fem.py
================
膜蛋白嵌入问题的有限元求解器。

核心数学内容：
  - 一维 Poisson-Boltzmann 方程的二次有限元离散化：
      $-\frac{d}{dx}\left(\epsilon(x)\frac{d\phi}{dx}\right) + \kappa^2(x)\phi = \rho(x)$
    其中 $\epsilon(x)$ 为介电常数，$\kappa(x)$ 为 Debye-Hückel 屏蔽参数，
    $\rho(x)$ 为电荷密度。
  - 齐次 Neumann 边界条件：
      $\frac{d\phi}{dx}\big|_{x=0} = 0$, $\frac{d\phi}{dx}\big|_{x=L} = 0$
  - 质量矩阵 M 与刚度矩阵 K 的组装
  - 反应-扩散非线性项的 FEM 离散（Allen-Cahn 型双稳态势）

种子项目映射：
  - 387_fem1d_bvp_quadratic  →  二次有限元、Gauss 积分、边界条件处理
  - 377_fem_neumann          →  Neumann 边界条件、反应-扩散非线性项
"""

import numpy as np
from typing import Callable, Tuple


# ---------------------------------------------------------------------------
# 二次有限元求解器（种子项目 387_fem1d_bvp_quadratic）
# ---------------------------------------------------------------------------
def fem1d_bvp_quadratic(
    n: int,
    a_func: Callable[[np.ndarray], np.ndarray],
    c_func: Callable[[np.ndarray], np.ndarray],
    f_func: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    left_bc: Tuple[str, float] = ("dirichlet", 0.0),
    right_bc: Tuple[str, float] = ("dirichlet", 0.0),
) -> np.ndarray:
    """
    使用二次有限元求解一维 BVP：
        $-\frac{d}{dx}\left(a(x)\frac{du}{dx}\right) + c(x) u = f(x)$

    参数边界：
        n        : 节点数，必须为奇数且 >= 3
        x        : 网格点坐标，单调递增
        a_func   : 扩散系数函数
        c_func   : 反应系数函数
        f_func   : 源项函数
        left_bc  : ("dirichlet"|"neumann", value)
        right_bc : ("dirichlet"|"neumann", value)

    返回：
        u        : 节点处的解向量
    """
    if n < 3:
        raise ValueError("fem1d_bvp_quadratic: n must be >= 3.")
    if n % 2 == 0:
        raise ValueError("fem1d_bvp_quadratic: n must be odd for quadratic elements.")
    if x.shape[0] != n:
        raise ValueError("fem1d_bvp_quadratic: x length must equal n.")
    if np.any(np.diff(x) <= 0):
        raise ValueError("fem1d_bvp_quadratic: x must be strictly increasing.")

    # 3-point Gauss-Legendre quadrature on [-1, 1]
    abscissa = np.array([-0.7745966692414834, 0.0, 0.7745966692414834], dtype=float)
    weight = np.array([0.5555555555555556, 0.8888888888888889, 0.5555555555555556], dtype=float)
    quad_num = 3

    A_mat = np.zeros((n, n), dtype=float)
    b_vec = np.zeros(n, dtype=float)

    e_num = (n - 1) // 2

    for e in range(e_num):
        l = 2 * e
        m = 2 * e + 1
        r = 2 * e + 2

        xl, xm, xr = x[l], x[m], x[r]
        h = xr - xl
        if h <= 0:
            raise ValueError("fem1d_bvp_quadratic: element length must be positive.")

        for q in range(quad_num):
            # 等参映射到物理坐标
            xi = abscissa[q]
            xq = 0.5 * ((1.0 - xi) * xl + (1.0 + xi) * xr)
            wq = weight[q] * h * 0.5

            axq = float(a_func(np.array([xq]))[0])
            cxq = float(c_func(np.array([xq]))[0])
            fxq = float(f_func(np.array([xq]))[0])

            # 二次形函数及其导数（在参考单元 [-1,1] 上）
            # N1 = xi*(xi-1)/2,  N2 = 1-xi^2,  N3 = xi*(xi+1)/2
            # dN1/dxi = xi-0.5,  dN2/dxi = -2*xi,  dN3/dxi = xi+0.5
            # dN/dx = (dN/dxi) * (2/h)
            N = np.array([0.5 * xi * (xi - 1.0), 1.0 - xi ** 2, 0.5 * xi * (xi + 1.0)], dtype=float)
            dN_dxi = np.array([xi - 0.5, -2.0 * xi, xi + 0.5], dtype=float)
            dN_dx = dN_dxi * (2.0 / h)

            # 组装局部刚度矩阵与载荷向量
            for i_local, i_global in enumerate([l, m, r]):
                for j_local, j_global in enumerate([l, m, r]):
                    A_mat[i_global, j_global] += wq * (
                        dN_dx[i_local] * axq * dN_dx[j_local]
                        + N[i_local] * cxq * N[j_local]
                    )
                b_vec[i_global] += wq * N[i_local] * fxq

    # 边界条件处理
    if left_bc[0] == "dirichlet":
        A_mat[0, :] = 0.0
        A_mat[0, 0] = 1.0
        b_vec[0] = left_bc[1]
    elif left_bc[0] == "neumann":
        # Neumann 条件自然满足（变分形式），此处无需修改矩阵
        pass
    else:
        raise ValueError("fem1d_bvp_quadratic: left_bc type must be 'dirichlet' or 'neumann'.")

    if right_bc[0] == "dirichlet":
        A_mat[-1, :] = 0.0
        A_mat[-1, -1] = 1.0
        b_vec[-1] = right_bc[1]
    elif right_bc[0] == "neumann":
        pass
    else:
        raise ValueError("fem1d_bvp_quadratic: right_bc type must be 'dirichlet' or 'neumann'.")

    # 求解线性系统
    u = np.linalg.solve(A_mat, b_vec)
    return u


# ---------------------------------------------------------------------------
# 反应-扩散方程的 FEM 离散（种子项目 377_fem_neumann）
# ---------------------------------------------------------------------------
def assemble_mass_stiffness_1d(n: int, L: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    在区间 [0, L] 上均匀划分 n 个单元（n+1 个节点），
    使用线性帽函数组装质量矩阵 M 和刚度矩阵 K。

    质量矩阵（一致质量）：
        $M_{ij} = \int_0^L \phi_i \phi_j \, dx$
    刚度矩阵：
        $K_{ij} = \int_0^L \phi_i' \phi_j' \, dx$

    参数边界：
        n >= 1, L > 0
    """
    if n < 1:
        raise ValueError("assemble_mass_stiffness_1d: n must be >= 1.")
    if L <= 0:
        raise ValueError("assemble_mass_stiffness_1d: L must be > 0.")

    h = L / n

    # 一致质量矩阵 (tridiagonal)
    M = np.zeros((n + 1, n + 1), dtype=float)
    main_diag_m = np.full(n + 1, 2.0 * h / 3.0)
    main_diag_m[0] = h / 3.0
    main_diag_m[-1] = h / 3.0
    off_diag_m = np.full(n, h / 6.0)
    M = np.diag(main_diag_m) + np.diag(off_diag_m, k=1) + np.diag(off_diag_m, k=-1)

    # 刚度矩阵 (tridiagonal)
    K = np.zeros((n + 1, n + 1), dtype=float)
    main_diag_k = np.full(n + 1, 2.0 / h)
    main_diag_k[0] = 1.0 / h
    main_diag_k[-1] = 1.0 / h
    off_diag_k = np.full(n, -1.0 / h)
    K = np.diag(main_diag_k) + np.diag(off_diag_k, k=1) + np.diag(off_diag_k, k=-1)

    return M, K


def reaction_diffusion_nonlinear(w: np.ndarray, c_array: np.ndarray,
                                  n: int, M: np.ndarray) -> np.ndarray:
    """
    计算反应-扩散方程中的非线性项 NL(w, c)。

    数学形式：
        $NL(w, c) = c_1 + c_2 w + c_3 w^2 + c_4 w^3$

    FEM 离散后的向量形式（一致质量加权）：
        $val = c_1 \mathbf{1} + c_2 M w + c_3 N_q(w) + c_4 N_c(w)$

    参数边界：
        c_array 长度 >= 4
        w 长度 == n+1
        M shape == (n+1, n+1)
    """
    if c_array.shape[0] < 4:
        raise ValueError("reaction_diffusion_nonlinear: c_array must have at least 4 elements.")
    if w.shape[0] != n + 1:
        raise ValueError("reaction_diffusion_nonlinear: w length must be n+1.")
    if M.shape != (n + 1, n + 1):
        raise ValueError("reaction_diffusion_nonlinear: M shape mismatch.")

    ones_vec = np.ones(n + 1, dtype=float)
    ones_vec[0] = 0.5
    ones_vec[-1] = 0.5
    ones_vec /= n

    Nq_val = _nonlinear_quadratic(w, n)
    Nc_val = _nonlinear_cubic(w, n)

    val = (c_array[0] * ones_vec
           + c_array[1] * (M @ w)
           + c_array[2] * Nq_val
           + c_array[3] * Nc_val)
    return val


def _nonlinear_quadratic(w: np.ndarray, n: int) -> np.ndarray:
    """
    二次非线性 FEM 项 $N_q(w)$。
    """
    w2 = w ** 2
    wx = (w[:-1] + w[1:]) ** 2
    val = np.zeros_like(w)
    val[0] = 2.0 * w2[0] + wx[0]
    val[1:-1] = wx[:-1] + 4.0 * w2[1:-1] + wx[1:]
    val[-1] = wx[-1] + 2.0 * w2[-1]
    val /= (12.0 * n)
    return val


def _nonlinear_cubic(w: np.ndarray, n: int) -> np.ndarray:
    """
    三次非线性 FEM 项 $N_c(w)$。
    """
    w2 = w ** 2
    w3 = w * w2
    wx = (w[:-1] + w[1:]) ** 3
    val = np.zeros_like(w)
    val[0] = 3.0 * w3[0] + wx[0] - w[0] * w[1] ** 2
    val[1:-1] = (wx[:-1] + 6.0 * w3[1:-1] + wx[1:]
                 - w[1:-1] * (w2[:-2] + w2[2:]))
    val[-1] = wx[-1] + 3.0 * w3[-1] - w[-1] * w[-2] ** 2
    val /= (20.0 * n)
    return val


# ---------------------------------------------------------------------------
# Poisson-Boltzmann 膜剖面求解器
# ---------------------------------------------------------------------------
def solve_poisson_boltzmann_membrane(
    n: int = 65,
    z_min: float = -30.0,  # Å
    z_max: float = 30.0,   # Å
    epsilon_water: float = 80.0,
    epsilon_protein: float = 4.0,
    epsilon_membrane: float = 2.0,
    kappa_water: float = 0.1,  # Å^{-1}
    protein_z_range: Tuple[float, float] = (-10.0, 10.0),
    membrane_z_range: Tuple[float, float] = (-15.0, -10.0),
    charge_density: Callable[[np.ndarray], np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    求解跨膜方向的 Poisson-Boltzmann 方程，得到静电势剖面。

    物理模型：
        膜法向（z 轴）被分为三个区域：
          - 水域 (z < -15 或 z > 10): 高介电常数，有屏蔽
          - 膜脂双层 (-15 <= z <= -10): 低介电常数，无屏蔽
          - 蛋白区域 (-10 < z < 10): 中等介电常数

    方程：
        $-\nabla \cdot (\epsilon(z) \nabla \phi) + \kappa^2(z) \phi = 4\pi \rho(z)$

    边界条件：
        Neumann 边界：$\frac{d\phi}{dz}\big|_{z_{\min}} = \frac{d\phi}{dz}\big|_{z_{\max}} = 0$

    返回：
        z      : 网格坐标 (Å)
        phi    : 电势 (kcal/mol/e)
        eps    : 介电常数剖面
        kappa  : 屏蔽参数剖面
    """
    if n % 2 == 0:
        n += 1
    if n < 3:
        n = 3

    z = np.linspace(z_min, z_max, n)

    def eps_profile(zz: np.ndarray) -> np.ndarray:
        eps = np.full_like(zz, epsilon_water)
        mask_mem = (zz >= membrane_z_range[0]) & (zz <= membrane_z_range[1])
        mask_prot = (zz > membrane_z_range[1]) & (zz < protein_z_range[1])
        eps[mask_mem] = epsilon_membrane
        eps[mask_prot] = epsilon_protein
        return eps

    def kappa_profile(zz: np.ndarray) -> np.ndarray:
        kap = np.full_like(zz, kappa_water)
        mask_mem = (zz >= membrane_z_range[0]) & (zz <= membrane_z_range[1])
        mask_prot = (zz > membrane_z_range[1]) & (zz < protein_z_range[1])
        kap[mask_mem] = 0.0
        kap[mask_prot] = 0.0
        return kap

    def rho_profile(zz: np.ndarray) -> np.ndarray:
        if charge_density is not None:
            return charge_density(zz)
        # 默认：高斯分布的蛋白电荷中心在 z=0
        return np.exp(-zz ** 2 / 10.0) / np.sqrt(10.0 * np.pi)

    phi = fem1d_bvp_quadratic(
        n=n,
        a_func=eps_profile,
        c_func=kappa_profile,
        f_func=rho_profile,
        x=z,
        left_bc=("neumann", 0.0),
        right_bc=("neumann", 0.0),
    )

    return z, phi, eps_profile(z), kappa_profile(z)
