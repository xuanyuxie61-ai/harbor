"""
equilibrium_solver.py
Grad-Shafranov 方程求解与磁面重构。

核心物理模型：
  在轴对称托卡马克中，磁通函数 ψ(R,Z) 满足 Grad-Shafranov 方程：

      Δ*ψ = R² ∇·( (1/R²) ∇ψ )
          = -μ₀ R² dp/dψ - F dF/dψ

  其中 p = p(ψ) 为等离子体压强，F = R B_φ = F(ψ) 为极向电流函数。

  本模块采用固定边界迭代法（Picard 迭代），在 (R,Z) 极坐标网格上
  通过有限差分近似求解。压力剖面与电流剖面采用典型 ITER 型参数化：

      p(ψ_n) = p_0 (1 - ψ_n^{α_p})^{β_p}
      F²(ψ_n) = R_0² B_0² [ 1 + (β_0 / (ε² κ²)) (1 - (1 - ψ_n^{α_I})^{β_I}) ]

  归一化磁通 ψ_n = (ψ - ψ_axis) / (ψ_edge - ψ_axis)。

边界处理：
  - 等离子体边界采用 Miller 参数化描述：
      R(θ) = R_0 + a cos(θ + δ sin θ)
      Z(θ) = κ a sin θ
  - Dirichlet 边界条件：ψ|_boundary = ψ_edge
"""

import numpy as np
from parameters import (
    MU0, R0, a_minor, B0, KAPPA, DELTA,
    NR_EQUIL, NTHETA_EQUIL
)


def miller_boundary(theta, R0=R0, a=a_minor, kappa=KAPPA, delta=DELTA):
    """
    Miller 参数化边界。

    参数
    ------
    theta : array_like
        极向角 [rad]。
    R0, a, kappa, delta : float
        Miller 几何参数。

    返回
    ------
    R, Z : ndarray
        边界上的 (R, Z) 坐标。
    """
    theta = np.asarray(theta, dtype=float)
    R = R0 + a * np.cos(theta + delta * np.sin(theta))
    Z = kappa * a * np.sin(theta)
    return R, Z


def pressure_profile(psi_norm, p0=1.0e5, alpha_p=2.0, beta_p=1.5):
    """
    等离子体压强剖面。

    公式
    ----
        p(ψ_n) = p_0 [ 1 - ψ_n^{α_p} ]^{β_p}

    参数
    ------
    psi_norm : ndarray
        归一化磁通，范围 [0, 1]。
    p0 : float
        轴心压强 [Pa]。
    alpha_p, beta_p : float
        剖面形状参数。

    返回
    ------
    p : ndarray
        压强值 [Pa]。
    """
    psi_norm = np.clip(np.asarray(psi_norm, dtype=float), 0.0, 1.0)
    val = 1.0 - np.power(psi_norm, alpha_p)
    val = np.maximum(val, 0.0)
    return p0 * np.power(val, beta_p)


def f_profile(psi_norm, R0=R0, B0=B0, epsilon=a_minor / R0,
              beta_pol=0.5, alpha_I=2.0, beta_I=1.5):
    """
    极向电流函数 F(ψ) = R B_φ。

    公式
    ----
        F²(ψ_n) = R_0² B_0² [ 1 + ν (1 - (1 - ψ_n^{α_I})^{β_I}) ]
        ν = β_pol / (ε² κ²)

    参数
    ------
    psi_norm : ndarray
        归一化磁通。
    R0, B0 : float
        轴心大半径与磁场。
    epsilon : float
        逆纵横比 a / R_0。
    beta_pol : float
        极向比压。
    alpha_I, beta_I : float
        电流剖面形状参数。

    返回
    ------
    F : ndarray
        极向电流函数 [T·m]。
    """
    # TODO: 实现极向电流函数 F(ψ) = R B_φ 的计算
    # 公式：F²(ψ_n) = R_0² B_0² [ 1 + ν (1 - (1 - ψ_n^{α_I})^{β_I}) ]
    # 其中 ν = β_pol / (ε² κ²)
    # 需要处理 psi_norm 的归一化、边界截断等数值稳定性问题
    raise NotImplementedError("此处需补全 f_profile 的科学公式实现")


def gs_operator(psi, R_grid, Z_grid):
    """
    Grad-Shafranov 椭圆算子 Δ*ψ 的有限差分近似。

    公式
    ----
        Δ*ψ = R ∂/∂R ( (1/R) ∂ψ/∂R ) + ∂²ψ/∂Z²

    采用二阶中心差分：
        ∂ψ/∂R|_{i,j} ≈ (ψ_{i+1,j} - ψ_{i-1,j}) / (2 dR)
        ∂²ψ/∂R²|_{i,j} ≈ (ψ_{i+1,j} - 2ψ_{i,j} + ψ_{i-1,j}) / dR²
        ∂²ψ/∂Z²|_{i,j} ≈ (ψ_{i,j+1} - 2ψ_{i,j} + ψ_{i,j-1}) / dZ²

    参数
    ------
    psi : ndarray, shape (nr, nz)
        磁通函数网格值。
    R_grid : ndarray, shape (nr,)
        径向坐标网格 [m]。
    Z_grid : ndarray, shape (nz,)
        垂直坐标网格 [m]。

    返回
    ------
    residual : ndarray, shape (nr, nz)
        Δ*ψ 的离散近似。
    """
    nr, nz = psi.shape
    if nr < 3 or nz < 3:
        raise ValueError("网格维度必须至少为 3×3")
    dR = R_grid[1] - R_grid[0]
    dZ = Z_grid[1] - Z_grid[0]
    if dR <= 0 or dZ <= 0:
        raise ValueError("网格间距必须为正")

    residual = np.zeros_like(psi)
    R = R_grid[:, np.newaxis]

    # 内部点二阶差分
    for i in range(1, nr - 1):
        for j in range(1, nz - 1):
            dpsi_dR = (psi[i + 1, j] - psi[i - 1, j]) / (2.0 * dR)
            d2psi_dR2 = (psi[i + 1, j] - 2.0 * psi[i, j] + psi[i - 1, j]) / (dR ** 2)
            d2psi_dZ2 = (psi[i, j + 1] - 2.0 * psi[i, j] + psi[i, j - 1]) / (dZ ** 2)
            # Δ*ψ = R * (d/dR (1/R dψ/dR)) + d²ψ/dZ²
            #      = d²ψ/dR² - (1/R) dψ/dR + d²ψ/dZ²
            residual[i, j] = d2psi_dR2 - (1.0 / R[i, 0]) * dpsi_dR + d2psi_dZ2

    return residual


def solve_grad_shafranov(max_iter=500, tol=1e-8, relaxation=0.3,
                         nr=NR_EQUIL, nz=NTHETA_EQUIL):
    """
    Picard 迭代求解固定边界 Grad-Shafranov 方程。

    算法
    ----
    1. 在矩形计算域 [R_min, R_max] × [Z_min, Z_max] 上建立网格。
    2. 初始化 ψ 为抛物线型试探函数。
    3. 每次迭代：
         a) 根据当前 ψ 计算归一化磁通 ψ_n。
         b) 计算源项 S = -μ₀ R² dp/dψ - F dF/dψ。
         c) 求解 Poisson 方程 Δ*ψ^{new} = S。
         d) 施加松弛与边界条件。
         e) 检查收敛性 ‖ψ^{new} - ψ^{old}‖_∞ < tol。

    参数
    ------
    max_iter : int
        最大迭代次数。
    tol : float
        收敛容差。
        
    relaxation : float
        Picard 松弛因子 (0 < ω < 1)。
    nr, nz : int
        径向与垂直网格数。

    返回
    ------
    psi : ndarray, shape (nr, nz)
        收敛后的磁通函数 [Wb/rad]。
    R_grid : ndarray, shape (nr,)
    Z_grid : ndarray, shape (nz,)
    info : dict
        包含迭代次数、残差、q-profile 等信息。
    """
    # 计算域
    R_min = R0 - 1.2 * a_minor
    R_max = R0 + 1.2 * a_minor
    Z_min = -1.2 * KAPPA * a_minor
    Z_max = 1.2 * KAPPA * a_minor

    R_grid = np.linspace(R_min, R_max, nr)
    Z_grid = np.linspace(Z_min, Z_max, nz)
    R, Z = np.meshgrid(R_grid, Z_grid, indexing='ij')

    # 初始化试探函数：在边界上 ψ = 0，轴心处 ψ = 1
    psi = np.zeros((nr, nz), dtype=float)
    R_axis = R0
    Z_axis = 0.0
    for i in range(nr):
        for j in range(nz):
            dist = np.sqrt(((R[i, j] - R_axis) / a_minor) ** 2 +
                           ((Z[i, j] - Z_axis) / (KAPPA * a_minor)) ** 2)
            psi[i, j] = max(0.0, 1.0 - dist ** 2)

    # 固定边界值 (Dirichlet)
    psi[0, :] = 0.0
    psi[-1, :] = 0.0
    psi[:, 0] = 0.0
    psi[:, -1] = 0.0

    dR = R_grid[1] - R_grid[0]
    dZ = Z_grid[1] - Z_grid[0]

    for it in range(max_iter):
        psi_old = psi.copy()

        # 归一化磁通
        psi_min = psi.min()
        psi_max = psi.max()
        if psi_max - psi_min < 1e-14:
            raise RuntimeError("磁通范围过小，数值发散")
        psi_norm = (psi - psi_min) / (psi_max - psi_min)
        psi_norm = np.clip(psi_norm, 0.0, 1.0)

        # 压强与电流剖面
        p = pressure_profile(psi_norm)
        F = f_profile(psi_norm)

        # 源项 S = -μ₀ R² dp/dψ - F dF/dψ
        # 采用有限差分近似 dp/dψ 与 d(F²)/dψ
        dp_dpsi = np.zeros_like(p)
        dF2_dpsi = np.zeros_like(F)
        for i in range(1, nr - 1):
            for j in range(1, nz - 1):
                dp_dpsi[i, j] = (p[i + 1, j] - p[i - 1, j]) / (psi_old[i + 1, j] - psi_old[i - 1, j] + 1e-20)
                dF2_dpsi[i, j] = (F[i + 1, j] ** 2 - F[i - 1, j] ** 2) / (psi_old[i + 1, j] - psi_old[i - 1, j] + 1e-20)

        source = -MU0 * R ** 2 * dp_dpsi - 0.5 * dF2_dpsi

        # Poisson 求解 (SOR 松弛)
        psi_new = psi_old.copy()
        for i in range(1, nr - 1):
            for j in range(1, nz - 1):
                # 离散 Laplacian 在 (R,Z) 坐标下
                term = (
                    (psi_old[i + 1, j] + psi_new[i - 1, j]) / (dR ** 2)
                    + (psi_old[i, j + 1] + psi_new[i, j - 1]) / (dZ ** 2)
                    - source[i, j]
                )
                # 考虑 1/R 项的修正：系数矩阵为 Aψ = source
                # 近似处理：使用标准 Poisson 求解器
                denom = 2.0 / (dR ** 2) + 2.0 / (dZ ** 2)
                psi_new[i, j] = term / denom

        # 松弛
        psi = relaxation * psi_new + (1.0 - relaxation) * psi_old

        # 边界固定
        psi[0, :] = 0.0
        psi[-1, :] = 0.0
        psi[:, 0] = 0.0
        psi[:, -1] = 0.0

        err = np.max(np.abs(psi - psi_old))
        if err < tol:
            break
    else:
        # 未收敛但继续返回最佳结果
        pass

    # 计算安全因子 q(ψ) 近似
    # q ≈ (F / (2π)) ∮ dl_p / (R² |∇ψ|)
    # 简化：q ≈ ε B_φ / B_θ，其中 B_θ ≈ (∂ψ/∂r) / R
    dpsi_dr = np.zeros(nr)
    r_mid = (nr // 2)
    for i in range(1, nr - 1):
        dpsi_dr[i] = (psi[i, nz // 2] - psi[i - 1, nz // 2]) / dR
    q_profile = np.zeros(nr)
    for i in range(1, nr - 1):
        R_loc = R_grid[i]
        B_theta = np.abs(dpsi_dr[i]) / (R_loc + 1e-10)
        q_profile[i] = (R_loc * B0 / (R0 + 1e-10)) / (B_theta + 1e-10)

    info = {
        "iterations": it + 1,
        "final_error": err,
        "R_grid": R_grid,
        "Z_grid": Z_grid,
        "q_profile": q_profile,
        "psi_axis": psi_max,
        "psi_edge": psi_min,
    }
    return psi, R_grid, Z_grid, info


def compute_magnetic_field(psi, R_grid, Z_grid):
    """
    由磁通函数计算磁场分量。

    公式
    ----
        B_R = - (1/R) ∂ψ/∂Z
        B_Z =  (1/R) ∂ψ/∂R
        B_φ = F(ψ) / R

    参数
    ------
    psi : ndarray
    R_grid, Z_grid : ndarray

    返回
    ------
    B_R, B_Z, B_phi : ndarray
        磁场分量 [T]。
    """
    nr, nz = psi.shape
    dR = R_grid[1] - R_grid[0]
    dZ = Z_grid[1] - Z_grid[0]
    R, Z = np.meshgrid(R_grid, Z_grid, indexing='ij')

    B_R = np.zeros_like(psi)
    B_Z = np.zeros_like(psi)
    psi_norm = (psi - psi.min()) / (psi.max() - psi.min() + 1e-20)
    F = f_profile(psi_norm)
    B_phi = F / (R + 1e-20)

    for i in range(1, nr - 1):
        for j in range(1, nz - 1):
            dpsi_dR = (psi[i + 1, j] - psi[i - 1, j]) / (2.0 * dR)
            dpsi_dZ = (psi[i, j + 1] - psi[i, j - 1]) / (2.0 * dZ)
            B_R[i, j] = -dpsi_dZ / (R[i, j] + 1e-20)
            B_Z[i, j] = dpsi_dR / (R[i, j] + 1e-20)

    return B_R, B_Z, B_phi
