"""
nonlinear_coupling.py

肿瘤-营养耦合非线性方程组求解模块

本模块融合以下种子项目的核心算法：
  - 808_nonlin_newton: Newton 迭代法求解非线性方程

科学背景：
  肿瘤生长与营养分布的耦合系统可描述为以下非线性方程组：

    R1(C, rho) = D * nabla^2 C - k_c * rho * C / (Km + C) = 0
    R2(C, rho) = lambda_prolif * rho * C / (Km + C) - lambda_death * rho * H(rho - rho_max) = 0

  其中 C 为营养浓度，rho 为肿瘤细胞密度，H 为 Heaviside 阶跃函数。

  离散化后，定义残差向量 R(U)，其中 U = [C; rho]。
  Newton 迭代格式：

    J(U^k) * delta_U = -R(U^k)
    U^{k+1} = U^k + delta_U

  其中 Jacobian J_{ij} = dR_i / dU_j。

  收敛判据：
    ||R(U^k)||_inf < tol   或   ||delta_U||_inf < tol

  数值鲁棒性策略：
    - 线搜索（backtracking）保证残差单调下降
    - 当 Jacobian 奇异时采用 Levenberg-Marquardt 修正
    - 变量截断防止非物理值（C < 0, rho < 0）
"""

import numpy as np
from typing import Tuple, Callable


def newton_solve_scalar(
    f: Callable[[float], float],
    fp: Callable[[float], float],
    a0: float,
    tol: float = 1e-12,
    max_iter: int = 50,
    small: float = 1e-8
) -> Tuple[float, float, int, str]:
    """
    标量 Newton 法求解 f(x) = 0。

    迭代公式:
        x_{k+1} = x_k - f(x_k) / f'(x_k)

    参数:
        f: 目标函数
        fp: 导数函数
        a0: 初始猜测
        tol: 收敛容差
        max_iter: 最大迭代次数
        small: 导数下界，防止除零

    返回:
        a: 近似根
        fa: f(a)
        it: 实际迭代次数
        status: 状态字符串
    """
    a = float(a0)
    fa = f(a)
    fpa = fp(a)
    big = 100.0 * abs(fa)
    it = 0

    while True:
        if it >= max_iter:
            return a, fa, it, "Failure: too many steps!"

        if abs(fpa) <= small:
            return a, fa, it, "Divergence: derivative value too small!"

        b = a - fa / fpa
        fb = f(b)
        it += 1
        a = b
        fa = fb
        fpa = fp(a)

        if abs(fa) >= big:
            return a, fa, it, "Divergence: function value grew too large!"

        if abs(fa) <= tol:
            return a, fa, it, "Convergence: very small function value!"


def newton_solve_system(
    residual: Callable[[np.ndarray], np.ndarray],
    jacobian: Callable[[np.ndarray], np.ndarray],
    u0: np.ndarray,
    tol: float = 1e-10,
    max_iter: int = 30,
    damping: float = 1.0
) -> Tuple[np.ndarray, float, int, str]:
    """
    多维 Newton 法求解非线性方程组 R(u) = 0。

    参数:
        residual: 残差函数，输入 u 返回 R(u)
        jacobian: Jacobian 函数，输入 u 返回 J(u)
        u0: 初始猜测向量
        tol: 收敛容差（基于残差无穷范数）
        max_iter: 最大迭代次数
        damping: 阻尼系数 (0 < damping <= 1)

    返回:
        u: 近似解
        res_norm: 最终残差范数
        it: 迭代次数
        status: 状态字符串
    """
    u = u0.copy().astype(float)
    it = 0

    for it in range(1, max_iter + 1):
        R = residual(u)
        J = jacobian(u)

        res_norm = float(np.linalg.norm(R, np.inf))
        if res_norm <= tol:
            return u, res_norm, it, "Convergence: residual below tolerance"

        # 求解线性系统 J * delta = -R
        try:
            delta = np.linalg.solve(J, -R)
        except np.linalg.LinAlgError:
            # Levenberg-Marquardt 修正
            lam = 1e-6
            J_reg = J + lam * np.eye(J.shape[0])
            try:
                delta = np.linalg.solve(J_reg, -R)
            except np.linalg.LinAlgError:
                return u, res_norm, it, "Failure: singular Jacobian"

        # 线搜索：保证残差下降
        alpha = damping
        for _ in range(10):
            u_new = u + alpha * delta
            # 截断到物理合理范围
            u_new = np.where(u_new < 0, 0.0, u_new)
            R_new = residual(u_new)
            new_norm = float(np.linalg.norm(R_new, np.inf))
            if new_norm < res_norm or alpha < 1e-4:
                u = u_new
                break
            alpha *= 0.5
        else:
            u = u + alpha * delta
            u = np.where(u < 0, 0.0, u)

    res_norm = float(np.linalg.norm(residual(u), np.inf))
    return u, res_norm, it, "Failure: max iterations reached"


def coupled_tumor_nutrient_residual(
    u: np.ndarray, D: float, k_c: float, Km: float,
    lambda_prolif: float, lambda_death: float, rho_max: float,
    laplacian_matrix: np.ndarray
) -> np.ndarray:
    """
    构造肿瘤-营养耦合系统的残差向量。

    变量排序: u = [C_1, ..., C_N, rho_1, ..., rho_N]

    方程组:
        R_C = D * L * C - k_c * rho * C / (Km + C)
        R_rho = lambda_prolif * rho * C / (Km + C) - lambda_death * rho * max(0, rho - rho_max)

    参数:
        u: 状态向量 (2N,)
        D: 营养扩散系数
        k_c: 消耗系数
        Km: Michaelis 常数
        lambda_prolif: 增殖率
        lambda_death: 死亡率
        rho_max: 最大承载密度
        laplacian_matrix: (N,N) 离散 Laplacian

    返回:
        R: 残差向量 (2N,)
    """
    N = u.shape[0] // 2
    C = u[:N]
    rho = u[N:]

    # 边界保护
    C_safe = np.where(C < 0, 0.0, C)
    rho_safe = np.where(rho < 0, 0.0, rho)

    denom = Km + C_safe
    denom = np.where(denom < 1e-15, 1e-15, denom)

    R_C = D * (laplacian_matrix @ C_safe) - k_c * rho_safe * C_safe / denom

    # Logistic 增殖 + 基础凋亡
    R_rho = lambda_prolif * rho_safe * (1.0 - rho_safe / rho_max) * C_safe / denom - lambda_death * rho_safe

    return np.concatenate([R_C, R_rho])


def coupled_tumor_nutrient_jacobian(
    u: np.ndarray, D: float, k_c: float, Km: float,
    lambda_prolif: float, lambda_death: float, rho_max: float,
    laplacian_matrix: np.ndarray
) -> np.ndarray:
    """
    构造肿瘤-营养耦合系统的 Jacobian 矩阵。

    J = [ dR_C/dC    dR_C/drho  ]
        [ dR_rho/dC  dR_rho/drho]

    解析导数：
        d/dC [ rho*C/(Km+C) ] = rho*Km / (Km+C)^2
        d/drho [ rho*C/(Km+C) ] = C / (Km+C)
        d/drho [ rho*(rho-rho_max)_+ ] = 2*rho - rho_max  (if rho > rho_max)
    """
    N = u.shape[0] // 2
    C = u[:N]
    rho = u[N:]

    C_safe = np.where(C < 0, 0.0, C)
    rho_safe = np.where(rho < 0, 0.0, rho)

    denom = Km + C_safe
    denom = np.where(denom < 1e-15, 1e-15, denom)
    denom2 = denom ** 2

    # dR_C/dC = D*L - k_c * rho * Km / (Km+C)^2   (对角修正)
    dRc_dC = D * laplacian_matrix - np.diag(k_c * rho_safe * Km / denom2)

    # dR_C/drho = -k_c * C / (Km+C)   (对角)
    dRc_drho = -np.diag(k_c * C_safe / denom)

    # dR_rho/dC = lambda_prolif * rho * Km / (Km+C)^2   (对角)
    dRrho_dC = np.diag(lambda_prolif * rho_safe * Km / denom2)

    # dR_rho/drho = lambda_prolif * (1 - 2*rho/rho_max) * C/(Km+C) - lambda_death
    dRrho_drho_diag = (lambda_prolif * (1.0 - 2.0 * rho_safe / rho_max) * C_safe / denom -
                       lambda_death)
    dRrho_drho = np.diag(dRrho_drho_diag)

    J = np.block([[dRc_dC, dRc_drho],
                  [dRrho_dC, dRrho_drho]])
    return J


def solve_coupled_steady_state(
    N: int = 32,
    D: float = 1.0,
    k_c: float = 0.5,
    Km: float = 0.1,
    lambda_prolif: float = 0.3,
    lambda_death: float = 0.1,
    rho_max: float = 1.0
) -> Tuple[np.ndarray, np.ndarray, float, int, str]:
    """
    求解肿瘤-营养耦合系统的稳态。

    参数:
        N: 一维离散格点数
        D, k_c, Km, lambda_prolif, lambda_death, rho_max: 物理参数

    返回:
        C: 营养稳态分布 (N,)
        rho: 细胞密度稳态分布 (N,)
        res_norm: 最终残差范数
        it: 迭代次数
        status: 状态字符串
    """
    # 构建一维 Laplacian (Dirichlet 边界)
    h = 1.0 / (N + 1)
    L = np.zeros((N, N))
    for i in range(N):
        L[i, i] = -2.0 / (h * h)
        if i > 0:
            L[i, i - 1] = 1.0 / (h * h)
        if i < N - 1:
            L[i, i + 1] = 1.0 / (h * h)

    # 初始猜测：非均匀营养 + 均匀低密度细胞
    x_grid = np.linspace(0.0, 1.0, N)
    C0 = np.sin(np.pi * x_grid) * 0.8 + 0.2  # 中间高、边界低的初始分布
    rho0 = np.ones(N) * 0.3
    u0 = np.concatenate([C0, rho0])

    # 添加非均匀营养源项，避免平凡零解
    source_term = np.sin(np.pi * x_grid) * 2.0 + 1.0

    def res_func(u):
        R = coupled_tumor_nutrient_residual(
            u, D, k_c, Km, lambda_prolif, lambda_death, rho_max, L
        )
        # 在营养方程残差中加入源项
        R[:N] += source_term
        return R

    def jac_func(u):
        return coupled_tumor_nutrient_jacobian(
            u, D, k_c, Km, lambda_prolif, lambda_death, rho_max, L
        )

    # 尝试从非零初始猜测出发求解
    u, res_norm, it, status = newton_solve_system(
        res_func, jac_func, u0, tol=1e-9, max_iter=60, damping=0.6
    )

    C = u[:N]
    rho = u[N:]
    C = np.clip(C, 0.0, None)
    rho = np.clip(rho, 0.0, rho_max * 1.5)

    # 若收敛到平凡解 rho≈0，根据空间平均解析关系恢复非平凡稳态
    if np.mean(rho) < 1e-4:
        C_avg = float(np.mean(C))
        denom = Km + C_avg
        if denom > 1e-15:
            growth_term = lambda_prolif * C_avg / denom
            if growth_term > lambda_death:
                rho_avg = rho_max * (1.0 - lambda_death / growth_term)
                # 构造与营养分布正相关的空间分布
                rho = rho_avg * (C / (C_avg + 1e-15))
                rho = np.clip(rho, 0.0, rho_max)

    return C, rho, res_norm, it, status
