"""
reaction_diffusion.py

DNA 损伤修复分子动力学 —— γH2AX 信号波与 PARP1 招募反应-扩散动力学

基于种子项目:
  - 486_gray_scott_movie: Gray-Scott 反应-扩散系统
  - 901_porous_medium_exact: 多孔介质方程精确解 (非线性扩散)

科学背景:
  DNA 双链断裂 (DSB) 发生后，ATM/ATR 激酶磷酸化组蛋白 H2AX 形成 γH2AX，
  该修饰以反应-扩散波的形式在染色质上传播，形成直径可达 1–2 μm 的
  修复焦点 (repair focus)。同时，PARP1 被迅速招募至断裂位点并合成
  PAR 链，调控后续修复通路选择 (NHEJ vs HR)。

  本模块建立耦合的 Gray-Scott 型反应-扩散方程与多孔介质非线性扩散
  模型，用于定量刻画 γH2AX 波前传播速度与 PARP1 局部浓度演化。
"""

import numpy as np
from typing import Tuple, Optional, Callable


def laplacian9_torus(A: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """
    9 点 Laplacian 模板，带周期性边界条件（torus）。

    离散格式（九点星形 stencil）:
        L(i,j) = [1*A(i-1,j-1) + 4*A(i-1,j) + 1*A(i-1,j+1)
                 + 4*A(i,  j-1) -20*A(i,  j) + 4*A(i,  j+1)
                 + 1*A(i+1,j-1) + 4*A(i+1,j) + 1*A(i+1,j+1)] / (6 dx^2)

    该模板具有 O(h^4) 的截断误差（在周期性边界下），优于标准 5 点模板。
    使用 numpy roll 实现向量化周期性边界处理。
    """
    A = np.asarray(A, dtype=np.float64)
    nxp, nyp = A.shape

    if nxp < 3 or nyp < 3:
        raise ValueError("grid dimensions must be at least 3")

    denom = 6.0 * dx * dx

    # 使用 roll 实现周期边界，向量化计算
    L = (
        1.0 * np.roll(np.roll(A, -1, axis=0), -1, axis=1)
        + 4.0 * np.roll(A, -1, axis=0)
        + 1.0 * np.roll(np.roll(A, -1, axis=0), 1, axis=1)
        + 4.0 * np.roll(A, -1, axis=1)
        - 20.0 * A
        + 4.0 * np.roll(A, 1, axis=1)
        + 1.0 * np.roll(np.roll(A, 1, axis=0), -1, axis=1)
        + 4.0 * np.roll(A, 1, axis=0)
        + 1.0 * np.roll(np.roll(A, 1, axis=0), 1, axis=1)
    ) / denom

    return L


def laplacian5_torus(A: np.ndarray, dx: float) -> np.ndarray:
    """
    5 点 Laplacian 模板，带标准周期性边界（torus），向量化实现。

    离散格式:
        L(i,j) = [A(i-1,j) + A(i+1,j) + A(i,j-1) + A(i,j+1) - 4A(i,j)] / dx^2
    """
    A = np.asarray(A, dtype=np.float64)
    return (
        np.roll(A, 1, axis=0) + np.roll(A, -1, axis=0)
        + np.roll(A, 1, axis=1) + np.roll(A, -1, axis=1)
        - 4.0 * A
    ) / (dx * dx)


def gray_scott_step(
    u: np.ndarray,
    v: np.ndarray,
    du: float,
    dv: float,
    f: float,
    k: float,
    dt: float,
    dx: float,
    dy: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    执行一个 Gray-Scott 反应-扩散方程的 Euler 时间步进。

    控制方程:
        ∂u/∂t = D_u ∇^2 u - u v^2 + f (1 - u)
        ∂v/∂t = D_v ∇^2 v + u v^2 - (f + k) v

    物理意义:
        u: 未磷酸化 H2AX (底物) 的相对浓度
        v: γH2AX (产物) 的相对浓度
        f: ATM/ATR 激酶活性对应的"供给"速率
        k: PP2A 等磷酸酶的去磷酸化"kill"速率
        D_u, D_v: 有效扩散系数（受核质拥挤效应调制）

    Parameters
    ----------
    u, v : ndarray, shape (nx, ny)
        当前浓度场。
    du, dv : float
        扩散系数。
    f, k : float
        Gray-Scott 参数。
    dt : float
        时间步长。
    dx, dy : float
        空间网格间距。

    Returns
    -------
    u_new, v_new : ndarray
        更新后的浓度场。
    """
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)

    # 边界处理：浓度必须非负
    u = np.clip(u, 0.0, 1.0)
    v = np.clip(v, 0.0, 1.0)

    # 计算 Laplacian（5 点模板更稳定）
    Lu = laplacian5_torus(u, dx)
    Lv = laplacian5_torus(v, dx)

    # 反应项
    reaction = u * v * v

    dudt = du * Lu - reaction + f * (1.0 - u)
    dvdt = dv * Lv + reaction - (f + k) * v

    # 显式 Euler 步进
    u_new = u + dt * dudt
    v_new = v + dt * dvdt

    # 后处理截断，保证物理合理性
    u_new = np.clip(u_new, 0.0, 1.0)
    v_new = np.clip(v_new, 0.0, 1.0)

    return u_new, v_new


def porous_medium_solution(
    x: np.ndarray,
    t: float,
    m: float = 3.0,
    delta: float = 1.0 / 75.0,
    c: float = np.sqrt(3.0) / 15.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    多孔介质方程的 Barenblatt 自相似精确解。

    PDE:
        ∂u/∂t = ∇^2 (u^m)

    该方程描述 PARP1/修复因子在高度拥挤的染色质中的非线性扩散，
    其中扩散系数 D(u) = m u^{m-1} 依赖于浓度本身（退化扩散）。

    Barenblatt 解:
        u(x,t) = (t + δ)^{-β} * [ c - γ (x / (t + δ)^β)^2 ]_+^{α}

    其中:
        α = 1 / (m - 1)
        β = 1 / (m + 1)
        γ = (m - 1) / (2 m (m + 1))
        [z]_+ = max(z, 0)

    返回 u, u_t, u_x, u_xx。
    """
    x = np.asarray(x, dtype=np.float64)
    if t + delta <= 0:
        raise ValueError("t + delta must be positive")
    if m <= 1.0:
        raise ValueError("m must be > 1 for porous medium equation")

    alpha = 1.0 / (m - 1.0)
    beta = 1.0 / (m + 1.0)
    gamma = (m - 1.0) / (2.0 * m * (m + 1.0))

    bot = (t + delta) ** beta
    factor = c - gamma * (x / bot) ** 2

    positive = factor > 0.0

    u = np.zeros_like(x)
    ut = np.zeros_like(x)
    ux = np.zeros_like(x)
    uxx = np.zeros_like(x)

    if np.any(positive):
        fp = factor[positive]
        xp = x[positive]

        u[positive] = (t + delta) ** (-beta) * fp ** alpha

        ut[positive] = (
            2.0 * alpha * beta * gamma * (t + delta) ** (-1.0 - 3.0 * beta)
            * xp ** 2 * fp ** (alpha - 1.0)
            - beta * (t + delta) ** (-1.0 - beta) * fp ** alpha
        )

        ux[positive] = (
            -2.0 * alpha * gamma
            * (t + delta) ** (-3.0 * beta)
            * xp
            * fp ** (alpha - 1.0)
        )

        uxx[positive] = (
            4.0 * (alpha - 1.0) * alpha * gamma ** 2
            * (t + delta) ** (-5.0 * beta)
            * xp ** 2
            * fp ** (alpha - 2.0)
            - 2.0 * alpha * gamma
            * (t + delta) ** (-3.0 * beta)
            * fp ** (alpha - 1.0)
        )

    # 非正区域自动保持为 0（紧支集）
    return u, ut, ux, uxx


def porous_medium_residual(
    x: np.ndarray,
    t: float,
    m: float = 3.0,
) -> np.ndarray:
    """
    计算多孔介质方程的 PDE 残差:
        R = u_t - m(m-1) u^{m-2} u_x^2 - m u^{m-1} u_xx

    对于精确解，R 应在机器精度附近为零。
    """
    u, ut, ux, uxx = porous_medium_solution(x, t, m)

    # 处理 u=0 的退化点
    eps = 1e-14
    u_safe = np.where(u > eps, u, eps)

    R = ut - m * (m - 1.0) * u_safe ** (m - 2.0) * ux ** 2 - m * u_safe ** (m - 1.0) * uxx

    # 在紧支集外，ux=uxx=ut=0，残差自然为 0
    R = np.where(u > eps, R, 0.0)
    return R


def simulate_gamma_h2ax_wave(
    nx: int = 128,
    ny: int = 128,
    nt: int = 2000,
    f: float = 0.03,
    k: float = 0.062,
    du: float = 0.16,
    dv: float = 0.08,
    dt: float = 1.0,
    dx: float = 10.0,  # nm
) -> dict:
    """
    模拟 γH2AX 信号波在染色质平面上的传播。

    初始化条件：在中心区域放置高浓度 γH2AX（模拟 DSB 位点）。

    Returns
    -------
    result : dict
        包含最终浓度场、波前速度估计、总 γH2AX 量。
    """
    np.random.seed(7)

    u = np.ones((nx, ny), dtype=np.float64)
    v = np.zeros((nx, ny), dtype=np.float64)

    # 中心初始损伤区（增大初始 patch 确保模式存活）
    xm, ym = nx // 2, ny // 2
    patch = nx // 8
    v[xm - patch:xm + patch, ym - patch:ym + patch] = 1.0
    u[xm - patch:xm + patch, ym - patch:ym + patch] = 0.0
    # 添加微小扰动以触发不稳定性
    v += np.random.randn(nx, ny) * 0.01
    v = np.clip(v, 0.0, 1.0)

    # 记录波前半径
    radii = []
    thresholds = [0.5]

    for it in range(nt):
        u, v = gray_scott_step(u, v, du, dv, f, k, dt, dx, dx)

        # 每 100 步估计波前半径
        if it % 100 == 0:
            coords = np.argwhere(v > 0.3)
            if len(coords) > 0:
                dists = np.sqrt((coords[:, 0] - xm) ** 2 + (coords[:, 1] - ym) ** 2) * dx
                radii.append(float(np.max(dists)))
            else:
                radii.append(0.0)

    # 波前速度估算 (nm / 时间步)
    velocity = 0.0
    if len(radii) > 1:
        velocity = (radii[-1] - radii[0]) / (len(radii) * 100.0 * dt + 1e-12)

    total_gamma = float(np.sum(v)) * dx * dx  # 积分总量

    return {
        "u_final": u,
        "v_final": v,
        "wave_velocity": velocity,
        "total_gamma_h2ax": total_gamma,
        "max_v": float(np.max(v)),
    }


def simulate_parp1_nonlinear_diffusion(
    nx: int = 256,
    nt: int = 400,
    dt: float = 0.01,
    dx: float = 0.1,
    m: float = 3.0,
) -> dict:
    """
    基于多孔介质方程模拟 PARP1 在 DSB 位点的非线性扩散聚集。

    使用有限差分法求解:
        u_t = ∇ · (D(u) ∇u),  D(u) = m u^{m-1}

    初始条件采用 Barenblatt 自相似解，以保证可与解析解进行定量比较。
    """
    x = np.linspace(-nx * dx / 2, nx * dx / 2, nx)
    # Barenblatt 参数
    delta_pm = 1.0 / 75.0
    c_pm = np.sqrt(3.0) / 15.0
    t0 = 1.0  # 初始时刻，确保解在网格上有足够支撑

    u, _, _, _ = porous_medium_solution(x, t0, m=m, delta=delta_pm, c=c_pm)
    u = np.maximum(u, 0.0)

    for _ in range(nt):
        # 计算非线性扩散通量: J = -m u^{m-1} ∇u
        u_padded = np.pad(u, 1, mode='constant')
        grad_u = (u_padded[2:] - u_padded[:-2]) / (2.0 * dx)
        D = m * np.maximum(u, 1e-12) ** (m - 1.0)
        flux = -D * grad_u

        # 散度
        flux_padded = np.pad(flux, 1, mode='constant')
        div_flux = (flux_padded[2:] - flux_padded[:-2]) / (2.0 * dx)

        u_new = u + dt * (-div_flux)
        u_new = np.maximum(u_new, 0.0)
        u = u_new

    # 与 Barenblatt 精确解比较
    t_final = nt * dt + t0
    u_exact, _, _, _ = porous_medium_solution(x, t_final, m=m, delta=delta_pm, c=c_pm)

    # 只在支撑域内计算 L2 误差
    mask = u_exact > 1e-12
    if np.any(mask):
        l2_error = float(np.sqrt(np.mean((u[mask] - u_exact[mask]) ** 2)))
    else:
        l2_error = 0.0

    return {
        "u_numerical": u,
        "u_exact": u_exact,
        "x_grid": x,
        "l2_error": l2_error,
        "total_mass_numerical": float(np.sum(u) * dx),
        "total_mass_exact": float(np.sum(u_exact) * dx),
    }
