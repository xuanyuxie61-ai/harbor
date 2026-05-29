"""
reaction_diffusion_dynamics.py
反应-扩散动力学模块

融合原项目:
  - 486_gray_scott_movie: Gray-Scott反应扩散方程 → DNA修复蛋白扩散-反应动力学
  - 901_porous_medium_exact: 多孔介质方程精确解 → 蛋白质密度非线性扩散

科学背景:
  在DNA损伤修复过程中，RAD51蛋白在细胞核中的扩散与结合遵循反应-扩散方程:

    ∂[RAD51]/∂t = D·∇²[RAD51] - k_on·[RAD51]·[site] + k_off·[RAD51·site]

  其中:
    D: 扩散系数 (~1-10 μm²/s 在细胞核中)
    k_on: 结合速率常数
    k_off: 解离速率常数
    [site]: 空余DNA结合位点浓度

  Gray-Scott模型是经典的反应-扩散系统:
    ∂u/∂t = D_u ∇²u - u·v² + f(1-u)
    ∂v/∂t = D_v ∇²v + u·v² - (f+k)v
    其中 u 为抑制剂 (此处类比游离蛋白), v 为激活剂 (类比结合蛋白)

  多孔介质方程描述非线性扩散:
    ∂ρ/∂t = Δ(ρ^m)
    其自相似解 (Barenblatt解):
        ρ(x,t) = (t+δ)^{-β} [ c - γ(x/(t+δ)^β)² ]^{1/(m-1)}_+
    其中 [·]_+ = max(·, 0), β = 1/(m+1), γ = (m-1)/(2m(m+1))
"""

import numpy as np


def gray_scott_step(u: np.ndarray, v: np.ndarray,
                    Du: float, Dv: float, f: float, k: float,
                    dx: float, dy: float, dt: float) -> tuple:
    """
    执行一步Gray-Scott反应扩散方程的时间推进

    方程:
        du/dt = Du * ∇²u - u*v² + f*(1-u)
        dv/dt = Dv * ∇²v + u*v² - (f+k)*v

    使用显式欧拉法 + 九点拉普拉斯 (torus边界)

    参数:
        u, v: 当前浓度场
        Du, Dv: 扩散系数
        f: 进料速率
        k: 衰减速率
        dx, dy: 空间步长
        dt: 时间步长

    Returns:
        u_new, v_new
    """
    if u.shape != v.shape:
        raise ValueError("u and v must have same shape")

    nx, ny = u.shape

    def laplacian9_torus(field: np.ndarray) -> np.ndarray:
        """
        九点拉普拉斯算子 (周期性边界)
        """
        lap = np.zeros_like(field)
        for i in range(nx):
            im = (i - 1) % nx
            ip = (i + 1) % nx
            for j in range(ny):
                jm = (j - 1) % ny
                jp = (j + 1) % ny
                lap[i, j] = (
                    field[im, jm] + field[im, j] + field[im, jp]
                    + field[i, jm] - 8.0 * field[i, j] + field[i, jp]
                    + field[ip, jm] + field[ip, j] + field[ip, jp]
                ) / (3.0 * dx * dy)
        return lap

    uLap = laplacian9_torus(u)
    vLap = laplacian9_torus(v)

    dudt = Du * uLap - u * v ** 2 + f * (1.0 - u)
    dvdt = Dv * vLap + u * v ** 2 - (f + k) * v

    # CFL稳定性检查
    max_dt_diff = 0.25 * min(dx * dx / (Du + 1e-12), dy * dy / (Dv + 1e-12))
    if dt > max_dt_diff:
        dt = max_dt_diff * 0.9

    u_new = np.clip(u + dt * dudt, 0.0, 1.0)
    v_new = np.clip(v + dt * dvdt, 0.0, 1.0)

    return u_new, v_new


def gray_scott_simulation(nx: int = 64, ny: int = 64,
                          f: float = 0.035, k: float = 0.060,
                          Du: float = 0.16, Dv: float = 0.08,
                          dx: float = 1.0, dy: float = 1.0,
                          dt: float = 1.0, n_steps: int = 2000) -> tuple:
    """
    运行Gray-Scott反应扩散模拟

    Returns:
        u, v: 最终浓度场
    """
    u = np.ones((nx, ny), dtype=float)
    v = np.zeros((nx, ny), dtype=float)

    # 初始条件: 中心方块的v=1
    xm = nx // 2
    ym = ny // 2
    u[xm - 5:xm + 5, ym - 5:ym + 5] = 0.0
    v[xm - 5:xm + 5, ym - 5:ym + 5] = 1.0

    for step in range(n_steps):
        u, v = gray_scott_step(u, v, Du, Dv, f, k, dx, dy, dt)

    return u, v


def porous_medium_exact(x: np.ndarray, t: float,
                        c: float = np.sqrt(3.0) / 15.0,
                        delta: float = 1.0 / 75.0,
                        m: float = 3.0) -> tuple:
    """
    基于 porous_medium_exact 的多孔介质方程Barenblatt精确解

    方程: ∂u/∂t = Δ(u^m)

    自相似解:
        α = 1/(m-1)
        β = 1/(m+1)
        γ = (m-1) / (2m(m+1))
        bot = (t+δ)^β
        factor = c - γ*(x/bot)²
        u = (t+δ)^{-β} * factor^α   (if factor > 0, else 0)

    参数:
        x: 位置数组
        t: 时间
        c, delta, m: 方程参数

    Returns:
        u, ut, ux, uxx
    """
    x = np.asarray(x, dtype=float)
    if t + delta <= 0:
        raise ValueError("t+delta must be positive")
    if m <= 1.0:
        raise ValueError("m must be > 1")

    alpha = 1.0 / (m - 1.0)
    beta = 1.0 / (m + 1.0)
    gamma = (m - 1.0) / (2.0 * m * (m + 1.0))

    bot = (t + delta) ** beta
    factor = c - gamma * (x / bot) ** 2

    u = np.zeros_like(x, dtype=float)
    ut = np.zeros_like(x, dtype=float)
    ux = np.zeros_like(x, dtype=float)
    uxx = np.zeros_like(x, dtype=float)

    mask = factor > 0.0
    if np.any(mask):
        f = factor[mask]
        u[mask] = (t + delta) ** (-beta) * f ** alpha
        ut[mask] = (2.0 * alpha * beta * gamma * (t + delta) ** (-1.0 - 3.0 * beta)
                    * x[mask] ** 2 * f ** (alpha - 1.0)
                    - beta * (t + delta) ** (-1.0 - beta) * f ** alpha)
        ux[mask] = (-2.0 * alpha * gamma * (t + delta) ** (-3.0 * beta)
                    * x[mask] * f ** (alpha - 1.0))
        uxx[mask] = (4.0 * (alpha - 1.0) * alpha * gamma ** 2
                     * (t + delta) ** (-5.0 * beta) * x[mask] ** 2 * f ** (alpha - 2.0)
                     - 2.0 * alpha * gamma * (t + delta) ** (-3.0 * beta) * f ** (alpha - 1.0))

    return u, ut, ux, uxx


def protein_diffusion_reaction_1d(n_sites: int = 100,
                                   n_steps: int = 5000,
                                   D: float = 0.1,
                                   k_on: float = 0.05,
                                   k_off: float = 0.01,
                                   dx: float = 1.0,
                                   dt: float = 0.1) -> tuple:
    """
    一维DNA上的蛋白质扩散-反应模拟

    方程组:
        ∂P/∂t = D·∂²P/∂x² - k_on·P·S + k_off·B
        ∂B/∂t = k_on·P·S - k_off·B
        S = S_total - B  (空余位点)

    其中 P 为游离蛋白浓度，B 为结合蛋白浓度，S 为空余位点

    Returns:
        P, B: 最终浓度分布
    """
    if n_sites <= 0:
        raise ValueError("n_sites must be positive")
    if D <= 0 or dx <= 0 or dt <= 0:
        raise ValueError("Physical parameters must be positive")

    # CFL条件
    cfl = D * dt / (dx ** 2)
    if cfl > 0.5:
        dt = 0.45 * dx ** 2 / D
        cfl = D * dt / (dx ** 2)

    P = np.zeros(n_sites, dtype=float)
    B = np.zeros(n_sites, dtype=float)
    S_total = np.ones(n_sites, dtype=float)  # 总位点密度

    # 初始条件: 两端有高浓度游离蛋白
    P[:10] = 1.0
    P[-10:] = 1.0

    for _ in range(n_steps):
        # 扩散项 (中心差分)
        P_diff = np.zeros_like(P)
        for i in range(1, n_sites - 1):
            P_diff[i] = D * (P[i - 1] - 2.0 * P[i] + P[i + 1]) / (dx ** 2)
        # Neumann边界
        P_diff[0] = D * (P[1] - P[0]) / (dx ** 2)
        P_diff[-1] = D * (P[-2] - P[-1]) / (dx ** 2)

        S = np.maximum(S_total - B, 0.0)
        binding = k_on * P * S
        unbinding = k_off * B

        P_new = P + dt * (P_diff - binding + unbinding)
        B_new = B + dt * (binding - unbinding)

        P = np.clip(P_new, 0.0, None)
        B = np.clip(B_new, 0.0, S_total)

    return P, B


def compute_reaction_front_velocity(D: float, k_on: float,
                                    P0: float = 1.0) -> float:
    """
    基于Fisher-KPP方程近似，计算反应前沿传播速度

    理论速度:
        v = 2 * sqrt(D * k_on * P0)
    """
    if D < 0 or k_on < 0 or P0 < 0:
        raise ValueError("Parameters must be non-negative")
    return 2.0 * np.sqrt(D * k_on * P0)
