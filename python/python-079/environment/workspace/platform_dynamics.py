"""
海洋平台刚体动力学与隐式时间积分模块

基于种子项目：
  - 074_bdf2：BDF2 隐式常微分方程求解器
  - 849_partition_brute：组合分区优化（用于多体子系统分解）

核心物理模型：
  1. 六自由度平台运动方程（刚体动力学 + 水动力附加质量/阻尼）：
       (M + A(ω)) · ξ¨(t) + B(ω) · ξ˙(t) + C · ξ(t) = F_wave(t) + F_moor(t) + F_drag(t)
     其中：
       - M : 6×6 刚体质量/惯性矩阵
       - A(ω) : 频率相关附加质量矩阵
       - B(ω) : 频率相关辐射阻尼矩阵
       - C : 静水恢复力矩阵（由水线面面积和惯性矩决定）
       - ξ = [ surge, sway, heave, roll, pitch, yaw ]^T

  2. Cummins 时域方程（将频域水动力系数转化为时域卷积）：
       (M + A_∞) · ξ¨(t) + ∫_0^∞ K(τ) ξ˙(t-τ) dτ + C · ξ(t) = F(t)
     其中 K(t) 为延迟函数（Retardation function），由频域阻尼 B(ω) 经
     傅里叶余弦变换得到：
       K(t) = (2/π) ∫_0^∞ B(ω) cos(ωt) dω

  3. BDF2 隐式时间积分（源自 074_bdf2）：
       3y^{n+1} - 4y^n + y^{n-1} = 2Δt · f(t^{n+1}, y^{n+1})
     起步步采用 Backward Euler + 外推校正：
       y^1 = 2 y_h - y^0,  其中 y_h 由半步 BE 得到。
     非线性方程使用拟牛顿迭代求解。

  4. 系泊力（拟静力悬链线模型）：
       T(H) = H · cosh( w·(z+h)/H ) / cosh( w·h/H )
     其中 H 为水平张力，w 为线重，h 为水深。

  5. 子系统分区（源自 849_partition_brute）：
     将 6-DOF 系统按质量和刚度耦合强度划分为子系统，
     优化负载均衡以支持并行求解。
"""

import numpy as np
from typing import Tuple, List, Optional, Callable
from utils import solve_quadratic, clamp_value


# ======================================================================
# 1. 六自由度质量与惯性矩阵
# ======================================================================

def build_rigid_body_mass_matrix(
    mass: float,
    cog: np.ndarray,
    I: np.ndarray,
) -> np.ndarray:
    """
    构建 6×6 刚体质量矩阵 M。
    设重心位置 r_cg = [x_g, y_g, z_g]，惯性张量 I = diag(I_xx, I_yy, Izz)。
    M = [  m·I_3×3       -m·S(r_cg)  ]
        [  m·S(r_cg)^T    I_3×3      ]
    其中 S(r) 为叉乘反对称矩阵。
    """
    M = np.zeros((6, 6), dtype=float)
    M[0, 0] = mass
    M[1, 1] = mass
    M[2, 2] = mass
    M[0, 4] = mass * cog[2]
    M[0, 5] = -mass * cog[1]
    M[1, 3] = -mass * cog[2]
    M[1, 5] = mass * cog[0]
    M[2, 3] = mass * cog[1]
    M[2, 4] = -mass * cog[0]
    M[3, 1] = -mass * cog[2]
    M[3, 2] = mass * cog[1]
    M[4, 0] = mass * cog[2]
    M[4, 2] = -mass * cog[0]
    M[5, 0] = -mass * cog[1]
    M[5, 1] = mass * cog[0]
    M[3, 3] = I[0]
    M[4, 4] = I[1]
    M[5, 5] = I[2]
    # 对称化
    M = 0.5 * (M + M.T)
    return M


def build_hydrostatic_restoring_matrix(
    rho: float,
    g: float,
    area_wp: float,
    I_xx: float,
    I_yy: float,
    I_xy: float,
    z_cob: float,
    z_cog: float,
) -> np.ndarray:
    """
    构建 6×6 静水恢复力矩阵 C。
    对于半潜式平台，主要非零元为：
      C_33 = ρg · A_wp          (垂向)
      C_34 = -ρg · A_wp · y_c  (垂向-横摇耦合)
      C_35 = ρg · A_wp · x_c   (垂向-纵摇耦合)
      C_44 = ρg · I_xx + ρg · V_disp · (z_cob - z_cog)  (横摇)
      C_55 = ρg · I_yy + ρg · V_disp · (z_cob - z_cog)  (纵摇)
      C_45 = -ρg · I_xy        (横摇-纵摇耦合)
    """
    C = np.zeros((6, 6), dtype=float)
    V_disp = area_wp * abs(z_cob)
    C[2, 2] = rho * g * area_wp
    C[3, 3] = rho * g * I_xx + rho * g * V_disp * (z_cob - z_cog)
    C[4, 4] = rho * g * I_yy + rho * g * V_disp * (z_cob - z_cog)
    C[5, 5] = rho * g * I_xx + rho * g * I_yy  # 近似偏航恢复
    C[3, 4] = -rho * g * I_xy
    C[4, 3] = C[3, 4]
    C[2, 3] = -rho * g * area_wp * 0.0  # 假设形心在原点
    C[3, 2] = C[2, 3]
    C[2, 4] = rho * g * area_wp * 0.0
    C[4, 2] = C[2, 4]
    return C


# ======================================================================
# 2. BDF2 隐式时间积分 (源自 074_bdf2)
# ======================================================================

def bdf2_solve(
    f_ode: Callable[[float, np.ndarray], np.ndarray],
    tspan: Tuple[float, float],
    y0: np.ndarray,
    n_steps: int,
    newton_tol: float = 1e-10,
    newton_max_iter: int = 50,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    使用 BDF2 隐式方法求解常微分方程初值问题 dy/dt = f(t, y)。
    非线性方程使用不动点迭代（简化牛顿法）求解。

    时间步进公式：
        (3y^{n+1} - 4y^n + y^{n-1}) / (2Δt) = f(t^{n+1}, y^{n+1})
    →  残差:  r(y) = 3y - 4y^n + y^{n-1} - 2Δt·f(t, y) = 0

    参数
    ----
    f_ode : (t, y) → y_dot 的右端函数
    tspan : (t0, tf)
    y0 : 初始条件向量
    n_steps : 时间步数
    newton_tol : 牛顿迭代容差
    newton_max_iter : 牛顿最大迭代次数

    返回
    ----
    t_arr : 时间数组 (n_steps+1,)
    y_arr : 解矩阵 (n_steps+1, n_dof)
    """
    y0 = np.asarray(y0, dtype=float)
    n_dof = len(y0)
    t0, tf = tspan
    if tf <= t0:
        raise ValueError("tf 必须大于 t0")
    if n_steps < 2:
        raise ValueError("n_steps 至少为 2")
    dt = (tf - t0) / n_steps

    t_arr = np.linspace(t0, tf, n_steps + 1)
    y_arr = np.zeros((n_steps + 1, n_dof), dtype=float)
    y_arr[0, :] = y0

    # ---- 起步步：Backward Euler + 外推 ----
    t1 = t_arr[1]
    # 半步 BE
    y_h = _backward_euler_step(f_ode, t0, y0, dt * 0.5, newton_tol, newton_max_iter)
    # 全步 BE
    y1_be = _backward_euler_step(f_ode, t0, y0, dt, newton_tol, newton_max_iter)
    # 外推校正：y1 = 2*y_h - y0
    y1 = 2.0 * y_h - y0
    # 边界保护
    y1 = _clamp_dofs(y1)
    y_arr[1, :] = y1

    # ---- BDF2 主循环 ----
    y_prev = y0
    y_curr = y1
    for n in range(1, n_steps):
        t_next = t_arr[n + 1]
        # 初猜：外推
        y_guess = 2.0 * y_curr - y_prev
        y_next = _bdf2_newton_solve(
            f_ode, t_next, dt, y_prev, y_curr, y_guess, newton_tol, newton_max_iter
        )
        y_next = _clamp_dofs(y_next)
        y_arr[n + 1, :] = y_next
        y_prev = y_curr
        y_curr = y_next

    return t_arr, y_arr


def _backward_euler_step(
    f: Callable,
    t0: float,
    y0: np.ndarray,
    dt: float,
    tol: float,
    max_iter: int,
) -> np.ndarray:
    """单步 Backward Euler：y = y0 + dt·f(t0+dt, y)。"""
    y = y0.copy()
    t = t0 + dt
    for _ in range(max_iter):
        f_val = f(t, y)
        if np.any(np.isnan(f_val)) or np.any(np.isinf(f_val)):
            # 若发散，减小步长重试
            return y0 + dt * f(t0, y0)
        y_new = y0 + dt * f_val
        if np.linalg.norm(y_new - y) < tol:
            return y_new
        # 阻尼
        y = 0.5 * y + 0.5 * y_new
    return y


def _bdf2_newton_solve(
    f: Callable,
    t: float,
    dt: float,
    y1: np.ndarray,
    y2: np.ndarray,
    y_guess: np.ndarray,
    tol: float,
    max_iter: int,
) -> np.ndarray:
    """
    求解 BDF2 隐式方程：r(y) = 3y - 4y2 + y1 - 2dt·f(t, y) = 0
    使用阻尼不动点迭代。
    """
    y = y_guess.copy()
    for it in range(max_iter):
        f_val = f(t, y)
        if np.any(np.isnan(f_val)) or np.any(np.isinf(f_val)):
            break
        r = 3.0 * y - 4.0 * y2 + y1 - 2.0 * dt * f_val
        if np.linalg.norm(r) < tol:
            break
        # 阻尼不动点迭代
        y_new = (4.0 * y2 - y1 + 2.0 * dt * f_val) / 3.0
        y_new = _clamp_dofs(y_new)
        diff = np.linalg.norm(y_new - y)
        # 自适应阻尼
        alpha = 0.7 if diff > 1.0 else 1.0
        y = alpha * y_new + (1.0 - alpha) * y
        if diff < tol:
            break
    return _clamp_dofs(y)


def _clamp_dofs(y: np.ndarray) -> np.ndarray:
    """
    对自由度进行物理边界限制：
      - 位移 surge/sway/heave：±50 m
      - 转角 roll/pitch/yaw：±0.5 rad (~30°)
    """
    y = y.copy()
    for i in range(3):
        y[i] = clamp_value(y[i], -50.0, 50.0)
    for i in range(3, 6):
        y[i] = clamp_value(y[i], -0.5, 0.5)
    return y


# ======================================================================
# 3. 系泊力模型
# ======================================================================

def catenary_mooring_force(
    x_platform: float,
    y_platform: float,
    anchor_pos: np.ndarray,
    unstretched_length: float,
    line_weight: float,
    EA: float,
    horizontal_pretension: float,
) -> np.ndarray:
    """
    拟静力悬链线系泊力计算。
    水平张力 H 满足：L = H/w · sinh( w·x_f / H ) + EA/w · [asinh(w·x_f/H) - w·x_f/H]
    使用牛顿迭代求解 H，然后计算平台受力。
    返回 [Fx, Fy, 0, 0, 0, 0]（水平面内）。
    """
    dx = x_platform - anchor_pos[0]
    dy = y_platform - anchor_pos[1]
    # 限制平台位移，防止数值溢出
    dx = max(-200.0, min(200.0, dx))
    dy = max(-200.0, min(200.0, dy))
    horiz_dist = np.sqrt(dx ** 2 + dy ** 2)
    if horiz_dist < 1e-6:
        return np.zeros(6)
    w = line_weight
    L0 = unstretched_length
    H = horizontal_pretension
    # 牛顿迭代求解水平张力
    for _ in range(50):
        if H < 1e-3:
            H = 1e-3
        arg = w * horiz_dist / H
        if arg > 50.0:
            # 大参数近似：悬链线趋近直线
            H = max(1e-3, w * horiz_dist / 50.0)
            break
        s = np.sinh(arg)
        c = np.cosh(arg)
        f = (H / w) * s + (EA / w) * (np.arcsinh(arg) - arg) - L0
        # 导数 df/dH
        ds_dH = c * (-w * horiz_dist / (H * H))
        df = (1.0 / w) * s + (H / w) * ds_dH
        df += (EA / w) * ((1.0 / np.sqrt(1.0 + arg * arg)) - 1.0) * (-w * horiz_dist / (H * H))
        if abs(df) < 1e-15:
            break
        delta = f / df
        H_new = H - delta
        if H_new < 1e-3:
            H_new = 1e-3
        if abs(H_new - H) < 1e-8:
            H = H_new
            break
        H = H_new

    # 计算系泊力方向分量
    angle = np.arctan2(dy, dx)
    Fx = -H * np.cos(angle)
    Fy = -H * np.sin(angle)
    force = np.zeros(6)
    force[0] = Fx
    force[1] = Fy
    return force


# ======================================================================
# 4. 分区优化（源自 849_partition_brute）
# ======================================================================

def partition_dofs_brute(
    coupling_weights: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """
    使用组合搜索将 6-DOF 平台运动分解为两个耦合最小的子系统。
    输入 coupling_weights[i,j] 为自由度 i 和 j 之间的耦合强度。
    目标：min |sum(W0) - sum(W1)|，其中 W0, W1 为两个子集内的总耦合权重。
    返回 (partition, discrepancy)。
    """
    n = coupling_weights.shape[0]
    if n > 10:
        # 对于大系统使用贪心近似
        return _partition_greedy(coupling_weights)
    best_disc = float('inf')
    best_mask = np.zeros(n, dtype=int)
    total_subsets = 1 << n
    for mask in range(total_subsets):
        # 保证两个子集都非空
        if mask == 0 or mask == (total_subsets - 1):
            continue
        sum0 = 0.0
        sum1 = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                w = coupling_weights[i, j]
                in0_i = (mask >> i) & 1
                in0_j = (mask >> j) & 1
                if in0_i == in0_j:
                    if in0_i:
                        sum0 += w
                    else:
                        sum1 += w
        disc = abs(sum0 - sum1)
        if disc < best_disc:
            best_disc = disc
            best_mask = np.array([(mask >> i) & 1 for i in range(n)])
    return best_mask, best_disc


def _partition_greedy(coupling_weights: np.ndarray) -> Tuple[np.ndarray, float]:
    """贪心分区近似。"""
    n = coupling_weights.shape[0]
    mask = np.zeros(n, dtype=int)
    # 按总耦合权重排序
    total_w = np.sum(coupling_weights, axis=1)
    order = np.argsort(-total_w)
    sum0 = 0.0
    sum1 = 0.0
    for idx in order:
        if sum0 <= sum1:
            mask[idx] = 0
            sum0 += total_w[idx]
        else:
            mask[idx] = 1
            sum1 += total_w[idx]
    disc = abs(sum0 - sum1)
    return mask, disc


def build_coupling_matrix_from_stiffness(
    K: np.ndarray, threshold: float = 1e-3
) -> np.ndarray:
    """
    从刚度矩阵构建耦合权重矩阵：
        W[i,j] = |K[i,j]| / sqrt(|K[i,i]·K[j,j]|)
    对角元设为零。
    """
    n = K.shape[0]
    W = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            kii = abs(K[i, i])
            kjj = abs(K[j, j])
            if kii > 1e-12 and kjj > 1e-12:
                W[i, j] = abs(K[i, j]) / np.sqrt(kii * kjj)
            if W[i, j] < threshold:
                W[i, j] = 0.0
    return W


# ======================================================================
# 5. 平台动力响应主驱动
# ======================================================================

def simulate_platform_response(
    mass: float = 3.5e7,  # kg，典型半潜平台排水量
    cog: Optional[np.ndarray] = None,
    inertia: Optional[np.ndarray] = None,
    A_add: Optional[np.ndarray] = None,
    B_rad: Optional[np.ndarray] = None,
    C_rest: Optional[np.ndarray] = None,
    wave_force_func: Optional[Callable] = None,
    mooring_config: Optional[List[dict]] = None,
    tspan: Tuple[float, float] = (0.0, 300.0),
    n_steps: int = 600,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    模拟海洋平台在波浪和系泊作用下的六自由度动力响应。
    返回 (t_arr, xi_arr, info_dict)。
    xi_arr 的列分别为 [surge, sway, heave, roll, pitch, yaw, surge_dot, sway_dot, ...]。
    """
    if cog is None:
        cog = np.array([0.0, 0.0, -10.0])
    if inertia is None:
        inertia = np.array([2.5e10, 2.5e10, 3.0e10])
    if A_add is None:
        # 典型附加质量（对角近似）
        A_add = np.diag([5.0e6, 5.0e6, 8.0e6, 1.0e9, 1.0e9, 5.0e8])
    if B_rad is None:
        # 典型辐射阻尼（对角近似）
        B_rad = np.diag([2.0e5, 2.0e5, 3.0e5, 5.0e7, 5.0e7, 2.0e7])
    if C_rest is None:
        C_rest = np.diag([0.0, 0.0, 2.5e8, 1.5e10, 1.5e10, 5.0e9])

    M = build_rigid_body_mass_matrix(mass, cog, inertia)
    M_total = M + A_add

    # 分区优化（源自 849_partition_brute）
    K_coupling = M_total + B_rad + C_rest
    W = build_coupling_matrix_from_stiffness(K_coupling)
    partition, disc = partition_dofs_brute(W)

    # 构建状态空间方程：y = [ξ; ξ_dot]，dim = 12
    n_dof = 6
    n_state = 2 * n_dof

    def ode_func(t: float, y: np.ndarray) -> np.ndarray:
        xi = y[:n_dof].copy()
        xi_dot = y[n_dof:].copy()
        # 物理边界限制，防止数值溢出
        for i in range(3):
            xi[i] = max(-50.0, min(50.0, xi[i]))
            xi_dot[i] = max(-10.0, min(10.0, xi_dot[i]))
        for i in range(3, 6):
            xi[i] = max(-0.5, min(0.5, xi[i]))
            xi_dot[i] = max(-0.3, min(0.3, xi_dot[i]))
        # 外力
        F_ext = np.zeros(n_dof)
        if wave_force_func is not None:
            F_ext += wave_force_func(t, xi)
        if mooring_config is not None:
            for mc in mooring_config:
                F_moor = catenary_mooring_force(
                    xi[0], xi[1], mc["anchor"], mc["length"],
                    mc["weight"], mc["EA"], mc["pretension"],
                )
                F_ext += F_moor[:n_dof]
        # TODO: 组装并求解六自由度运动方程
        # 关键物理：
        #   (M + A_add)·ξ¨ = -B_rad·ξ˙ - C_rest·ξ + F_ext
        #   其中 M 为刚体质量/惯性矩阵，A_add 为附加质量，
        #   B_rad 为辐射阻尼，C_rest 为静水恢复力矩阵。
        #   状态空间形式：y_dot = [ξ_dot; ξ¨]
        raise NotImplementedError("ode_func 中的运动方程组装需要实现")

    y0 = np.zeros(n_state)
    t_arr, y_arr = bdf2_solve(ode_func, tspan, y0, n_steps)

    info = {
        "partition": partition,
        "partition_discrepancy": disc,
        "M_total": M_total,
        "B_rad": B_rad,
        "C_rest": C_rest,
    }
    return t_arr, y_arr, info
