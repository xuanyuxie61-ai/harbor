"""
bandgap_analysis.py
===================
光子晶体带隙分析引擎

融合原项目:
  - 1196_task_division : 任务分配算法 (用于 k 点并行化分区)
  - 1036_rk4           : 四阶 Runge-Kutta ODE 求解器
  - 1070_shallow_water_1d: 一维守恒律差分格式 (用于耦合模方程)

本模块实现:
  1. k 空间布洛赫模的色散关系计算
  2. 带隙边界定位与宽度分析
  3. 缺陷态/波导模式的耦合模理论 (CMT) 模拟
  4. 基于 RK4 的时域耦合模传播
"""

import numpy as np
from physics_core import (
    C_0, coupled_mode_equations, bragg_reflectivity,
    normalized_frequency, bandgap_ratio, cavity_q_factor
)


# =============================================================================
# 基于 1196_task_division 的 k 点任务分区
# =============================================================================

def task_division(task_number, proc_first, proc_last):
    """
    将任务分配给多个处理器 —— 基于 task_division.m
    
    用于将布里渊区的大量 k 点计算分配给多个并行任务。
    
    分配策略:
        每个处理器获得连续的任务区间 [i_lo, i_hi]
        任务数尽可能均匀分配
    
    Parameters
    ----------
    task_number : int
        总任务数
    proc_first : int
        起始处理器编号
    proc_last : int
        终止处理器编号
    
    Returns
    -------
    divisions : list of tuple
        每个处理器分配的任务区间 [(proc, n_tasks, i_lo, i_hi), ...]
    """
    if task_number < 1:
        raise ValueError("任务数必须 >= 1")
    if proc_first > proc_last:
        raise ValueError("proc_first 必须 <= proc_last")
    
    p = proc_last + 1 - proc_first
    divisions = []
    i_hi = 0
    task_remain = task_number
    proc_remain = p
    
    for proc in range(proc_first, proc_last + 1):
        task_proc = int(np.round(task_remain / proc_remain))
        proc_remain -= 1
        task_remain -= task_proc
        
        i_lo = i_hi + 1
        i_hi = i_hi + task_proc
        
        divisions.append((proc, task_proc, i_lo, i_hi))
    
    return divisions


def divide_k_points(k_points, n_workers):
    """
    将 k 点路径划分为多个子集用于并行计算
    
    Parameters
    ----------
    k_points : ndarray
        k 点数组
    n_workers : int
        工作进程数
    
    Returns
    -------
    list of ndarray
        每个 worker 的 k 点子集
    """
    N = len(k_points)
    if n_workers < 1:
        raise ValueError("工作进程数必须 >= 1")
    if N < n_workers:
        n_workers = N
    
    divisions = task_division(N, 0, n_workers - 1)
    subsets = []
    for _, _, i_lo, i_hi in divisions:
        subsets.append(k_points[i_lo - 1:i_hi])
    return subsets


# =============================================================================
# 基于 1036_rk4 的耦合模方程传播
# =============================================================================

def rk4(dydt, tspan, y0, n_steps):
    """
    四阶 Runge-Kutta ODE 求解器 —— 基于 rk4.m
    
    RK4 递推公式:
        k₁ = h·f(tₙ, yₙ)
        k₂ = h·f(tₙ + h/2, yₙ + k₁/2)
        k₃ = h·f(tₙ + h/2, yₙ + k₂/2)
        k₄ = h·f(tₙ + h, yₙ + k₃)
        y_{n+1} = yₙ + (k₁ + 2k₂ + 2k₃ + k₄)/6
    
    局部截断误差: O(h⁵)
    全局累积误差: O(h⁴)
    
    Parameters
    ----------
    dydt : callable
        右端项函数 f(t, y)，返回 ndarray
    tspan : tuple
        (t_start, t_end)
    y0 : ndarray
        初始条件
    n_steps : int
        时间步数
    
    Returns
    -------
    t : ndarray
        时间节点
    y : ndarray
        解轨迹
    """
    y0 = np.asarray(y0, dtype=complex).flatten()
    m = len(y0)
    
    t = np.zeros(n_steps + 1)
    y = np.zeros((n_steps + 1, m), dtype=complex)
    
    tfirst, tlast = tspan
    dt = (tlast - tfirst) / n_steps
    
    t[0] = tfirst
    y[0, :] = y0
    
    for i in range(n_steps):
        f1 = dydt(t[i], y[i, :])
        f2 = dydt(t[i] + dt / 2.0, y[i, :] + dt * f1 / 2.0)
        f3 = dydt(t[i] + dt / 2.0, y[i, :] + dt * f2 / 2.0)
        f4 = dydt(t[i] + dt, y[i, :] + dt * f3)
        
        t[i + 1] = t[i] + dt
        y[i + 1, :] = y[i, :] + dt * (f1 + 2.0 * f2 + 2.0 * f3 + f4) / 6.0
    
    return t, y


def propagate_bragg_grating(kappa, delta_beta, L, n_z=200):
    """
    用 RK4 + 线性打靶法模拟光在布拉格光栅中的传播
    
    求解耦合模方程:
        dA⁺/dz =  iδβ A⁺ + iκ A⁻
        dA⁻/dz = -iδβ A⁻ + iκ* A⁺
    
    边界条件: A⁺(0)=1, A⁻(L)=0 (单端入射)
    
    线性打靶法:
        1. 求解初值问题 u: u⁺(0)=1, u⁻(0)=0
        2. 求解初值问题 v: v⁺(0)=0, v⁻(0)=1
        3. 由 A⁻(L)=0 得: g = -u⁻(L) / v⁻(L)
        4. 真解 A = u + g·v
    
    Parameters
    ----------
    kappa : complex
        耦合系数 [m⁻¹]
    delta_beta : float
        传播常数失谐 [m⁻¹]
    L : float
        光栅长度 [m]
    n_z : int
        空间离散点数
    
    Returns
    -------
    z : ndarray
        空间坐标 [m]
    A_plus : ndarray
        前向模振幅
    A_minus : ndarray
        后向模振幅
    reflectivity : float
        功率反射率
    """
    # TODO: Implement the linear shooting method for the Bragg grating boundary-value problem.
    #
    # Key numerical method:
    #   1. Define RHS using coupled_mode_equations(kappa, delta_beta, alpha=0.0)
    #   2. Solve two IVPs with RK4:
    #        u: u⁺(0)=1, u⁻(0)=0
    #        v: v⁺(0)=0, v⁻(0)=1
    #   3. Enforce boundary condition A⁻(L)=0 at z=L:
    #        g = -u⁻(L) / v⁻(L)   (if v⁻(L) ≠ 0)
    #   4. Construct true solution: A = u + g·v
    #   5. Compute numerical reflectivity: R = |A⁻(0)/A⁺(0)|²
    #
    # Return: z, A_plus, A_minus, reflectivity
    raise NotImplementedError("Hole 2: propagate_bragg_grating shooting method needs to be implemented.")


# =============================================================================
# 基于 1070_shallow_water_1d 思想的一维守恒律差分
# =============================================================================

def coupled_mode_fdtm(kappa_profile, delta_beta, L, nz, dt_factor=0.5):
    """
    一维耦合模方程的有限差分时域方法 (FDTM)
    
    受 shallow_water_1d 中守恒律差分格式的启发，采用
    Lax-Wendroff 型守恒格式:
    
        ∂A⁺/∂z =  iδβ(z) A⁺ + iκ(z) A⁻
        ∂A⁻/∂z = -iδβ(z) A⁻ + iκ(z) A⁺
    
    改写为拟守恒形式:
        ∂U/∂z + ∂F/∂z = S
    
    其中 U = [A⁺, A⁻]ᵀ, F = [-iδβ A⁺, iδβ A⁻]ᵀ, S = [iκ A⁻, iκ A⁺]ᵀ
    
    Parameters
    ----------
    kappa_profile : callable or ndarray
        沿 z 方向的耦合系数分布 [m⁻¹]
    delta_beta : float
        平均传播常数失谐 [m⁻¹]
    L : float
        总长度 [m]
    nz : int
        空间网格数
    dt_factor : float
        CFL 稳定性因子 (建议 < 1)
    
    Returns
    -------
    z : ndarray
        空间网格 [m]
    A_plus : ndarray
        前向模
    A_minus : ndarray
        后向模
    """
    if L <= 0 or nz < 3:
        raise ValueError("参数超出允许范围")
    
    dz = L / (nz - 1)
    z = np.linspace(0, L, nz)
    
    # 生成耦合系数分布
    if callable(kappa_profile):
        kappa_z = np.array([kappa_profile(zi) for zi in z], dtype=complex)
    else:
        kappa_z = np.full(nz, kappa_profile, dtype=complex)
    
    # 稳定性条件
    beta_max = abs(delta_beta) + np.max(np.abs(kappa_z))
    if beta_max < 1e-15:
        beta_max = 1.0
    
    # 场量初始化
    A_plus = np.zeros(nz, dtype=complex)
    A_minus = np.zeros(nz, dtype=complex)
    A_plus[0] = 1.0
    
    # 逐层传播 (类似于 shallow water 的逐时间步推进)
    for i in range(nz - 1):
        # 局部耦合系数
        kappa_local = 0.5 * (kappa_z[i] + kappa_z[i + 1])
        
        # Lax-Wendroff 型更新
        Ap_mid = 0.5 * (A_plus[i] + A_plus[i + 1]) if i < nz - 2 else A_plus[i]
        Am_mid = 0.5 * (A_minus[i] + A_minus[i + 1]) if i < nz - 2 else A_minus[i]
        
        # 前向模更新
        dAp = 1j * delta_beta * A_plus[i] + 1j * kappa_local * A_minus[i]
        A_plus[i + 1] = A_plus[i] + dz * dAp
        
        # 后向模更新 (从右边界向内传播)
        # 使用预估-校正
        dAm = -1j * delta_beta * A_minus[i] + 1j * np.conj(kappa_local) * A_plus[i]
        A_minus[i + 1] = A_minus[i] + dz * dAm
    
    return z, A_plus, A_minus


# =============================================================================
# 带隙检测与分析
# =============================================================================

def detect_bandgaps(omega_bands, threshold_ratio=0.05):
    """
    从能带数据中检测光子带隙
    
    判定准则:
        对于所有 k 点，若第 n 带的最高频率 < 第 n+1 带的最低频率，
        则存在带隙。
    
    带隙中心频率:
        ω_c = (ω_{n,max} + ω_{n+1,min}) / 2
    
    带隙相对宽度:
        Δω/ω_c = 2(ω_{n+1,min} - ω_{n,max}) / (ω_{n+1,min} + ω_{n,max})
    
    Parameters
    ----------
    omega_bands : ndarray, shape (N_k, n_bands)
        能带频率数据
    threshold_ratio : float
        最小可识别带隙相对宽度
    
    Returns
    -------
    gaps : list of dict
        每个带隙的信息字典
    """
    N_k, n_bands = omega_bands.shape
    if n_bands < 2:
        return []
    
    gaps = []
    
    for band_idx in range(n_bands - 1):
        band_n_max = np.max(omega_bands[:, band_idx])
        band_np1_min = np.min(omega_bands[:, band_idx + 1])
        
        if band_np1_min > band_n_max:
            gap_width = band_np1_min - band_n_max
            gap_center = 0.5 * (band_n_max + band_np1_min)
            gap_ratio = bandgap_ratio(band_n_max, band_np1_min)
            
            if gap_ratio >= threshold_ratio:
                gaps.append({
                    'lower_band': band_idx,
                    'upper_band': band_idx + 1,
                    'omega_lower': band_n_max,
                    'omega_upper': band_np1_min,
                    'omega_center': gap_center,
                    'gap_width': gap_width,
                    'relative_width': gap_ratio,
                    'mid_gap_frequency': gap_center
                })
    
    return gaps


def gap_mismatch_parameter(eps_bg, eps_hole, fill_factor):
    """
    光子晶体带隙的介电常数失配参数
    
    经验公式 (来自 Yablonovitch-John 理论):
        Δω/ω ≈ (4/π) |√ε₁ - √ε₂| / (√ε₁ + √ε₂) · |sin(π·f)|
    
    其中 f 为填充因子。
    
    Parameters
    ----------
    eps_bg : float
        背景介电常数
    eps_hole : float
        孔内介电常数
    fill_factor : float
        高介电材料填充因子 [0, 1]
    
    Returns
    -------
    float
        预估最大相对带隙宽度
    """
    if eps_bg <= 0 or eps_hole <= 0:
        raise ValueError("介电常数必须为正")
    if not (0 <= fill_factor <= 1):
        raise ValueError("填充因子必须在 [0, 1] 区间内")
    
    n_bg = np.sqrt(eps_bg)
    n_hole = np.sqrt(eps_hole)
    
    mismatch = abs(n_bg - n_hole) / (n_bg + n_hole)
    geometric_factor = abs(np.sin(np.pi * fill_factor))
    
    return (4.0 / np.pi) * mismatch * geometric_factor


def defect_mode_frequency(omega_gap_center, defect_strength, Q_factor):
    """
    点缺陷微腔的共振频率估计
    
    对于介电常数局域微扰 δε(r) 引入的缺陷态:
        ω_d ≈ ω_c + δω
    
    其中频率移动:
        δω/ω_c ≈ -∫ δε(r)|E(r)|² d³r / (2∫ ε(r)|E(r)|² d³r)
    
    Parameters
    ----------
    omega_gap_center : float
        带隙中心频率 [rad/s]
    defect_strength : float
        缺陷强度参数 (介电常数相对变化)
    Q_factor : float
        微腔品质因子
    
    Returns
    -------
    omega_defect : float
        缺陷态频率 [rad/s]
    delta_omega : float
        线宽 [rad/s]
    """
    if omega_gap_center <= 0 or Q_factor <= 0:
        raise ValueError("频率和 Q 值必须为正")
    
    # 简化的缺陷频率移动模型
    omega_defect = omega_gap_center * (1.0 - 0.5 * defect_strength)
    delta_omega = omega_defect / Q_factor
    
    return omega_defect, delta_omega


def slow_light_group_index(omega, k, band_index=0):
    """
    计算慢光群折射率
    
    群速度:
        v_g = dω/dk
    
    群折射率:
        n_g = c / v_g = c · dk/dω
    
    在能带边缘 v_g → 0，因此 n_g → ∞。
    
    Parameters
    ----------
    omega : ndarray
        频率数组
    k : ndarray
        波矢数组 (标量，沿路径的累积距离)
    band_index : int
        能带索引
    
    Returns
    -------
    n_g : ndarray
        群折射率
    """
    if len(omega) < 3 or len(k) < 3:
        raise ValueError("数据点必须至少 3 个")
    if len(omega) != len(k):
        raise ValueError("omega 和 k 长度必须一致")
    
    # 数值微分 dk/dω (用中心差分)
    dk_domega = np.zeros(len(omega))
    
    domega_0 = omega[1] - omega[0]
    if abs(domega_0) > 1e-18:
        dk_domega[0] = (k[1] - k[0]) / domega_0
    
    domega_end = omega[-1] - omega[-2]
    if abs(domega_end) > 1e-18:
        dk_domega[-1] = (k[-1] - k[-2]) / domega_end
    
    for i in range(1, len(omega) - 1):
        domega = omega[i + 1] - omega[i - 1]
        if abs(domega) < 1e-18:
            dk_domega[i] = 0.0
        else:
            dk_domega[i] = (k[i + 1] - k[i - 1]) / domega
    
    n_g = C_0 * dk_domega
    # 边界处理: 限制群折射率范围
    n_g = np.clip(n_g, -1e6, 1e6)
    
    return n_g
