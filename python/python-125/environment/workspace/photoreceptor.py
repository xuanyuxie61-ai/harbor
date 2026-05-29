"""
photoreceptor.py
光感受器光转导与光适应模型

基于以下种子项目合成：
- 512_heated_plate: 稳态热方程有限差分求解（Jacobi迭代）
- 144_cc_project: Clenshaw-Curtis求积公式

科学背景：
光感受器（视杆/视锥细胞）将光信号转换为电信号。
本模块模拟：
1. 光适应的稳态方程（基于 heated_plate 的有限差分）
2. 光转导级联的微分方程
3. 光电流的数值积分（基于 Clenshaw-Curtis 求积）

关键生物学公式：
- 光感受器外段电流：I_dark - I_light = g_max * cGMP^3 / (cGMP^3 + K_cGMP^3)
- 光适应遵循Weber-Fechner定律：ΔI/I = 常数
"""

import numpy as np
from typing import Tuple


# =============================================================================
# 光适应稳态方程（基于512_heated_plate的有限差分Jacobi迭代）
# =============================================================================

def solve_light_adaptation_steady_state(
    nx: int, ny: int,
    I_top: float, I_bottom: float, I_left: float, I_right: float,
    source_term: np.ndarray,
    epsilon: float = 1e-8,
    max_iter: int = 50000
) -> Tuple[np.ndarray, int, float]:
    """
    求解光感受器外段中的光适应稳态方程。
    
    光适应在光感受器外段（outer segment）中可建模为稳态扩散-反应方程：
    
        D * ∇²C(x,y) + S(x,y) = 0
    
    其中：
    - C(x,y) 为光感受器外段中的钙离子浓度（或cGMP浓度）
    - D 为扩散系数
    - S(x,y) 为光源项（由入射光强度决定）
    
    使用中心差分离散Laplacian：
        ∇²C_{i,j} ≈ (C_{i-1,j} + C_{i+1,j} + C_{i,j-1} + C_{i,j+1} - 4*C_{i,j}) / h²
    
    采用Jacobi迭代求解：
        C_{i,j}^{(new)} = (C_{i-1,j} + C_{i+1,j} + C_{i,j-1} + C_{i,j+1} + h²*S_{i,j}/D) / 4
    
    参数:
        nx, ny: 网格维度
        I_top, I_bottom, I_left, I_right: 边界Dirichlet条件
        source_term: (nx, ny) 光源项矩阵
        epsilon: 收敛容限
        max_iter: 最大迭代次数
    
    返回:
        C: (nx, ny) 稳态浓度分布
        iterations: 实际迭代次数
        final_error: 最终误差
    """
    # 初始化
    C = np.zeros((nx, ny), dtype=np.float64)
    
    # 设置边界条件
    C[0, :] = I_top       # 上边界
    C[-1, :] = I_bottom   # 下边界
    C[:, 0] = I_left      # 左边界
    C[:, -1] = I_right    # 右边界
    
    C_new = C.copy()
    h2 = 1.0  # 假设 h = 1，h²*S/D 已包含在source_term中
    
    for iteration in range(1, max_iter + 1):
        # Jacobi迭代：只更新内部点
        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                C_new[i, j] = 0.25 * (
                    C[i - 1, j] + C[i + 1, j] +
                    C[i, j - 1] + C[i, j + 1] +
                    h2 * source_term[i, j]
                )
        
        # 计算误差
        diff = np.max(np.abs(C_new - C))
        C, C_new = C_new, C  # 交换，避免复制
        
        if diff < epsilon:
            return C, iteration, diff
    
    return C, max_iter, diff


# =============================================================================
# Clenshaw-Curtis求积（基于144_cc_project）
# =============================================================================

def clenshaw_curtis_nodes_weights(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成标准Clenshaw-Curtis求积的节点和权重。
    
    在区间 [-1, 1] 上，节点为Chebyshev极端点：
        x_j = cos(j * π / (n-1)),  j = 0, 1, ..., n-1
    
    权重通过离散余弦变换（DCT）计算：
        w_j = c_j / (n-1) * [1 - Σ_{k=1}^{⌊(n-1)/2⌋} b_k * cos(2kθ_j) / (4k² - 1)]
    
    其中：
        θ_j = j * π / (n-1)
        c_0 = c_{n-1} = 1/2,  c_j = 1  (其他)
        b_k = 1  (k < (n-1)/2),  b_k = 1/2  (k = (n-1)/2)
    
    参数:
        n: 求积节点数（n ≥ 2）
    
    返回:
        x: (n,) 求积节点
        w: (n,) 求积权重
    """
    if n < 2:
        raise ValueError("n must be at least 2")
    
    # Chebyshev极端点
    j = np.arange(n, dtype=np.float64)
    x = np.cos(j * np.pi / (n - 1))
    
    # 计算权重
    theta = j * np.pi / (n - 1)
    
    c = np.ones(n, dtype=np.float64)
    c[0] = 0.5
    c[-1] = 0.5
    
    w = np.zeros(n, dtype=np.float64)
    
    half_nm1 = (n - 1) / 2.0
    
    for j_idx in range(n):
        sum_val = 0.0
        for k in range(1, int(np.floor(half_nm1)) + 1):
            if k < half_nm1:
                b_k = 1.0
            else:
                b_k = 0.5
            sum_val += b_k * np.cos(2.0 * k * theta[j_idx]) / (4.0 * k * k - 1.0)
        w[j_idx] = c[j_idx] / (n - 1.0) * (1.0 - sum_val)
    
    return x, w


def integrate_photocurrent_clenshaw_curtis(
    intensity_profile: callable,
    a: float, b: float,
    n: int = 64
) -> float:
    """
    使用Clenshaw-Curtis求积计算光感受器的光电流积分。
    
    光感受器的光电流响应可表示为：
        I_photo = ∫_a^b R(I(x)) dx
    
    其中R(I)为光响应函数（非线性），I(x)为沿外段轴的光强度分布。
    
    通过变量替换 x = (b-a)/2 * t + (b+a)/2, t∈[-1,1]：
        I_photo = (b-a)/2 * ∫_{-1}^{1} R(I((b-a)/2 * t + (b+a)/2)) dt
    
    参数:
        intensity_profile: 光强度分布函数 I(x)
        a, b: 积分区间
        n: CC求积阶数
    
    返回:
        photocurrent: 光电流估计值
    """
    x_nodes, w = clenshaw_curtis_nodes_weights(n)
    
    # 区间变换
    scale = (b - a) / 2.0
    shift = (b + a) / 2.0
    t_nodes = scale * x_nodes + shift
    
    # 求积
    f_vals = np.array([intensity_profile(t) for t in t_nodes], dtype=np.float64)
    photocurrent = scale * np.sum(w * f_vals)
    
    return float(photocurrent)


# =============================================================================
# 光转导级联微分方程
# =============================================================================

def phototransduction_ode(
    t: float,
    y: np.ndarray,
    I_light: float,
    params: dict
) -> np.ndarray:
    """
    光转导级联的常微分方程组。
    
    基于Lamb-TD模型（简化版），描述视锥细胞中：
    - cGMP浓度 [cGMP]
    - 钙离子浓度 [Ca²⁺]
    - 激活的磷酸二酯酶浓度 [PDE*]
    
    微分方程组：
    
    d[PDE*]/dt = α * I_light - β * [PDE*]
    
    d[cGMP]/dt = α_gc_max / (1 + ([Ca²⁺]/K_gc)^n_gc) - γ * [PDE*] * [cGMP]
    
    d[Ca²⁺]/dt = -η * I_Ca + k_ex * ([Ca²⁺]_out - [Ca²⁺]) - k_NCX * [Ca²⁺]
    
    其中光电流：
    I_Ca = g_max * [cGMP]^3 / ([cGMP]^3 + K_cGMP^3) * (V_m - E_Ca)
    
    参数:
        t: 时间
        y: [PDE*, cGMP, Ca]
        I_light: 入射光强度
        params: 模型参数字典
    
    返回:
        dy/dt: 状态变量的时间导数
    """
    # TODO: Hole 1 — 实现光转导级联ODE系统的核心科学计算
    # 状态变量: y = [PDE*, cGMP, Ca]
    # 需要基于Lamb-TD简化模型计算三个状态变量的时间导数:
    #   1. d[PDE*]/dt = α·I_light - β·[PDE*]
    #   2. d[cGMP]/dt = α_gc_max/(1+([Ca²⁺]/K_gc)^n_gc) - γ·[PDE*]·[cGMP]
    #   3. d[Ca²⁺]/dt = -η·I_Ca
    # 其中光电流 I_Ca = g_max·[cGMP]³/([cGMP]³+K_cGMP³)·(V_m-E_Ca)
    # 所有参数从 params 字典中提取，需处理默认值和非负约束
    raise NotImplementedError("Hole 1: phototransduction_ode 核心科学计算待实现")


def solve_phototransduction_rk4(
    I_light_func: callable,
    y0: np.ndarray,
    t_span: Tuple[float, float],
    dt: float,
    params: dict
) -> Tuple[np.ndarray, np.ndarray]:
    """
    使用四阶Runge-Kutta方法求解光转导ODE。
    
    RK4公式：
        k1 = f(t_n, y_n)
        k2 = f(t_n + dt/2, y_n + dt/2 * k1)
        k3 = f(t_n + dt/2, y_n + dt/2 * k2)
        k4 = f(t_n + dt, y_n + dt * k3)
        y_{n+1} = y_n + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    
    参数:
        I_light_func: 光强度随时间变化函数 I(t)
        y0: 初始状态 [PDE*, cGMP, Ca]
        t_span: (t_start, t_end)
        dt: 时间步长
        params: 模型参数
    
    返回:
        t_array: 时间点数组
        y_array: (N, 3) 状态变量历史
    """
    t_start, t_end = t_span
    n_steps = int(np.ceil((t_end - t_start) / dt))
    dt = (t_end - t_start) / n_steps  # 调整dt使恰好整除
    
    t_array = np.zeros(n_steps + 1, dtype=np.float64)
    y_array = np.zeros((n_steps + 1, 3), dtype=np.float64)
    
    t_array[0] = t_start
    y_array[0] = y0
    
    y = y0.copy()
    
    for n in range(n_steps):
        t = t_array[n]
        I_light = I_light_func(t)
        
        # TODO: Hole 2 — 实现四阶Runge-Kutta步进公式
        # k1 = f(t_n, y_n)
        # k2 = f(t_n + dt/2, y_n + dt/2 * k1)
        # k3 = f(t_n + dt/2, y_n + dt/2 * k2)
        # k4 = f(t_n + dt, y_n + dt * k3)
        # y_{n+1} = y_n + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        # 需调用 phototransduction_ode(t, y, I_light, params) 作为右端函数 f
        # 注意数值保护: 确保状态变量非负，且 cGMP 不能为 0
        raise NotImplementedError("Hole 2: RK4步进公式待实现")
    
    return t_array, y_array
