"""
vertical_profiles.py
================================================================================
海水垂直剖面的 Hermite 三次样条插值

融合项目：
    - 518_hermite_cubic : Hermite 三次多项式与样条

核心科学问题：
    在海水垂直柱中，温度、盐度、DIC、TA 等物理化学量随深度的变化往往具有
    显著的非线性特征（跃层、混合层等）。Hermite 三次样条保证函数值和一阶
    导数在节点处连续，适用于需要同时估计梯度（如层化稳定性 N²）的场景。

科学背景：
    在深度区间 [z_k, z_{k+1}] 上，设已知：
        f_k = f(z_k),    f_{k+1} = f(z_{k+1})
        d_k = f'(z_k),   d_{k+1} = f'(z_{k+1})
    
    Hermite 三次多项式：
        P(z) = f_k + (z-z_k)·[d_k + (z-z_k)·(c2 + (z-z_k)·c3)]
    
    其中 h = z_{k+1} - z_k, df = (f_{k+1} - f_k)/h：
        c2 = -(2·d_k - 3·df + d_{k+1}) / h
        c3 =  (d_k - 2·df + d_{k+1}) / h²
    
    一阶导数连续性保证可以精确计算 Brunt-Väisälä 频率：
        N² = -(g/ρ₀)·(dρ/dz)
    
    密度由 UNESCO 状态方程计算：
        ρ = ρ(T, S, p) = ρ₀ / (1 - p/K(T,S,p))
================================================================================
"""

import numpy as np


# =============================================================================
# Hermite 三次多项式基础
# =============================================================================

def hermite_cubic_coefficients(z1, z2, f1, d1, f2, d2):
    """
    计算 Hermite 三次多项式在 [z1, z2] 上的系数 (c0, c1, c2, c3)。
    
    表示为：
        P(z) = c0 + c1·(z-z1) + c2·(z-z1)² + c3·(z-z1)³
    
    其中 c0 = f1。
    
    参数:
        z1, z2 : float, 区间端点
        f1, f2 : float, 端点函数值
        d1, d2 : float, 端点导数值
    
    返回:
        c0, c1, c2, c3 : float
    """
    h = z2 - z1
    if abs(h) < 1e-14:
        raise ValueError("区间长度 h 必须为正")
    
    df = (f2 - f1) / h
    c0 = f1
    c1 = d1
    c2 = -(2.0 * d1 - 3.0 * df + d2) / h
    c3 = (d1 - 2.0 * df + d2) / (h * h)
    return c0, c1, c2, c3


def hermite_cubic_value(z1, z2, f1, d1, f2, d2, z_query):
    """
    在查询点 z_query 处求 Hermite 三次多项式的值及其 1-3 阶导数。
    
    参数:
        z1, z2   : float, 区间端点
        f1, f2   : float, 端点函数值
        d1, d2   : float, 端点导数值
        z_query  : float or ndarray, 查询点（必须在 [z1,z2] 内或外推）
    
    返回:
        f, df, d2f, d3f : 函数值与 1-3 阶导数
    """
    c0, c1, c2, c3 = hermite_cubic_coefficients(z1, z2, f1, d1, f2, d2)
    dz = z_query - z1
    
    f = c0 + dz * (c1 + dz * (c2 + dz * c3))
    df = c1 + dz * (2.0 * c2 + dz * 3.0 * c3)
    d2f = 2.0 * c2 + dz * 6.0 * c3
    d3f = 6.0 * c3
    return f, df, d2f, d3f


def hermite_cubic_integral(z1, z2, f1, d1, f2, d2):
    """
    计算 Hermite 三次多项式在 [z1, z2] 上的定积分。
    
    ∫_{z1}^{z2} P(z) dz = h·[f1 + h/2·d1 + h²/3·c2 + h³/4·c3]
    """
    h = z2 - z1
    c0, c1, c2, c3 = hermite_cubic_coefficients(z1, z2, f1, d1, f2, d2)
    return h * (c0 + h / 2.0 * c1 + h**2 / 3.0 * c2 + h**3 / 4.0 * c3)


# =============================================================================
# 分段 Hermite 三次样条
# =============================================================================

def build_hermite_spline(z_nodes, f_nodes, d_nodes):
    """
    构建分段 Hermite 三次样条。
    
    参数:
        z_nodes : ndarray, shape (n,), 单调递增的节点坐标
        f_nodes : ndarray, shape (n,), 节点函数值
        d_nodes : ndarray, shape (n,), 节点导数值
    
    返回:
        dict: 包含节点和系数的样条表示
    """
    n = len(z_nodes)
    if len(f_nodes) != n or len(d_nodes) != n:
        raise ValueError("z_nodes, f_nodes, d_nodes 长度必须相等")
    if n < 2:
        raise ValueError("至少需要 2 个节点")
    
    # 检查单调性
    if not np.all(np.diff(z_nodes) > 0):
        raise ValueError("z_nodes 必须严格单调递增")
    
    coeffs = []
    for k in range(n - 1):
        c0, c1, c2, c3 = hermite_cubic_coefficients(
            z_nodes[k], z_nodes[k+1],
            f_nodes[k], d_nodes[k],
            f_nodes[k+1], d_nodes[k+1]
        )
        coeffs.append((c0, c1, c2, c3))
    
    return {
        'z_nodes': z_nodes.copy(),
        'f_nodes': f_nodes.copy(),
        'd_nodes': d_nodes.copy(),
        'coeffs': coeffs,
        'n_segments': n - 1,
    }


def evaluate_hermite_spline(spline, z_query):
    """
    求分段 Hermite 三次样条在查询点的值与导数。
    
    参数:
        spline   : dict, build_hermite_spline 的输出
        z_query  : float or ndarray
    
    返回:
        f, df, d2f, d3f
    """
    z_nodes = spline['z_nodes']
    coeffs = spline['coeffs']
    
    is_scalar = np.isscalar(z_query)
    zq = np.atleast_1d(z_query)
    
    f = np.zeros_like(zq, dtype=float)
    df = np.zeros_like(zq, dtype=float)
    d2f = np.zeros_like(zq, dtype=float)
    d3f = np.zeros_like(zq, dtype=float)
    
    z_min = z_nodes[0]
    z_max = z_nodes[-1]
    
    for i, z in enumerate(zq):
        # 边界处理：外推使用最近区间
        if z < z_min:
            seg = 0
        elif z >= z_max:
            seg = len(coeffs) - 1
        else:
            # 二分查找区间
            seg = np.searchsorted(z_nodes, z, side='right') - 1
            seg = max(0, min(seg, len(coeffs) - 1))
        
        c0, c1, c2, c3 = coeffs[seg]
        dz = z - z_nodes[seg]
        f[i] = c0 + dz * (c1 + dz * (c2 + dz * c3))
        df[i] = c1 + dz * (2.0 * c2 + dz * 3.0 * c3)
        d2f[i] = 2.0 * c2 + dz * 6.0 * c3
        d3f[i] = 6.0 * c3
    
    if is_scalar:
        return f[0], df[0], d2f[0], d3f[0]
    return f, df, d2f, d3f


def integrate_hermite_spline(spline, a, b):
    """
    计算样条在 [a, b] 上的积分。
    """
    z_nodes = spline['z_nodes']
    coeffs = spline['coeffs']
    
    if a > b:
        return -integrate_hermite_spline(spline, b, a)
    
    total = 0.0
    n = len(z_nodes)
    
    for k in range(n - 1):
        z1, z2 = z_nodes[k], z_nodes[k+1]
        c0, c1, c2, c3 = coeffs[k]
        
        # 计算 [a,b] 与 [z1,z2] 的交集
        left = max(a, z1)
        right = min(b, z2)
        if left >= right:
            continue
        
        # ∫_{left}^{right} [c0 + c1·(z-z1) + c2·(z-z1)² + c3·(z-z1)³] dz
        dl = left - z1
        dr = right - z1
        
        def F(dz):
            return c0 * dz + c1 * dz**2 / 2.0 + c2 * dz**3 / 3.0 + c3 * dz**4 / 4.0
        
        total += F(dr) - F(dl)
    
    return total


# =============================================================================
# 海洋垂直剖面的具体应用
# =============================================================================

def estimate_derivatives_central(z_nodes, f_nodes):
    """
    用中心差分估计节点导数（用于构建样条）。
    
    边界用前向/后向差分：
        d₀ = (f₁ - f₀) / (z₁ - z₀)
        d_k = (f_{k+1} - f_{k-1}) / (z_{k+1} - z_{k-1})
        d_n = (f_n - f_{n-1}) / (z_n - z_{n-1})
    """
    n = len(z_nodes)
    d_nodes = np.zeros(n)
    
    d_nodes[0] = (f_nodes[1] - f_nodes[0]) / (z_nodes[1] - z_nodes[0])
    d_nodes[-1] = (f_nodes[-1] - f_nodes[-2]) / (z_nodes[-1] - z_nodes[-2])
    
    for k in range(1, n - 1):
        d_nodes[k] = (f_nodes[k+1] - f_nodes[k-1]) / (z_nodes[k+1] - z_nodes[k-1])
    
    return d_nodes


def compute_brunt_vaisala_frequency(z_nodes, T_nodes, S_nodes, lat=30.0):
    """
    利用 Hermite 三次样条计算 Brunt-Väisälä 频率（浮力频率）。
    
    N² = -(g/ρ₀)·(dρ/dz)
    
    其中密度 ρ 由 UNESCO 状态方程近似（线性化形式）：
        ρ ≈ ρ₀ + α_T·(T-T₀) + β_S·(S-S₀)
        α_T ≈ -0.15 kg/(m³·°C)   (热膨胀系数)
        β_S ≈  0.78 kg/(m³·psu)   (盐度收缩系数)
    
    参数:
        z_nodes : ndarray, 深度 (m, 负值或正值均可，向上为正)
        T_nodes : ndarray, 温度 (°C)
        S_nodes : ndarray, 盐度 (psu)
        lat     : float, 纬度 (°), 用于计算局部重力加速度
    
    返回:
        N2      : ndarray, 浮力频率平方 (s⁻²)
        z_mid   : ndarray, 层间中点深度
    """
    n = len(z_nodes)
    if n < 2:
        raise ValueError("至少需要两个深度层")
    
    # 局部重力加速度 (WGS84 近似)
    phi = np.radians(lat)
    g = 9.780327 * (1.0 + 0.0053024 * np.sin(phi)**2 - 0.0000058 * np.sin(2*phi)**2)
    
    rho0 = 1025.0  # 参考密度 kg/m³
    alpha_T = -0.15  # kg/(m³·°C)
    beta_S = 0.78    # kg/(m³·psu)
    
    # 构建 T(z) 和 S(z) 的 Hermite 样条
    dT = estimate_derivatives_central(z_nodes, T_nodes)
    dS = estimate_derivatives_central(z_nodes, S_nodes)
    
    T_spline = build_hermite_spline(z_nodes, T_nodes, dT)
    S_spline = build_hermite_spline(z_nodes, S_nodes, dS)
    
    # 在层间中点计算 dρ/dz
    z_mid = 0.5 * (z_nodes[:-1] + z_nodes[1:])
    _, dT_dz, _, _ = evaluate_hermite_spline(T_spline, z_mid)
    _, dS_dz, _, _ = evaluate_hermite_spline(S_spline, z_mid)
    
    drho_dz = alpha_T * dT_dz + beta_S * dS_dz
    
    # N² = -g/ρ₀ · dρ/dz
    N2 = -(g / rho0) * drho_dz
    
    # 数值稳定性：若 N² < 0（不稳定层化），标记为 0 或保持负值
    N2 = np.where(N2 < -1e-6, N2, np.maximum(N2, 0.0))
    
    return N2, z_mid


def mixed_layer_depth(z_nodes, T_nodes, threshold=0.5):
    """
    基于温度阈值法估算混合层深度 (MLD)。
    
    MLD 定义为表层温度下降 threshold °C 的深度。
    
    使用 Hermite 样条精确插值找到阈值深度。
    """
    if len(z_nodes) < 2:
        return z_nodes[0] if len(z_nodes) > 0 else 0.0
    
    T_surface = T_nodes[0]
    T_target = T_surface - threshold
    
    # 寻找第一个使 T < T_target 的层
    for k in range(len(z_nodes) - 1):
        if (T_nodes[k] - T_target) * (T_nodes[k+1] - T_target) <= 0:
            # 线性插值找到精确深度
            z1, z2 = z_nodes[k], z_nodes[k+1]
            T1, T2 = T_nodes[k], T_nodes[k+1]
            if abs(T2 - T1) > 1e-10:
                frac = (T_target - T1) / (T2 - T1)
                return z1 + frac * (z2 - z1)
    
    return z_nodes[-1]
