"""
detector.py
引力波探测器响应与多探测器天球定位模块。

融合种子项目:
- 307_distance_to_position_sphere: 球面距离到位置反演 → 引力波源天球定位
- 872_ply_display: 3D几何张量计算 → 探测器臂方向张量与响应函数

核心公式:
1. 探测器响应张量 (LIGO/Virgo 型干涉仪):
   D^{ab} = (1/2)(u^a u^b - v^a v^b)
   
   其中 u, v 为两条臂的单位方向矢量。
   对引力波度规微扰 h_{ab} 的响应:
   h(t) = D^{ab} h_{ab}(t)

2. 引力波度规微扰 (TT 规范):
   h_{ab} = h_+ e^+_{ab} + h_× e^×_{ab}
   
   其中偏振张量:
     e^+_{ab} = l_a l_b - m_a m_b
     e^×_{ab} = l_a m_b + m_a l_b
   
   (l, m, n) 为波传播方向 n 与两个正交横向方向构成的右手标架。

3. 天线方向函数 ( detector pattern functions ):
   F^+(θ, φ, ψ) = (1/2)(1 + cos^2θ) cos(2φ) cos(2ψ) - cosθ sin(2φ) sin(2ψ)
   F^×(θ, φ, ψ) = (1/2)(1 + cos^2θ) cos(2φ) sin(2ψ) + cosθ sin(2φ) cos(2ψ)
   
   其中 (θ, φ) 为源方向的天球坐标，ψ 为偏振角。

4. 多探测器到达时间差定位:
   对于两个探测器 i, j:
   Δt_{ij} = (r_i - r_j) · n / c
   
   其中 r_i 为探测器位置，n 为波传播方向。
   球面几何约束: |n| = 1
"""

import numpy as np
from numpy.linalg import lstsq, norm


# ---------------------------------------------------------------------------
# 探测器臂方向张量 (源自 872_ply_display 的3D几何思想)
# ---------------------------------------------------------------------------

def detector_tensor(arm1, arm2):
    """
    计算探测器响应张量 D^{ab} = 0.5*(u^a u^b - v^a v^b)。
    
    参数:
        arm1, arm2: 长度为3的单位向量，表示两条臂的方向。
    
    在数值相对论波形分析中，此张量将 TT 规范下的度规微扰
    投影到探测器输出信号。
    """
    u = np.asarray(arm1, dtype=np.float64)
    v = np.asarray(arm2, dtype=np.float64)
    
    if u.shape != (3,) or v.shape != (3,):
        raise ValueError("臂方向必须为三维向量")
    
    # 归一化
    u_norm = norm(u)
    v_norm = norm(v)
    if u_norm < 1e-12 or v_norm < 1e-12:
        raise ValueError("臂方向向量不能为零")
    u = u / u_norm
    v = v / v_norm
    
    # 确保正交性（LIGO臂近似正交）
    dot_uv = np.dot(u, v)
    if np.abs(dot_uv) > 0.01:
        # Gram-Schmidt 正交化
        v = v - dot_uv * u
        v_norm = norm(v)
        if v_norm < 1e-12:
            raise ValueError("臂方向线性相关")
        v = v / v_norm
    
    D = 0.5 * (np.outer(u, u) - np.outer(v, v))
    return D


def antenna_pattern_functions(theta, phi, psi, arm1, arm2):
    """
    计算探测器的天线方向函数 F^+ 和 F^×。
    
    公式推导:
    设波传播方向 n = (sinθ cosφ, sinθ sinφ, cosθ)。
    选择横向标架:
        l = (cosθ cosφ, cosθ sinφ, -sinθ)
        m = (-sinφ, cosφ, 0)
    偏振角 ψ 旋转: (l', m') = (l cosψ + m sinψ, -l sinψ + m cosψ)
    
    则:
        F^+ = D^{ab} (l'_a l'_b - m'_a m'_b)
        F^× = D^{ab} (l'_a m'_b + m'_a l'_b)
    """
    theta = float(theta)
    phi = float(phi)
    psi = float(psi)
    
    # 波传播方向
    n = np.array([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta)
    ], dtype=np.float64)
    
    # 横向标架
    l = np.array([
        np.cos(theta) * np.cos(phi),
        np.cos(theta) * np.sin(phi),
        -np.sin(theta)
    ], dtype=np.float64)
    m = np.array([-np.sin(phi), np.cos(phi), 0.0], dtype=np.float64)
    
    # 偏振角旋转
    lp = l * np.cos(psi) + m * np.sin(psi)
    mp = -l * np.sin(psi) + m * np.cos(psi)
    
    # 偏振张量
    e_plus = np.outer(lp, lp) - np.outer(mp, mp)
    e_cross = np.outer(lp, mp) + np.outer(mp, lp)
    
    D = detector_tensor(arm1, arm2)
    
    F_plus = np.sum(D * e_plus)
    F_cross = np.sum(D * e_cross)
    
    return F_plus, F_cross


# ---------------------------------------------------------------------------
# 球面几何与距离反演 (源自 307_distance_to_position_sphere)
# ---------------------------------------------------------------------------

def spherical_distance(lat1, lon1, lat2, lon2, radius=1.0):
    """
    计算球面上两点的大圆距离。
    
    Haversine 公式:
        a = sin^2(Δφ/2) + cosφ1 cosφ2 sin^2(Δλ/2)
        c = 2 * atan2(√a, √(1-a))
        d = R * c
    """
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
    a = min(1.0, max(0.0, a))
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return radius * c


def latlon_to_xyz(lat, lon, radius=1.0):
    """
    将纬度/经度转换为笛卡尔坐标。
    
    公式:
        x = R cos(lon) cos(lat)
        y = R sin(lon) cos(lat)
        z = R sin(lat)
    """
    x = radius * np.cos(lon) * np.cos(lat)
    y = radius * np.sin(lon) * np.cos(lat)
    z = radius * np.sin(lat)
    return np.array([x, y, z], dtype=np.float64)


def xyz_to_latlon(xyz, radius=1.0):
    """
    笛卡尔坐标转纬度/经度。
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    if xyz.shape[0] != 3:
        raise ValueError("xyz 必须为三维向量")
    
    r = norm(xyz)
    if r < 1e-12:
        return 0.0, 0.0
    
    lat = np.arcsin(np.clip(xyz[2] / r, -1.0, 1.0))
    lon = np.arctan2(xyz[1], xyz[0])
    return lat, lon


def compute_sky_position_from_time_delays(detector_positions, time_delays, radius=1.0, max_iter=100):
    """
    基于多探测器到达时间差反演引力波源在天球上的位置。
    
    数学模型:
        对于 N 个探测器，有 N(N-1)/2 个独立的时间差约束:
        Δt_{ij} = (r_i - r_j) · n / c
        
        其中 n 为单位方向向量 (未知)，r_i 为已知探测器位置。
        
        转化为非线性最小二乘问题:
        min_n Σ_{i<j} [ Δt_{ij} - (r_i - r_j)·n/c ]^2
        subject to |n| = 1
    
    求解方法: 迭代最小二乘 + 球面投影。
    
    参数:
        detector_positions: 形状 (N, 3) 的数组，探测器位置（光秒单位）
        time_delays: 形状 (N,) 的数组，相对于某个参考的到达时间
        radius: 球面半径（通常取 1，表示单位球面）
    """
    detector_positions = np.asarray(detector_positions, dtype=np.float64)
    time_delays = np.asarray(time_delays, dtype=np.float64)
    
    N = detector_positions.shape[0]
    if N < 3:
        raise ValueError("至少需要 3 个探测器进行天球定位")
    if time_delays.shape[0] != N:
        raise ValueError("time_delays 长度必须与探测器数量一致")
    
    c = 1.0  # 几何单位制
    
    # 构建约束矩阵
    # 使用探测器 0 作为参考
    n_constraints = N - 1
    A = np.zeros((n_constraints, 3), dtype=np.float64)
    b = np.zeros(n_constraints, dtype=np.float64)
    
    for i in range(1, N):
        A[i - 1, :] = (detector_positions[i, :] - detector_positions[0, :]) / c
        b[i - 1] = time_delays[i] - time_delays[0]
    
    # 初始最小二乘估计（忽略单位约束）
    n_est, residuals, rank, s = lstsq(A, b, rcond=None)
    
    # 投影到单位球面
    n_norm = norm(n_est)
    if n_norm < 1e-12:
        n_est = np.array([1.0, 0.0, 0.0])
    else:
        n_est = n_est / n_norm
    
    # 迭代细化：Gauss-Newton 迭代
    for _ in range(max_iter):
        # 残差
        residual = A.dot(n_est) - b
        
        # Jacobian（包含球面约束的拉格朗日乘子）
        J = np.vstack([A, 2.0 * n_est.reshape(1, 3)])
        rhs = np.hstack([-residual, [1.0 - norm(n_est)**2]])
        
        delta, _, _, _ = lstsq(J, rhs, rcond=None)
        n_est = n_est + delta
        
        # 重新投影到单位球面
        n_norm = norm(n_est)
        if n_norm > 1e-12:
            n_est = n_est / n_norm
        
        if norm(delta) < 1e-12:
            break
    
    lat, lon = xyz_to_latlon(n_est, radius)
    return lat, lon, n_est


def network_snr(F_plus_list, F_cross_list, h_plus, h_cross, noise_psd=1.0):
    """
    计算探测器网络的合成信噪比。
    
    公式:
        ρ_{network}^2 = Σ_k ρ_k^2
        ρ_k^2 = (4/S_n) ∫ |F_k^+ h_+ + F_k^× h_×|^2 df
    """
    rho_sq = 0.0
    for Fp, Fc in zip(F_plus_list, F_cross_list):
        h_detector = Fp * h_plus + Fc * h_cross
        rho_sq += np.sum(np.abs(h_detector)**2) / noise_psd
    
    return np.sqrt(max(rho_sq, 0.0))


# ---------------------------------------------------------------------------
# 标准探测器配置
# ---------------------------------------------------------------------------

LIGO_HANFORD_ARMS = {
    'name': 'LIGO Hanford',
    'arm1': np.array([-0.2239, 0.7998, 0.5569], dtype=np.float64),
    'arm2': np.array([-0.9140, 0.0261, -0.4049], dtype=np.float64),
    'position': np.array([-2.1614, -3.8347, 4.6005], dtype=np.float64) * 1e6 / 299792458.0  # 光秒
}

LIGO_LIVINGSTON_ARMS = {
    'name': 'LIGO Livingston',
    'arm1': np.array([-0.9546, -0.1416, 0.2622], dtype=np.float64),
    'arm2': np.array([0.2977, -0.4879, 0.8205], dtype=np.float64),
    'position': np.array([-7.4276, -0.2470, 0.5849], dtype=np.float64) * 1e6 / 299792458.0
}

VIRGO_ARMS = {
    'name': 'Virgo',
    'arm1': np.array([-0.7005, 0.2085, 0.6826], dtype=np.float64),
    'arm2': np.array([-0.0538, -0.9691, 0.2408], dtype=np.float64),
    'position': np.array([4.5464, 0.8429, 0.9877], dtype=np.float64) * 1e6 / 299792458.0
}


def get_standard_detector_network():
    """返回标准的三探测器网络配置。"""
    return [LIGO_HANFORD_ARMS, LIGO_LIVINGSTON_ARMS, VIRGO_ARMS]
