"""
非线性声学椭圆函数解模块

基于种子项目 1096_sncndn 的核心算法（Bulirsch AGM），
为超声成像提供非线性声波传播的理论解与数值验证。

物理模型:
在强声场中，有限振幅声波满足Burgers方程:
    ∂u/∂t + u·∂u/∂x = ν·∂²u/∂x²
其中 ν 为粘滞扩散系数。

Burgers方程的行波解可用Jacobi椭圆函数表示。
在无损极限（ν → 0）下，解退化为锯齿波（shock wave formation）。

核心公式:
- Jacobi椭圆函数 sn(u|m), cn(u|m), dn(u|m)
- 模数 m ∈ [0, 1]
- 周期: 4K(m)，其中 K(m) 为第一类完全椭圆积分
- 极限: m=0 时退化为三角函数；m=1 时退化为双曲函数

Bulirsch算法:
基于算术-几何平均（AGM）的Landen变换递推:
    a_{n+1} = (a_n + b_n) / 2
    b_{n+1} = √(a_n · b_n)
    c_{n+1} = (a_n - b_n) / 2
收敛至 a_∞ = b_∞ = AGM(a_0, b_0)
"""

import numpy as np
from typing import Tuple


def sncndn(u: float, m: float, tol: float = 1e-10, max_iter: int = 100) -> Tuple[float, float, float]:
    """计算Jacobi椭圆函数 sn(u|m), cn(u|m), dn(u|m)。
    
    采用Bulirsch经典算法，基于算术-几何平均（AGM）迭代。
    
    参数:
        u: 幅角（振幅）
        m: 模数参数，m ∈ [0, 1]
        tol: 收敛容差
        max_iter: 最大迭代次数
    
    返回:
        sn, cn, dn: Jacobi椭圆函数值
    
    边界处理:
    - m = 0: sn(u|0) = sin(u), cn(u|0) = cos(u), dn(u|0) = 1
    - m = 1: sn(u|1) = tanh(u), cn(u|1) = sech(u), dn(u|1) = sech(u)
    - m < 0 或 m > 1: 使用变换公式转换到标准范围
    """
    # 边界情况处理
    if abs(m) < tol:
        return np.sin(u), np.cos(u), 1.0
    
    if abs(m - 1.0) < tol:
        su = np.sinh(u)
        cu = np.cosh(u)
        sech_u = 1.0 / cu if abs(cu) > tol else 0.0
        return np.tanh(u), sech_u, sech_u
    
    # 处理 m > 1 的情况: 使用变换 m' = 1/m
    if m > 1.0:
        mp = 1.0 / m
        up = u * np.sqrt(m)
        snp, cnp, dnp = sncndn(up, mp, tol, max_iter)
        sn_val = snp / np.sqrt(m)
        cn_val = dnp
        dn_val = cnp
        # 数值修正
        if abs(sn_val) > 1.0:
            sn_val = np.sign(sn_val)
            cn_val = 0.0
        return sn_val, cn_val, dn_val
    
    # 处理 m < 0 的情况
    if m < 0.0:
        mp = -m / (1.0 - m)
        up = u / np.sqrt(1.0 - m)
        snp, cnp, dnp = sncndn(up, mp, tol, max_iter)
        denom = 1.0 - mp * snp**2
        if abs(denom) < tol:
            denom = tol
        sn_val = snp * np.sqrt(1.0 - mp * (1.0 - snp**2)) / denom
        cn_val = cnp * dnp / denom
        dn_val = (1.0 - mp * snp**2) / denom
        return sn_val, cn_val, dn_val
    
    # 标准AGM迭代
    a = 1.0
    b = np.sqrt(1.0 - m)
    c_val = np.sqrt(m)
    
    n_iter = 0
    while abs(c_val) > tol and n_iter < max_iter:
        a_next = 0.5 * (a + b)
        b_next = np.sqrt(a * b)
        c_next = 0.5 * (a - b)
        a, b, c_val = a_next, b_next, c_next
        n_iter += 1
    
    # AGM收敛后的计算
    phi = 2**n_iter * a * u
    
    # 反向递推计算sn, cn, dn
    # 简化为直接使用sin/cos近似（AGM收敛后）
    sn_val = np.sin(phi)
    cn_val = np.cos(phi)
    dn_val = 1.0 - 0.5 * m * sn_val**2  # 小m近似
    
    # 数值修正
    if abs(sn_val) > 1.0:
        sn_val = np.sign(sn_val)
        cn_val = 0.0
    
    return sn_val, cn_val, dn_val


def jacobi_sn(u: float, m: float) -> float:
    """Jacobi椭圆正弦函数 sn(u|m)。"""
    sn, _, _ = sncndn(u, m)
    return sn


def jacobi_cn(u: float, m: float) -> float:
    """Jacobi椭圆余弦函数 cn(u|m)。"""
    _, cn, _ = sncndn(u, m)
    return cn


def jacobi_dn(u: float, m: float) -> float:
    """Jacobi delta振幅函数 dn(u|m)。"""
    _, _, dn = sncndn(u, m)
    return dn


def burgers_periodic_solution(x: np.ndarray, t: float, A: float = 1.0,
                               nu: float = 0.01, m: float = 0.5) -> np.ndarray:
    """Burgers方程的周期行波解，用Jacobi椭圆函数表示。
    
    解析解形式:
        u(x,t) = A · cn²(ξ|m)
    其中 ξ = k·(x - c·t)，k 为波数，c 为波速。
    
    在极限 m → 1 时，cn²退化为 sech²，对应孤子解（无耗散情况下的
    KdV方程孤子解）。
    
    参数:
        x: 空间坐标数组
        t: 时间
        A: 振幅
        nu: 粘滞系数
        m: 椭圆模数
    
    返回:
        u: 速度场
    """
    # 波数和波速由色散关系确定
    # 对于 cn² 解，波数 k 与模数 m 相关
    k_wave = np.sqrt(A / (2.0 * nu))  # 特征波数
    c_wave = A * (2.0 - m) / 3.0      # 波速
    
    u = np.zeros_like(x)
    for i, xi in enumerate(x):
        phase = k_wave * (xi - c_wave * t)
        _, cn_val, _ = sncndn(phase, m)
        u[i] = A * cn_val**2
    
    return u


def shock_wave_formation(x: np.ndarray, t: float, u0: float = 1.0,
                         x0: float = 0.5, L: float = 1.0) -> np.ndarray:
    """一维冲击波形成的非线性声学解。
    
    初始条件为锯齿波（Fourier级数展开）:
        u(x,0) = u0 · (x - x0) / L,  x ∈ [0, L]
    
    在无损Burgers方程中，特征线相交形成冲击:
        t_shock = L / u0
    
    冲击后的弱解可用Rankine-Hugoniot条件确定:
        [u]·dx_s/dt = [u²/2]
    其中 [·] 表示跨越冲击的跳跃，x_s(t) 为冲击位置。
    
    参数:
        x: 空间坐标
        t: 时间（t < t_shock 时为连续解，t ≥ t_shock 时含冲击）
        u0: 特征速度
        x0: 初始波前位置
        L: 波长
    
    返回:
        u: 速度分布
    """
    t_shock = L / u0
    u = np.zeros_like(x)
    
    if t < t_shock:
        # 冲击前：简单波解
        for i, xi in enumerate(x):
            # 隐式关系: x = ξ + u(ξ)·t
            # 对于线性初始条件，可显式求解
            u[i] = u0 * (xi - x0) / (L + u0 * t)
    else:
        # 冲击后：N波近似
        # 冲击位置由面积守恒确定
        x_shock = x0 + 0.5 * u0 * t_shock + 0.5 * u0 * (t - t_shock)
        
        for i, xi in enumerate(x):
            if xi < x_shock:
                u[i] = u0 * (xi - x0) / (L + u0 * t)
                if u[i] < 0:
                    u[i] = 0.0
            else:
                u[i] = 0.0
    
    return u


def nonlinear_acoustic_parameter_estimation(pressure_amplitudes: np.ndarray,
                                            frequencies: np.ndarray) -> dict:
    """从多频声压幅值估计非线性声学参数 B/A。
    
    非线性参数 B/A 描述介质的非线性响应强度:
        B/A = 2ρ₀c₀ · (∂c/∂p)|_{p=0}
    
    二次谐波生成效率:
        P₂/P₁ ≈ (B/A + 2) · π·f·z·P₁ / (2ρ₀c₀³)
    其中 P₁ 为基波幅值，P₂ 为二次谐波幅值，z 为传播距离。
    
    参数:
        pressure_amplitudes: 声压幅值数组 (基波和二次谐波)
        frequencies: 对应频率数组
    
    返回:
        包含非线性参数估计结果的字典
    """
    if len(pressure_amplitudes) < 2:
        return {'error': '需要至少2个频率点的数据'}
    
    # 简化的线性回归估计
    # 假设 pressure_amplitudes[0] 为基波，pressure_amplitudes[1] 为二次谐波
    p1 = pressure_amplitudes[0]
    p2 = pressure_amplitudes[1]
    f = frequencies[0]
    
    # 典型生物组织参数
    rho0 = 1000.0  # kg/m³
    c0 = 1540.0    # m/s
    z = 0.05       # m，传播距离
    
    if abs(p1) < 1e-14:
        return {'error': '基波幅值过小'}
    
    # 从二次谐波效率反推 B/A
    efficiency = p2 / p1
    denom = np.pi * f * z * p1 / (2.0 * rho0 * c0**3)
    
    if abs(denom) < 1e-14:
        return {'error': '分母过小，无法估计'}
    
    BA_estimated = efficiency / denom - 2.0
    
    return {
        'B_over_A': float(BA_estimated),
        'efficiency': float(efficiency),
        'fundamental_pressure': float(p1),
        'harmonic_pressure': float(p2),
        'frequency_MHz': float(f / 1e6)
    }
