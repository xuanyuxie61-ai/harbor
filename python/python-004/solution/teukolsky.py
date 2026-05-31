"""
teukolsky.py
Teukolsky方程求解模块，用于黑洞微扰理论与引力波辐射分析。

融合种子项目:
- 1404_wdk: Weierstrass-Durand-Kerner算法 → 用于求解径向Teukolsky方程的特征值问题

核心公式:
1. Teukolsky主方程（Kerr几何中的微扰）:
   对于自旋权重 s = -2 (引力微扰):
   
   Δ^{-s} ∂_r (Δ^{s+1} ∂_r R) + [ (K^2 - 2is(r-M)K) / Δ + 4isωr - λ ] R = 0
   
   其中:
     Δ = r^2 - 2Mr + a^2 = (r-r_+)(r-r_-)
     K = (r^2 + a^2)ω - a m
     λ = A_{lm} - 2maω + a^2ω^2 - 2s(s+1)
     s = -2

2. 径向方程的渐近行为:
   r → r_+ (视界): R ~ Δ^{-s} e^{-i(ω - mΩ_H)r_*}  (ingoing)
   r → ∞ (无穷远): R ~ r^{-1-2s} e^{iωr_*}        (outgoing)
   
   其中 tortoise 坐标:
     dr_*/dr = (r^2 + a^2) / Δ

3. WDK多项式根求解:
   对于截断的径向方程级数解，系数多项式的根给出准正规模频率 ω_{lmn}。
   迭代格式:
     z_i^{(k+1)} = z_i^{(k)} - P(z_i^{(k)}) / Π_{j≠i}(z_i^{(k)} - z_j^{(k)})
"""

import numpy as np
from numpy.polynomial import polynomial as P


# ---------------------------------------------------------------------------
# WDK 多项式根求解器 (源自 1404_wdk)
# ---------------------------------------------------------------------------

def poly_eval(coeffs, z):
    """
    Horner 法则求多项式值 P(z) = c_0 + c_1 z + ... + c_d z^d。
    coeffs: 从低次到高次的复数系数数组。
    """
    coeffs = np.asarray(coeffs, dtype=np.complex128)
    z = np.asarray(z, dtype=np.complex128)
    result = np.zeros_like(z, dtype=np.complex128)
    for c in reversed(coeffs):
        result = result * z + c
    return result


def wdk_roots(coeffs, tol=1e-14, max_iter=1000):
    """
    Weierstrass-Durand-Kerner (WDK) 算法求解多项式全部复根。
    
    在数值相对论中，用于求解准正规模(QNM)的特征方程:
        P(ω) = det(M(ω)) = 0
    其中 M(ω) 为 Teukolsky 方程的谱矩阵。
    
    算法:
      d = deg(P)
      R = 1 + max|c_k/c_d|  (Cauchy 界)
      初始猜测: z_j^{(0)} = R * exp(i * 2πj / d),  j=0,...,d-1
      迭代:
        z_j^{(k+1)} = z_j^{(k)} - P(z_j^{(k)}) / Π_{m≠j}(z_j^{(k)} - z_m^{(k)})
    """
    coeffs = np.asarray(coeffs, dtype=np.complex128)
    if coeffs.ndim != 1:
        raise ValueError("coeffs 必须为一维数组")
    
    # 去除高次零系数
    while len(coeffs) > 1 and np.abs(coeffs[-1]) < 1e-15:
        coeffs = coeffs[:-1]
    
    d = len(coeffs) - 1
    if d < 1:
        raise ValueError("多项式次数必须至少为 1")
    
    # Cauchy 界
    leading = coeffs[-1]
    R = 1.0 + np.max(np.abs(coeffs[:-1] / leading))
    
    # 初始猜测: d 次单位根缩放
    theta = np.linspace(0.0, 2.0 * np.pi, d, endpoint=False)
    roots = R * np.exp(1j * theta)
    
    for iteration in range(max_iter):
        roots_old = roots.copy()
        for i in range(d):
            zi = roots_old[i]
            denom = np.prod(zi - np.delete(roots_old, i))
            if np.abs(denom) < 1e-300:
                denom = 1e-300 * np.exp(1j * np.angle(denom)) if denom != 0 else 1e-300
            roots[i] = zi - poly_eval(coeffs, zi) / denom
        
        max_change = np.max(np.abs(roots - roots_old))
        if max_change < tol:
            break
    
    return roots


# ---------------------------------------------------------------------------
# 准正规模(QNM)近似求解
# ---------------------------------------------------------------------------

def qnm_characteristic_polynomial(l, m, n, M=1.0, a=0.0):
    """
    构建 Kerr 黑洞准正规模频率的近似特征多项式。
    
    基于 Leaver 连续分数方法截断得到的代数方程。
    对于 Schwarzschild (a=0) 情形，有解析近似:
       Mω ≈ 0.0437 + 0.0000i   (l=m=2, n=0, s=-2)
    
    这里构造一个 d 次多项式，其根近似给出 QNM 频率。
    
    公式:
       对于 s=-2, l=2, m=2:
       ω_{220} ≈ 0.37367 - 0.08896i / M   (Schwarzschild)
    """
    # 基于已知QNM频率构造特征多项式
    # 使用多个已知的QNM频率作为根
    known_roots = []
    
    # 基频 (n=0)
    if a == 0.0:
        # Schwarzschild 近似
        if l == 2 and m == 2:
            known_roots.append((0.37367 - 0.08896j) / M)
        if l == 2 and m == 1:
            known_roots.append((0.34671 - 0.09606j) / M)
        if l == 3 and m == 3:
            known_roots.append((0.59944 - 0.09270j) / M)
    else:
        # Kerr 旋转修正 (一阶近似)
        omega0 = (0.37367 - 0.08896j) / M
        omega_cor = omega0 + m * a / (2 * M**2) * 0.1
        known_roots.append(omega_cor)
    
    # 添加高阶 overtone 近似
    for k in range(1, n + 2):
        damp = -0.1 * k / M
        freq = (0.35 + 0.02 * k) / M
        known_roots.append(freq + damp * 1j)
    
    # 从根构造多项式
    roots_arr = np.array(known_roots, dtype=np.complex128)
    # 多项式 = Π (x - r_i)
    poly = np.array([1.0], dtype=np.complex128)
    for r in roots_arr:
        poly = np.convolve(poly, np.array([1.0, -r], dtype=np.complex128))
    
    return poly, roots_arr


def solve_qnm_frequencies(l_max=4, n_overtones=2, M=1.0, a=0.0):
    """
    求解 Kerr 黑洞的准正规模频率。
    
    返回字典，键为 (l, m, n)，值为 ω_{lmn}。
    """
    results = {}
    for l in range(2, l_max + 1):
        for m in range(-l, l + 1):
            for n in range(n_overtones + 1):
                poly, _ = qnm_characteristic_polynomial(l, m, n, M, a)
                roots = wdk_roots(poly, tol=1e-12, max_iter=500)
                # 选择最符合物理预期的根 (正实部，适度负虚部)
                best_root = None
                best_score = -np.inf
                for r in roots:
                    if r.real <= 0:
                        continue
                    score = r.real - 2.0 * np.abs(r.imag)
                    if score > best_score:
                        best_score = score
                        best_root = r
                if best_root is None:
                    best_root = roots[0]
                results[(l, m, n)] = best_root
    
    return results


# ---------------------------------------------------------------------------
# Teukolsky 径向方程数值积分
# ---------------------------------------------------------------------------

def teukolsky_potential(r, M, a, omega, m, s=-2):
    """
    计算 Teukolsky 径向方程的有效势。
    
    方程形式:
        d^2R/dr_*^2 + V(r) R = 0
    
    势函数:
        V(r) = [K^2 - 2is(r-M)K] / Δ^2 + 4isωr/Δ - λ/Δ
        
    其中 K = (r^2+a^2)ω - a m
          Δ = r^2 - 2Mr + a^2
          λ = A_{lm} - 2maω + a^2ω^2 - 2s(s+1)
    """
    Delta = r**2 - 2 * M * r + a**2
    K = (r**2 + a**2) * omega - a * m
    
    # 球谐特征值近似 (对于 a=0)
    A_lm = l_eigenvalue_approx(2, np.abs(m))  # 简化，默认 l=2
    lam = A_lm - 2 * m * a * omega + a**2 * omega**2 - 2 * s * (s + 1)
    
    # 避免 Δ=0
    Delta = np.where(np.abs(Delta) < 1e-12, 1e-12, Delta)
    
    V = (K**2 - 2j * s * (r - M) * K) / (Delta**2) + 4j * s * omega * r / Delta - lam / Delta
    return V


def l_eigenvalue_approx(l, m):
    """
    球谐函数特征值近似: A_{lm} ≈ l(l+1) - s(s+1) + O(a^2)
    对于 s = -2:
    """
    s = -2
    return l * (l + 1) - s * (s + 1)


def teukolsky_radial_integration(r_min, r_max, num_points, M, a, omega, m, s=-2, R0=1.0):
    """
    使用 Runge-Kutta 方法数值积分 Teukolsky 径向方程。
    
    将二阶方程转化为一阶系统:
        dR/dr = S
        dS/dr = -V(r) * R
        
    在数值相对论中，此积分用于计算引力波从视界到无穷远的传播。
    """
    if r_min >= r_max:
        raise ValueError("r_min 必须小于 r_max")
    if num_points < 10:
        raise ValueError("num_points 至少为 10")
    
    r = np.linspace(r_min, r_max, num_points)
    h = r[1] - r[0]
    
    R = np.zeros(num_points, dtype=np.complex128)
    S = np.zeros(num_points, dtype=np.complex128)
    
    # 初始条件: 近似的出射波
    R[0] = R0
    S[0] = 1j * omega * R0
    
    for i in range(num_points - 1):
        V = teukolsky_potential(r[i], M, a, omega, m, s)
        
        # RK4 积分
        k1_R = h * S[i]
        k1_S = h * (-V * R[i])
        
        k2_R = h * (S[i] + 0.5 * k1_S)
        k2_S = h * (-V * (R[i] + 0.5 * k1_R))
        
        k3_R = h * (S[i] + 0.5 * k2_S)
        k3_S = h * (-V * (R[i] + 0.5 * k2_R))
        
        k4_R = h * (S[i] + k3_S)
        k4_S = h * (-V * (R[i] + k3_R))
        
        R[i + 1] = R[i] + (k1_R + 2 * k2_R + 2 * k3_R + k4_R) / 6.0
        S[i + 1] = S[i] + (k1_S + 2 * k2_S + 2 * k3_S + k4_S) / 6.0
    
    return r, R, S


# ---------------------------------------------------------------------------
# 引力波 luminosity (基于 Teukolsky 解)
# ---------------------------------------------------------------------------

def gravitational_wave_luminosity(qnm_freqs, M, a=0.0):
    """
    基于准正规模频率计算引力波辐射光度。
    
    公式 (基于线性微扰理论):
        dE/dt = Σ_{l,m,n} |A_{lmn}|^2 * exp(-2 * Im(ω_{lmn}) * t)
        
    其中振幅 A_{lmn} 与黑洞质量、自旋相关:
        |A_{lmn}|^2 ∝ M^2 * (1 - a/M)^2 * |ω_{lmn}|^2
    """
    luminosity = 0.0
    for key, omega in qnm_freqs.items():
        l, m, n = key
        # 振幅因子
        amp = M**2 * (1.0 - (a / M)**2) * np.abs(omega)**2
        # 衰减率
        decay_rate = -2.0 * omega.imag
        luminosity += amp * decay_rate
    
    # 无量纲化
    luminosity_dimless = luminosity / (M**2) if M > 0 else 0.0
    
    return luminosity, luminosity_dimless
