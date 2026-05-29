"""
matrix_element.py
希格斯玻色子 H -> ZZ* -> 4l 的螺旋度振幅计算

基于两个种子项目:
  - 950_quadrature_weights_vandermonde: Vandermonde 线性系统求解
  - 200_collocation: Horner 多项式快速求值

物理内容:
  树图水平的矩阵元 (标准模型):
    M = (g^2 / (2 * cos^2(theta_W))) * (g_{mu nu} - q1_mu q1_nu / m_Z^2)
        * (v_ffbar / (s - m_H^2 + i m_H Gamma_H)) * ...
  
  这里我们实现一个简化但物理正确的模型:
    |M|^2 ~ g_{HZZ}^2 * |P_Z(q1)|^2 * |P_Z(q2)|^2 * (leptonic_current)^2
  
  其中 P_Z(q) = 1 / (q^2 - m_Z^2 + i m_Z Gamma_Z) 为 Z 传播子
"""
import numpy as np
from constants import M_HIGGS, M_Z, GAMMA_Z, GAMMA_H, G_F, ALPHA_EM, TINY
from utils import horner_eval, lu_factor_scaled, lu_solve, safe_divide

# ============================================================
# 1. 传播子函数
# ============================================================
def z_propagator(s, m_z=M_Z, gamma_z=GAMMA_Z):
    """
    Z 玻色子传播子 (Breit-Wigner):
      P_Z(s) = 1 / (s - m_Z^2 + i * m_Z * Gamma_Z)
    
    s: 不变质量平方 [GeV^2]
    返回: 复数传播子值
    """
    denom = s - m_z ** 2 + 1j * m_z * gamma_z
    if abs(denom) < TINY:
        return 0.0 + 0.0j
    return 1.0 / denom


def higgs_propagator(s, m_h=M_HIGGS, gamma_h=GAMMA_H):
    """
    希格斯传播子:
      P_H(s) = 1 / (s - m_H^2 + i * m_H * Gamma_H)
    """
    denom = s - m_h ** 2 + 1j * m_h * gamma_h
    if abs(denom) < TINY:
        return 0.0 + 0.0j
    return 1.0 / denom


# ============================================================
# 2. 耦合常数
# ============================================================
def g_hzz_coupling():
    """
    希格斯-Z-Z 耦合 (树图水平):
      g_{HZZ} = 2 * m_Z^2 / v = g * m_Z / (2 * cos(theta_W))
      其中 v = (sqrt(2) G_F)^{-1/2} ~ 246 GeV
    """
    v = 1.0 / np.sqrt(np.sqrt(2.0) * G_F)
    return 2.0 * M_Z ** 2 / v


def g_zff_coupling(g_v, g_a):
    """
    Z-费米子耦合: g_V (矢量), g_A (轴矢量)
    对轻子: g_V = -1/2 + 2 sin^2(theta_W), g_A = -1/2
    
    返回 (g_V^2 + g_A^2) 因子
    """
    return g_v ** 2 + g_a ** 2


# ============================================================
# 3. 简化振幅 |M|^2 计算
# ============================================================
def matrix_element_squared_hzz4l(m_z1, m_z2, m_higgs=M_HIGGS):
    """
    H -> ZZ* -> 4l 的树图水平 |M|^2 (对角度积分后)
    
    简化模型 (窄宽度近似下归一化):
      |M|^2 ~ |g_{HZZ}|^2 * |P_Z(m_z1^2)|^2 * |P_Z(m_z2^2)|^2 * F(m_z1, m_z2)
    
    其中形状因子 F 包含相空间体积和轻子流:
      F ~ (m_z1^2 + m_z2^2) * (m_higgs^2 - m_z1^2 - m_z2^2)^2
    
    物理约束: m_z1 + m_z2 <= m_higgs
    
    参数:
        m_z1, m_z2: Z 玻色子不变质量 [GeV]
    返回:
        |M|^2 的标量值 [GeV^{-4} 量级]
    """
    # === HOLE 1 BEGIN ===
    # TODO: 实现 H->ZZ*->4l 树图水平 |M|^2 计算
    # 物理要求:
    #   1. 检查运动学约束: m_z1 > 0, m_z2 > 0, m_z1 + m_z2 <= m_higgs
    #   2. 计算 g_HZZ 耦合 (调用 g_hzz_coupling)
    #   3. 计算 Z 传播子 |P_Z(m_z1^2)|^2 和 |P_Z(m_z2^2)|^2
    #   4. 计算 Higgs 传播子 |P_H(m_higgs^2)|^2
    #   5. 计算轻子耦合因子 (g_V^2 + g_A^2)^2, 其中 g_V = -1/2 + 2*sin^2(theta_W), g_A = -1/2
    #   6. 计算 Källén 函数 lambda(m_higgs^2, m_z1^2, m_z2^2) = (s - s1 - s2)^2 - 4*s1*s2
    #   7. 计算相空间 Jacobian = sqrt(lambda) / m_higgs^2
    #   8. 组合为 |M|^2 = g_HZZ^2 * |P_Z1|^2 * |P_Z2|^2 * |P_H|^2 * coupling * lambda/s^2 * Jacobian
    raise NotImplementedError("HOLE 1: 请实现 matrix_element_squared_hzz4l")
    # === HOLE 1 END ===


# ============================================================
# 4. Vandermonde 权重用于多项式振幅逼近 (映射 950_quadrature_weights_vandermonde)
# ============================================================
def vandermonde_quadrature_weights(n, a, b, nodes):
    """
    给定 n 个节点 x_i in [a,b]，通过 Vandermonde 系统求解权重 w_i，
    使得求积规则对 1, x, x^2, ..., x^{n-1} 精确。
    
    线性系统:
      V_{k,i} = x_i^{k},  k = 0,...,n-1
      rhs_k = int_a^b x^k dx = (b^{k+1} - a^{k+1}) / (k+1)
      V * w = rhs
    
    参数:
        n: 节点数
        a, b: 积分区间
        nodes: 节点位置数组
    返回:
        weights: 权重数组
    """
    nodes = np.asarray(nodes, dtype=float)
    vander = np.zeros((n, n))
    rhs = np.zeros(n)
    
    for k in range(n):
        for i in range(n):
            vander[k, i] = nodes[i] ** k
        rhs[k] = (b ** (k + 1) - a ** (k + 1)) / (k + 1.0)
    
    # 使用 LU 分解求解 (映射 209_conte_deboor)
    lu, pivot, iflag = lu_factor_scaled(vander)
    if iflag != 0:
        # 矩阵奇异，回退到最小二乘
        weights = np.linalg.lstsq(vander, rhs, rcond=None)[0]
    else:
        weights = lu_solve(lu, pivot, rhs)
    
    # 保证非负权重 (物理要求)
    weights = np.maximum(weights, 0.0)
    
    return weights


# ============================================================
# 5. 多项式振幅逼近 (映射 200_collocation: Horner 求值)
# ============================================================
def fit_amplitude_polynomial(m_z1_vals, m_z2_vals, amplitude_vals, degree=5):
    """
    用二元多项式拟合振幅 |M(m1,m2)|^2
    
    基函数: T_i(x) * T_j(y) (Chebyshev 多项式)
    
    为简化，这里对两个变量分别做 1D 多项式拟合，然后取乘积逼近。
    更精确的做法是用 Padua 点 (映射 1279_toms886) 做真正的双变量插值。
    
    返回:
        coeffs_1d: 两个方向的 Chebyshev 系数
    """
    from utils import cooley_tukey_fft
    
    m_z1_vals = np.asarray(m_z1_vals, dtype=float)
    m_z2_vals = np.asarray(m_z2_vals, dtype=float)
    amp = np.asarray(amplitude_vals, dtype=float)
    
    # 检查单调性
    if len(m_z1_vals) < degree + 1 or len(m_z2_vals) < degree + 1:
        degree = min(len(m_z1_vals), len(m_z2_vals)) - 1
    
    # 简单方法: 对每个变量方向分别采样并做 Chebyshev 变换
    # 使用 DCT 近似
    n1 = min(len(m_z1_vals), degree + 1)
    n2 = min(len(m_z2_vals), degree + 1)
    
    # 选取均匀子集
    idx1 = np.linspace(0, len(m_z1_vals) - 1, n1, dtype=int)
    idx2 = np.linspace(0, len(m_z2_vals) - 1, n2, dtype=int)
    
    # 1D 投影平均值
    amp_proj_1 = np.zeros(n1)
    for i, ii in enumerate(idx1):
        mask = np.abs(m_z2_vals - m_z2_vals[len(m_z2_vals)//2]) < 10.0
        if np.any(mask):
            amp_proj_1[i] = np.mean(amp[ii, mask]) if amp.ndim > 1 else amp[ii]
        else:
            amp_proj_1[i] = amp[ii] if amp.ndim == 1 else amp[ii, 0]
    
    # 使用 numpy 的 polyfit 作为稳定回退
    x_norm = 2.0 * (m_z1_vals[idx1] - np.min(m_z1_vals[idx1])) / (np.max(m_z1_vals[idx1]) - np.min(m_z1_vals[idx1]) + TINY) - 1.0
    coeffs = np.polyfit(x_norm, amp_proj_1, min(degree, len(x_norm)-1))
    
    return coeffs


def eval_amplitude_polynomial(coeffs, x):
    """
    用 Horner 方法快速求值多项式振幅 (映射 200_collocation)
    
    x 需先归一化到 [-1, 1]
    """
    return horner_eval(coeffs, x)


# ============================================================
# 6. 螺旋度振幅 (完整但简化)
# ============================================================
def helicity_amplitude_zzstar(m_z1, m_z2, cos_theta, phi, m_higgs=M_HIGGS):
    """
    H -> ZZ* 衰变的螺旋度振幅 (在希格斯静止系)
    
    螺旋度态: lambda1, lambda2 in {0, +/-1}
    振幅公式 (有效拉氏量):
      M(lambda1, lambda2) ~ epsilon_{mu}(p1, lambda1) * epsilon_{nu}(p2, lambda2)
                           * (g^{mu nu} - p1^mu p2^nu / m_Z^2)
    
    对无质量轻子末态，角分布:
      |M|^2 ~ 1 + cos^2(theta)  (纵向极化主导)
    
    参数:
        m_z1, m_z2: Z 不变质量
        cos_theta: 极角余弦
        phi: 方位角
    返回:
        |M|^2 标量值
    """
    if abs(cos_theta) > 1.0:
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
    
    # 传播子因子
    pz1_sq = abs(z_propagator(m_z1 ** 2)) ** 2
    pz2_sq = abs(z_propagator(m_z2 ** 2)) ** 2
    
    # 角分布: 标准模型 H->ZZ 预测 ~ (1 + cos^2(theta))
    angular = 1.0 + cos_theta ** 2
    
    # 归一化因子
    norm = 1.0 / (m_higgs ** 4)
    
    return float(pz1_sq * pz2_sq * angular * norm)
