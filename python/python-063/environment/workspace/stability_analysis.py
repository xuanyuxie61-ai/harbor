"""
stability_analysis.py
气候系统稳定性分析模块

使用 Weierstrass-Durand-Kerner (WDK) 方法（融合种子项目 1404_wdk）
求解能量平衡模型在平衡态附近的特征多项式，从而判断气候模式的稳定性。
"""

import numpy as np

SIGMA = 5.670374419e-8
Q_SOLAR = 1361.0 / 4.0


def poly_eval(coeffs, z):
    """
    多项式求值（Horner 法则）:
        p(z) = c_0 + c_1 z + ... + c_n z^n
    """
    result = np.zeros_like(z, dtype=np.complex128)
    for c in reversed(coeffs):
        result = result * z + c
    return result


def wdk_roots(coeffs, tol=1e-12, max_iter=50):
    """
    Weierstrass-Durand-Kerner 同时求根法。

    迭代公式:
        z_i^{new} = z_i - p(z_i) / prod_{j!=i}(z_i - z_j)

    初始猜测为按 Cauchy 界缩放的单位根:
        R = 1 + max|c_k/c_n|
        z_i^{(0)} = R * exp(2*pi*i * k / n)
    """
    n = len(coeffs) - 1
    if n <= 0:
        return np.array([], dtype=np.complex128)

    denom = max(np.abs(coeffs[-1]), 1e-300)
    R = 1.0 + np.max(np.abs(coeffs[:-1])) / denom
    roots = R * np.exp(2j * np.pi * np.arange(n) / n)

    for _ in range(max_iter):
        p_vals = poly_eval(coeffs, roots)
        corrections = np.zeros(n, dtype=np.complex128)
        for i in range(n):
            denom_prod = 1.0 + 0j
            for j in range(n):
                if i != j:
                    diff = roots[i] - roots[j]
                    if abs(diff) < 1e-15:
                        diff = 1e-15 * (1.0 + 1.0j)
                    denom_prod *= diff
            corrections[i] = p_vals[i] / denom_prod
        roots -= corrections
        if np.max(np.abs(corrections)) < tol:
            break
    return roots


def faddeev_leverrier(A, max_dim=20):
    """
    Faddeev-LeVerrier 算法计算特征多项式系数:
        det(lambda I - A) = lambda^n + c_1 lambda^{n-1} + ... + c_n
    对大于 max_dim 的矩阵自动截断左上角子块。
    """
    n_full = A.shape[0]
    if n_full > max_dim:
        n = max_dim
        A = A[:n, :n]
    else:
        n = n_full

    coeffs = np.zeros(n + 1, dtype=np.float64)
    coeffs[0] = 1.0
    B = np.eye(n)
    for k in range(1, n + 1):
        C = A @ B
        coeffs[k] = -np.trace(C) / k
        B = C + coeffs[k] * np.eye(n)
    return coeffs


def build_ebm_jacobian(n_nodes, diffusion_coeff=0.55, epsilon=0.6, T_eq=288.0):
    """
    构造 EBM 在平衡态 T_eq 附近的 Jacobian 矩阵 J = d(dT/dt)/dT。
    采用简化环形最近邻耦合来近似球面扩散。
    """
    # TODO: Build the EBM Jacobian matrix J = d(dT/dt)/dT at equilibrium T_eq.
    # The Jacobian must be consistent with the physics in ebm_dynamics.ebm_rhs().
    # Key derivatives at T_eq:
    #   d(alpha)/dT from ice_albedo_feedback(T)
    #   d(OLR)/dT   from outgoing_longwave_radiation(T, epsilon)
    #   d(shortwave)/dT = Q_SOLAR * d(alpha)/dT
    #   local_feedback = -(d_sw_dT - d_olr_dT) / C_heat
    # Add nearest-neighbor diffusion coupling with periodic boundaries.
    # HINT: Ensure the feedback signs and heat capacity match ebm_dynamics.py.
    pass


def analyze_climate_stability(n_nodes, vertices, T_eq=288.0, **kwargs):
    """
    分析气候系统稳定性，返回特征值与稳定性判据。
    对中小规模矩阵直接调用 numpy.linalg.eigvals（数值稳定），
    同时保留 WDK 求根函数作为理论展示。
    """
    A = build_ebm_jacobian(n_nodes, **kwargs)

    # 对中小矩阵使用 numpy 直接求特征值，避免多项式求根的数值不稳定
    if n_nodes <= 50:
        eigenvalues = np.linalg.eigvals(A)
    else:
        coeffs = faddeev_leverrier(A)
        eigenvalues = wdk_roots(coeffs)

    max_real = float(np.max(np.real(eigenvalues)))
    if max_real > 0.01:
        stability = 'unstable'
    elif max_real > -0.01:
        stability = 'marginally_stable'
    else:
        stability = 'stable'

    oscillatory = bool(np.any(np.abs(np.imag(eigenvalues)) > 0.01))
    if oscillatory:
        stability += '_oscillatory'

    dom_idx = int(np.argmax(np.real(eigenvalues)))
    return {
        'eigenvalues': eigenvalues,
        'stability_type': stability,
        'dominant_mode': eigenvalues[dom_idx],
        'max_real_part': max_real
    }
