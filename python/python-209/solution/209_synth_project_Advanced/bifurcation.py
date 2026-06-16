"""
bifurcation.py — 分岔检测与临界参数搜索
==========================================

本模块实现随机 PDE 解的分岔行为分析:
  1. Newton-Maehly 同时求根法 (映射自 801_newton_maehly)
  2. 分岔点检测: 特征值穿越零点
  3. 临界参数搜索: 寻找系统相变点

数学框架:
  非线性 PDE 的特征值问题:
    L(u,λ)·v = μ(λ)·v
  分岔条件: ∃λ* 使 μ(λ*) = 0
  Newton-Maehly: 同时寻找特征方程的多个根

映射种子项目:
  - 801_newton_maehly: Newton-Maehly 同时求根法
"""

import numpy as np


# ============================================================
# 第1部分: Newton-Maehly 同时求根
# (映射自 801_newton_maehly)
# ============================================================

def poly_and_derivative(coeffs, x):
    """
    Horner 法则同时计算多项式值和导数:
        p(x) = c_d x^d + ... + c_0
        p'(x) = d·c_d x^{d-1} + ...

    Horner 递推:
        p = c_d
        dp = 0
        for k = d-1,...,0:
            dp = dp·x + p
            p = p·x + c_k

    返回: (p(x), p'(x))
    """
    coeffs = np.asarray(coeffs, dtype=complex)
    d = len(coeffs) - 1
    p = coeffs[0]
    dp = complex(0)
    for k in range(1, len(coeffs)):
        dp = dp * x + p
        p = p * x + coeffs[k]
    return p, dp


def newton_maehly_roots(coeffs, max_iter=100, tol=1e-10):
    """
    Newton-Maehly 方法同时求多项式的所有根:
        z_i ← z_i - P(z_i) / (P'(z_i) - P(z_i)·Σ_{j≠i} 1/(z_i - z_j))

    修正项 Σ 1/(z_i - z_j) 防止多个初值收敛到同一根。

    初值: Cauchy 界内的单位根:
        |z| ≤ 1 + max|c_k/c_d|
        z_i⁰ = R · exp(2πi·k/d)

    返回:
        roots: (d,) 复数根数组
        converged: 是否收敛
        iterations: 迭代次数
    """
    coeffs = np.asarray(coeffs, dtype=complex)
    # 去除前导零
    while len(coeffs) > 1 and abs(coeffs[0]) < 1e-15:
        coeffs = coeffs[1:]
    d = len(coeffs) - 1
    if d <= 0:
        return np.array([]), True, 0

    # Cauchy 界
    normalized = np.abs(coeffs[1:] / max(np.abs(coeffs[0]), 1e-15))
    radius = 1.0 + np.max(normalized) if len(normalized) > 0 else 2.0

    # 初值: 半径内的单位根
    roots = np.array([radius * np.exp(2j * np.pi * k / d) for k in range(d)])

    converged = False
    for iteration in range(max_iter):
        max_change = 0.0
        for i in range(d):
            p_val, dp_val = poly_and_derivative(coeffs, roots[i])
            # Maehly 修正
            correction_sum = complex(0)
            for j in range(d):
                if i != j:
                    diff = roots[i] - roots[j]
                    if abs(diff) > 1e-15:
                        correction_sum += 1.0 / diff
            denom = dp_val - p_val * correction_sum
            if abs(denom) < 1e-15:
                continue
            delta = p_val / denom
            roots[i] -= delta
            max_change = max(max_change, abs(delta))

        # 收敛检测
        max_poly = max(abs(poly_and_derivative(coeffs, roots[i])[0])
                        for i in range(d))
        if max_change < tol and max_poly < tol:
            converged = True
            return roots, converged, iteration + 1

    return roots, converged, max_iter


# ============================================================
# 第2部分: 特征值分岔检测
# ============================================================

def compute_jacobian_eigenvalues(forward_model, xi, eps=1e-5):
    """
    计算前向模型的 Jacobian 特征值:
        J_{ij} = ∂G_i/∂ξ_j ≈ (G_i(ξ+εe_j) - G_i(ξ-εe_j))/(2ε)

    分岔条件: det(J) = 0 ↔ ∃ 零特征值
    返回:
        eigenvalues: Jacobian 的特征值
        jacobian: J 矩阵
    """
    xi = np.asarray(xi, dtype=np.float64)
    d = len(xi)
    try:
        G0 = np.asarray(forward_model(xi), dtype=np.float64)
    except Exception:
        return np.array([0.0]), np.zeros((1, d))
    m = len(G0)
    J = np.zeros((m, d))
    for j in range(d):
        xi_p = xi.copy()
        xi_m = xi.copy()
        xi_p[j] += eps
        xi_m[j] -= eps
        try:
            Gp = np.asarray(forward_model(xi_p), dtype=np.float64)
            Gm = np.asarray(forward_model(xi_m), dtype=np.float64)
            J[:, j] = (Gp - Gm) / (2 * eps)
        except Exception:
            J[:, j] = 0.0
    # 特征值
    if m == d:
        eigenvalues = np.linalg.eigvals(J)
    elif m > d:
        JTJ = J.T @ J
        eigenvalues = np.linalg.eigvalsh(JTJ)
        eigenvalues = np.sqrt(np.maximum(eigenvalues, 0.0))
    else:
        JJT = J @ J.T
        eigenvalues = np.linalg.eigvalsh(JJT)
        eigenvalues = np.sqrt(np.maximum(eigenvalues, 0.0))
    return eigenvalues, J


def detect_bifurcation_along_path(forward_model, xi_start, xi_end, n_steps=20):
    """
    沿参数路径检测分岔点:
        ξ(t) = (1-t)·ξ_start + t·ξ_end, t ∈ [0,1]

    分岔检测: 特征值穿越零点 (符号变化)。

    返回:
        t_values: 参数路径采样点
        min_eigenvalues: 每点的最小特征值模
        bifurcation_detected: 是否检测到分岔
        bifurcation_t: 分岔位置估计
    """
    t_values = np.linspace(0, 1, n_steps)
    min_eigs = np.zeros(n_steps)

    for i, t in enumerate(t_values):
        xi_t = (1 - t) * xi_start + t * xi_end
        eigs, _ = compute_jacobian_eigenvalues(forward_model, xi_t)
        min_eigs[i] = np.min(np.abs(eigs))

    # 检测: 最小特征值是否穿越零点
    bifurcation_detected = False
    bifurcation_t = None
    for i in range(1, n_steps):
        # 如果特征值显著下降后又上升, 可能接近分岔点
        if min_eigs[i] < 0.1 * min_eigs[0] and min_eigs[i] < min_eigs[i - 1]:
            bifurcation_detected = True
            bifurcation_t = t_values[i]
            break

    return t_values, min_eigs, bifurcation_detected, bifurcation_t


# ============================================================
# 第3部分: 临界参数搜索
# ============================================================

def critical_parameter_search(forward_model, param_range, n_search=20,
                               stability_threshold=1e-3, seed=42):
    """
    搜索临界参数: 使系统从稳定变为不稳定的参数值。

    策略:
      1. 在参数范围内均匀采样
      2. 在每个采样点计算最小特征值
      3. 寻找特征值穿越零的位置

    参数:
        forward_model: G(ξ) → ℝ^m
        param_range: (low, high) 参数范围
        n_search: 搜索点数

    返回:
        critical_value: 临界参数值
        stability_profile: (n_search, 2) 参数-稳定性数据
    """
    low, high = param_range
    t_values = np.linspace(low, high, n_search)
    min_eigs = np.zeros(n_search)
    rng = np.random.default_rng(seed)

    for i, t in enumerate(t_values):
        # 测试参数点 (一维简化)
        xi = np.array([t])
        eigs, _ = compute_jacobian_eigenvalues(forward_model, xi)
        min_eigs[i] = np.min(np.abs(eigs)) if len(eigs) > 0 else 0.0

    # 寻找临界点
    critical_idx = np.argmin(min_eigs)
    critical_value = t_values[critical_idx]
    stability_profile = np.column_stack([t_values, min_eigs])

    return critical_value, stability_profile


# ============================================================
# 第4部分: 分岔多项式特征方程
# ============================================================

def bifurcation_polynomial_from_eigenvalues(eigenvalues_trajectory):
    """
    从特征值轨迹构造分岔特征多项式:
        p(λ) = Π (λ - λᵢ(t*))
    其中 t* 为疑似分岔点。

    使用 Newton-Maehly 求解此多项式的根, 确定分岔类型。
    返回:
        roots: 多项式的根
        bifurcation_type: 'saddle-node'/'pitchfork'/'transcritical'
    """
    if len(eigenvalues_trajectory) == 0:
        return np.array([]), 'unknown'
    # 构造多项式系数 (Vieta 公式)
    n = len(eigenvalues_trajectory)
    coeffs = np.zeros(n + 1, dtype=complex)
    coeffs[0] = 1.0
    for i in range(n):
        new_coeffs = np.zeros(n + 1, dtype=complex)
        for j in range(i + 2):
            if j < len(coeffs):
                new_coeffs[j] += coeffs[j]
            if j > 0 and j - 1 < len(coeffs):
                new_coeffs[j] -= eigenvalues_trajectory[i] * coeffs[j - 1]
        coeffs = new_coeffs[:i + 2]
    # 补齐
    full_coeffs = np.zeros(n + 1, dtype=complex)
    full_coeffs[:len(coeffs)] = coeffs
    # Newton-Maehly 求根
    roots, converged, iters = newton_maehly_roots(full_coeffs)

    # 分岔类型判断
    real_roots = np.real(roots[np.abs(np.imag(roots)) < 0.1])
    n_zero = np.sum(np.abs(real_roots) < 0.1)
    n_positive = np.sum(real_roots > 0.1)

    if n_zero >= 2:
        bif_type = 'pitchfork'
    elif n_zero == 1 and n_positive > 0:
        bif_type = 'transcritical'
    elif n_zero == 1:
        bif_type = 'saddle-node'
    else:
        bif_type = 'unknown'

    return roots, bif_type
