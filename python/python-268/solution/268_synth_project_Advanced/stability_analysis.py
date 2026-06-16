"""
stability_analysis.py - DQMC 数值稳定性分析与条件数诊断
=========================================================

科学背景 (Scientific Background):
    DQMC 的数值稳定性是关键挑战. B 矩阵乘积的条件数随 β 指数增长:
        κ(B_L ... B_0) ~ e^{β W}
    其中 W 为带宽 (~8t 对三角晶格).

    当 κ > 1/ε_machine ~ 10^{16} 时, 矩阵求逆灾难性失效.

    稳定化方案:
        1. SVD 稳定化: 每 p 步做一次 SVD, 分离大/小奇异值
            B = U S V†,  S = diag(s_1, ..., s_Ns)
            将 s_i 分为 "大" (s > √ε) 和 "小" (s ≤ √ε) 两组
        2. UDT 分解: B = U D T (U 酉, D 对角, T 上三角)
            更高效, 但实现更复杂
        3. 多精度算术: 使用 quad precision (昂贵)

    本模块实现:
        - B 矩阵乘积的条件数追踪
        - SVD 稳定化验证
        - 能量守恒检查
        - 自洽性测试 (Dyson 方程残差)

融合种子项目:
    - 无直接对应, 但使用 362_fd1d_heat_steady 中的对角占优概念

核心公式 (Key Formulas):
    条件数:
        κ(A) = ||A|| · ||A^{-1}|| = σ_max / σ_min

    SVD 稳定化判据:
        若 σ_min / σ_max < ε_stable = √(ε_machine) ≈ 10^{-8}
        则需要稳定化

    Green 函数的误差界:
        ||δG|| / ||G|| ≤ κ(M) · ε_machine
    其中 M = I + B_total
"""

import numpy as np
from typing import Tuple, Dict, List, Optional


# ==========================================================================
#  B 矩阵乘积的条件数分析
# ==========================================================================

def compute_b_product_svd(B_list: List[np.ndarray]
                          ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    计算 B 矩阵乘积 B_{L-1} ... B_0 的 SVD 分解.

    使用增量 SVD: 每次乘一个 B_l 后重新分解.
    B_total = U S V†

    返回:
        U:  (Ns, Ns) 左奇异向量
        S:  (Ns,) 奇异值
        Vt: (Ns, Ns) 右奇异向量的转置
    """
    Ns = B_list[0].shape[0]
    U = np.eye(Ns)
    S = np.ones(Ns)
    Vt = np.eye(Ns)

    for B_l in B_list:
        M = B_l @ (U * S) @ Vt
        U, S, Vt = np.linalg.svd(M)
        # 归一化防止溢出
        S_max = S.max()
        if S_max > 1e100:
            S = S / S_max
        elif S_max < 1e-100 and S_max > 0:
            S = S / S_max

    return U, S, Vt


def condition_number_from_svd(S: np.ndarray) -> float:
    """
    从奇异值计算条件数.

    κ = σ_max / σ_min

    若 σ_min = 0 (或极小), κ = ∞ (奇异).
    """
    s_max = S.max()
    s_min = S.min()
    if s_min < 1e-300:
        return np.inf
    return s_max / s_min


def analyze_stability(B_list: List[np.ndarray]) -> Dict[str, object]:
    """
    对 B 矩阵乘积进行完整稳定性分析.

    返回:
        'condition_number': 条件数
        'singular_values': 奇异值谱
        'svd_rank': 数值秩 (σ > ε_threshold 的数量)
        'is_stable': 是否需要稳定化
        'log_condition': log10(κ)
    """
    U, S, Vt = compute_b_product_svd(B_list)
    kappa = condition_number_from_svd(S)
    eps_machine = 2.2e-16
    rank = int(np.sum(S > eps_machine * S.max()))

    return {
        'condition_number': kappa,
        'singular_values': S,
        'svd_rank': rank,
        'is_stable': kappa < 1.0 / np.sqrt(eps_machine),
        'log_condition': np.log10(kappa) if kappa > 0 else float('inf'),
    }


# ==========================================================================
#  能量守恒与自洽性检查
# ==========================================================================

def check_green_function_idempotency(G: np.ndarray, tol: float = 1e-8) -> Dict[str, float]:
    """
    检查格林函数的幂等性 (idempotency).

    等时格林函数满足:
        G² = G  (在零温极限)

    对有限温度:
        G(I - G) = T(τ)  →  非零但小

    残差:
        R = ||G² - G||_F / ||G||_F

    若 R > tol, 说明格林函数计算有数值误差.
    """
    G2 = G @ G
    residual = np.linalg.norm(G2 - G, ord='fro')
    G_norm = np.linalg.norm(G, ord='fro')
    relative_residual = residual / max(G_norm, 1e-300)

    return {
        'residual_norm': residual,
        'relative_residual': relative_residual,
        'is_idempotent': relative_residual < tol,
        'trace_G': np.trace(G).real,
        'trace_G2': np.trace(G2).real,
    }


def check_particle_number(G: np.ndarray, target_density: float = 1.0,
                          tol: float = 0.05) -> Dict[str, float]:
    """
    检查粒子数守恒.

    总粒子数:
        N = Σ_i ⟨n_i⟩ = Σ_i (1 - G_{ii})
        n = N / Ns  (每格点平均粒子数)

    半填充: n = 1.0 (考虑自旋, 实际为 ⟨n_↑ + n_↓⟩ = 1)

    对自旋对称情况:
        n_σ = 1 - (1/Ns) Tr[G]
        n_total = 2 n_σ
    """
    Ns = G.shape[0]
    n_sigma = 1.0 - np.trace(G).real / Ns
    n_total = 2.0 * n_sigma  # 自旋因子

    deviation = abs(n_total - target_density)
    return {
        'n_sigma': n_sigma,
        'n_total': n_total,
        'target_density': target_density,
        'deviation': deviation,
        'is_conserved': deviation < tol,
    }


def check_hermiticity(G: np.ndarray, tol: float = 1e-10) -> Dict[str, float]:
    """
    检查格林函数的厄米性.

    物理要求: G = G† (厄米共轭)

    残差: ||G - G†||_F / ||G||_F
    """
    G_dagger = G.conj().T
    diff = G - G_dagger
    residual = np.linalg.norm(diff, ord='fro')
    G_norm = np.linalg.norm(G, ord='fro')
    relative = residual / max(G_norm, 1e-300)
    return {
        'hermiticity_residual': residual,
        'relative_residual': relative,
        'is_hermitian': relative < tol,
    }


def check_dyson_self_consistency(G: np.ndarray, T_mat: np.ndarray,
                                 Sigma: np.ndarray, mu: float,
                                 omega_n: np.ndarray,
                                 tol: float = 0.1) -> Dict[str, float]:
    """
    检查 Dyson 方程的自洽性.

    Dyson 方程:
        G^{-1}(iω_n) = iω_n + μ - ε_k - Σ(iω_n)

    残差 (在矩阵形式):
        R = ||I - G^{-1} G_0 - G^{-1} Σ G_0||
    其中 G_0 = (iω_n + μ - T)^{-1} 为非相互作用格林函数.

    简化版本 (仅检查迹的一致性):
        tr[G^{-1}] = tr[iω_n + μ - T - Σ]
    """
    Ns = G.shape[0]
    try:
        G_inv = np.linalg.inv(G)
    except np.linalg.LinAlgError:
        G_inv = np.linalg.pinv(G)

    # 非相互作用格林函数 (取第一个 Matsubara 频率)
    if len(omega_n) > 0:
        iw = 1j * omega_n[0]
        G0_inv = iw * np.eye(Ns) + mu * np.eye(Ns) - T_mat
        G0 = np.linalg.inv(G0_inv)

        # Dyson 残差
        dyson_lhs = G_inv
        dyson_rhs = G0_inv - Sigma
        residual = np.linalg.norm(dyson_lhs - dyson_rhs, ord='fro')
        relative = residual / max(np.linalg.norm(dyson_lhs, ord='fro'), 1e-300)
    else:
        relative = 0.0

    return {
        'dyson_residual': relative,
        'is_self_consistent': relative < tol,
    }


# ==========================================================================
#  有限差分稳定性 (CFL 条件)
# ==========================================================================

def cfl_condition_check(delta_tau: float, delta_x: float,
                        diffusion_coeff: float = 1.0,
                        scheme: str = 'explicit') -> Dict[str, float]:
    """
    CFL (Courant-Friedrichs-Lewy) 稳定性条件检查.

    对显式差分格式求解扩散方程 u_t = D u_xx:
        稳定性要求:  D Δτ / (Δx)² ≤ 1/2  (1D)
                     D Δτ / (Δx)² ≤ 1/4  (2D)
                     D Δτ / (Δx)² ≤ 1/6  (3D)

    在 DQMC 语境下:
        Δτ = β/L (虚时步长)
        Δx = 晶格常数
        D = 有效扩散系数 ~ t a²

    对隐式格式 (如中点法), 无条件稳定.
    """
    if scheme == 'explicit_1d':
        cfl_limit = 0.5
    elif scheme == 'explicit_2d':
        cfl_limit = 0.25
    elif scheme == 'explicit_3d':
        cfl_limit = 1.0 / 6.0
    else:
        # 隐式格式
        return {
            'cfl_number': 0.0,
            'cfl_limit': float('inf'),
            'is_stable': True,
            'scheme': scheme,
        }

    cfl_number = diffusion_coeff * delta_tau / (delta_x * delta_x)
    return {
        'cfl_number': cfl_number,
        'cfl_limit': cfl_limit,
        'is_stable': cfl_number <= cfl_limit,
        'scheme': scheme,
    }


def von_neumann_stability(delta_tau: float, delta_x: float,
                          wave_numbers: np.ndarray,
                          diffusion_coeff: float = 1.0) -> Dict[str, float]:
    """
    von Neumann 稳定性分析.

    对 FTCS (Forward Time Centered Space) 格式:
        放大因子: G(k) = 1 - 4 D Δτ / (Δx)² sin²(k Δx / 2)

    稳定性要求: |G(k)| ≤ 1 for all k

    最危险模式: k = π / Δx → sin² = 1
        → G = 1 - 4 D Δτ / (Δx)²
        → 需要: -1 ≤ 1 - 4 D Δτ / (Δx)² → D Δτ / (Δx)² ≤ 1/2
    """
    amplification = np.zeros_like(wave_numbers, dtype=float)
    for i, k in enumerate(wave_numbers):
        amplification[i] = 1.0 - 4.0 * diffusion_coeff * delta_tau / (delta_x ** 2) * \
                           np.sin(k * delta_x / 2.0) ** 2

    return {
        'max_amplification': float(np.max(np.abs(amplification))),
        'min_amplification': float(np.min(amplification)),
        'is_stable': bool(np.all(np.abs(amplification) <= 1.0 + 1e-12)),
        'amplification_spectrum': amplification,
    }


# ==========================================================================
#  矩阵指数精度分析
# ==========================================================================

def matrix_exp_error_analysis(A: np.ndarray, methods: List[str] = None
                              ) -> Dict[str, Dict[str, float]]:
    """
    比较不同矩阵指数方法的精度.

    方法:
        1. Taylor 展开 (不同阶数)
        2. Padé 近似
        3. 特征值分解 (精确, 对对称矩阵)

    参考解: 使用 scipy.linalg.expm (若可用) 或高精度 Taylor.
    """
    Ns = A.shape[0]
    if methods is None:
        methods = ['taylor_10', 'taylor_20', 'eigendecomp']

    # 参考解: 高精度 Taylor
    ref = np.eye(Ns)
    term = np.eye(Ns)
    for k in range(1, 30):
        term = term @ A / k
        ref = ref + term
        if np.linalg.norm(term, ord='fro') < 1e-16:
            break

    results = {}
    for method in methods:
        if method == 'taylor_10':
            approx = np.eye(Ns)
            term = np.eye(Ns)
            for k in range(1, 11):
                term = term @ A / k
                approx = approx + term
        elif method == 'taylor_20':
            approx = np.eye(Ns)
            term = np.eye(Ns)
            for k in range(1, 21):
                term = term @ A / k
                approx = approx + term
        elif method == 'eigendecomp':
            evals, evecs = np.linalg.eigh(A)
            approx = evecs @ np.diag(np.exp(evals)) @ evecs.T
        else:
            continue

        error = np.linalg.norm(approx - ref, ord='fro')
        rel_error = error / max(np.linalg.norm(ref, ord='fro'), 1e-300)
        results[method] = {
            'absolute_error': float(error),
            'relative_error': float(rel_error),
        }

    return results
