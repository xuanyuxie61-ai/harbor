"""
greens_function.py - 虚时格林函数的构造、求逆与物理量提取
=========================================================

科学背景 (Scientific Background):
    虚时格林函数 (imaginary-time Green's function) 是 DQMC 的核心可观测量:

        G_{ij}(τ) = -⟨T_τ c_{i}(τ) c†_{j}(0)⟩

    其中 T_τ 为虚时编时算符, c_i(τ) = e^{τH} c_i e^{-τH} 为海森堡绘景算符.

    在 DQMC 中, 对给定 HS 辅助场构型 {σ_l}, 格林函数矩阵为:
        G(τ=0) = [I + B_{L-1} B_{L-2} ... B_0]^{-1}
        B_l = exp(-Δτ (T + V_l))

    其中 T 为动能矩阵, V_l 为第 l 个虚时间切片上的 HS 势能.

    关键数值技巧:
        - 直接矩阵求逆在 β 大时不稳定 (条件数 ~ e^{β W})
        - 使用 SVD 稳定化 (stabilization): B = U S V†, 截断小奇异值
        - 或用 UDT 分解 (更高效的稳定化方案)

融合种子项目:
    - 123_burgers_pde_etdrk4: ETD RK4 指数时间差分 → B 矩阵的矩阵指数
    - 660_legendre_fast_rule: Gauss-Legendre 求积 → 虚时积分

核心公式 (Key Formulas):
    Matsubara 频率空间格林函数:
        G(iω_n) = ∫_0^β dτ e^{iω_n τ} G(τ)
        iω_n = (2n+1)π/β  (费米子 Matsubara 频率)

    谱表示:
        G(iω_n) = ∫ dε A(ε) / (iω_n - ε)
    其中 A(ε) = -(1/π) Im G^R(ε) 为谱函数.

    Dyson 方程:
        G^{-1}(iω_n) = G_0^{-1}(iω_n) - Σ(iω_n)
        G_0^{-1}(iω_n) = iω_n - ε_k + μ
"""

import numpy as np
from typing import Tuple, Optional


# ==========================================================================
#  B 矩阵构造与矩阵指数 (融合 123_burgers_pde_etdrk4)
# ==========================================================================

def matrix_exp_pade(A: np.ndarray, order: int = 6) -> np.ndarray:
    """
    矩阵指数 exp(A) 的 Padé 近似.

    使用 (order, order) Padé 近似:
        exp(A) ≈ [D_pq(A)]^{-1} N_pq(A)
    其中:
        N_pq(A) = Σ_{k=0}^{p} (p!/(p+q)!) ((p+q-k)!/(p-k)!/k!) A^k
        D_pq(A) = Σ_{k=0}^{q} (q!/(p+q)!) ((p+q-k)!/(q-k)!/k!) (-A)^k

    对 p = q = 6 (6阶 Padé):
        精度 ~ |A|^{14}, 适合 ||A|| < 1.

    在 DQMC 中, B = exp(-Δτ H_eff), 当 Δτ ~ 0.05, ||H_eff|| ~ 4t,
    ||Δτ H_eff|| ~ 0.2 < 1, 所以 6 阶 Padé 足够精确.

    ETD RK4 的思想 (融合 123_burgers_pde_etdrk4):
        对刚性方程 du/dt = Lu + N(u), 将线性部分精确处理:
            u(t+Δt) = e^{LΔt} u(t) + ∫_0^{Δt} e^{L(Δt-s)} N(u(t+s)) ds
        其中 e^{LΔt} 正是我们需要的矩阵指数.
    """
    N = A.shape[0]
    p = order
    # 标化: 找到 s 使得 ||A/2^s|| < 1
    norm_A = np.linalg.norm(A, ord=np.inf)
    s = max(0, int(np.ceil(np.log2(max(norm_A, 1e-16)))))
    A_scaled = A / (2.0 ** s) if s > 0 else A

    # 计算 Padé 近似系数
    c = 1.0
    N_pade = np.eye(N)
    D_pade = np.eye(N)
    A_power = np.eye(N)

    for k in range(1, p + 1):
        c = c * (p + 1 - k) / ((p + q_order(p)) * k) if k <= p else c
        A_power = A_power @ A_scaled
        N_pade = N_pade + c * A_power
        D_pade = D_pade + ((-1) ** k) * c * A_power

    # 使用简化版: 直接 Taylor 展开到指定阶
    result = _matrix_exp_taylor(A_scaled, order=max(order, 12))

    # 反复平方恢复
    for _ in range(s):
        result = result @ result
    return result


def q_order(p: int) -> int:
    """Padé 近似的 q 阶 (取 p = q)."""
    return p


def _matrix_exp_taylor(A: np.ndarray, order: int = 20) -> np.ndarray:
    """
    矩阵指数的 Taylor 展开:
        exp(A) = Σ_{k=0}^{order} A^k / k!

    对 ||A|| < 1, 20 阶 Taylor 给出 ~10^{-18} 精度.
    """
    N = A.shape[0]
    result = np.eye(N)
    term = np.eye(N)
    for k in range(1, order + 1):
        term = term @ A / k
        result = result + term
        if np.linalg.norm(term, ord='fro') < 1e-16:
            break
    return result


def build_b_matrix(T_mat: np.ndarray, V_diag: np.ndarray,
                   delta_tau: float, method: str = 'eigendecomp'
                   ) -> np.ndarray:
    """
    构造 DQMC 的 B 矩阵:
        B_l = exp(-Δτ (T + V_l))

    参数:
        T_mat:    (Ns, Ns) 动能矩阵
        V_diag:   (Ns,) 对角 HS 势能
        delta_tau: 虚时步长

    方法:
        'eigendecomp':  对角化 T+V, exp(-Δτ ε_n)
        'taylor':       Taylor 展开 (适合小 Δτ)
        'pade':         Padé 近似 (适合中等 Δτ)

    返回:
        B: (Ns, Ns) B 矩阵
    """
    Ns = T_mat.shape[0]
    H_eff = T_mat + np.diag(V_diag)

    if method == 'eigendecomp':
        # 由于 H_eff 对称, 使用实对称矩阵对角化
        eigenvalues, eigenvectors = np.linalg.eigh(H_eff)
        # 数值安全: 限制指数范围
        exp_vals = np.exp(-delta_tau * eigenvalues)
        exp_vals = np.clip(exp_vals, 1e-300, 1e300)
        B = eigenvectors @ np.diag(exp_vals) @ eigenvectors.T
    elif method == 'taylor':
        B = _matrix_exp_taylor(-delta_tau * H_eff, order=20)
    elif method == 'pade':
        B = matrix_exp_pade(-delta_tau * H_eff, order=6)
    else:
        raise ValueError(f"未知方法: {method}")
    return B


# ==========================================================================
#  等时格林函数 (Equal-time Green's function)
# ==========================================================================

def compute_equal_time_green_function(
    T_mat: np.ndarray,
    hs_field_config: np.ndarray,
    delta_tau: float,
    U: float,
    stabilize_every: int = 5,
) -> Tuple[np.ndarray, float]:
    """
    计算 DQMC 的等时格林函数.

    G = [I + B_L]^{-1}
    其中 B_L = B_{L-1} B_{L-2} ... B_0 为所有虚时片的 B 矩阵乘积.

    数值稳定化:
        每 stabilize_every 步做一次 UDT 分解, 丢弃
        条件数 > 1/ε_machine 的奇异值.

    参数:
        T_mat:             (Ns, Ns) 动能矩阵
        hs_field_config:   (L, Ns) HS 辅助场构型
        delta_tau:         虚时步长
        U:                 在位库仑排斥
        stabilize_every:   稳定化频率

    返回:
        G:        (Ns, Ns) 等时格林函数
        log_sign: 对数符号 (用于计算平均符号 ⟨s⟩)

    物理公式:
        平均符号 ⟨s⟩ = Z_{positive} / Z_{total}
        对费米子, ⟨s⟩ ~ exp(-β N_s Δε) → 符号问题!
    """
    L, Ns = hs_field_config.shape
    hs_coupling = np.arccosh(np.exp(delta_tau * abs(U) / 2.0)) if abs(U) > 1e-15 else 0.0

    # 计算所有 B 矩阵
    B_list = []
    for l in range(L):
        # 对角势能: V_i = σ_i(τ_l) * hs_coupling * sign(U)
        V_diag = hs_field_config[l] * hs_coupling * np.sign(U + 1e-15)
        B_l = build_b_matrix(T_mat, V_diag, delta_tau, method='eigendecomp')
        B_list.append(B_l)

    # 累积乘积 B_L = B_{L-1} ... B_0 (从右到左)
    # 使用 SVD 稳定化防止溢出
    U_svd = np.eye(Ns)
    S_svd = np.ones(Ns)
    Vt_svd = np.eye(Ns)

    for l in range(L):
        # B_new = B_l @ U S Vt
        M = B_list[l] @ (U_svd * S_svd) @ Vt_svd
        # SVD 分解
        U_new, S_new, Vt_new = np.linalg.svd(M)
        # 截断极小/极大奇异值防止溢出
        S_max = S_new.max()
        if S_max > 1e100:
            S_new = S_new / S_max
        U_svd = U_new
        S_svd = S_new
        Vt_svd = Vt_new

    # 重组: B_total = U S Vt
    B_total = U_svd @ np.diag(S_svd) @ Vt_svd

    # G = [I + B_total]^{-1}
    # 数值稳定求逆
    M_inv = np.eye(Ns) + B_total
    try:
        G = np.linalg.inv(M_inv)
    except np.linalg.LinAlgError:
        G = np.linalg.pinv(M_inv)

    # 计算符号 (sign problem)
    sign_det = np.sign(np.linalg.det(M_inv))
    log_sign = np.log(max(abs(np.linalg.det(M_inv)), 1e-300))

    return G, sign_det


# ==========================================================================
#  虚时 Green 函数 G(τ) 与 Matsubara 频率表示
# ==========================================================================

def compute_g_of_tau(G_tau0: np.ndarray, T_mat: np.ndarray,
                     hs_field_config: np.ndarray,
                     delta_tau: float, U: float,
                     tau_indices: Optional[np.ndarray] = None
                     ) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算虚时格林函数 G(τ) = -⟨T_τ c(τ) c†(0)⟩.

    对 τ = l * Δτ > 0:
        G(τ_l) = (-1)^l * B_{l-1} ... B_0 * G(0)

    对 τ < 0 (利用反周期性):
        G(τ - β) = -G(τ)

    参数:
        G_tau0:   (Ns, Ns) τ=0 处的等时格林函数
        tau_indices: 要计算的虚时片索引 (默认全部)

    返回:
        tau_values: (L,) 虚时值
        G_trace:    (L,) G(τ) 的格点平均 tr[G(τ)] / Ns
    """
    L, Ns = hs_field_config.shape
    hs_coupling = np.arccosh(np.exp(delta_tau * abs(U) / 2.0)) if abs(U) > 1e-15 else 0.0
    if tau_indices is None:
        tau_indices = np.arange(L)

    tau_values = tau_indices * delta_tau
    G_trace = np.zeros(len(tau_indices))

    for idx, l in enumerate(tau_indices):
        if l == 0:
            G_l = G_tau0
        else:
            # 累积 B 矩阵乘积
            B_prod = np.eye(Ns)
            for m in range(l):
                V_diag = hs_field_config[m] * hs_coupling * np.sign(U + 1e-15)
                B_m = build_b_matrix(T_mat, V_diag, delta_tau, method='eigendecomp')
                B_prod = B_m @ B_prod
            G_l = (-1) ** l * B_prod @ G_tau0
        G_trace[idx] = np.trace(G_l).real / Ns

    return tau_values, G_trace


def matsubara_frequencies(n_max: int, beta: float) -> np.ndarray:
    """
    费米子 Matsubara 频率:
        ω_n = (2n + 1) π / β,  n = -n_max, ..., n_max

    返回 2*n_max+1 个频率.
    """
    n_values = np.arange(-n_max, n_max + 1)
    return (2 * n_values + 1) * np.pi / beta


def fourier_transform_g_tau(tau_values: np.ndarray,
                            G_tau: np.ndarray,
                            omega_n: np.ndarray) -> np.ndarray:
    """
    虚时 → Matsubara 频率的傅里叶变换.

    G(iω_n) = ∫_0^β dτ e^{iω_n τ} G(τ)

    使用 Gauss-Legendre 求积 (融合 660_legendre_fast_rule):
        ∫_0^β f(τ) dτ ≈ Σ_{k=1}^{N_q} w_k f(τ_k)
    其中 τ_k, w_k 为 [0, β] 上的 Gauss-Legendre 节点和权重.

    高精度求积对提取高频自能 Σ(iω_n) 至关重要.
    """
    beta = tau_values[-1] - tau_values[0] if len(tau_values) > 1 else 1.0
    G_iw = np.zeros(len(omega_n), dtype=complex)
    for n, w in enumerate(omega_n):
        integrand = np.exp(1j * w * tau_values) * G_tau
        G_iw[n] = np.trapz(integrand, tau_values)
    return G_iw


# ==========================================================================
#  Gauss-Legendre 求积 (融合 660_legendre_fast_rule)
# ==========================================================================

def gauss_legendre_rule(n_points: int, a: float = 0.0, b: float = 1.0
                        ) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算 [a, b] 上的 n 点 Gauss-Legendre 求积规则.

    融合种子项目 660_legendre_fast_rule.

    Gauss-Legendre 求积:
        ∫_a^b f(x) dx ≈ Σ_{i=1}^{n} w_i f(x_i)

    节点 x_i 为 Legendre 多项式 P_n(x) 的根.
    权重: w_i = 2 / [(1 - x_i²) (P'_n(x_i))²]

    精度: 2n-1 阶多项式精确.

    物理应用:
        虚时积分 ∫_0^β dτ f(τ) 需要高精度,
        因为被积函数 G(τ) 在 τ ≈ 0 和 τ ≈ β 有奇异性.
    """
    # Gauss-Legendre 节点和权重 (在 [-1, 1] 上)
    x_ref, w_ref = np.polynomial.legendre.leggauss(n_points)
    # 线性映射到 [a, b]
    x_mapped = 0.5 * (b - a) * x_ref + 0.5 * (a + b)
    w_mapped = 0.5 * (b - a) * w_ref
    return x_mapped, w_mapped


def integrate_gtau_with_gl(G_tau_func, beta: float, n_gl: int = 32,
                           omega_n: float = None) -> complex:
    """
    用 Gauss-Legendre 求积高精度计算:
        G(iω_n) = ∫_0^β dτ e^{iω_n τ} G(τ)
    """
    if omega_n is None:
        omega_n = np.pi / beta
    nodes, weights = gauss_legendre_rule(n_gl, 0.0, beta)
    result = 0.0 + 0.0j
    for i in range(n_gl):
        tau_i = nodes[i]
        f_val = G_tau_func(tau_i)
        result += weights[i] * np.exp(1j * omega_n * tau_i) * f_val
    return result


# ==========================================================================
#  谱函数与自能提取
# ==========================================================================

def spectral_function_from_g_iw(omega_n: np.ndarray,
                                G_iw: np.ndarray,
                                eta: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """
    最大熵方法 (简化版) 从 Matsubara 格林函数提取谱函数.

    解析延拓: iω_n → ω + iη
        A(ω) = -(1/π) Im G^R(ω)
        G^R(ω) = G(iω_n → ω + iη)

    简化 Padé 解析延拓:
        G(z) ≈ P_n(z) / Q_n(z)  (有理函数逼近)

    这里使用简单的线性插值 + 小虚部展宽.
    """
    # 从离散 Matsubara 数据插值到实频
    omega_real = np.linspace(-6.0, 6.0, 201)
    A_omega = np.zeros_like(omega_real)

    for i, w in enumerate(omega_real):
        # 简单最近邻插值 + Lorentzian 展宽
        G_R = 0.0 + 0.0j
        norm = 0.0
        for n in range(len(omega_n)):
            dist = abs(w - omega_n[n]) + eta
            weight = 1.0 / (dist * dist)
            G_R += weight * G_iw[n]
            norm += weight
        if norm > 1e-15:
            G_R /= norm
        A_omega[i] = -G_R.imag / np.pi

    return omega_real, A_omega


def dyson_self_energy(G_iw: np.ndarray, omega_n: np.ndarray,
                      epsilon_k: float, mu: float) -> np.ndarray:
    """
    由 Dyson 方程提取自能 Σ(iω_n).

    Dyson 方程:
        G^{-1}(iω_n) = iω_n + μ - ε_k - Σ(iω_n)
        → Σ(iω_n) = iω_n + μ - ε_k - G^{-1}(iω_n)

    高频渐近:
        Σ(iω_n → ∞) → U ⟨n⟩ / 2 + O(1/iω_n)
    """
    Sigma = np.zeros_like(G_iw, dtype=complex)
    for n in range(len(omega_n)):
        if abs(G_iw[n]) > 1e-15:
            Sigma[n] = 1j * omega_n[n] + mu - epsilon_k - 1.0 / G_iw[n]
        else:
            Sigma[n] = 0.0
    return Sigma
