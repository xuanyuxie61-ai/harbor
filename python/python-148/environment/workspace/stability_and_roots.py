"""
stability_and_roots.py — 神经动力系统稳定性与根界分析
=======================================================
融合 log_norm（矩阵对数范数）与 polynomial_root_bound（多项式根界）
两个项目的核心算法。

科学背景：
脑机接口闭环系统的稳定性至关重要。神经质量模型、神经场方程
离散化后的线性化系统，其特征值分布决定了解码器的稳定性边界。

核心数学：
---
**矩阵对数范数 μ_p(A)：**
对矩阵 A ∈ C^{n×n}，l_p 对数范数定义为：

    μ_p(A) = lim_{h→0+} (||I + hA||_p - 1) / h

对 p=1, ∞ 有显式公式：
    μ_1(A) = max_j ( Re(a_jj) + sum_{i≠j} |a_ij| )
    μ_∞(A) = max_i ( Re(a_ii) + sum_{j≠i} |a_ij| )

对 p=2：μ_2(A) = λ_max( (A + A^H) / 2 )，即 Hermite 部分的最大特征值。

对数范数的重要性质：
    ||exp(At)||_p ≤ exp(μ_p(A) * t)

若 μ_p(A) < 0，则系统指数稳定。

---
**Cauchy 多项式根界：**
对多项式 P(z) = c_1 z^n + c_2 z^{n-1} + ... + c_{n+1}，
所有根 z 满足 |z| ≤ r，其中 r 是以下多项式的唯一正根：

    q(x) = |c_1| x^n - |c_2| x^{n-1} - ... - |c_{n+1}| = 0

通过区间加倍找到上界 [0, R]，再用二分法精确求解 r。

---
**神经闭环系统稳定性：**
将 E-I 神经质量模型在平衡点 (E*, I*) 附近线性化：

    d/dt [δE; δI] = J * [δE; δI]

其中 Jacobian J = [[∂f_E/∂E, ∂f_E/∂I],
                   [∂f_I/∂E, ∂f_I/∂I]]
在平衡点处取值。

闭环 BCI 系统增加反馈增益 K：

    d/dt x = (J - BKC) x

稳定性要求 μ_2(J - BKC) < 0 或所有特征值实部为负。
"""

import numpy as np
from numpy.linalg import eigvalsh, norm


def logarithmic_norm(A, p=2):
    """
    计算矩阵 A 的 l_p 对数范数 μ_p(A)。
    p 可为 1, 2, np.inf。
    """
    A = np.asarray(A, dtype=np.complex128 if np.iscomplexobj(A) else float)
    if p == 1:
        # μ_1(A) = max_j ( Re(a_jj) + sum_{i≠j} |a_ij| )
        diag = np.diag(A).real
        col_sums = np.sum(np.abs(A), axis=0) - np.abs(diag)
        return float(np.max(diag + col_sums))
    elif p == np.inf:
        # μ_∞(A) = max_i ( Re(a_ii) + sum_{j≠i} |a_ij| )
        diag = np.diag(A).real
        row_sums = np.sum(np.abs(A), axis=1) - np.abs(diag)
        return float(np.max(diag + row_sums))
    elif p == 2:
        # μ_2(A) = λ_max( (A + A^H)/2 )
        sym_part = 0.5 * (A + A.conj().T)
        eigenvalues = eigvalsh(sym_part)
        return float(np.max(eigenvalues))
    else:
        raise ValueError("p must be 1, 2, or np.inf")


def exponential_bound_estimate(A, t, p=2):
    """
    利用对数范数估计 ||exp(At)||_p 的上界：
        ||exp(At)||_p ≤ exp(μ_p(A) * t)
    返回估计上界。
    """
    mu = logarithmic_norm(A, p)
    return np.exp(mu * t)


def cauchy_polynomial_root_bound(coeffs):
    """
    计算多项式根界：对 P(z) = c[0] z^n + c[1] z^{n-1} + ... + c[n]
    所有根满足 |z| ≤ r，r 为 q(x)=|c[0]|x^n - sum_{k=1}^{n} |c[k]| x^{n-k} 的唯一正根。

    算法：
      1. 通过区间加倍找到上界 R（使得 q(R) > 0）
      2. 在 [0, R] 上用二分法找到 q(x)=0 的根
    """
    c = np.asarray(coeffs, dtype=float)
    n = len(c) - 1
    if n < 0:
        return 0.0
    if n == 0:
        return 0.0
    abs_c = np.abs(c)

    def q(x):
        if x <= 0:
            return -abs_c[-1]
        val = abs_c[0] * (x ** n)
        for k in range(1, n + 1):
            val -= abs_c[k] * (x ** (n - k))
        return val

    # 找上界 R
    R = 1.0
    while q(R) < 0:
        R *= 2.0
        if R > 1e12:
            break

    # 二分法
    lo, hi = 0.0, R
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if q(mid) > 0:
            hi = mid
        else:
            lo = mid
        if hi - lo < 1e-12:
            break
    return 0.5 * (lo + hi)


def polynomial_from_matrix(A):
    """
    由特征多项式系数构造：det(λI - A) = λ^n + a_{n-1}λ^{n-1} + ... + a_0
    使用 Faddeev-LeVerrier 算法。
    返回系数 [1, a_{n-1}, ..., a_0]。
    """
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    coeffs = np.zeros(n + 1, dtype=float)
    coeffs[0] = 1.0
    B = np.eye(n)
    for k in range(1, n + 1):
        B = A @ B
        trace_Bk = np.trace(B)
        coeffs[k] = -trace_Bk / k
        B = B + coeffs[k] * np.eye(n)
    return coeffs


def neural_mass_jacobian(ei_oscillator, E_star, I_star):
    """
    计算 E-I 神经质量模型在平衡点 (E*, I*) 处的 Jacobian 矩阵。
    对模型：
        dE/dt = -E + S_e(a_ee E - a_ei I + P_e + k_e s(t))
        dI/dt = -I + S_i(a_ie E - a_ii I + P_i + k_i s(t))
    Jacobian 元素为：
        J_11 = -1 + a_ee * S_e'
        J_12 = -a_ei * S_e'
        J_21 = a_ie * S_i'
        J_22 = -1 - a_ii * S_i'
    其中 S' = S(x)*(1-S(x))/sigma 为 sigmoid 导数。
    """
    # TODO_HOLE_2: implement Jacobian computation for the E-I neural mass model
    # Compute the 2x2 Jacobian matrix J at equilibrium (E_star, I_star).
    # Must use the sigmoid derivative formula consistent with utils.sigmoid_activation.
    # Must return J as a 2x2 numpy array.
    pass


class BCIStabilityAnalyzer:
    """
    BCI 闭环系统稳定性分析器。
    """

    def __init__(self, ei_oscillator, feedback_gain=0.5):
        self.ei_osc = ei_oscillator
        self.K = feedback_gain

    def find_equilibrium(self, E_guess=0.3, I_guess=0.1, max_iter=100, tol=1e-10):
        """
        用 Newton-Raphson 找 E-I 系统的平衡点（假设平均外部驱动为零）。
        """
        from utils import sigmoid_activation
        x = np.array([E_guess, I_guess], dtype=float)
        for _ in range(max_iter):
            E, I = x
            s_val = 0.0
            x_e = self.ei_osc.a_ee * E - self.ei_osc.a_ei * I + self.ei_osc.P_e + self.ei_osc.k_e * s_val
            x_i = self.ei_osc.a_ie * E - self.ei_osc.a_ii * I + self.ei_osc.P_i + self.ei_osc.k_i * s_val
            f1 = -E + sigmoid_activation(x_e, self.ei_osc.theta_e, self.ei_osc.sigma_e)
            f2 = -I + sigmoid_activation(x_i, self.ei_osc.theta_i, self.ei_osc.sigma_i)
            F = np.array([f1, f2], dtype=float)
            J = neural_mass_jacobian(self.ei_osc, E, I)
            try:
                dx = np.linalg.solve(J, -F)
            except np.linalg.LinAlgError:
                break
            x = x + dx
            if norm(dx) < tol:
                break
        return x

    def analyze_open_loop_stability(self):
        """
        分析开环 E-I 系统的稳定性：计算 Jacobian 特征值、对数范数、根界。
        """
        eq = self.find_equilibrium()
        J = neural_mass_jacobian(self.ei_osc, eq[0], eq[1])
        eigenvalues = np.linalg.eigvals(J)
        max_real = float(np.max(eigenvalues.real))
        mu1 = logarithmic_norm(J, p=1)
        mu2 = logarithmic_norm(J, p=2)
        mu_inf = logarithmic_norm(J, p=np.inf)
        # 特征多项式根界
        char_poly_coeffs = polynomial_from_matrix(J)
        root_bound = cauchy_polynomial_root_bound(char_poly_coeffs)
        return {
            'equilibrium': eq,
            'jacobian': J,
            'eigenvalues': eigenvalues,
            'max_real_part': max_real,
            'mu_1': mu1,
            'mu_2': mu2,
            'mu_inf': mu_inf,
            'characteristic_polynomial': char_poly_coeffs,
            'cauchy_root_bound': root_bound,
            'is_stable': max_real < 0
        }

    def analyze_closed_loop_stability(self, B=np.array([[1.0], [0.0]]),
                                      C=np.array([[1.0, 0.0]])):
        """
        分析闭环系统稳定性，假设状态反馈 u = -K*C*x，
        闭环矩阵 A_cl = J - B*K*C。
        """
        eq = self.find_equilibrium()
        J = neural_mass_jacobian(self.ei_osc, eq[0], eq[1])
        A_cl = J - B * self.K * C
        eigenvalues = np.linalg.eigvals(A_cl)
        max_real = float(np.max(eigenvalues.real))
        mu2 = logarithmic_norm(A_cl, p=2)
        char_poly_coeffs = polynomial_from_matrix(A_cl)
        root_bound = cauchy_polynomial_root_bound(char_poly_coeffs)
        return {
            'closed_loop_matrix': A_cl,
            'eigenvalues': eigenvalues,
            'max_real_part': max_real,
            'mu_2': mu2,
            'cauchy_root_bound': root_bound,
            'is_stable': max_real < 0
        }

    def compute_lyapunov_exponents(self, n_steps=5000, dt=0.001):
        """
        使用 QR 分解法计算有限时间 Lyapunov 指数（FTLE）。
        对非线性系统：
            Q_{k+1} R_{k+1} = Df(x_k) Q_k
            λ_i ≈ (1 / (n*dt)) * sum_k log(R_k[i,i])
        """
        n_dim = 2
        Q = np.eye(n_dim)
        exponents = np.zeros(n_dim)
        # 初始状态在平衡点
        x = self.find_equilibrium()
        for step in range(n_steps):
            # 在当前状态计算 Jacobian
            J = neural_mass_jacobian(self.ei_osc, x[0], x[1])
            M = J @ Q
            # QR 分解
            Q, R = np.linalg.qr(M)
            for i in range(n_dim):
                exponents[i] += np.log(max(abs(R[i, i]), 1e-15))
            # 积分一步非线性动力学（简化：保持在平衡附近）
            # 这里我们只需在平衡附近线性化，因此 x 保持近似不变
        exponents /= (n_steps * dt)
        return exponents
