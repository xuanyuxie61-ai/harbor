"""
order_parameters.py
取向序参数与谱分析模块

本模块利用 Jacobi 多项式作为角向谱基函数，对脂质分子的取向分布
进行高阶展开，计算各阶序参数以及取向分布函数（ODF）。

参考种子项目: 607_jacobi_polynomial (Jacobi 多项式求值与零点)
                078_bernstein_polynomial (Bernstein 基函数)

物理背景:
    脂质链的取向序可用球谐函数 Y_l^m(θ,φ) 展开。
    对于轴对称近似（绕膜法线），只保留 m=0 项，此时关联函数退化为
    Legendre 多项式 P_l(cosθ)。
    然而，实际双层膜存在面内关联，因此引入广义 Jacobi 多项式
    P_n^{(α,β)}(x) 作为角向分布的加权正交基，权重函数
        w(x) = (1-x)^α (1+x)^β
    其中 x = cosθ，参数 α, β 由膜曲率和脂质头基化学性质决定。
"""

import numpy as np
from scipy.special import gamma as scipy_gamma


def jacobi_polynomial(n, alpha, beta, x):
    """
    计算 Jacobi 多项式 P_n^{(α,β)}(x) 及其前 n 项。

    递推关系（Abramowitz & Stegun 22.7.1）:
        P_0^{(α,β)}(x) = 1
        P_1^{(α,β)}(x) = [(α+β+2)x + (α-β)] / 2

        对于 k ≥ 1:
        a_k P_{k+1} = (b_k + c_k x) P_k - d_k P_{k-1}

    其中:
        a_k = 2(k+1)(k+α+β+1)(2k+α+β)
        b_k = (2k+α+β+1)(α²-β²)
        c_k = (2k+α+β+1)(2k+α+β)(2k+α+β+2)
        d_k = 2(k+α)(k+β)(2k+α+β+2)

    Parameters
    ----------
    n : int
        最高阶数。
    alpha, beta : float
        Jacobi 参数，要求 α > -1, β > -1。
    x : ndarray
        求值点。

    Returns
    -------
    P : ndarray, shape (len(x), n+1)
        P[:, k] = P_k^{(α,β)}(x)。
    """
    if alpha <= -1.0 or beta <= -1.0:
        raise ValueError("alpha, beta 必须大于 -1。")
    if n < 0:
        return np.zeros((len(np.atleast_1d(x)), 0))

    x = np.asarray(x, dtype=float)
    m = x.size
    P = np.zeros((m, n + 1))
    P[:, 0] = 1.0

    if n == 0:
        return P

    P[:, 1] = ((alpha + beta + 2.0) * x + (alpha - beta)) / 2.0

    for k in range(1, n):
        ak = 2.0 * (k + 1.0) * (k + alpha + beta + 1.0) * (2.0 * k + alpha + beta)
        bk = (2.0 * k + alpha + beta + 1.0) * (alpha**2 - beta**2)
        ck = (2.0 * k + alpha + beta + 1.0) * (2.0 * k + alpha + beta) * (2.0 * k + alpha + beta + 2.0)
        dk = 2.0 * (k + alpha) * (k + beta) * (2.0 * k + alpha + beta + 2.0)

        if abs(ak) < 1e-15:
            raise RuntimeError("Jacobi 递推分母 ak 接近零。")

        P[:, k + 1] = ((bk + ck * x) * P[:, k] - dk * P[:, k - 1]) / ak

    return P


def jacobi_norm_constant(n, alpha, beta):
    """
    Jacobi 多项式的归一化常数:
        h_n = ∫_{-1}^{1} (1-x)^α (1+x)^β [P_n^{(α,β)}(x)]² dx
          = 2^{α+β+1} Γ(n+α+1) Γ(n+β+1) / [(2n+α+β+1) n! Γ(n+α+β+1)]
    """
    if n < 0:
        return 0.0
    num = (2.0 ** (alpha + beta + 1.0)) * scipy_gamma(n + alpha + 1.0) * scipy_gamma(n + beta + 1.0)
    den = (2.0 * n + alpha + beta + 1.0) * scipy_gamma(n + 1.0) * scipy_gamma(n + alpha + beta + 1.0)
    if den == 0 or not np.isfinite(den):
        return 0.0
    return float(num / den)


def jacobi_zeros_guess(n, alpha, beta):
    """
    利用 Newton-Maehly 思想（种子项目 801_newton_maehly 的简化版），
    通过高斯-雅可比求积的近似零点作为初始猜测。
    此处使用已知的渐近分布:
        x_k ≈ cos[(k - 0.25)π / (n + 0.5*(α+β+1))]
    """
    if n <= 0:
        return np.array([])
    k = np.arange(1, n + 1)
    denom = n + 0.5 * (alpha + beta + 1.0)
    if denom <= 0:
        denom = 1.0
    return np.cos((k - 0.25) * np.pi / denom)


class OrientationalOrderAnalysis:
    """
    脂质取向序分析器。

    利用 Jacobi 多项式展开取向分布函数 f(cosθ):
        f(x) = Σ_{n=0}^{N_max} c_n P_n^{(α,β)}(x)

    展开系数:
        c_n = (1/h_n) ∫_{-1}^{1} w(x) f(x) P_n^{(α,β)}(x) dx

    其中权重 w(x) = (1-x)^α (1+x)^β 反映了链熔化的非对称性。
    在凝胶相，取向集中于 x≈1（θ≈0），因此取 α 较小、β 较大可
    增强对 x≈1 区域的分辨率。
    """

    def __init__(self, n_max=12, alpha=0.0, beta=2.0):
        if n_max < 0:
            raise ValueError("n_max 必须非负。")
        if alpha <= -1.0 or beta <= -1.0:
            raise ValueError("alpha, beta 必须大于 -1。")
        self.n_max = n_max
        self.alpha = alpha
        self.beta = beta

    def expand_odf(self, cos_theta_samples):
        """
        从取向样本估计 ODF 展开系数。

        Parameters
        ----------
        cos_theta_samples : ndarray
            cosθ 样本值，范围 [-1, 1]。

        Returns
        -------
        coeffs : ndarray
            Jacobi 展开系数 c_n。
        """
        cos_theta_samples = np.asarray(cos_theta_samples)
        cos_theta_samples = np.clip(cos_theta_samples, -1.0, 1.0)

        coeffs = np.zeros(self.n_max + 1)
        P_vals = jacobi_polynomial(self.n_max, self.alpha, self.beta,
                                   cos_theta_samples)
        w = ((1.0 - cos_theta_samples) ** self.alpha *
             (1.0 + cos_theta_samples) ** self.beta)
        w = np.where(w < 0, 0.0, w)

        for n in range(self.n_max + 1):
            h_n = jacobi_norm_constant(n, self.alpha, self.beta)
            if h_n <= 0:
                coeffs[n] = 0.0
                continue
            integrand = w * P_vals[:, n]
            coeffs[n] = np.mean(integrand) / h_n

        return coeffs

    def order_parameters_from_coeffs(self, coeffs):
        """
        由展开系数提取各阶序参数。

        对于 m=0 的球谐函数，序参数:
            S_l = c_l * √(4π/(2l+1)) * 归一化因子
        在 Jacobi 基下，近似:
            S_l ≈ coeffs[l] / coeffs[0]   （对于 l ≥ 1）

        Returns
        -------
        s_params : ndarray
            S_0, S_1, ..., S_{n_max}。
        """
        coeffs = np.asarray(coeffs)
        if coeffs[0] == 0:
            return np.zeros_like(coeffs)
        s_params = coeffs / coeffs[0]
        return s_params

    def reconstruct_odf(self, x_grid, coeffs):
        """
        由展开系数重构取向分布函数 f(x)。

        f(x) = Σ_n c_n P_n^{(α,β)}(x)
        """
        x_grid = np.asarray(x_grid)
        x_grid = np.clip(x_grid, -1.0, 1.0)
        P_vals = jacobi_polynomial(self.n_max, self.alpha, self.beta, x_grid)
        return P_vals @ coeffs

    def compute_entropy(self, coeffs):
        """
        由 ODF 计算取向熵（近似）。

        S_orient = -k_B ∫ f(x) ln[f(x)] dx
        采用 Gauss-Jacobi 求积近似。
        """
        # 使用 Jacobi 多项式零点作为求积节点（简化：用等距节点）
        n_quad = max(2 * self.n_max + 1, 20)
        x_nodes = np.linspace(-0.999, 0.999, n_quad)
        dx = x_nodes[1] - x_nodes[0]
        f = self.reconstruct_odf(x_nodes, coeffs)
        f = np.clip(f, 1e-12, None)
        entropy = -np.sum(f * np.log(f)) * dx
        return float(entropy)


def spherical_harmonic_y20_approx(cos_theta):
    """
    球谐函数 Y_2^0 的近似（轴对称情况）。

    Y_2^0(θ) = √(5/(16π)) * (3cos²θ - 1)
             = √(5/(4π)) * P_2(cosθ)
    """
    coeff = np.sqrt(5.0 / (16.0 * np.pi))
    return coeff * (3.0 * cos_theta ** 2 - 1.0)


def debye_waller_factor(order_param, temperature, moment_inertia=1.0):
    """
    Debye-Waller 因子，描述热振动对 X-ray/中子散射强度的衰减。

    B = (8π²/3) <u²>
    其中 <u²> = k_B T / (I * ω_0²) * (1 - S_2)

    参数:
        order_param : float
            二阶序参数 S_2。
        temperature : float
            温度 T（K）。
        moment_inertia : float
            转动惯量 I。

    Returns
    -------
    B : float
        Debye-Waller 因子。
    """
    if moment_inertia <= 0:
        raise ValueError("转动惯量必须为正。")
    kb = 1.380649e-23
    u2 = kb * temperature / moment_inertia * (1.0 - order_param)
    B = (8.0 * np.pi ** 2 / 3.0) * u2
    return float(B)
