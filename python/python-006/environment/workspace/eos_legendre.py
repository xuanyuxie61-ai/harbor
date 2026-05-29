"""
eos_legendre.py
中子星致密物质状态方程 (Equation of State, EOS) 计算模块

基于连带Legendre多项式展开与Skyrme-Hartree-Fock框架，
计算非对称核物质的状态方程 P(ε, δ)，其中 δ = (ρ_n - ρ_p)/ρ_b 为同位旋不对称度。

原项目映射:
- 661_legendre_polynomial  -> 连带Legendre多项式 pm_polynomial_value，
                              用于展开核物质相互作用势的角度依赖性
"""

import numpy as np
import math
from utils_physics import safe_sqrt, safe_divide, fermi_momentum_to_density


# =============================================================================
# 连带Legendre多项式 P_n^m(x)
# =============================================================================
def associated_legendre_polynomial_value(mm: int, n: int, m: int, x: np.ndarray) -> np.ndarray:
    """
    计算连带Legendre多项式 P_n^m(x) 的值。

    源自 661_legendre_polynomial 中 pm_polynomial_value.m 的核心递推算法。
    在核物理中，P_n^m(cosθ) 用于展开核子-核子相互作用势的角分布，
    例如 Argonne v18 势中的中心力、张量力等分波展开。

    微分方程:
        (1 - x^2) y'' - 2x y' + [n(n+1) - m^2/(1-x^2)] y = 0

    递推关系:
        P_m^m(x)   = -(2m-1)!! (1-x^2)^{m/2}
        P_{m+1}^m(x) = x (2m+1) P_m^m(x)
        (n-m) P_n^m(x) = x(2n-1) P_{n-1}^m(x) - (n+m-1) P_{n-2}^m(x)

    Parameters
    ----------
    mm : int
        计算点的数量。
    n : int
        最高阶数（>= 0）。
    m : int
        阶数（0 <= m <= n）。
    x : np.ndarray
        计算点，形状 (mm,)，取值范围 [-1, 1]。

    Returns
    -------
    cx : np.ndarray
        函数值，形状 (mm, n+1)。
    """
    x = np.asarray(x)
    if x.ndim != 1:
        x = x.reshape(-1)
    mm = x.size

    # 边界检查
    if np.any(np.abs(x) > 1.0 + 1e-12):
        raise ValueError("Associated Legendre argument x must be in [-1, 1].")

    cx = np.zeros((mm, n + 1), dtype=float)

    if m <= n:
        cx[:, m] = 1.0
        fact = 1.0
        for j in range(1, m + 1):
            cx[:, m] = -cx[:, m] * fact * safe_sqrt(1.0 - x**2)
            fact += 2.0

    if m + 1 <= n:
        cx[:, m + 1] = (2 * m + 1) * x * cx[:, m]

    for j in range(m + 2, n + 1):
        cx[:, j] = (
            (2 * j - 1) * x * cx[:, j - 1]
            + (-j - m + 1) * cx[:, j - 2]
        ) / (j - m)

    return cx


def legendre_angular_expansion(coeffs: np.ndarray, cos_theta: np.ndarray) -> np.ndarray:
    """
    利用Legendre多项式展开计算角度依赖的相互作用势。

    V(cosθ) = Σ_{l=0}^{L_max} c_l P_l(cosθ)

    Parameters
    ----------
    coeffs : np.ndarray
        展开系数 c_l。
    cos_theta : np.ndarray
        角度余弦值。

    Returns
    -------
    V : np.ndarray
        展开后的势函数值。
    """
    L_max = len(coeffs) - 1
    mm = cos_theta.size
    P_vals = associated_legendre_polynomial_value(mm, L_max, 0, cos_theta)
    V = np.zeros_like(cos_theta)
    for l in range(L_max + 1):
        V += coeffs[l] * P_vals[:, l]
    return V


# =============================================================================
# Skyrme型状态方程
# =============================================================================
class SkyrmeEOS:
    """
    Skyrme有效相互作用参数化的核物质状态方程。

    能量密度泛函 (单位: MeV fm^-3):
        ε = (ħ^2/2m) τ + (3/8) t0 ρ^2 + (1/16) t3 ρ^{α+2}
            + (3/80) (3t1 + 5t2) ρ τ + (1/64) (9t1 - 5t2) (\nabla ρ)^2

    压强由热力学关系给出:
        P = ρ^2 ∂(ε/ρ)/∂ρ|_S

    参考参数（SLy4 参数组）:
        t0 = -2488.91  MeV·fm^3
        t1 = 486.82    MeV·fm^5
        t2 = -546.39   MeV·fm^5
        t3 = 13777.0   MeV·fm^{3+3α}
        α  = 1/6
        x0 = 0.834
        x1 = -0.344
        x2 = -1.0
        x3 = 1.354
    """

    def __init__(self, t0: float = -2488.91, t1: float = 486.82,
                 t2: float = -546.39, t3: float = 13777.0,
                 alpha: float = 1.0 / 6.0,
                 x0: float = 0.834, x1: float = -0.344,
                 x2: float = -1.0, x3: float = 1.354):
        self.t0 = t0
        self.t1 = t1
        self.t2 = t2
        self.t3 = t3
        self.alpha = alpha
        self.x0 = x0
        self.x1 = x1
        self.x2 = x2
        self.x3 = x3
        self.hbar2_over_2m = 20.73553  # MeV·fm^2 (核子平均)

    def energy_density(self, rho_b: float, delta: float = 0.0) -> float:
        """
        计算非对称核物质的能量密度 ε (MeV/fm^3)。

        公式:
            ε = ε_{kin} + ε_{pot}

            ε_{kin} = (3/5) (ħ^2/2m) k_F^2 [(1+δ)^{5/3} + (1-δ)^{5/3}]/2

            ε_{pot} = (3/8) t0 ρ^2 [ (1 + x0/2) - (x0 + 1/2) δ^2 ]
                      + (1/16) t3 ρ^{α+2} [ (1 + x3/2) - (x3 + 1/2) δ^2 ]
                      + ... (t1, t2 项对有效质量修正)

        Parameters
        ----------
        rho_b : float
            重子数密度 (fm^-3)，必须 >= 0。
        delta : float
            同位旋不对称度，|δ| <= 1。

        Returns
        -------
        float
            能量密度 (MeV/fm^3)。
        """
        if rho_b < 0.0:
            raise ValueError("Baryon density rho_b must be non-negative.")
        if abs(delta) > 1.0 + 1e-12:
            raise ValueError("Isospin asymmetry delta must be in [-1, 1].")
        delta = np.clip(delta, -1.0, 1.0)

        kf = (3.0 * math.pi**2 * rho_b)**(1.0 / 3.0)

        # 动能项
        kin_term = 0.0
        if rho_b > 1e-15:
            kin_term = (3.0 / 5.0) * self.hbar2_over_2m * kf**2 * 0.5 * (
                (1.0 + delta)**(5.0 / 3.0) + (1.0 - delta)**(5.0 / 3.0)
            )

        # 势能项 (简化的 Skyrme 中心力)
        pot_term = (3.0 / 8.0) * self.t0 * rho_b**2 * (
            (1.0 + self.x0 / 2.0) - (self.x0 + 0.5) * delta**2
        )
        pot_term += (1.0 / 16.0) * self.t3 * rho_b**(self.alpha + 2.0) * (
            (1.0 + self.x3 / 2.0) - (self.x3 + 0.5) * delta**2
        )

        return kin_term + pot_term

    def pressure(self, rho_b: float, delta: float = 0.0) -> float:
        """
        计算压强 P (MeV/fm^3)。

        通过热力学关系 P = ρ_b^2 ∂(ε/ρ_b)/∂ρ_b 数值微分求得，
        或解析求导（此处采用解析导数以提高精度）。

        P = P_{kin} + P_{pot}

        P_{kin} = (2/5) (ħ^2/2m) k_F^2 ρ_b [(1+δ)^{5/3} + (1-δ)^{5/3}]/2

        P_{pot} = (3/8) t0 ρ_b^2 [ ... ] * (某些导数关系)
        """
        if rho_b < 1e-15:
            return 0.0
        if abs(delta) > 1.0 + 1e-12:
            raise ValueError("Isospin asymmetry delta must be in [-1, 1].")
        delta = np.clip(delta, -1.0, 1.0)

        kf = (3.0 * math.pi**2 * rho_b)**(1.0 / 3.0)

        # 动能压强
        p_kin = (2.0 / 5.0) * self.hbar2_over_2m * kf**2 * rho_b * 0.5 * (
            (1.0 + delta)**(5.0 / 3.0) + (1.0 - delta)**(5.0 / 3.0)
        )

        # TODO: 修复此处被挖空的势能压强公式
        # 势能压强由 Skyrme 能量密度对密度的热力学导数给出
        # p_pot = ???
        raise NotImplementedError("Hole 1: 请补全 Skyrme EOS 的势能压强计算公式")

        return p_kin + p_pot

    def chemical_potential(self, rho_b: float, delta: float = 0.0) -> tuple:
        """
        计算中子和质子的化学势 μ_n, μ_p (MeV)。

        μ_i = ∂ε/∂ρ_i

        Returns
        -------
        (mu_n, mu_p) : tuple of float
        """
        d = 1e-6
        eps_p = self.energy_density(rho_b + d, delta)
        eps_m = self.energy_density(rho_b - d, delta)
        deps_drho = (eps_p - eps_m) / (2.0 * d)

        mu_avg = deps_drho
        # 对称能分离项（近似）
        E_sym = self.symmetry_energy(rho_b)
        mu_n = mu_avg + E_sym * delta
        mu_p = mu_avg - E_sym * delta
        return mu_n, mu_p

    def symmetry_energy(self, rho_b: float) -> float:
        """
        计算对称能 E_sym (MeV) 在密度 rho_b 处的值。

        E_sym(ρ) = (1/2) ∂^2(ε/ρ)/∂δ^2|_{δ=0}
        """
        d = 1e-4
        e0 = self.energy_density(rho_b, 0.0)
        ep = self.energy_density(rho_b, d)
        em = self.energy_density(rho_b, -d)
        second_deriv = (ep - 2.0 * e0 + em) / (d**2)
        return 0.5 * second_deriv / rho_b if rho_b > 1e-15 else 0.0

    def sound_speed_squared(self, rho_b: float, delta: float = 0.0) -> float:
        """
        计算声速平方 c_s^2 / c^2 (无量纲)。

        c_s^2 = ∂P/∂ε = (∂P/∂ρ) / (∂ε/∂ρ)
        """
        dr = 1e-5 * rho_b if rho_b > 1e-15 else 1e-8
        p_plus = self.pressure(rho_b + dr, delta)
        p_minus = self.pressure(rho_b - dr, delta)
        e_plus = self.energy_density(rho_b + dr, delta)
        e_minus = self.energy_density(rho_b - dr, delta)

        dp_dr = (p_plus - p_minus) / (2.0 * dr)
        de_dr = (e_plus - e_minus) / (2.0 * dr)

        cs2 = safe_divide(dp_dr, de_dr, default=0.0)
        # 因果性限制: c_s^2 <= c^2
        return min(max(cs2, 0.0), 1.0)


class PolytropicEOS:
    """
    分段多方物态方程，用于与Skyrme EOS拼接，描述超核饱和密度以上的行为。

    P = K ρ^Γ

    能量密度与压强的关系:
        ε = ρ + P/(Γ - 1)   (自然单位 c = 1)
    """

    def __init__(self, K: float, Gamma: float):
        if Gamma <= 1.0:
            raise ValueError("Polytropic index Gamma must be > 1.")
        self.K = K
        self.Gamma = Gamma

    def pressure_from_density(self, rho: float) -> float:
        if rho < 0.0:
            raise ValueError("Density must be non-negative.")
        return self.K * rho**self.Gamma

    def energy_density_from_density(self, rho: float) -> float:
        P = self.pressure_from_density(rho)
        return rho + P / (self.Gamma - 1.0)

    def pressure_from_energy_density(self, eps: float) -> float:
        """
        由能量密度反解压强，需数值求解。
        使用牛顿迭代法。
        """
        if eps < 0.0:
            raise ValueError("Energy density must be non-negative.")
        # 初始猜测
        rho = eps
        for _ in range(50):
            P = self.pressure_from_density(rho)
            f = rho + P / (self.Gamma - 1.0) - eps
            df = 1.0 + self.K * self.Gamma * rho**(self.Gamma - 1.0) / (self.Gamma - 1.0)
            if abs(df) < 1e-30:
                break
            drho = -f / df
            rho += drho
            if abs(drho) < 1e-12:
                break
        return self.pressure_from_density(rho)


def build_composite_eos():
    """
    构建组合物态方程：低密区用Skyrme，高密区用多段多方拼接。
    返回一个可调用对象，输入能量密度 ε (MeV/fm^3)，输出压强 P (MeV/fm^3)。
    """
    skyrme = SkyrmeEOS()
    # 高密拼接参数
    poly1 = PolytropicEOS(K=0.05, Gamma=2.5)

    def eos_func(eps: float) -> float:
        if eps < 0.0:
            raise ValueError("Energy density must be non-negative.")
        # 饱和密度附近能量密度约 150 MeV/fm^3
        if eps < 300.0:
            # 反向查找密度（简化为近似）
            rho = eps / 150.0 * 0.16  # 粗略反演
            rho = max(rho, 1e-10)
            return skyrme.pressure(rho)
        else:
            return poly1.pressure_from_energy_density(eps)

    return eos_func
