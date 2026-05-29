"""
cosmology.py
============
宇宙学背景演化模块

基于 ΛCDM 平坦宇宙模型，提供：
1. 尺度因子 a(t) 的背景演化（Friedmann 方程数值积分，融入 RK12 思想）
2. Hubble 参数 H(a)、物质密度参数 Ω_m(a)、宇宙学距离
3. 线性增长因子 D(a) 的自适应 RK12 积分
4. 非线性方程二分法求根（融入 nonlin_bisect）求解特定红移对应的宇宙时间

核心物理公式
------------
Friedmann 方程（平坦 ΛCDM）：
    H²(a) = H₀² [ Ω_m a⁻³ + Ω_r a⁻⁴ + Ω_Λ ]

尺度因子演化：
    da/dt = a H(a) = H₀ a √[ Ω_m a⁻³ + Ω_r a⁻⁴ + Ω_Λ ]

线性增长因子方程：
    d²D/da² + (3 + d ln H / d ln a)/(2a) · dD/da - (3 Ω_m(a))/(2a²) D = 0

    其中 Ω_m(a) = Ω_m a⁻³ / [Ω_m a⁻³ + Ω_r a⁻⁴ + Ω_Λ]

临界过密度（球对称坍缩模型）：
    δ_c(z) ≈ 1.686 / D(a) · (1 + 0.0123 log₁₀ Ω_m(a))
"""

import numpy as np
from typing import Tuple, Callable


class Cosmology:
    """
    ΛCDM 平坦宇宙学模型参数与背景演化计算。
    """

    def __init__(
        self,
        h: float = 0.6732,
        Omega_m: float = 0.3158,
        Omega_b: float = 0.0494,
        Omega_Lambda: float = 0.6842,
        Omega_r: float = 9.2e-5,
        T_cmb: float = 2.7255,
        sigma8: float = 0.811,
        ns: float = 0.965,
    ):
        """
        初始化 Planck 2018 基准宇宙学参数。

        Parameters
        ----------
        h : float
            Hubble 常数 H₀ = 100 h km/s/Mpc
        Omega_m : float
            当前物质密度参数（含重子与暗物质）
        Omega_b : float
            当前重子密度参数
        Omega_Lambda : float
            当前暗能量密度参数
        Omega_r : float
            当前辐射密度参数（含光子与中微子）
        T_cmb : float
            CMB 温度（K）
        sigma8 : float
            8 h⁻¹ Mpc 尺度上的质量涨落幅度
        ns : float
            原初功率谱谱指数
        """
        self.h = h
        self.H0 = 100.0 * h  # km/s/Mpc
        self.Omega_m = Omega_m
        self.Omega_b = Omega_b
        self.Omega_Lambda = Omega_Lambda
        self.Omega_r = Omega_r
        self.T_cmb = T_cmb
        self.sigma8 = sigma8
        self.ns = ns

        # 物理常数
        self.G = 4.30091e-9  # Mpc M⊙⁻¹ (km/s)²
        self.c = 299792.458  # km/s
        self.rho_crit_0 = 3.0 * self.H0 ** 2 / (8.0 * np.pi * self.G)  # M⊙ / Mpc³

        # 一致性校验
        total_omega = Omega_m + Omega_Lambda + Omega_r
        if abs(total_omega - 1.0) > 0.02:
            raise ValueError(f"平坦性破坏: Ω_total = {total_omega:.4f} ≠ 1")
        if h <= 0.0 or h > 2.0:
            raise ValueError(f"不合理的 h = {h}")
        if Omega_m <= 0.0 or Omega_m > 1.5:
            raise ValueError(f"不合理的 Ω_m = {Omega_m}")

    def H(self, a: float) -> float:
        """
        计算给定尺度因子 a 处的 Hubble 参数 H(a)。

        公式:
            H(a) = H₀ √[ Ω_m a⁻³ + Ω_r a⁻⁴ + Ω_Λ ]

        Parameters
        ----------
        a : float
            尺度因子，a > 0

        Returns
        -------
        float
            Hubble 参数，单位 km/s/Mpc
        """
        if a <= 0.0:
            raise ValueError("尺度因子 a 必须为正")
        a2 = a * a
        a3 = a2 * a
        a4 = a3 * a
        term = self.Omega_m / a3 + self.Omega_r / a4 + self.Omega_Lambda
        return self.H0 * np.sqrt(term)

    def dH_da(self, a: float) -> float:
        """
        H(a) 对 a 的导数，用于线性增长因子方程。

        dH/da = H₀²/(2H) · (-3Ω_m a⁻⁴ - 4Ω_r a⁻⁵)
        """
        if a <= 0.0:
            raise ValueError("尺度因子 a 必须为正")
        ha = self.H(a)
        if ha == 0.0:
            return 0.0
        return (
            self.H0 ** 2
            / (2.0 * ha)
            * (-3.0 * self.Omega_m / a ** 4 - 4.0 * self.Omega_r / a ** 5)
        )

    def dlnH_dlna(self, a: float) -> float:
        """
        d ln H / d ln a = a / H · dH/da
        """
        ha = self.H(a)
        if ha == 0.0:
            return 0.0
        return a * self.dH_da(a) / ha

    def Omega_m_a(self, a: float) -> float:
        """
        随尺度因子演化的物质密度参数:
            Ω_m(a) = Ω_m a⁻³ / E²(a)
        其中 E(a) = H(a)/H₀
        """
        e2 = (
            self.Omega_m / a ** 3
            + self.Omega_r / a ** 4
            + self.Omega_Lambda
        )
        return (self.Omega_m / a ** 3) / e2

    def scale_factor_evolution_rhs(self, t: float, y: np.ndarray) -> np.ndarray:
        """
        尺度因子演化的 ODE 右端项:
            dy/dt = [da/dt] = a H(a) = y[0] * H(y[0])
        """
        # TODO: 实现尺度因子演化方程的右端项
        raise NotImplementedError("请实现 scale_factor_evolution_rhs 方法")

    def linear_growth_rhs(self, a: float, y: np.ndarray) -> np.ndarray:
        """
        线性增长因子 D(a) 的二阶 ODE 化为一阶系统:
            y = [D, dD/da]
            dy/da = [y[1], f(a, y[0], y[1])]

        其中:
            d²D/da² = - (3 + dlnH/dlna)/(2a) · dD/da + (3 Ω_m(a))/(2a²) D
        """
        D, dD_da = y[0], y[1]
        if a <= 0.0:
            a = 1e-10
        coeff1 = -(3.0 + self.dlnH_dlna(a)) / (2.0 * a)
        coeff2 = (3.0 * self.Omega_m_a(a)) / (2.0 * a * a)
        return np.array([dD_da, coeff1 * dD_da + coeff2 * D])

    def rk12_integrate(
        self,
        rhs: Callable[[float, np.ndarray], np.ndarray],
        t_span: Tuple[float, float],
        y0: np.ndarray,
        n_steps: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        显式 Runge-Kutta 1-2 阶自适应积分器（融入 rk12 核心思想）。

        算法:
            k1 = dt · rhs(t_n, y_n)
            ỹ_{n+1} = y_n + k1                (Euler, 一阶)
            k2 = dt · rhs(t_n + dt, y_n + k1)
            y_{n+1} = y_n + (k1 + k2)/2        (Heun, 二阶)
            e_{n+1} = y_{n+1} - ỹ_{n+1}        (局部截断误差估计)

        Parameters
        ----------
        rhs : callable
            右端函数 f(t, y)
        t_span : (t0, t1)
            时间区间
        y0 : np.ndarray
            初始条件
        n_steps : int
            步数

        Returns
        -------
        t : np.ndarray
            时间节点
        y : np.ndarray
            数值解
        e : np.ndarray
            误差估计
        """
        t0, t1 = t_span
        if n_steps <= 0:
            raise ValueError("步数必须为正")
        dt = (t1 - t0) / n_steps
        dim = len(y0)
        t = np.zeros(n_steps + 1)
        y = np.zeros((n_steps + 1, dim))
        e = np.zeros((n_steps + 1, dim))
        t[0] = t0
        y[0, :] = y0
        e[0, :] = 0.0

        for i in range(n_steps):
            ti = t[i]
            yi = y[i, :]
            k1 = dt * rhs(ti, yi)
            y_euler = yi + k1
            k2 = dt * rhs(ti + dt, y_euler)
            y[i + 1, :] = yi + 0.5 * (k1 + k2)
            e[i + 1, :] = y[i + 1, :] - y_euler
            t[i + 1] = ti + dt

        return t, y, e

    def compute_scale_factor_history(
        self, t_Gyr_span: Tuple[float, float] = (0.0, 13.8), n_steps: int = 2000
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        从大爆炸后 t_Gyr_span[0] 到 t_Gyr_span[1] 积分尺度因子演化。

        宇宙时间单位为 Gyr，需转换为内部单位:
            1 Gyr = 10⁹ yr ≈ 3.086e19 s
            H₀ 单位 km/s/Mpc = (km/s) / (3.086e19 km) ≈ 1/(3.086e16 s)
            因此 a H(a) 的单位转换因子约为 1e-3 / Gyr
        """
        # 转换系数: km/s/Mpc → Gyr⁻¹
        conv = 1.02271e-3
        y0 = np.array([1e-8])  # 大爆炸时 a ≈ 0

        def rhs_local(t: float, y: np.ndarray) -> np.ndarray:
            a = y[0]
            if a <= 0.0:
                a = 1e-10
            return np.array([a * self.H(a) * conv])

        t, y, e = self.rk12_integrate(rhs_local, t_Gyr_span, y0, n_steps)
        a = y[:, 0]
        # 确保 a 不越界
        a = np.clip(a, 1e-10, None)
        return t, a, e[:, 0]

    def compute_linear_growth_factor(
        self, a_min: float = 1e-4, a_max: float = 1.0, n_steps: int = 2000
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        计算线性增长因子 D(a) 从早期宇宙到今天的演化。

        早期宇宙近似（物质-辐射平衡前）:
            D(a) ≈ a  (当 a << a_eq)
            dD/da ≈ 1

        因此初始条件取 a_min 处 D = a_min, dD/da = 1。
        """
        y0 = np.array([a_min, 1.0])
        a_arr, y, e = self.rk12_integrate(
            self.linear_growth_rhs, (a_min, a_max), y0, n_steps
        )
        D = y[:, 0]
        dD_da = y[:, 1]
        # 归一化到 D(a=1)=1
        if len(D) > 0 and D[-1] != 0.0:
            D_norm = D / D[-1]
        else:
            D_norm = D
        return a_arr, D_norm, e[:, 0]

    def bisect_root_finder(
        self,
        func: Callable[[float], float],
        a: float,
        b: float,
        tol: float = 1e-12,
        max_iter: int = 100,
    ) -> Tuple[float, float, int]:
        """
        二分法求根（融入 nonlin_bisect 核心算法）。

        要求 func(a) 与 func(b) 异号。
        迭代格式:
            c = (a + b)/2
            若 sign(f(c)) == sign(f(a)): a = c
            否则: b = c
            直至 |b - a| < tol

        Parameters
        ----------
        func : callable
            目标函数
        a, b : float
            含根区间端点
        tol : float
            容差
        max_iter : int
            最大迭代次数

        Returns
        -------
        a, b : float
            收缩后的区间
        it : int
            实际迭代次数
        """
        fa = func(a)
        fb = func(b)
        if np.sign(fa) == np.sign(fb):
            raise ValueError(
                f"二分法要求 f(a) 与 f(b) 异号，但得到 {fa:.4e} 与 {fb:.4e}"
            )
        it = 0
        while abs(b - a) > tol and it < max_iter:
            c = (a + b) * 0.5
            fc = func(c)
            it += 1
            if np.sign(fc) == np.sign(fa):
                a = c
                fa = fc
            else:
                b = c
                fb = fc
        return a, b, it

    def age_of_universe(self, a_target: float = 1.0) -> float:
        """
        计算宇宙达到目标尺度因子 a_target 时的年龄。

        通过数值积分:
            t(a) = ∫_0^a da' / (a' H(a'))

        采用复合 Simpson 法则，积分变量改为 ln(a) 以改善小 a 行为:
            t = ∫_{-∞}^{ln a} du / H(e^u)
        """
        if a_target <= 0.0 or a_target > 2.0:
            raise ValueError("a_target 必须在 (0, 2] 范围内")

        # 转换因子: km/s/Mpc -> Gyr⁻¹
        conv = 1.02271e-3
        n_steps = 2000
        # 积分区间 [u_min, u_max] = [ln(a_min), ln(a_target)]
        u_min = np.log(1e-10)
        u_max = np.log(a_target)
        du = (u_max - u_min) / n_steps
        us = np.linspace(u_min, u_max, n_steps + 1)
        a_vals = np.exp(us)
        H_vals = np.array([self.H(a) for a in a_vals]) * conv
        H_vals = np.clip(H_vals, 1e-30, None)
        integrand = 1.0 / H_vals

        # Simpson 积分
        S = integrand[0] + integrand[-1]
        S += 4.0 * np.sum(integrand[1:-1:2])
        S += 2.0 * np.sum(integrand[2:-1:2])
        t_age = S * du / 3.0
        return t_age

    def delta_c(self, z: float) -> float:
        """
        球对称坍缩模型的线性临界过密度 δ_c(z)。

        在 Einstein-de Sitter 宇宙中 δ_c ≈ 1.686。
        对于 ΛCDM，采用近似:
            δ_c(z) ≈ 1.686 / D(z) · [1 + 0.0123 log₁₀ Ω_m(z)]

        其中 D(z) = D(a=1/(1+z)) 为线性增长因子。
        """
        a = 1.0 / (1.0 + z)
        a_arr, D_arr, _ = self.compute_linear_growth_factor(
            a_min=1e-4, a_max=1.0, n_steps=2000
        )
        # 插值求 D(a)
        D = np.interp(a, a_arr, D_arr)
        if D <= 0.0:
            D = 1e-10
        omega_m_z = self.Omega_m_a(a)
        return 1.686 * (1.0 + 0.0123 * np.log10(omega_m_z)) / D

    def comoving_distance(self, z: float, n_int: int = 1000) -> float:
        """
        计算到红移 z 的共动距离:
            D_C(z) = c ∫₀^z dz' / H(z')

        使用复合 Simpson 积分。
        """
        if z < 0.0:
            raise ValueError("红移 z 不能为负")
        if z == 0.0:
            return 0.0
        zs = np.linspace(0.0, z, n_int + 1)
        # H(z) = H₀ E(z), E(z) = √[Ω_m(1+z)³ + Ω_r(1+z)⁴ + Ω_Λ]
        zp1 = 1.0 + zs
        E = np.sqrt(
            self.Omega_m * zp1 ** 3
            + self.Omega_r * zp1 ** 4
            + self.Omega_Lambda
        )
        integrand = self.c / (self.H0 * E)  # Mpc
        # Simpson 法则
        h_step = z / n_int
        S = integrand[0] + integrand[-1]
        S += 4.0 * np.sum(integrand[1:-1:2])
        S += 2.0 * np.sum(integrand[2:-1:2])
        return S * h_step / 3.0


if __name__ == "__main__":
    # 快速自检
    cosmo = Cosmology()
    print(f"H0 = {cosmo.H0:.2f} km/s/Mpc")
    print(f"H(a=1) = {cosmo.H(1.0):.2f} km/s/Mpc")
    print(f"Ω_m(a=1) = {cosmo.Omega_m_a(1.0):.4f}")
    age = cosmo.age_of_universe(a_target=1.0)
    print(f"宇宙年龄（a=1）≈ {age:.3f} Gyr")
    dc = cosmo.delta_c(z=0.0)
    print(f"δ_c(z=0) ≈ {dc:.4f}")
    dcz = cosmo.comoving_distance(z=1.0)
    print(f"共动距离 D_C(z=1) ≈ {dcz:.2f} Mpc")
