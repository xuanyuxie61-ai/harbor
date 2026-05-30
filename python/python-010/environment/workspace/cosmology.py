
import numpy as np
from typing import Tuple, Callable


class Cosmology:

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
        self.h = h
        self.H0 = 100.0 * h
        self.Omega_m = Omega_m
        self.Omega_b = Omega_b
        self.Omega_Lambda = Omega_Lambda
        self.Omega_r = Omega_r
        self.T_cmb = T_cmb
        self.sigma8 = sigma8
        self.ns = ns


        self.G = 4.30091e-9
        self.c = 299792.458
        self.rho_crit_0 = 3.0 * self.H0 ** 2 / (8.0 * np.pi * self.G)


        total_omega = Omega_m + Omega_Lambda + Omega_r
        if abs(total_omega - 1.0) > 0.02:
            raise ValueError(f"平坦性破坏: Ω_total = {total_omega:.4f} ≠ 1")
        if h <= 0.0 or h > 2.0:
            raise ValueError(f"不合理的 h = {h}")
        if Omega_m <= 0.0 or Omega_m > 1.5:
            raise ValueError(f"不合理的 Ω_m = {Omega_m}")

    def H(self, a: float) -> float:
        if a <= 0.0:
            raise ValueError("尺度因子 a 必须为正")
        a2 = a * a
        a3 = a2 * a
        a4 = a3 * a
        term = self.Omega_m / a3 + self.Omega_r / a4 + self.Omega_Lambda
        return self.H0 * np.sqrt(term)

    def dH_da(self, a: float) -> float:
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
        ha = self.H(a)
        if ha == 0.0:
            return 0.0
        return a * self.dH_da(a) / ha

    def Omega_m_a(self, a: float) -> float:
        e2 = (
            self.Omega_m / a ** 3
            + self.Omega_r / a ** 4
            + self.Omega_Lambda
        )
        return (self.Omega_m / a ** 3) / e2

    def scale_factor_evolution_rhs(self, t: float, y: np.ndarray) -> np.ndarray:

        raise NotImplementedError("请实现 scale_factor_evolution_rhs 方法")

    def linear_growth_rhs(self, a: float, y: np.ndarray) -> np.ndarray:
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

        conv = 1.02271e-3
        y0 = np.array([1e-8])

        def rhs_local(t: float, y: np.ndarray) -> np.ndarray:
            a = y[0]
            if a <= 0.0:
                a = 1e-10
            return np.array([a * self.H(a) * conv])

        t, y, e = self.rk12_integrate(rhs_local, t_Gyr_span, y0, n_steps)
        a = y[:, 0]

        a = np.clip(a, 1e-10, None)
        return t, a, e[:, 0]

    def compute_linear_growth_factor(
        self, a_min: float = 1e-4, a_max: float = 1.0, n_steps: int = 2000
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        y0 = np.array([a_min, 1.0])
        a_arr, y, e = self.rk12_integrate(
            self.linear_growth_rhs, (a_min, a_max), y0, n_steps
        )
        D = y[:, 0]
        dD_da = y[:, 1]

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
        if a_target <= 0.0 or a_target > 2.0:
            raise ValueError("a_target 必须在 (0, 2] 范围内")


        conv = 1.02271e-3
        n_steps = 2000

        u_min = np.log(1e-10)
        u_max = np.log(a_target)
        du = (u_max - u_min) / n_steps
        us = np.linspace(u_min, u_max, n_steps + 1)
        a_vals = np.exp(us)
        H_vals = np.array([self.H(a) for a in a_vals]) * conv
        H_vals = np.clip(H_vals, 1e-30, None)
        integrand = 1.0 / H_vals


        S = integrand[0] + integrand[-1]
        S += 4.0 * np.sum(integrand[1:-1:2])
        S += 2.0 * np.sum(integrand[2:-1:2])
        t_age = S * du / 3.0
        return t_age

    def delta_c(self, z: float) -> float:
        a = 1.0 / (1.0 + z)
        a_arr, D_arr, _ = self.compute_linear_growth_factor(
            a_min=1e-4, a_max=1.0, n_steps=2000
        )

        D = np.interp(a, a_arr, D_arr)
        if D <= 0.0:
            D = 1e-10
        omega_m_z = self.Omega_m_a(a)
        return 1.686 * (1.0 + 0.0123 * np.log10(omega_m_z)) / D

    def comoving_distance(self, z: float, n_int: int = 1000) -> float:
        if z < 0.0:
            raise ValueError("红移 z 不能为负")
        if z == 0.0:
            return 0.0
        zs = np.linspace(0.0, z, n_int + 1)

        zp1 = 1.0 + zs
        E = np.sqrt(
            self.Omega_m * zp1 ** 3
            + self.Omega_r * zp1 ** 4
            + self.Omega_Lambda
        )
        integrand = self.c / (self.H0 * E)

        h_step = z / n_int
        S = integrand[0] + integrand[-1]
        S += 4.0 * np.sum(integrand[1:-1:2])
        S += 2.0 * np.sum(integrand[2:-1:2])
        return S * h_step / 3.0


if __name__ == "__main__":

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
