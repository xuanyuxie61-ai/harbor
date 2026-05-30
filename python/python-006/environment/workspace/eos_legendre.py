
import numpy as np
import math
from utils_physics import safe_sqrt, safe_divide, fermi_momentum_to_density





def associated_legendre_polynomial_value(mm: int, n: int, m: int, x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim != 1:
        x = x.reshape(-1)
    mm = x.size


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
    L_max = len(coeffs) - 1
    mm = cos_theta.size
    P_vals = associated_legendre_polynomial_value(mm, L_max, 0, cos_theta)
    V = np.zeros_like(cos_theta)
    for l in range(L_max + 1):
        V += coeffs[l] * P_vals[:, l]
    return V





class SkyrmeEOS:

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
        self.hbar2_over_2m = 20.73553

    def energy_density(self, rho_b: float, delta: float = 0.0) -> float:
        if rho_b < 0.0:
            raise ValueError("Baryon density rho_b must be non-negative.")
        if abs(delta) > 1.0 + 1e-12:
            raise ValueError("Isospin asymmetry delta must be in [-1, 1].")
        delta = np.clip(delta, -1.0, 1.0)

        kf = (3.0 * math.pi**2 * rho_b)**(1.0 / 3.0)


        kin_term = 0.0
        if rho_b > 1e-15:
            kin_term = (3.0 / 5.0) * self.hbar2_over_2m * kf**2 * 0.5 * (
                (1.0 + delta)**(5.0 / 3.0) + (1.0 - delta)**(5.0 / 3.0)
            )


        pot_term = (3.0 / 8.0) * self.t0 * rho_b**2 * (
            (1.0 + self.x0 / 2.0) - (self.x0 + 0.5) * delta**2
        )
        pot_term += (1.0 / 16.0) * self.t3 * rho_b**(self.alpha + 2.0) * (
            (1.0 + self.x3 / 2.0) - (self.x3 + 0.5) * delta**2
        )

        return kin_term + pot_term

    def pressure(self, rho_b: float, delta: float = 0.0) -> float:
        if rho_b < 1e-15:
            return 0.0
        if abs(delta) > 1.0 + 1e-12:
            raise ValueError("Isospin asymmetry delta must be in [-1, 1].")
        delta = np.clip(delta, -1.0, 1.0)

        kf = (3.0 * math.pi**2 * rho_b)**(1.0 / 3.0)


        p_kin = (2.0 / 5.0) * self.hbar2_over_2m * kf**2 * rho_b * 0.5 * (
            (1.0 + delta)**(5.0 / 3.0) + (1.0 - delta)**(5.0 / 3.0)
        )




        raise NotImplementedError("Hole 1: 请补全 Skyrme EOS 的势能压强计算公式")

        return p_kin + p_pot

    def chemical_potential(self, rho_b: float, delta: float = 0.0) -> tuple:
        d = 1e-6
        eps_p = self.energy_density(rho_b + d, delta)
        eps_m = self.energy_density(rho_b - d, delta)
        deps_drho = (eps_p - eps_m) / (2.0 * d)

        mu_avg = deps_drho

        E_sym = self.symmetry_energy(rho_b)
        mu_n = mu_avg + E_sym * delta
        mu_p = mu_avg - E_sym * delta
        return mu_n, mu_p

    def symmetry_energy(self, rho_b: float) -> float:
        d = 1e-4
        e0 = self.energy_density(rho_b, 0.0)
        ep = self.energy_density(rho_b, d)
        em = self.energy_density(rho_b, -d)
        second_deriv = (ep - 2.0 * e0 + em) / (d**2)
        return 0.5 * second_deriv / rho_b if rho_b > 1e-15 else 0.0

    def sound_speed_squared(self, rho_b: float, delta: float = 0.0) -> float:
        dr = 1e-5 * rho_b if rho_b > 1e-15 else 1e-8
        p_plus = self.pressure(rho_b + dr, delta)
        p_minus = self.pressure(rho_b - dr, delta)
        e_plus = self.energy_density(rho_b + dr, delta)
        e_minus = self.energy_density(rho_b - dr, delta)

        dp_dr = (p_plus - p_minus) / (2.0 * dr)
        de_dr = (e_plus - e_minus) / (2.0 * dr)

        cs2 = safe_divide(dp_dr, de_dr, default=0.0)

        return min(max(cs2, 0.0), 1.0)


class PolytropicEOS:

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
        if eps < 0.0:
            raise ValueError("Energy density must be non-negative.")

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
    skyrme = SkyrmeEOS()

    poly1 = PolytropicEOS(K=0.05, Gamma=2.5)

    def eos_func(eps: float) -> float:
        if eps < 0.0:
            raise ValueError("Energy density must be non-negative.")

        if eps < 300.0:

            rho = eps / 150.0 * 0.16
            rho = max(rho, 1e-10)
            return skyrme.pressure(rho)
        else:
            return poly1.pressure_from_energy_density(eps)

    return eos_func
