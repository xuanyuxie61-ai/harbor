
import numpy as np


HBAR_C = 197.3269804
MASS_NUCLEON = 939.5654133
ELEM_CHARGE2 = 1.43996448


class OpticalPotentialParameters:

    def __init__(self, projectile='n', target_A=56, target_Z=26, E_lab=14.0):
        self.projectile = projectile
        self.target_A = target_A
        self.target_Z = target_Z
        self.E_lab = float(E_lab)


        if projectile == 'n':
            self.proj_mass = 1.008665
            self.proj_Z = 0
        elif projectile == 'p':
            self.proj_mass = 1.007276
            self.proj_Z = 1
        elif projectile == 'alpha':
            self.proj_mass = 4.002603
            self.proj_Z = 2
        else:
            raise ValueError(f"不支持的入射粒子类型: {projectile}")


        self.reduced_mass = (self.proj_mass * target_A) / (self.proj_mass + target_A)

        self.mu_MeV = self.reduced_mass * MASS_NUCLEON



        self.k = np.sqrt(2.0 * self.mu_MeV * self.E_lab) / HBAR_C


        self.r0 = 1.25
        self.r0_so = 1.25
        self.rC = 1.25
        self.R_v = self.r0 * (target_A ** (1.0 / 3.0))
        self.R_w = self.r0 * (target_A ** (1.0 / 3.0))
        self.R_d = self.r0 * (target_A ** (1.0 / 3.0))
        self.R_so = self.r0_so * (target_A ** (1.0 / 3.0))
        self.R_C = self.rC * (target_A ** (1.0 / 3.0))


        self.a_v = 0.65
        self.a_w = 0.65
        self.a_d = 0.47
        self.a_so = 0.65




        self.V0 = 51.5 - 0.3 * self.E_lab
        self.W0 = 2.5 + 0.15 * self.E_lab
        self.WD = 6.0 - 0.05 * self.E_lab
        self.Vso0 = 6.2


        self._validate_parameters()

    def _validate_parameters(self):
        assert self.target_A > 0, "靶核质量数必须为正"
        assert self.target_Z >= 0, "靶核电荷数必须非负"
        assert self.E_lab > 0.0, "入射能量必须为正"
        assert self.a_v > 0.05, "弥散参数过小会导致数值不稳定"
        assert self.a_w > 0.05
        assert self.a_d > 0.05
        assert self.a_so > 0.05
        eps = 1e-12
        if abs(self.V0) < eps:
            self.V0 = eps
        if abs(self.W0) < eps:
            self.W0 = eps
        if abs(self.WD) < eps:
            self.WD = eps

    def __repr__(self):
        return (
            f"OpticalPotentialParameters("
            f"{self.projectile}+{self.target_A}{self._element_symbol()}, "
            f"E_lab={self.E_lab:.2f} MeV, k={self.k:.4f} fm^-1)"
        )

    def _element_symbol(self):
        symbols = {
            1: 'H', 2: 'He', 6: 'C', 8: 'O', 13: 'Al', 20: 'Ca',
            26: 'Fe', 28: 'Ni', 50: 'Sn', 82: 'Pb', 92: 'U'
        }
        return symbols.get(self.target_Z, f"Z{self.target_Z}")


def woods_saxon(r, V0, R, a):
    r = np.asarray(r, dtype=float)

    arg = (r - R) / a


    f = np.empty_like(arg)

    mask_pos = arg > 700
    mask_neg = arg < -700
    mask_mid = ~mask_pos & ~mask_neg
    f[mask_pos] = 0.0
    f[mask_neg] = 1.0
    f[mask_mid] = 1.0 / (1.0 + np.exp(arg[mask_mid]))
    return f


def woods_saxon_derivative(r, V0, R, a):
    r = np.asarray(r, dtype=float)
    arg = (r - R) / a
    g = np.empty_like(arg)
    mask_pos = arg > 700
    mask_neg = arg < -700
    mask_mid = ~mask_pos & ~mask_neg
    g[mask_pos] = 0.0
    g[mask_neg] = 0.0
    e = np.exp(arg[mask_mid])
    g[mask_mid] = 4.0 * e / (1.0 + e) ** 2
    return g


def thomas_spin_orbit_factor(r, R_so, a_so):
    r = np.asarray(r, dtype=float)
    eps = 1e-15

    r_safe = np.where(np.abs(r) < eps, eps, r)
    arg = (r_safe - R_so) / a_so
    f = np.empty_like(arg)
    mask_pos = arg > 700
    mask_neg = arg < -700
    mask_mid = ~mask_pos & ~mask_neg
    f[mask_pos] = 0.0
    f[mask_neg] = 0.0
    e = np.exp(arg[mask_mid])

    f[mask_mid] = -(1.0 / (r_safe[mask_mid] * a_so)) * e / (1.0 + e) ** 2
    return f


def coulomb_potential(r, Zp, Zt, RC):
    r = np.asarray(r, dtype=float)
    VC = np.zeros_like(r)
    if Zp == 0 or Zt == 0:
        return VC
    prefactor = Zp * Zt * ELEM_CHARGE2
    mask_in = r <= RC
    mask_out = r > RC
    VC[mask_in] = prefactor / (2.0 * RC) * (3.0 - (r[mask_in] / RC) ** 2)

    r_out = r[mask_out]
    VC[mask_out] = prefactor / np.where(r_out > 0, r_out, 1e-15)
    return VC


def build_optical_potential(r, params, l=0, j=None):
    r = np.asarray(r, dtype=float)
    if np.any(r < 0):
        raise ValueError("径向坐标 r 必须非负")


    V = -params.V0 * woods_saxon(r, params.V0, params.R_v, params.a_v)


    Wv = -params.W0 * woods_saxon(r, params.W0, params.R_w, params.a_w)


    Wd = -params.WD * woods_saxon_derivative(r, params.WD, params.R_d, params.a_d)


    Vso = np.zeros_like(r)
    if j is not None and l > 0:



        lambda_pi = HBAR_C / 138.0
        ls_coupling = 0.5 * (j * (j + 1.0) - l * (l + 1.0) - 0.5 * 1.5)
        Vso = params.Vso0 * (lambda_pi ** 2) * thomas_spin_orbit_factor(r, params.R_so, params.a_so) * ls_coupling


    VC = coulomb_potential(r, params.proj_Z, params.target_Z, params.R_C)


    U = V + 1j * (Wv + Wd) + Vso + VC
    return U


def effective_potential(r, params, l=0, j=None):
    r = np.asarray(r, dtype=float)
    U = build_optical_potential(r, params, l, j)


    return U


if __name__ == "__main__":

    params = OpticalPotentialParameters('n', 56, 26, 14.0)
    print(params)
    r = np.linspace(0.01, 15.0, 200)
    U = build_optical_potential(r, params, l=2, j=2.5)
    print("势场实部范围:", np.real(U).min(), np.real(U).max())
    print("势场虚部范围:", np.imag(U).min(), np.imag(U).max())
