
import numpy as np


class THMParameters:

    def __init__(self):

        self.reservoir_length = 500.0
        self.reservoir_height = 200.0
        self.reservoir_width  = 100.0


        self.nx = 64
        self.nz = 32
        self.ny = 16
        self.dx = self.reservoir_length / self.nx
        self.dz = self.reservoir_height / self.nz
        self.dy = self.reservoir_width / self.ny


        self.porosity = 0.15
        self.matrix_permeability = 1.0e-14
        self.density_rock = 2700.0
        self.heat_capacity_rock = 850.0
        self.thermal_conductivity_rock = 2.5
        self.young_modulus = 30.0e9
        self.poisson_ratio = 0.25
        self.biot_coefficient = 0.8
        self.thermal_expansion_rock = 1.0e-5


        self.density_fluid_ref = 1000.0
        self.viscosity_fluid_ref = 1.0e-3
        self.heat_capacity_fluid = 4180.0
        self.thermal_conductivity_fluid = 0.6
        self.compressibility_fluid = 4.5e-10


        self.T_initial = 423.15
        self.T_injection = 323.15
        self.p_initial = 20.0e6
        self.p_production = 15.0e6


        self.injection_rate = 0.05
        self.production_pressure = 15.0e6


        self.heat_source_magnitude = 5.0e3


        self.dt = 86400.0 * 30.0
        self.num_time_steps = 12


        self._compute_effective_properties()

    def _compute_effective_properties(self):
        phi = self.porosity
        self.rho_eff = phi * self.density_fluid_ref + (1.0 - phi) * self.density_rock
        self.cp_eff = (phi * self.density_fluid_ref * self.heat_capacity_fluid
                       + (1.0 - phi) * self.density_rock * self.heat_capacity_rock) / self.rho_eff
        self.lambda_eff = phi * self.thermal_conductivity_fluid + (1.0 - phi) * self.thermal_conductivity_rock
        self.drained_bulk_modulus = (self.young_modulus
                                      / (3.0 * (1.0 - 2.0 * self.poisson_ratio)))
        self.solid_bulk_modulus = 50.0e9
        self.biot_modulus = 1.0 / (self.porosity / self.fluid_bulk_modulus()
                                   + (self.biot_coefficient - self.porosity)
                                   / self.solid_bulk_modulus)

    def fluid_bulk_modulus(self):
        return 1.0 / self.compressibility_fluid

    def lame_lambda(self):
        E = self.young_modulus
        nu = self.poisson_ratio
        return E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

    def lame_mu(self):
        return self.young_modulus / (2.0 * (1.0 + self.poisson_ratio))

    def grid_shape(self):
        return (self.nx, self.nz, self.ny)

    def grid_size(self):
        return self.dx, self.dz, self.dy


class THMState:

    def __init__(self, params: THMParameters):
        shape = params.grid_shape()
        self.p = np.full(shape, params.p_initial, dtype=np.float64)
        self.T = np.full(shape, params.T_initial, dtype=np.float64)
        self.u_x = np.zeros(shape, dtype=np.float64)
        self.u_y = np.zeros(shape, dtype=np.float64)
        self.u_z = np.zeros(shape, dtype=np.float64)
        self.time = 0.0

    def copy(self):
        import copy
        return copy.deepcopy(self)


def darcy_velocity_scalar(k, mu, dp_dx, rho_f, g, dz_sign=1.0):
    if np.any(mu <= 0.0):
        raise ValueError("Viscosity must be strictly positive.")
    if np.any(k < 0.0):
        raise ValueError("Permeability cannot be negative.")
    q = -(k / mu) * (dp_dx + rho_f * g * dz_sign)
    return q


def effective_heat_capacity(phi, rho_f, cp_f, rho_r, cp_r):
    phi = np.asarray(phi)
    if np.any(phi < 0.0) or np.any(phi > 1.0):
        raise ValueError("Porosity must lie in [0, 1].")
    return phi * rho_f * cp_f + (1.0 - phi) * rho_r * cp_r


def effective_thermal_conductivity(phi, lam_f, lam_r):
    phi = np.asarray(phi)
    if np.any(phi < 0.0) or np.any(phi > 1.0):
        raise ValueError("Porosity must lie in [0, 1].")
    return phi * lam_f + (1.0 - phi) * lam_r


def biot_modulus(phi, K_f, alpha, K_s):
    if K_f <= 0.0 or K_s <= 0.0:
        raise ValueError("Bulk moduli must be positive.")
    return 1.0 / (phi / K_f + (alpha - phi) / K_s)


def thermal_diffusivity(lambda_eff, rho_eff, cp_eff):



    raise NotImplementedError("thermal_diffusivity is missing — scientific knowledge required.")


def strain_tensor_2d(dux_dx, duz_dz, dux_dz, duz_dx):
    exx = dux_dx
    ezz = duz_dz
    exz = 0.5 * (dux_dz + duz_dx)
    return exx, ezz, exz


def poroelastic_stress_2d(exx, ezz, exz, p, T, T0, params: THMParameters):
    lam = params.lame_lambda()
    mu = params.lame_mu()
    alpha = params.biot_coefficient
    beta = params.thermal_expansion_rock
    KT = params.drained_bulk_modulus
    delta_T = T - T0
    sxx = (lam + 2.0 * mu) * exx + lam * ezz - alpha * p - beta * KT * delta_T
    szz = lam * exx + (lam + 2.0 * mu) * ezz - alpha * p - beta * KT * delta_T
    sxz = 2.0 * mu * exz
    return sxx, szz, sxz


def fluid_density_temperature(p, T, params: THMParameters):
    beta_p = params.compressibility_fluid
    beta_T = 2.1e-4
    rho0 = params.density_fluid_ref
    p0 = params.p_initial
    T0 = params.T_initial
    rho = rho0 * (1.0 + beta_p * (p - p0) - beta_T * (T - T0))
    rho = np.clip(rho, 500.0, 1500.0)
    return rho


def fluid_viscosity_temperature(T, params=None):
    T = np.asarray(T)
    if np.any(T <= 140.0):
        raise ValueError("Temperature below singular point in viscosity formula.")
    A = 2.414e-5
    B = 247.8
    C = 140.0
    mu = A * np.exp(B / (T - C))
    mu = np.clip(mu, 1.0e-4, 5.0e-3)
    return mu
