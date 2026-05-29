"""
thm_model.py
============
Core governing equations for geothermal reservoir Thermal-Hydro-Mechanical (THM)
coupled simulation.

Scientific Domain: Energy Systems — Geothermal Reservoir Thermal-Hydro-Mechanical Coupling

Governing Equations:
--------------------
1. Fluid Mass Conservation (continuity):
   \partial(\phi \rho_f)/\partial t + \nabla\cdot(\rho_f \mathbf{q}) = Q_m

   where \phi is porosity, \rho_f is fluid density, \mathbf{q} is Darcy flux,
   and Q_m is mass source term.

2. Darcy's Law (momentum balance for fluid in porous medium):
   \mathbf{q} = -\frac{\mathbf{k}}{\mu} \left( \nabla p + \rho_f g \nabla z \right)

   where \mathbf{k} is permeability tensor (m^2), \mu is dynamic viscosity (Pa·s),
   p is pore pressure (Pa), g is gravitational acceleration (m/s^2).

3. Energy Conservation (heat transport):
   (\rho c)_{\text{eff}} \frac{\partial T}{\partial t}
   + \rho_f c_f \mathbf{q}\cdot\nabla T
   = \nabla\cdot(\lambda_{\text{eff}} \nabla T) + Q_T

   where (\rho c)_{\text{eff}} = \phi \rho_f c_f + (1-\phi) \rho_r c_r
   is the effective heat capacity,
   \lambda_{\text{eff}} = \phi \lambda_f + (1-\phi) \lambda_r
   is the effective thermal conductivity,
   and Q_T is heat source (W/m^3).

4. Poroelastic Mechanical Equilibrium:
   \nabla\cdot\boldsymbol{\sigma} + \rho_b \mathbf{g} = \mathbf{0}

   with total stress:
   \boldsymbol{\sigma} = \mathbf{C} : \boldsymbol{\varepsilon}
   - \alpha p \mathbf{I} - \beta K_T (T - T_0) \mathbf{I}

   where \alpha = 1 - K_T/K_s is the Biot coefficient,
   \beta is the thermal expansion coefficient of the solid matrix,
   K_T is the drained bulk modulus, K_s is the solid grain bulk modulus,
   \mathbf{C} is the elasticity tensor, and \boldsymbol{\varepsilon}
   is the strain tensor.

5. Constitutive Relation (strain-displacement):
   \varepsilon_{ij} = \frac{1}{2}\left(\frac{\partial u_i}{\partial x_j}
   + \frac{\partial u_j}{\partial x_i}\right)

The fully coupled THM system requires simultaneous solution of:
  - pressure field p(\mathbf{x}, t)
  - temperature field T(\mathbf{x}, t)
  - displacement field \mathbf{u}(\mathbf{x}, t)
"""

import numpy as np


class THMParameters:
    """
    Physical and numerical parameters for the THM geothermal model.
    All units are SI (m, kg, s, K, Pa).
    """

    def __init__(self):
        # --- Reservoir geometry ---
        self.reservoir_length = 500.0       # m (x-direction)
        self.reservoir_height = 200.0       # m (z-direction)
        self.reservoir_width  = 100.0       # m (y-direction)

        # --- Grid discretization ---
        self.nx = 64
        self.nz = 32
        self.ny = 16
        self.dx = self.reservoir_length / self.nx
        self.dz = self.reservoir_height / self.nz
        self.dy = self.reservoir_width / self.ny

        # --- Rock properties ---
        self.porosity = 0.15
        self.matrix_permeability = 1.0e-14  # m^2 (10 millidarcies)
        self.density_rock = 2700.0          # kg/m^3 (granite)
        self.heat_capacity_rock = 850.0     # J/(kg·K)
        self.thermal_conductivity_rock = 2.5 # W/(m·K)
        self.young_modulus = 30.0e9         # Pa
        self.poisson_ratio = 0.25
        self.biot_coefficient = 0.8
        self.thermal_expansion_rock = 1.0e-5 # 1/K

        # --- Fluid properties (reference values) ---
        self.density_fluid_ref = 1000.0     # kg/m^3
        self.viscosity_fluid_ref = 1.0e-3   # Pa·s
        self.heat_capacity_fluid = 4180.0   # J/(kg·K)
        self.thermal_conductivity_fluid = 0.6 # W/(m·K)
        self.compressibility_fluid = 4.5e-10 # 1/Pa

        # --- Initial conditions ---
        self.T_initial = 423.15             # K (150°C)
        self.T_injection = 323.15           # K (50°C)
        self.p_initial = 20.0e6             # Pa (20 MPa)
        self.p_production = 15.0e6          # Pa (15 MPa)

        # --- Boundary conditions ---
        self.injection_rate = 0.05          # kg/s per unit area
        self.production_pressure = 15.0e6   # Pa

        # --- Thermal source ---
        self.heat_source_magnitude = 5.0e3  # W/m^3

        # --- Numerical ---
        self.dt = 86400.0 * 30.0            # 30 days in seconds
        self.num_time_steps = 12            # 12 months

        # --- Precomputed effective properties ---
        self._compute_effective_properties()

    def _compute_effective_properties(self):
        """Compute effective thermal and mechanical properties."""
        phi = self.porosity
        self.rho_eff = phi * self.density_fluid_ref + (1.0 - phi) * self.density_rock
        self.cp_eff = (phi * self.density_fluid_ref * self.heat_capacity_fluid
                       + (1.0 - phi) * self.density_rock * self.heat_capacity_rock) / self.rho_eff
        self.lambda_eff = phi * self.thermal_conductivity_fluid + (1.0 - phi) * self.thermal_conductivity_rock
        self.drained_bulk_modulus = (self.young_modulus
                                      / (3.0 * (1.0 - 2.0 * self.poisson_ratio)))
        self.solid_bulk_modulus = 50.0e9    # Pa (assumed for granite)
        self.biot_modulus = 1.0 / (self.porosity / self.fluid_bulk_modulus()
                                   + (self.biot_coefficient - self.porosity)
                                   / self.solid_bulk_modulus)

    def fluid_bulk_modulus(self):
        """Fluid bulk modulus K_f = 1/\beta_f."""
        return 1.0 / self.compressibility_fluid

    def lame_lambda(self):
        """Lamé first parameter:
        \lambda = \frac{E \nu}{(1+\nu)(1-2\nu)}
        """
        E = self.young_modulus
        nu = self.poisson_ratio
        return E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

    def lame_mu(self):
        """Lamé second parameter (shear modulus):
        \mu = \frac{E}{2(1+\nu)}
        """
        return self.young_modulus / (2.0 * (1.0 + self.poisson_ratio))

    def grid_shape(self):
        return (self.nx, self.nz, self.ny)

    def grid_size(self):
        return self.dx, self.dz, self.dy


class THMState:
    """
    Container for the primary field variables at a given time step:
      - pressure p(x,y,z)
      - temperature T(x,y,z)
      - displacement vector u(x,y,z) = (u_x, u_y, u_z)
    """

    def __init__(self, params: THMParameters):
        shape = params.grid_shape()
        self.p = np.full(shape, params.p_initial, dtype=np.float64)
        self.T = np.full(shape, params.T_initial, dtype=np.float64)
        self.u_x = np.zeros(shape, dtype=np.float64)
        self.u_y = np.zeros(shape, dtype=np.float64)
        self.u_z = np.zeros(shape, dtype=np.float64)
        self.time = 0.0

    def copy(self):
        """Return a deep copy of the state."""
        import copy
        return copy.deepcopy(self)


def darcy_velocity_scalar(k, mu, dp_dx, rho_f, g, dz_sign=1.0):
    """
    Scalar Darcy velocity:
    q = -(k / mu) * (dp/dx + rho_f * g * dz_sign)

    Parameters
    ----------
    k : float or np.ndarray
        Permeability (m^2).
    mu : float or np.ndarray
        Dynamic viscosity (Pa·s).
    dp_dx : float or np.ndarray
        Pressure gradient (Pa/m).
    rho_f : float or np.ndarray
        Fluid density (kg/m^3).
    g : float
        Gravitational acceleration (m/s^2).
    dz_sign : float
        +1 for upward gradient, -1 for downward.

    Returns
    -------
    q : float or np.ndarray
        Darcy flux (m/s).
    """
    if np.any(mu <= 0.0):
        raise ValueError("Viscosity must be strictly positive.")
    if np.any(k < 0.0):
        raise ValueError("Permeability cannot be negative.")
    q = -(k / mu) * (dp_dx + rho_f * g * dz_sign)
    return q


def effective_heat_capacity(phi, rho_f, cp_f, rho_r, cp_r):
    """
    (\rho c)_{\text{eff}} = \phi \rho_f c_f + (1-\phi) \rho_r c_r
    """
    phi = np.asarray(phi)
    if np.any(phi < 0.0) or np.any(phi > 1.0):
        raise ValueError("Porosity must lie in [0, 1].")
    return phi * rho_f * cp_f + (1.0 - phi) * rho_r * cp_r


def effective_thermal_conductivity(phi, lam_f, lam_r):
    """
    \lambda_{\text{eff}} = \phi \lambda_f + (1-\phi) \lambda_r
    """
    phi = np.asarray(phi)
    if np.any(phi < 0.0) or np.any(phi > 1.0):
        raise ValueError("Porosity must lie in [0, 1].")
    return phi * lam_f + (1.0 - phi) * lam_r


def biot_modulus(phi, K_f, alpha, K_s):
    """
    Biot modulus M:
    1/M = \phi/K_f + (\alpha - \phi)/K_s
    """
    if K_f <= 0.0 or K_s <= 0.0:
        raise ValueError("Bulk moduli must be positive.")
    return 1.0 / (phi / K_f + (alpha - phi) / K_s)


def thermal_diffusivity(lambda_eff, rho_eff, cp_eff):
    """
    Thermal diffusivity:
    \kappa_T = \lambda_{\text{eff}} / (\rho_{\text{eff}} c_{\text{eff}})
    """
    # TODO: Implement thermal diffusivity formula.
    # Hint: \kappa = \lambda_{\text{eff}} / (\rho_{\text{eff}} c_{\text{eff}})
    # Ensure physical validity (denominator must be positive).
    raise NotImplementedError("thermal_diffusivity is missing — scientific knowledge required.")


def strain_tensor_2d(dux_dx, duz_dz, dux_dz, duz_dx):
    """
    2D infinitesimal strain tensor components:
    \varepsilon_{xx} = \partial u_x / \partial x
    \varepsilon_{zz} = \partial u_z / \partial z
    \varepsilon_{xz} = \varepsilon_{zx} = 0.5 (\partial u_x / \partial z + \partial u_z / \partial x)
    """
    exx = dux_dx
    ezz = duz_dz
    exz = 0.5 * (dux_dz + duz_dx)
    return exx, ezz, exz


def poroelastic_stress_2d(exx, ezz, exz, p, T, T0, params: THMParameters):
    """
    2D plane-strain poroelastic stress:
    \sigma_{xx} = (\lambda + 2\mu) \varepsilon_{xx} + \lambda \varepsilon_{zz}
                  - \alpha p - \beta K_T (T - T_0)
    \sigma_{zz} = \lambda \varepsilon_{xx} + (\lambda + 2\mu) \varepsilon_{zz}
                  - \alpha p - \beta K_T (T - T_0)
    \sigma_{xz} = 2\mu \varepsilon_{xz}
    """
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
    """
    Temperature- and pressure-dependent fluid density using a linearized
    equation of state:
    \rho_f(p,T) = \rho_{f0} \left[1 + \beta_p (p - p_0)
                                  - \beta_T (T - T_0)\right]
    where \beta_p is compressibility and \beta_T is thermal expansion.
    """
    beta_p = params.compressibility_fluid
    beta_T = 2.1e-4  # 1/K thermal expansion for water
    rho0 = params.density_fluid_ref
    p0 = params.p_initial
    T0 = params.T_initial
    rho = rho0 * (1.0 + beta_p * (p - p0) - beta_T * (T - T0))
    rho = np.clip(rho, 500.0, 1500.0)  # physical bounds for water
    return rho


def fluid_viscosity_temperature(T, params=None):
    """
    Water viscosity as function of temperature (Poiseuille-type fit):
    \mu(T) = A \exp\left(\frac{B}{T - C}\right)
    with A=2.414e-5 Pa·s, B=247.8 K, C=140 K.
    Valid for 273 K < T < 623 K.
    """
    T = np.asarray(T)
    if np.any(T <= 140.0):
        raise ValueError("Temperature below singular point in viscosity formula.")
    A = 2.414e-5
    B = 247.8
    C = 140.0
    mu = A * np.exp(B / (T - C))
    mu = np.clip(mu, 1.0e-4, 5.0e-3)
    return mu
