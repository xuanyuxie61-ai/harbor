"""
mhd_operators.py
================
Discrete differential operators for the resistive MHD equations on a
2D Cartesian grid (x, z) representing a magnetic reconnection layer.

The resistive MHD equations in dimensionless form:
    drho/dt + div(rho v) = 0                              (continuity)
    d(rho v)/dt + div(rho v v - B B / mu0) = -grad(p)     (momentum)
    dB/dt = curl(v x B) - curl(eta curl B)                (induction)
    dE/dt + div((E + p) v - (v.B) B / mu0) = div(eta J x B) (energy)

where J = curl(B) / mu0, and E = p/(gamma-1) + rho v^2/2 + B^2/(2 mu0).

In the 2D incompressible/reduced MHD limit (reconnection plane x-z):
    dB_z/dt = -d/dx(E_y)        where E_y = eta J_y - (v x B)_y
    d psi/dt = -E_y             (in terms of flux function psi)

Maps seed projects:
  - 127_burgers_time_viscous: time integration, conservation form,
    flux evaluation f(u) = 0.5 u^2, viscous diffusion nu d^2u/dx^2
  - 875_poisson_1d: Gauss-Seidel relaxation for the Poisson solve
  - 990_r8poly: polynomial flux reconstruction
"""

import numpy as np


# ============================================================
# Grid Setup
# ============================================================

class ReconnectionGrid:
    """
    2D Cartesian grid for the reconnection layer (x, z).
    x: outflow direction
    z: inflow direction (reconnection direction)

    The current sheet is initially aligned along x at z=0.
    """

    def __init__(self, nx, nz, Lx, Lz, bc_type='periodic_x_dirichlet_z'):
        self.nx = nx
        self.nz = nz
        self.Lx = Lx
        self.Lz = Lz
        self.dx = Lx / max(nx, 1)
        self.dz = Lz / max(nz, 1)
        self.bc_type = bc_type

        # Cell-centered coordinates
        self.x = np.linspace(0.5 * self.dx, Lx - 0.5 * self.dx, nx)
        self.z = np.linspace(0.5 * self.dz, Lz - 0.5 * self.dz, nz)
        self.X, self.Z = np.meshgrid(self.x, self.z, indexing='ij')

        # Area element
        self.dA = self.dx * self.dz
        # Total domain
        self.volume = Lx * Lz

    def __repr__(self):
        return (f"ReconnectionGrid(nx={self.nx}, nz={self.nz}, "
                f"Lx={self.Lx:.2e}, Lz={self.Lz:.2e})")


# ============================================================
# Differential Operators (High-Order Central Differences)
# ============================================================

def d_dx_4th(f, dx):
    """
    4th-order central difference for df/dx with periodic BC in x.
    Uses 5-point stencil:
        f'_i = (-f_{i+2} + 8 f_{i+1} - 8 f_{i-1} + f_{i-2}) / (12 dx)
    """
    nx, nz = f.shape
    df = np.zeros_like(f)
    # Interior (4th-order central)
    df[2:-2, :] = (-f[4:, :] + 8.0 * f[3:-1, :]
                    - 8.0 * f[1:-3, :] + f[:-4, :]) / (12.0 * dx)
    # Periodic boundaries in x
    for i in range(2):
        df[i, :] = (-np.roll(f, -2, axis=0)[i, :]
                     + 8.0 * np.roll(f, -1, axis=0)[i, :]
                     - 8.0 * np.roll(f, 1, axis=0)[i, :]
                     + np.roll(f, 2, axis=0)[i, :]) / (12.0 * dx)
    return df


def d_dz_4th(f, dz, bc_z='dirichlet'):
    """
    4th-order central difference for df/dz.
    Near z-boundaries, fall back to one-sided 4th-order stencils
    for Dirichlet BCs, or periodic for periodic BCs.
    """
    nx, nz = f.shape
    df = np.zeros_like(f)
    # Interior
    if nz > 4:
        df[:, 2:-2] = (-f[:, 4:] + 8.0 * f[:, 3:-1]
                       - 8.0 * f[:, 1:-3] + f[:, :-4]) / (12.0 * dz)
    # Boundaries: 2nd-order central (fallback for stability)
    if nz > 2:
        df[:, 0] = (f[:, 1] - f[:, 0]) / dz
        df[:, -1] = (f[:, -1] - f[:, -2]) / dz
        df[:, 1] = (f[:, 2] - f[:, 0]) / (2.0 * dz)
        df[:, -2] = (f[:, -1] - f[:, -3]) / (2.0 * dz)
    return df


def d2_dx2_4th(f, dx):
    """
    4th-order central difference for d^2f/dx^2.
    5-point stencil:
        f''_i = (-f_{i+2} + 16 f_{i+1} - 30 f_i + 16 f_{i-1} - f_{i-2})
                / (12 dx^2)
    """
    nx, nz = f.shape
    d2f = np.zeros_like(f)
    if nx > 4:
        d2f[2:-2, :] = (-f[4:, :] + 16.0 * f[3:-1, :]
                        - 30.0 * f[2:-2, :]
                        + 16.0 * f[1:-3, :] - f[:-4, :]) / (12.0 * dx ** 2)
    # Periodic fallback
    for i in range(2):
        d2f[i, :] = (-np.roll(f, -2, axis=0)[i, :]
                      + 16.0 * np.roll(f, -1, axis=0)[i, :]
                      - 30.0 * f[i, :]
                      + 16.0 * np.roll(f, 1, axis=0)[i, :]
                      - np.roll(f, 2, axis=0)[i, :]) / (12.0 * dx ** 2)
    return d2f


def d2_dz2_4th(f, dz):
    """
    4th-order central difference for d^2f/dz^2.
    """
    nx, nz = f.shape
    d2f = np.zeros_like(f)
    if nz > 4:
        d2f[:, 2:-2] = (-f[:, 4:] + 16.0 * f[:, 3:-1]
                        - 30.0 * f[:, 2:-2]
                        + 16.0 * f[:, 1:-3] - f[:, :-4]) / (12.0 * dz ** 2)
    # Boundary fallback (2nd order)
    if nz > 2:
        d2f[:, 0] = (f[:, 1] - 2.0 * f[:, 0] + f[:, 0]) / dz ** 2
        d2f[:, -1] = (f[:, -1] - 2.0 * f[:, -1] + f[:, -2]) / dz ** 2
    return d2f


# ============================================================
# Vector Calculus Operators (2D, x-z plane)
# ============================================================

def curl_2d(Bx, Bz, grid):
    """
    Compute the y-component of curl(B) in 2D:
        J_y = (curl B)_y = dBx/dz - dBz/dx

    In the reconnection context, J_y is the out-of-plane current sheet
    current. This is the primary driver of magnetic reconnection.
    """
    dBx_dz = d_dz_4th(Bx, grid.dz)
    dBz_dx = d_dx_4th(Bz, grid.dx)
    Jy = dBx_dz - dBz_dx
    return Jy


def div_2d(Ax, Az, grid):
    """
    Divergence of a 2D vector field:
        div(A) = dAx/dx + dAz/dz

    Physical constraint: div(B) = 0 must be maintained at all times.
    Used for divergence cleaning diagnostics.
    """
    dAx_dx = d_dx_4th(Ax, grid.dx)
    dAz_dz = d_dz_4th(Az, grid.dz)
    return dAx_dx + dAz_dz


def laplacian_2d(f, grid):
    """
    2D Laplacian operator (4th-order):
        nabla^2 f = d^2f/dx^2 + d^2f/dz^2

    Appears in the resistive diffusion term: eta nabla^2 B
    """
    return d2_dx2_4th(f, grid.dx) + d2_dz2_4th(f, grid.dz)


def grad_2d(f, grid):
    """
    2D gradient: (df/dx, df/dz)
    """
    return d_dx_4th(f, grid.dx), d_dz_4th(f, grid.dz)


# ============================================================
# Induction Equation Terms
# ============================================================

def induction_emf(vx, vz, Bx, Bz, eta, grid):
    """
    Compute the y-component of the electric field in the induction equation:

        E_y = -(v x B)_y + eta J_y
            = -(vx Bz - vz Bx) + eta (dBx/dz - dBz/dx)

    The first term is the ideal MHD convective EMF.
    The second term is the resistive diffusion EMF (Ohmic).

    In the generalized Ohm's law:
        E + v x B = eta J + (d_i / rho)(J x B - grad p_e) + ...

    Here we use the single-fluid resistive MHD approximation.

    Returns:
        Ey: shape (nx, nz)
        E_conv: convective part
        E_res: resistive part
    """
    Jy = curl_2d(Bx, Bz, grid)

    # Convective EMF: -(v x B)_y = vz Bx - vx Bz
    E_conv = vz * Bx - vx * Bz

    # Resistive EMF: eta J_y
    if np.isscalar(eta):
        E_res = eta * Jy
    else:
        E_res = eta * Jy

    Ey = E_conv + E_res
    return Ey, E_conv, E_res


def induction_rhs(Bx, Bz, vx, vz, eta, grid):
    """
    Right-hand side of the induction equation:
        dBx/dt = -dE_y/dz   (or equivalently, curl(v x B - eta J)_x)
        dBz/dt =  dE_y/dx

    In terms of the flux function psi (2D, no guide field):
        Bx = dpsi/dz, Bz = -dpsi/dx
        dpsi/dt = -E_y
    """
    Ey, _, _ = induction_emf(vx, vz, Bx, Bz, eta, grid)

    # dBx/dt = -dEy/dz
    dBx_dt = -d_dz_4th(Ey, grid.dz)
    # dBz/dt = dEy/dx
    dBz_dt = d_dx_4th(Ey, grid.dx)

    return dBx_dt, dBz_dt


# ============================================================
# Momentum Equation Terms
# ============================================================

def lorentz_force(Bx, Bz, grid):
    """
    Lorentz force: F_L = J x B / mu0

    In 2D with B = (Bx, 0, Bz), J = (0, Jy, 0):
        (J x B)_x = Jy Bz
        (J x B)_z = -Jy Bx

    (In normalized units with mu0 = 1)

    This is the driving force for plasma acceleration in reconnection
    outflows. The J x B force converts magnetic energy to kinetic energy.
    """
    Jy = curl_2d(Bx, Bz, grid)
    Fx = Jy * Bz
    Fz = -Jy * Bx
    return Fx, Fz


def viscous_force(vx, vz, nu, grid):
    """
    Viscous diffusion (analogous to Burgers equation viscous term):
        F_visc_x = nu nabla^2 vx
        F_visc_z = nu nabla^2 vz

    This provides numerical dissipation and physical viscosity.
    Maps to the nu d^2u/dx^2 term in burgers_time_viscous.
    """
    Fx = nu * laplacian_2d(vx, grid)
    Fz = nu * laplacian_2d(vz, grid)
    return Fx, Fz


# ============================================================
# Divergence Cleaning Diagnostics
# ============================================================

def divergence_error(Bx, Bz, grid):
    """
    Measure the divergence-free constraint violation:
        err_div = max|div(B)| / max|B|

    For a valid MHD solution, this should remain ~ machine epsilon.
    """
    divB = div_2d(Bx, Bz, grid)
    B_mag = np.sqrt(Bx ** 2 + Bz ** 2) + 1e-30
    err = np.max(np.abs(divB)) / np.max(B_mag)
    return err


# ============================================================
# Magnetic Flux Function
# ============================================================

def compute_flux_function(Bx, Bz, grid):
    """
    Compute the magnetic flux function psi from B:
        Bx = dpsi/dz  =>  psi = integral(Bx, dz)
        Bz = -dpsi/dx =>  psi = -integral(Bz, dx)

    We use Bx for integration along z, starting from z=0.
    """
    psi = np.zeros_like(Bx)
    for iz in range(1, grid.nz):
        psi[:, iz] = psi[:, iz - 1] + 0.5 * (Bx[:, iz] + Bx[:, iz - 1]) * grid.dz
    return psi
