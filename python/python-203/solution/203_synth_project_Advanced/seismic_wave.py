"""
seismic_wave.py -- Discontinuous Galerkin Solver for 1D Stochastic Seismic Wave Equation
=========================================================================================
Solves the 1D elastic wave equation with spatially varying uncertain velocity:
    rho(x) * d^2u/dt^2 = d/dx[mu(x,xi) * du/dx] + f(x,t)
where mu(x,xi) = rho(x) * c^2(x,xi) is the uncertain shear modulus,
c(x,xi) is the random velocity field from KL expansion.

The DG discretization uses piecewise quadratic elements with interior penalty
coupling, following the SIPG/NIPG/IIPG framework.

Seed references:
  - 275_dg1d_poisson: DG formulation, local matrices, penalty, Gauss quadrature
  - 061_b1g3: B1G3 time integration (used via ode_integrator)

Scientific context:
  Seismic wave propagation in heterogeneous geological media is a fundamental
  problem in computational geophysics. The velocity field c(x) is uncertain
  due to limited subsurface information, requiring stochastic methods.
"""
import numpy as np
from typing import Tuple


class DGWaveSolver1D:
    """
    Discontinuous Galerkin finite element solver for the 1D wave equation:
        rho * u_tt = (mu * u_x)_x + f(x,t)

    Uses piecewise quadratic basis functions on each element with
    interior penalty flux coupling across element boundaries.

    The DG formulation follows:
        ss =  1.0: NIPG (nonsymmetric interior penalty Galerkin)
        ss =  0.0: IIPG (incomplete interior penalty Galerkin)
        ss = -1.0: SIPG (symmetric interior penalty Galerkin)
    """

    def __init__(self, n_elements: int, penalty_factor: float = 10.0,
                 ss: float = -1.0):
        """
        Parameters
        ----------
        n_elements : int
            Number of elements in [0, 1].
        penalty_factor : float
            Factor for interior penalty parameter: sigma = factor * p^2 / h.
        ss : float
            Symmetrization parameter: 1 (NIPG), 0 (IIPG), -1 (SIPG).
        """
        self.nel = n_elements
        self.h = 1.0 / n_elements
        self.ss = ss
        self.locdim = 3  # quadratic basis
        self.penalty = penalty_factor * 4.0 / self.h  # p=2 -> p^2=4
        self.n_dof = self.locdim * n_elements

        # Precompute Gauss quadrature (2 points, exact for degree 3)
        self.gp = np.array([0.5 - np.sqrt(3.0) / 6.0,
                            0.5 + np.sqrt(3.0) / 6.0])
        self.gw = np.array([0.5, 0.5])

    def basis(self, xref: float) -> np.ndarray:
        """
        Evaluate quadratic basis functions on reference element [0,1]:
            phi_0(x) = (1-x)(1-2x) = 1 - 3x + 2x^2
            phi_1(x) = 4x(1-x)     = 4x - 4x^2
            phi_2(x) = x(2x-1)     = -x + 2x^2
        """
        return np.array([
            1.0 - 3.0 * xref + 2.0 * xref ** 2,
            4.0 * xref - 4.0 * xref ** 2,
            -xref + 2.0 * xref ** 2
        ])

    def basis_deriv(self, xref: float) -> np.ndarray:
        """Derivatives of basis functions w.r.t. reference coordinate."""
        return np.array([
            -3.0 + 4.0 * xref,
            4.0 - 8.0 * xref,
            -1.0 + 4.0 * xref
        ])

    def assemble_stiffness(self, mu_field: np.ndarray) -> np.ndarray:
        """
        Assemble the global stiffness matrix K for the DG wave equation.

        K_{ij} = sum_e integral_e mu(x) phi_i'(x) phi_j'(x) dx
                 + penalty terms at interfaces

        Parameters
        ----------
        mu_field : ndarray, shape (n_elements,)
            Element-wise shear modulus values (may be uncertain).

        Returns
        -------
        K : ndarray, shape (n_dof, n_dof)
            Global stiffness matrix.
        """
        K = np.zeros((self.n_dof, self.n_dof))

        for elem in range(self.nel):
            mu_e = mu_field[elem] if elem < len(mu_field) else 1.0
            dof_s = elem * self.locdim

            # Element stiffness: integral mu * phi_i' * phi_j' / h
            for ig in range(len(self.gp)):
                bd = self.basis_deriv(self.gp[ig])
                K[dof_s:dof_s + 3, dof_s:dof_s + 3] += (
                    self.gw[ig] * mu_e / self.h * np.outer(bd, bd))

        # Interior penalty and flux terms at element interfaces
        for iface in range(self.nel - 1):
            dof_left = iface * self.locdim
            dof_right = (iface + 1) * self.locdim

            # mu at interface (average of adjacent elements)
            mu_if = 0.5 * (mu_field[iface] + mu_field[iface + 1]) \
                if iface + 1 < len(mu_field) else mu_field[iface]

            sigma = self.penalty * mu_if

            # Consistency term: -<mu du/dx . n, [v]>
            # Symmetry term:   -ss * <[u], mu dv/dx . n>
            # Penalty term:    sigma * <[u], [v]>

            # Right basis of left element: phi_i(1) = delta_{i,2}
            # Left basis of right element: phi_j(0) = delta_{j,0}
            # Derivatives at boundaries
            bd_right = self.basis_deriv(1.0) / self.h  # left elem, right face
            bd_left = self.basis_deriv(0.0) / self.h   # right elem, left face

            # [u] = u_right(0) - u_left(1)
            # {mu du/dx.n} = 0.5*(mu_left * u_left'(1) + mu_right * u_right'(0))
            for i in range(self.locdim):
                for j in range(self.locdim):
                    # Penalty: sigma * [v_i] * [u_j]
                    # v_i contribution: v_i at left face = (1 if i==0 else 0) for left elem
                    # v_i at right face = (1 if i==2 else 0) for left elem
                    vi_right = 1.0 if i == 2 else 0.0  # v_i(1) for left elem
                    vi_left = 1.0 if i == 0 else 0.0   # v_i(0) for right elem
                    uj_right = 1.0 if j == 2 else 0.0
                    uj_left = 1.0 if j == 0 else 0.0

                    # [v_i] = vi_left - vi_right  (for left elem DOFs)
                    # [u_j] = uj_left - uj_right

                    # Penalty term for left-left block
                    K[dof_left + i, dof_left + j] += sigma * vi_right * uj_right
                    # Right-right block
                    K[dof_right + i, dof_right + j] += sigma * vi_left * uj_left
                    # Cross terms
                    K[dof_left + i, dof_right + j] -= sigma * vi_right * uj_left
                    K[dof_right + i, dof_left + j] -= sigma * vi_left * uj_right

                    # Consistency: -{mu u',n} * [v]
                    K[dof_left + i, dof_left + j] -= (
                        0.5 * mu_if * vi_right * bd_right[j])
                    K[dof_left + i, dof_right + j] -= (
                        0.5 * mu_if * vi_right * bd_left[j])
                    K[dof_right + i, dof_left + j] += (
                        0.5 * mu_if * vi_left * bd_right[j])
                    K[dof_right + i, dof_right + j] += (
                        0.5 * mu_if * vi_left * bd_left[j])

                    # Symmetry: -ss * [u] * {mu v',n}
                    K[dof_left + i, dof_left + j] -= (
                        self.ss * 0.5 * mu_if * bd_right[i] * uj_right)
                    K[dof_left + i, dof_right + j] += (
                        self.ss * 0.5 * mu_if * bd_right[i] * uj_left)
                    K[dof_right + i, dof_left + j] -= (
                        self.ss * 0.5 * mu_if * bd_left[i] * uj_right)
                    K[dof_right + i, dof_right + j] += (
                        self.ss * 0.5 * mu_if * bd_left[i] * uj_left)

        return K

    def assemble_mass(self, rho_field: np.ndarray) -> np.ndarray:
        """
        Assemble the global mass matrix M.
        M_{ij} = sum_e integral_e rho(x) phi_i(x) phi_j(x) dx
        """
        M = np.zeros((self.n_dof, self.n_dof))
        for elem in range(self.nel):
            rho_e = rho_field[elem] if elem < len(rho_field) else 1.0
            dof_s = elem * self.locdim
            for ig in range(len(self.gp)):
                b = self.basis(self.gp[ig])
                M[dof_s:dof_s + 3, dof_s:dof_s + 3] += (
                    self.gw[ig] * rho_e * self.h * np.outer(b, b))
        return M

    def assemble_source(self, f_func, t: float) -> np.ndarray:
        """
        Assemble the source vector: F_i = integral f(x,t) phi_i(x) dx
        """
        F = np.zeros(self.n_dof)
        for elem in range(self.nel):
            x_left = elem * self.h
            dof_s = elem * self.locdim
            for ig in range(len(self.gp)):
                x_g = x_left + self.h * self.gp[ig]
                f_val = f_func(x_g, t)
                b = self.basis(self.gp[ig])
                F[dof_s:dof_s + 3] += self.gw[ig] * f_val * b * self.h
        return F

    def solve_wave_equation(self, mu_field: np.ndarray, rho_field: np.ndarray,
                            f_func, tspan: Tuple[float, float],
                            n_steps: int, ic_func=None
                            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Solve the wave equation using Newmark-beta time integration.

        M * u_tt + K * u = F(t)

        Newmark parameters: beta = 0.25, gamma = 0.5 (unconditionally stable,
        second-order accurate, no numerical dissipation).

        The effective stiffness at each step:
            K_eff = K + 1/(beta*dt^2) * M
            F_eff = F(t_{n+1}) + M * (u_n/(beta*dt^2) + v_n/(beta*dt)
                      + (1/(2*beta)-1)*a_n)

        Parameters
        ----------
        mu_field : ndarray, shape (n_elements,)
            Shear modulus per element.
        rho_field : ndarray, shape (n_elements,)
            Density per element.
        f_func : callable
            Source function f(x, t).
        tspan : tuple
            (t_start, t_end).
        n_steps : int
            Number of time steps.
        ic_func : callable or None
            Initial condition u(x, 0). Default: zero.

        Returns
        -------
        t_array : ndarray, shape (n_steps+1,)
        u_array : ndarray, shape (n_steps+1, n_dof)
        x_nodes : ndarray
        """
        K = self.assemble_stiffness(mu_field)
        M = self.assemble_mass(rho_field)

        dt = (tspan[1] - tspan[0]) / n_steps
        beta_nm = 0.25
        gamma_nm = 0.5

        # Initial conditions
        u = np.zeros(self.n_dof)
        v = np.zeros(self.n_dof)
        if ic_func is not None:
            for elem in range(self.nel):
                x_left = elem * self.h
                dof_s = elem * self.locdim
                for ig in range(len(self.gp)):
                    x_g = x_left + self.h * self.gp[ig]
                    b = self.basis(self.gp[ig])
                    u[dof_s:dof_s + 3] += self.gw[ig] * ic_func(x_g) * b * self.h

        # Initial acceleration: M * a_0 = F(0) - K * u_0
        F0 = self.assemble_source(f_func, tspan[0])
        rhs = F0 - K @ u
        # Solve for initial acceleration with regularization
        M_reg = M + 1e-12 * np.eye(self.n_dof)
        a = np.linalg.solve(M_reg, rhs)

        # Effective stiffness (constant for constant time step)
        K_eff = K + M / (beta_nm * dt ** 2)
        # Add boundary conditions
        K_eff[0, :] = 0.0
        K_eff[0, 0] = 1.0
        K_eff[-1, :] = 0.0
        K_eff[-1, -1] = 1.0

        # Pre-factorize
        try:
            K_eff_lu = np.linalg.inv(K_eff)
        except np.linalg.LinAlgError:
            K_eff_lu = np.linalg.pinv(K_eff)

        t_array = np.linspace(tspan[0], tspan[1], n_steps + 1)
        u_array = np.zeros((n_steps + 1, self.n_dof))
        u_array[0] = u.copy()

        for step in range(n_steps):
            t_new = t_array[step + 1]
            F_new = self.assemble_source(f_func, t_new)

            # Effective force
            F_eff = F_new + M @ (
                u / (beta_nm * dt ** 2) +
                v / (beta_nm * dt) +
                (1.0 / (2.0 * beta_nm) - 1.0) * a
            )

            # Apply BCs
            F_eff[0] = 0.0
            F_eff[-1] = 0.0

            # Solve
            u_new = K_eff_lu @ F_eff

            # Update acceleration and velocity
            a_new = (u_new - u) / (beta_nm * dt ** 2) - v / (beta_nm * dt) - \
                    (1.0 / (2.0 * beta_nm) - 1.0) * a
            v_new = v + dt * ((1.0 - gamma_nm) * a + gamma_nm * a_new)

            u = u_new
            v = v_new
            a = a_new
            u_array[step + 1] = u.copy()

        x_nodes = np.linspace(0, 1, self.nel * 10 + 1)
        return t_array, u_array, x_nodes


def compute_wave_energy(u: np.ndarray, v: np.ndarray,
                        K: np.ndarray, M: np.ndarray) -> float:
    """
    Compute the total wave energy: E = 0.5 * v^T M v + 0.5 * u^T K u
    (kinetic + potential energy). Used for conservation monitoring.
    """
    kinetic = 0.5 * v @ M @ v
    potential = 0.5 * u @ K @ u
    return kinetic + potential


def generate_random_velocity_field(n_elements: int, mean_c: float = 3000.0,
                                   std_c: float = 300.0,
                                   correlation_length: float = 0.2,
                                   seed: int = None) -> np.ndarray:
    """
    Generate a random velocity field c(x) for seismic wave propagation.
    Uses a squared-exponential covariance model with KL-like discretization:
        c(x_i) = mean_c + std_c * sum_m sqrt(lambda_m) phi_m(x_i) xi_m

    Parameters
    ----------
    n_elements : int
        Number of spatial elements.
    mean_c : float
        Mean velocity (m/s).
    std_c : float
        Standard deviation of velocity.
    correlation_length : float
        Spatial correlation length (fraction of domain).
    seed : int or None
        Random seed.

    Returns
    -------
    c_field : ndarray, shape (n_elements,)
        Velocity at each element.
    """
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()

    n_kl = min(10, n_elements)
    x = np.linspace(0, 1, n_elements)

    # Covariance matrix (squared exponential)
    lc = correlation_length
    C = np.zeros((n_elements, n_elements))
    for i in range(n_elements):
        for j in range(n_elements):
            C[i, j] = np.exp(-0.5 * ((x[i] - x[j]) / lc) ** 2)

    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(C)
    idx = np.argsort(-eigenvalues)
    eigenvalues = eigenvalues[idx][:n_kl]
    eigenvectors = eigenvectors[:, idx][:, :n_kl]
    eigenvalues = np.maximum(eigenvalues, 0.0)

    # Random coefficients
    xi = rng.randn(n_kl)

    # Construct field
    c_field = mean_c + std_c * eigenvectors @ (np.sqrt(eigenvalues) * xi)
    c_field = np.maximum(c_field, mean_c * 0.1)  # Ensure positivity

    return c_field
