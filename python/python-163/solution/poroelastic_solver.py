"""
poroelastic_solver.py
=====================
Poroelastic mechanical solver for the geothermal reservoir THM model.

Mathematical formulation:
For a poroelastic medium, the equilibrium equation is:

  \nabla\cdot\boldsymbol{\sigma} + \rho_b \mathbf{g} = \mathbf{0}

with total stress:
  \boldsymbol{\sigma} = \mathbf{C} : \boldsymbol{\varepsilon}
  - \alpha p \mathbf{I} - \beta K_T (T - T_0) \mathbf{I}

In 2D plane strain (x-z), the stress-strain relations are:

  \sigma_{xx} = \frac{E(1-\nu)}{(1+\nu)(1-2\nu)} \varepsilon_{xx}
                + \frac{E \nu}{(1+\nu)(1-2\nu)} \varepsilon_{zz}
                - \alpha p - \beta K_T \Delta T

  \sigma_{zz} = \frac{E \nu}{(1+\nu)(1-2\nu)} \varepsilon_{xx}
                + \frac{E(1-\nu)}{(1+\nu)(1-2\nu)} \varepsilon_{zz}
                - \alpha p - \beta K_T \Delta T

  \sigma_{xz} = \frac{E}{2(1+\nu)} \gamma_{xz}
              = 2 G \varepsilon_{xz}

The Navier-type equation for displacement:

  G \nabla^2 \mathbf{u} + (\lambda + G) \nabla(\nabla\cdot\mathbf{u})
  = \alpha \nabla p + \beta K_T \nabla T - \rho_b \mathbf{g}

Finite difference discretization on a staggered grid yields a system
of linear equations solved by iterative methods.
"""

import numpy as np


class PoroelasticSolver2D:
    """
    2D poroelastic solver on a regular grid.
    """

    def __init__(self, nx, nz, dx, dz, E, nu, alpha, beta, rho_b, g):
        """
        Parameters
        ----------
        nx, nz : int
            Grid dimensions.
        dx, dz : float
            Grid spacing.
        E : float
            Young's modulus (Pa).
        nu : float
            Poisson's ratio.
        alpha : float
            Biot coefficient.
        beta : float
            Thermal expansion coefficient (1/K).
        rho_b : float
            Bulk density (kg/m^3).
        g : float
            Gravitational acceleration (m/s^2).
        """
        self.nx = nx
        self.nz = nz
        self.dx = dx
        self.dz = dz
        self.E = E
        self.nu = nu
        self.alpha = alpha
        self.beta = beta
        self.rho_b = rho_b
        self.g = g

        # Lamé parameters
        self.lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        self.mu = E / (2.0 * (1.0 + nu))
        self.K_T = E / (3.0 * (1.0 - 2.0 * nu))  # drained bulk modulus

    def solve_displacement(self, p, T, T0, num_iterations=100, tol=1.0e-8,
                           fixed_bottom=True, fixed_sides=True):
        """
        Solve for displacement using a finite-difference iterative scheme
        (Gauss-Seidel relaxation).

        Parameters
        ----------
        p : np.ndarray, shape (nx, nz)
            Pore pressure field.
        T : np.ndarray, shape (nx, nz)
            Temperature field.
        T0 : float
            Reference temperature.
        num_iterations : int
        tol : float
        fixed_bottom : bool
            Fix bottom boundary (u=0).
        fixed_sides : bool
            Fix side boundaries horizontally.

        Returns
        -------
        u_x, u_z : np.ndarray
            Displacement fields.
        """
        nx, nz = self.nx, self.nz
        dx, dz = self.dx, self.dz
        lam, mu = self.lam, self.mu
        alpha, beta, K_T = self.alpha, self.beta, self.K_T
        rho_b_g = self.rho_b * self.g

        u_x = np.zeros((nx, nz), dtype=np.float64)
        u_z = np.zeros((nx, nz), dtype=np.float64)

        # Compute forcing terms
        dp_dx = np.zeros_like(p)
        dp_dz = np.zeros_like(p)
        dT_dx = np.zeros_like(T)
        dT_dz = np.zeros_like(T)

        dp_dx[1:-1, :] = (p[2:, :] - p[:-2, :]) / (2.0 * dx)
        dp_dz[:, 1:-1] = (p[:, 2:] - p[:, :-2]) / (2.0 * dz)
        dT_dx[1:-1, :] = (T[2:, :] - T[:-2, :]) / (2.0 * dx)
        dT_dz[:, 1:-1] = (T[:, 2:] - T[:, :-2]) / (2.0 * dz)

        fx = alpha * dp_dx + beta * K_T * dT_dx
        fz = alpha * dp_dz + beta * K_T * dT_dz - rho_b_g

        # Iterative solver (Gauss-Seidel)
        coeff = 2.0 * (lam + 2.0 * mu) / dx**2 + 2.0 * mu / dz**2
        coeff_z = 2.0 * mu / dx**2 + 2.0 * (lam + 2.0 * mu) / dz**2

        for it in range(num_iterations):
            u_x_old = u_x.copy()
            u_z_old = u_z.copy()

            for i in range(1, nx - 1):
                for k in range(1, nz - 1):
                    # x-displacement
                    rhs_x = (fx[i, k]
                             + (lam + 2.0 * mu) * (u_x[i - 1, k] + u_x[i + 1, k]) / dx**2
                             + mu * (u_x[i, k - 1] + u_x[i, k + 1]) / dz**2
                             + (lam + mu) * (u_z[i + 1, k + 1] - u_z[i + 1, k - 1]
                                             - u_z[i - 1, k + 1] + u_z[i - 1, k - 1])
                             / (4.0 * dx * dz))
                    u_x[i, k] = rhs_x / coeff

                    # z-displacement
                    rhs_z = (fz[i, k]
                             + mu * (u_z[i - 1, k] + u_z[i + 1, k]) / dx**2
                             + (lam + 2.0 * mu) * (u_z[i, k - 1] + u_z[i, k + 1]) / dz**2
                             + (lam + mu) * (u_x[i + 1, k + 1] - u_x[i - 1, k + 1]
                                             - u_x[i + 1, k - 1] + u_x[i - 1, k - 1])
                             / (4.0 * dx * dz))
                    u_z[i, k] = rhs_z / coeff_z

            # Boundary conditions
            if fixed_bottom:
                u_x[:, 0] = 0.0
                u_z[:, 0] = 0.0
            else:
                # Free surface at bottom: zero normal stress
                u_z[:, 0] = u_z[:, 1]
                u_x[:, 0] = u_x[:, 1]

            # Top boundary: free surface
            u_x[:, -1] = u_x[:, -2]
            u_z[:, -1] = u_z[:, -2]

            if fixed_sides:
                u_x[0, :] = 0.0
                u_x[-1, :] = 0.0
            else:
                u_x[0, :] = u_x[1, :]
                u_x[-1, :] = u_x[-2, :]

            # Side z-displacements
            u_z[0, :] = u_z[1, :]
            u_z[-1, :] = u_z[-2, :]

            # Check convergence
            err_x = np.max(np.abs(u_x - u_x_old))
            err_z = np.max(np.abs(u_z - u_z_old))
            if max(err_x, err_z) < tol:
                break

        return u_x, u_z

    def compute_strain_stress(self, u_x, u_z, p, T, T0):
        """
        Compute strain and stress tensors from displacement.

        Returns
        -------
        strain : dict
            {'exx', 'ezz', 'exz'}
        stress : dict
            {'sxx', 'szz', 'sxz'}
        """
        nx, nz = self.nx, self.nz
        dx, dz = self.dx, self.dz
        lam, mu = self.lam, self.mu
        alpha, beta, K_T = self.alpha, self.beta, self.K_T

        exx = np.zeros((nx, nz))
        ezz = np.zeros((nx, nz))
        exz = np.zeros((nx, nz))

        exx[1:-1, :] = (u_x[2:, :] - u_x[:-2, :]) / (2.0 * dx)
        ezz[:, 1:-1] = (u_z[:, 2:] - u_z[:, :-2]) / (2.0 * dz)
        exz[1:-1, 1:-1] = 0.5 * ((u_x[1:-1, 2:] - u_x[1:-1, :-2]) / (2.0 * dz)
                                 + (u_z[2:, 1:-1] - u_z[:-2, 1:-1]) / (2.0 * dx))

        delta_T = T - T0
        sxx = (lam + 2.0 * mu) * exx + lam * ezz - alpha * p - beta * K_T * delta_T
        szz = lam * exx + (lam + 2.0 * mu) * ezz - alpha * p - beta * K_T * delta_T
        sxz = 2.0 * mu * exz

        return {"exx": exx, "ezz": ezz, "exz": exz}, {"sxx": sxx, "szz": szz, "sxz": sxz}

    def von_mises_stress(self, stress):
        """
        Compute 2D von Mises equivalent stress:
        \sigma_{vm} = \sqrt{\sigma_{xx}^2 + \sigma_{zz}^2 - \sigma_{xx}\sigma_{zz}
                         + 3\sigma_{xz}^2}
        """
        sxx = stress["sxx"]
        szz = stress["szz"]
        sxz = stress["sxz"]
        svm = np.sqrt(sxx**2 + szz**2 - sxx * szz + 3.0 * sxz**2)
        return svm
