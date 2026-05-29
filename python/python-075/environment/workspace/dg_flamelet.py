"""
dg_flamelet.py
==============
High-Order Discontinuous Galerkin (DG) Solver for 1D Laminar Flamelet Equations.

Based on seed project 273 (dg1d_heat):
- Jacobi polynomial basis (JacobiP.m)
- Vandermonde matrix construction (Vandermonde1D.m)
- DG discretization with central fluxes (HeatCRHS1D.m)
- Low-storage RK4 time stepping (Heat1D.m)

Extended to reacting flamelet equations for combustion DNS.

Scientific Context:
-------------------
The laminar flamelet equation for a scalar Y_k (species mass fraction or temperature)
in mixture-fraction space Z is:

  ∂Y_k/∂t = (χ/2) * ∂²Y_k/∂Z² + ω̇_k/ρ

where χ = 2D|∇Z|² is the scalar dissipation rate [1/s].
In physical space, the 1D flamelet equation reads:

  ρ ∂Y_k/∂t + ρu ∂Y_k/∂x = ∂/∂x(ρD ∂Y_k/∂x) + ω̇_k

For the DG formulation on element K with local coordinate ξ∈[-1,1]:
  ∫_K ∂Y_k/∂t φ_j dξ = -∫_K (ρD/L) ∂Y_k/∂ξ ∂φ_j/∂ξ dξ
                     + [φ_j (ρD/L) ∂Y_k/∂ξ]|_{∂K}
                     + ∫_K (ω̇_k/ρ) φ_j dξ

where L is the Jacobian mapping factor, φ_j are the Np-th order Lagrange basis functions.

The central numerical flux for diffusion is:
  q* = {q} = (q⁺ + q⁻)/2
"""

import numpy as np


def jacobi_polynomial(x, alpha, beta, N):
    """
    Evaluate the orthonormal Jacobi polynomial P_N^{(α,β)}(x).
    Based on seed 273 (JacobiP.m).

    Normalization:
      γ_0 = 2^(α+β+1) / (α+β+1) * Γ(α+1)Γ(β+1) / Γ(α+β+1)

    Recurrence coefficients for i = 1,...,N-1:
      a_{i+1} = 2/(h_1+2) * sqrt((i+1)(i+1+α+β)(i+1+α)(i+1+β) / ((h_1+1)(h_1+3)))
      b_{i+1} = -(α²-β²) / (h_1(h_1+2))
      where h_1 = 2i + α + β

    Three-term recurrence:
      P_{i+1} = (1/a_{i+1}) * [ -(x - b_{i+1}) P_i - a_i P_{i-1} ]
    """
    x = np.atleast_1d(x)
    if N == 0:
        return np.ones_like(x)

    # Normalization constant γ_0
    from math import gamma as math_gamma
    gamma0 = (2.0**(alpha + beta + 1.0) / (alpha + beta + 1.0)
              * math_gamma(alpha + 1.0) * math_gamma(beta + 1.0)
              / math_gamma(alpha + beta + 1.0))

    # Storage for all orders up to N
    P = np.zeros((N + 1, len(x)))
    P[0, :] = 1.0 / np.sqrt(gamma0)

    if N >= 1:
        gamma1 = (alpha + 1.0) * (beta + 1.0) / (alpha + beta + 3.0) * gamma0
        P[1, :] = ((alpha + beta + 2.0) * x / 2.0 + (alpha - beta) / 2.0) / np.sqrt(gamma1)

    aold = 2.0 / (2.0 + alpha + beta) * np.sqrt((alpha + 1.0) * (beta + 1.0) / (alpha + beta + 3.0))

    for i in range(1, N):
        h1 = 2.0 * i + alpha + beta
        anew = (2.0 / (h1 + 2.0)
                * np.sqrt((i + 1.0) * (i + 1.0 + alpha + beta)
                          * (i + 1.0 + alpha) * (i + 1.0 + beta)
                          / ((h1 + 1.0) * (h1 + 3.0))))
        bnew = -(alpha**2 - beta**2) / (h1 * (h1 + 2.0))
        P[i + 1, :] = (1.0 / anew) * (-aold * P[i - 1, :] + (x - bnew) * P[i, :])
        aold = anew

    return P[N, :]


def vandermonde_1d(N, r):
    """
    Initialize 1D Vandermonde matrix V_{ij} = P_j(r_i).
    Based on seed 273 (Vandermonde1D.m).
    """
    r = np.atleast_1d(r)
    V = np.zeros((len(r), N + 1))
    for j in range(N + 1):
        V[:, j] = jacobi_polynomial(r, 0.0, 0.0, j)
    return V


def jacobi_gauss_lobatto(alpha, beta, N):
    """
    Compute the N+1 Gauss-Lobatto nodes and weights for Jacobi polynomials.
    Nodes include endpoints -1 and 1.
    """
    if N == 0:
        return np.array([-1.0, 1.0]), np.array([1.0, 1.0])
    if N == 1:
        return np.array([-1.0, 0.0, 1.0]), np.array([1.0/3.0, 4.0/3.0, 1.0/3.0])

    # Interior nodes are eigenvalues of modified Jacobi matrix
    from numpy.linalg import eigvalsh
    # Construct symmetric Jacobi matrix
    n = N - 1
    diag = np.zeros(n)
    offdiag = np.zeros(n - 1)
    for i in range(n):
        diag[i] = (beta**2 - alpha**2) / ((2.0 * i + alpha + beta + 2.0)
                                          * (2.0 * i + alpha + beta + 4.0))
    for i in range(n - 1):
        num = 4.0 * (i + 1.0) * (i + 1.0 + alpha + beta + 1.0) \
              * (i + 1.0 + alpha) * (i + 1.0 + beta)
        den = ((2.0 * i + alpha + beta + 2.0)**2
               * (2.0 * i + alpha + beta + 3.0)
               * (2.0 * i + alpha + beta + 1.0))
        offdiag[i] = np.sqrt(num / den)

    J = np.diag(diag) + np.diag(offdiag, 1) + np.diag(offdiag, -1)
    x_int = eigvalsh(J)
    x = np.concatenate([[-1.0], np.sort(x_int), [1.0]])

    # Weights via quadrature formula
    V = vandermonde_1d(N, x)
    w = np.zeros(N + 1)
    # Integral of P_0 = 1 is 2
    # Solve V^T w = [2, 0, 0, ...]
    rhs = np.zeros(N + 1)
    rhs[0] = np.sqrt(2.0)
    w = np.linalg.solve(V.T, rhs)
    w = w**2  # Adjust for orthonormal basis
    return x, w


class DGFlameletSolver:
    """
    1D Discontinuous Galerkin solver for reacting scalar transport.
    Solves:  ∂Y/∂t = D ∂²Y/∂x² + S(Y)
    on domain [xmin, xmax] with Neumann BCs (zero flux).
    """

    def __init__(self, N, K, xmin, xmax, D, bc_type='neumann'):
        """
        Parameters
        ----------
        N : int
            Polynomial order per element.
        K : int
            Number of elements.
        xmin, xmax : float
            Domain bounds.
        D : float
            Diffusion coefficient.
        bc_type : str
            'neumann' or 'dirichlet'.
        """
        self.N = N
        self.K = K
        self.xmin = xmin
        self.xmax = xmax
        self.D = D
        self.bc_type = bc_type

        # Gauss-Lobatto nodes and weights in [-1,1]
        self.r, self.w = jacobi_gauss_lobatto(0.0, 0.0, N)
        self.Np = N + 1  # nodes per element

        # Build Vandermonde and differentiation matrix
        self.V = vandermonde_1d(N, self.r)
        self.Vinv = np.linalg.inv(self.V)

        # Derivative matrix: D_{ij} = dφ_j/dr at r_i
        # dV_{ij} = dP_j/dr at r_i
        dV = np.zeros((self.Np, self.Np))
        for j in range(self.Np):
            # Derivative of P_j
            if j == 0:
                dV[:, j] = 0.0
            else:
                # Recurrence for derivative of Legendre polynomial
                dV[:, j] = jacobi_polynomial(self.r, 1.0, 1.0, j - 1) * np.sqrt(j * (j + 1.0))
        self.Dr = np.dot(dV, self.Vinv)

        # Mass matrix (diagonal for Gauss-Lobatto)
        self.M = np.diag(self.w)
        self.Minv = np.diag(1.0 / self.w)

        # Element size
        self.dx_elem = (xmax - xmin) / K

        # Global node coordinates
        self.x = np.zeros((self.Np, K))
        for k in range(K):
            self.x[:, k] = xmin + (k + 0.5) * self.dx_elem + 0.5 * self.dx_elem * self.r

        # Connectivity maps for faces
        self._build_maps()

        # Lift matrix (surface to volume)
        # Emat = [φ(-1), φ(1)] at nodes
        Emat = np.zeros((self.Np, 2))
        V_face = vandermonde_1d(N, np.array([-1.0, 1.0]))
        Emat = np.dot(self.Vinv.T, V_face.T)
        self.LIFT = np.dot(self.V, np.dot(self.V.T, Emat))

        # RK4 coefficients (low-storage)
        self.rk4a = np.array([0.0,
                              -567301805773.0 / 1357537059087.0,
                              -2404267990393.0 / 2016746695238.0,
                              -3550918686646.0 / 2091501179385.0,
                              -1275806237668.0 / 842570457699.0])
        self.rk4b = np.array([1432997174477.0 / 9575080441755.0,
                              5161836677717.0 / 13612068292357.0,
                              1720146321549.0 / 2090206949498.0,
                              3134564353537.0 / 4481467310338.0,
                              2277821191437.0 / 14882151754819.0])
        self.rk4c = np.array([0.0,
                              1432997174477.0 / 9575080441755.0,
                              2526269341429.0 / 6820363962896.0,
                              2006345519317.0 / 3224310063776.0,
                              2802321613138.0 / 2924317926251.0])

    def _build_maps(self):
        """Build face connectivity maps for DG."""
        self.vmapM = np.zeros((2, self.K), dtype=int)
        self.vmapP = np.zeros((2, self.K), dtype=int)
        self.mapI = 0
        self.mapO = 1

        for k in range(self.K):
            self.vmapM[0, k] = k * self.Np          # left face
            self.vmapM[1, k] = (k + 1) * self.Np - 1  # right face

        for k in range(self.K):
            if k == 0:
                self.vmapP[0, k] = self.vmapM[1, self.K - 1]  # periodic left
            else:
                self.vmapP[0, k] = self.vmapM[1, k - 1]

            if k == self.K - 1:
                self.vmapP[1, k] = self.vmapM[0, 0]  # periodic right
            else:
                self.vmapP[1, k] = self.vmapM[0, k + 1]

        self.vmapM = self.vmapM.flatten()
        self.vmapP = self.vmapP.flatten()

    def compute_rhs(self, u, source_func=None, t=0.0):
        """
        Compute DG right-hand side for diffusion with central flux.
        Based on HeatCRHS1D.m from seed 273.

        q = D * du/dx (auxiliary variable for LDG)
        rhs = D * dq/dx + source
        """
        # Reshape u to (Np, K)
        u_local = u.reshape((self.Np, self.K), order='F')

        # Compute derivative in reference element
        ux = np.dot(self.Dr, u_local)

        # Map to physical: d/dx = (2/dx) * d/dr
        ux = (2.0 / self.dx_elem) * ux

        # Build jumps at faces: du = (u^- - u^+)/2
        du = np.zeros(2 * self.K)
        du[:] = (u[self.vmapM] - u[self.vmapP]) / 2.0

        # Boundary conditions
        if self.bc_type == 'neumann':
            du[self.mapI] = 0.0
            du[self.mapO] = 0.0
        elif self.bc_type == 'dirichlet':
            uin = -u[self.vmapI]
            du[self.mapI] = (u[self.vmapI] - uin) / 2.0
            uout = -u[self.vmapO]
            du[self.mapO] = (u[self.vmapO] - uout) / 2.0

        # Compute q = D * ux - LIFT * du
        q = self.D * ux - np.dot(self.LIFT, du.reshape((2, self.K), order='F'))

        # Derivative of q
        qx = np.dot(self.Dr, q)
        qx = (2.0 / self.dx_elem) * qx

        # Jumps in q
        dq = np.zeros(2 * self.K)
        dq[:] = (q.flatten('F')[self.vmapM] - q.flatten('F')[self.vmapP]) / 2.0

        if self.bc_type == 'neumann':
            dq[self.mapI] = 0.0
            dq[self.mapO] = 0.0
        elif self.bc_type == 'dirichlet':
            qin = q.flatten('F')[self.vmapI]
            dq[self.mapI] = (q.flatten('F')[self.vmapI] - qin) / 2.0
            qout = q.flatten('F')[self.vmapO]
            dq[self.mapO] = (q.flatten('F')[self.vmapO] - qout) / 2.0

        # RHS
        rhs = self.D * qx - np.dot(self.LIFT, dq.reshape((2, self.K), order='F'))
        rhs = rhs.flatten('F')

        # Add source term
        if source_func is not None:
            x_flat = self.x.flatten('F')
            src = source_func(x_flat, t)
            rhs = rhs + src

        return rhs

    def step(self, u, dt, source_func=None, t=0.0):
        """Low-storage RK4 step with clipping for stability."""
        resu = np.zeros_like(u)
        time = t
        for intrk in range(5):
            timelocal = time + self.rk4c[intrk] * dt
            rhsu = self.compute_rhs(u, source_func, timelocal)
            # Clip extreme RHS values to prevent overflow
            rhsu = np.clip(rhsu, -1e6, 1e6)
            resu = self.rk4a[intrk] * resu + dt * rhsu
            resu = np.clip(resu, -1e6, 1e6)
            u = u + self.rk4b[intrk] * resu
            u = np.clip(u, -1e3, 1e3)
        return u

    def solve_steady_flamelet(self, source_func, max_iter=1000, tol=1e-8):
        """
        Solve steady flamelet equation: D d²Y/dx² + S(Y) = 0
        using pseudo-time stepping.
        """
        u = np.zeros(self.Np * self.K)
        dt = 0.25 * (self.dx_elem / self.Np) ** 2 / max(self.D, 1e-12)
        for _ in range(max_iter):
            u_new = self.step(u, dt, source_func)
            if np.linalg.norm(u_new - u) < tol:
                return u_new
            u = u_new
        return u
