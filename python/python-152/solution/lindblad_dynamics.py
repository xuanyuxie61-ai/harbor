"""
Open quantum system dynamics for error propagation.

Incorporates:
- 827_ode_euler_system: forward Euler ODE solver
- 274_dg1d_maxwell: Discontinuous Galerkin spectral methods for spatial derivative
"""
import numpy as np
import math
from scipy.linalg import expm
from utils import chop_array, pauli_operators, depolarizing_channel, trace_inner_product


def lindbladian_superoperator(H: np.ndarray, jump_ops: list, hbar: float = 1.0) -> np.ndarray:
    """
    Build the Liouvillian superoperator L in vectorized form.
    For density matrix ρ, the Lindblad master equation is:

        dρ/dt = -(i/ℏ)[H, ρ] + Σ_k ( L_k ρ L_k† - ½{L_k† L_k, ρ} )

    Vectorized form: |ρ⟩⟩ is column-stacked vector.
    L = -i/ℏ (I ⊗ H - H^T ⊗ I) + Σ_k (L_k^* ⊗ L_k - ½ I ⊗ L_k† L_k - ½ (L_k^T L_k^*) ⊗ I)

    Parameters
    ----------
    H : Hamiltonian (d×d).
    jump_ops : list of jump operators L_k.
    hbar : Planck constant.

    Returns
    -------
    L : (d² × d²) complex matrix.
    """
    d = H.shape[0]
    I = np.eye(d, dtype=complex)
    # Unitary part
    L = -(1j / hbar) * (np.kron(I, H) - np.kron(H.T, I))
    # Dissipative part
    for Lk in jump_ops:
        Lk_dag = Lk.conj().T
        L += np.kron(Lk.conj(), Lk)
        L -= 0.5 * np.kron(I, Lk_dag @ Lk)
        L -= 0.5 * np.kron((Lk_dag @ Lk).T, I)
    return L


def forward_euler_rho(rho0: np.ndarray, L: np.ndarray, t_final: float, n_steps: int) -> np.ndarray:
    """
    Solve d|ρ⟩⟩/dt = L |ρ⟩⟩ via forward Euler (from 827_ode_euler_system).

        |ρ_{i+1}⟩⟩ = |ρ_i⟩⟩ + h * L |ρ_i⟩⟩

    with step size h = t_final / n_steps.
    """
    h = t_final / n_steps
    d2 = rho0.size
    rho_vec = rho0.reshape(d2, 1).copy()
    for _ in range(n_steps):
        rho_vec = rho_vec + h * (L @ rho_vec)
        # Numerical renormalization to preserve trace
        rho_mat = rho_vec.reshape(int(np.sqrt(d2)), int(np.sqrt(d2)))
        tr = np.trace(rho_mat)
        if abs(tr) > 1e-15:
            rho_mat = rho_mat / tr
        rho_vec = rho_mat.reshape(d2, 1)
    return rho_vec.reshape(int(np.sqrt(d2)), int(np.sqrt(d2)))


def exact_lindblad_evolution(rho0: np.ndarray, L: np.ndarray, t: float) -> np.ndarray:
    """Exact solution |ρ(t)⟩⟩ = exp(L t) |ρ_0⟩⟩."""
    d2 = rho0.size
    rho_vec = rho0.reshape(d2, 1)
    rho_t_vec = expm(L * t) @ rho_vec
    rho_t = rho_t_vec.reshape(int(np.sqrt(d2)), int(np.sqrt(d2)))
    tr = np.trace(rho_t)
    if abs(tr) > 1e-15:
        rho_t = rho_t / tr
    return rho_t


class DGLindbladSolver:
    """
    Discontinuous Galerkin spectral solver for Lindblad dynamics on a 1D chain
    of qubits with nearest-neighbor coupling.

    Spatial discretization (from 274_dg1d_maxwell):
        - Nodal DG on Legendre-Gauss-Lobatto (LGL) points.
        - Vandermonde matrix V, differentiation matrix D.
        - Upwind numerical flux for advection-like terms.

    The effective equation for error probability density p(x,t) on a 1D lattice:

        ∂p/∂t + v ∂p/∂x = D ∂²p/∂x² + γ(1 - 2p)

    where v is error propagation velocity, D is diffusion, γ is local error rate.
    """

    def __init__(self, n_elements: int, poly_order: int, domain: tuple = (0.0, 1.0)):
        self.n_elements = n_elements
        self.poly_order = poly_order
        self.domain = domain
        self.Np = poly_order + 1  # nodes per element
        self.x, self.V, self.Vr, self.Dr, self.M, self.Minv = self._dg_operators()
        self.rx, self.J = self._geometric_factors()
        self.lift = self._lift_operator()

    def _jacobi_polynomial(self, x: np.ndarray, alpha: float, beta: float, N: int) -> np.ndarray:
        """Evaluate Jacobi polynomial P_n^{(α,β)}(x) for n=0..N."""
        xp = x.reshape(-1, 1)
        PL = np.zeros((xp.shape[0], N + 1))
        gamma0 = 2 ** (alpha + beta + 1) / (alpha + beta + 1) * math.gamma(alpha + 1) * math.gamma(beta + 1) / math.gamma(alpha + beta + 1)
        PL[:, 0] = 1.0 / np.sqrt(gamma0)
        if N == 0:
            return PL
        gamma1 = (alpha + 1) * (beta + 1) / (alpha + beta + 2) * gamma0
        PL[:, 1] = (((alpha + beta + 2) * xp.flatten() / 2 + (alpha - beta) / 2) / np.sqrt(gamma1)).flatten()
        if N == 1:
            return PL
        aold = 2.0 / (2.0 + alpha + beta) * np.sqrt((alpha + 1) * (beta + 1) / (alpha + beta + 3))
        for i in range(1, N):
            h1 = 2 * i + alpha + beta
            anew = 2.0 / (h1 + 2) * np.sqrt((i + 1) * (i + 1 + alpha + beta) * (i + 1 + alpha) * (i + 1 + beta) / (h1 + 1) / (h1 + 3))
            bnew = -(alpha ** 2 - beta ** 2) / h1 / (h1 + 2)
            PL[:, i + 1] = (1.0 / anew * (-aold * PL[:, i - 1] + (xp.flatten() - bnew) * PL[:, i])).flatten()
            aold = anew
        return PL

    def _jacobi_gauss_lobatto(self, alpha: float, beta: float, N: int) -> np.ndarray:
        """LGL nodes for Jacobi polynomial of order N."""
        if N == 0:
            return np.array([-1.0, 1.0])
        if N == 1:
            return np.array([-1.0, 0.0, 1.0])
        x = np.zeros(N + 1)
        x[0], x[N] = -1.0, 1.0
        # Interior nodes are roots of P'_{N}^{(α+1,β+1)}
        from numpy.polynomial.legendre import legroots
        # Approximate using Legendre derivative roots for α=β=0
        PL = self._jacobi_polynomial(np.linspace(-1, 1, 1000), alpha, beta, N)
        # Use numpy roots for derivative approximation
        # For simplicity, use numpy's polynomial fitting
        t = np.linspace(-1 + 1e-8, 1 - 1e-8, 10000)
        PLt = self._jacobi_polynomial(t, alpha, beta, N).flatten()
        # Better: use scipy special functions
        from scipy.special import roots_jacobi
        xi, _ = roots_jacobi(N - 1, alpha + 1, beta + 1)
        x[1:N] = np.sort(xi)
        return x

    def _dg_operators(self):
        """Build DG operators: nodes, Vandermonde, differentiation matrix."""
        N = self.poly_order
        x_local = self._jacobi_gauss_lobatto(0.0, 0.0, N)
        # Vandermonde
        V = np.zeros((N + 1, N + 1))
        for j in range(N + 1):
            PL = self._jacobi_polynomial(x_local, 0.0, 0.0, j)
            V[:, j] = PL[:, j].flatten()
        # Derivative matrix
        Vr = np.zeros((N + 1, N + 1))
        for j in range(1, N + 1):
            # Derivative of P_j is proportional to P_{j-1} with coefficient
            # d/dx P_j^{(0,0)}(x) = (j+1)/2 * P_{j-1}^{(1,1)}(x)
            # Simpler: numerical differentiation via polynomial fit
            pass
        # Use more stable approach: D = Vr / V where Vr_{ij} = dϕ_j/dr(r_i)
        Vr_poly = np.zeros((N + 1, N + 1))
        for i in range(N + 1):
            for j in range(N + 1):
                # d/dx x^j = j x^{j-1}
                if j > 0:
                    Vr_poly[i, j] = j * (x_local[i] ** (j - 1))
        # Better: use scipy's approach
        from numpy.polynomial.legendre import legvander, legder
        # Legendre Vandermonde
        V_leg = legvander(x_local, N)
        # Derivative matrix: map coefficients to nodal derivatives
        D_coeff = np.zeros((N + 1, N + 1))
        for j in range(N + 1):
            c = np.zeros(N + 1)
            c[j] = 1.0
            dc = legder(c)
            dc_padded = np.zeros(N + 1)
            dc_padded[:len(dc)] = dc
            D_coeff[:, j] = dc_padded
        Dr = V_leg @ D_coeff @ np.linalg.inv(V_leg)
        # Mass matrix and inverse
        M = np.linalg.inv(V_leg @ V_leg.T)
        Minv = np.linalg.inv(M)
        # Global coordinates
        a, b = self.domain
        dx_elem = (b - a) / self.n_elements
        x_global = np.zeros((self.n_elements, N + 1))
        for e in range(self.n_elements):
            x_center = a + (e + 0.5) * dx_elem
            x_global[e, :] = x_center + 0.5 * dx_elem * x_local
        return x_global, V_leg, Vr_poly, Dr, M, Minv

    def _geometric_factors(self):
        """Jacobian and metric terms."""
        a, b = self.domain
        dx = (b - a) / self.n_elements
        J = dx / 2.0
        rx = 1.0 / J
        return rx, J

    def _lift_operator(self):
        """Lifting operator for surface flux contributions."""
        # Simplified lift: only face contributions
        N = self.poly_order
        lift = np.zeros((N + 1, 2))
        # Mass matrix at faces
        Mface = np.array([1.0, 1.0])  # approximate
        lift[0, 0] = 1.0
        lift[N, 1] = 1.0
        return lift

    def rhs(self, u: np.ndarray, v: float, D: float, gamma: float) -> np.ndarray:
        """
        Compute right-hand side for DG discretization:
            du/dt = -v ∂u/∂x + D ∂²u/∂x² + γ(1 - 2u)
        """
        n_elem, Np = u.shape
        du = np.zeros_like(u)
        rx = self.rx
        # Volume terms
        for e in range(n_elem):
            ux = rx * (self.Dr @ u[e, :])
            uxx = rx * (self.Dr @ ux)
            du[e, :] = -v * ux + D * uxx + gamma * (1.0 - 2.0 * u[e, :])
        # Interface fluxes (upwind for advection, central for diffusion)
        for e in range(n_elem):
            eL = (e - 1) % n_elem
            eR = (e + 1) % n_elem
            # Left interface
            uM = u[eL, -1] if e > 0 else u[e, 0]
            uP = u[e, 0]
            flux = 0.5 * v * (uM + uP) - 0.5 * abs(v) * (uP - uM)
            du[e, 0] += -flux / self.J
            # Right interface
            uM = u[e, -1]
            uP = u[eR, 0] if e < n_elem - 1 else u[e, -1]
            flux = 0.5 * v * (uM + uP) - 0.5 * abs(v) * (uP - uM)
            du[e, -1] += flux / self.J
        return du

    def evolve(self, u0: np.ndarray, t_final: float, n_steps: int,
               v: float = 1.0, D: float = 0.01, gamma: float = 0.1) -> np.ndarray:
        """
        Evolve initial condition u0 using low-storage RK4 (5-stage, 4th order)
        adapted from Maxwell1D.
        """
        dt = t_final / n_steps
        u = u0.copy()
        # LSERK coefficients
        rk4a = np.array([0.0, -0.417890474499852, -1.192151694642677,
                         -1.697784692471528, -1.514183444257156])
        rk4b = np.array([0.149659021999229, 0.379210312999627, 0.822955029386982,
                         0.699450455949122, 0.153057247968152])
        for _ in range(n_steps):
            res = np.zeros_like(u)
            for s in range(5):
                rhs_val = self.rhs(u, v, D, gamma)
                res = rk4a[s] * res + dt * rhs_val
                u = u + rk4b[s] * res
            # Clip to physical range [0,1]
            u = np.clip(u, 0.0, 1.0)
        return u
