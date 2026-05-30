
import numpy as np
from mesh_geometry import TetrahedralMesh
from quadrature_engine import tetrahedron_unit_o04, tetrahedron_unit_volume
from sparse_linear_algebra import r83s_cg


class AdvectionDiffusionSolver:

    def __init__(self, mesh: TetrahedralMesh, D: float = 0.01,
                 velocity: np.ndarray = None, reaction_rate: float = 0.0):
        self.mesh = mesh
        self.D = D
        if velocity is None:
            velocity = np.array([0.0, 0.0, 0.0])
        self.velocity = np.asarray(velocity, dtype=float)
        self.reaction_rate = reaction_rate
        self.M_lumped = None
        self.K = None
        self.boundary_nodes = None
        self._build_matrices()

    def _build_matrices(self):
        n = self.mesh.n_nodes
        self.M_lumped = np.zeros(n)



        raise NotImplementedError("lumped mass assembly not implemented (Hole 2)")


        x, y, z = self.mesh.nodes[:, 0], self.mesh.nodes[:, 1], self.mesh.nodes[:, 2]
        tol = 1.0e-10
        self.boundary_nodes = np.where(
            (x <= tol) | (x >= 1.0 - tol) |
            (y <= tol) | (y >= 1.0 - tol) |
            (z <= tol) | (z >= 1.0 - tol)
        )[0]


        self.K = self._build_diffusion_operator()

    def _build_diffusion_operator(self):
        n = self.mesh.n_nodes

        order = np.argsort(self.mesh.nodes[:, 0])
        self._perm = order
        self._inv_perm = np.argsort(order)


        h = self.mesh.element_diameter()
        if h < 1.0e-14:
            h = 1.0
        vol_avg = np.mean(self.mesh.compute_volumes()) if self.mesh.n_elements > 0 else 1.0
        scale = self.D * vol_avg / (h * h)
        diag = 2.0 * scale
        off = -scale

        a_sub = np.full(n, off)
        a_diag = np.full(n, diag)
        a_sup = np.full(n, off)
        return np.vstack([a_sub, a_diag, a_sup]).T

    def reaction_term(self, u: np.ndarray) -> np.ndarray:
        return self.reaction_rate * u * (1.0 - u)

    def advection_term(self, u: np.ndarray) -> np.ndarray:
        n = len(u)
        adv = np.zeros(n)
        nodes = self.mesh.nodes[self._perm]
        u_perm = u[self._perm]

        dx = np.diff(nodes[:, 0])
        dx = np.append(dx, dx[-1])
        dx[dx < 1.0e-14] = 1.0e-14
        v = self.velocity[0]
        for i in range(n):
            if v > 0.0 and i < n - 1:
                adv[i] = -v * (u_perm[i + 1] - u_perm[i]) / dx[i]
            elif v < 0.0 and i > 0:
                adv[i] = -v * (u_perm[i] - u_perm[i - 1]) / dx[i - 1]
        return adv[self._inv_perm]

    def step_explicit(self, u: np.ndarray, dt: float) -> np.ndarray:
        u = np.asarray(u, dtype=float)

        u_perm = u[self._perm]
        rhs_diff_perm = np.zeros_like(u_perm)
        n = len(u_perm)
        for i in range(n):
            sub, diag, sup = self.K[i]
            val = diag * u_perm[i]
            if i > 0:
                val += sub * u_perm[i - 1]
            if i < n - 1:
                val += sup * u_perm[i + 1]
            rhs_diff_perm[i] = val
        rhs_diff = rhs_diff_perm[self._inv_perm]

        rhs = rhs_diff + self.advection_term(u) + self.reaction_term(u)

        rhs = rhs / self.M_lumped
        u_new = u + dt * rhs

        u_new[self.boundary_nodes] = 0.0

        u_new = np.clip(u_new, -1.0e3, 1.0e3)
        return u_new

    def initial_condition(self, mode: str = "gaussian") -> np.ndarray:
        x = self.mesh.nodes[:, 0]
        y = self.mesh.nodes[:, 1]
        z = self.mesh.nodes[:, 2]
        if mode == "gaussian":
            u = np.exp(-((x - 0.5) ** 2 + (y - 0.5) ** 2 + (z - 0.5) ** 2) / 0.05)
        elif mode == "random":
            rng = np.random.default_rng(42)
            u = rng.random(self.mesh.n_nodes)
        else:
            u = np.zeros(self.mesh.n_nodes)

        u[self.boundary_nodes] = 0.0
        return u

    def compute_energy(self, u: np.ndarray) -> float:



        raise NotImplementedError("compute_energy not implemented (Hole 1)")
