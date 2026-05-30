
import numpy as np
from typing import Tuple


class ConvectionDynamics:

    def __init__(self, nx: int = 128, ny: int = 128, dx: float = 2000.0,
                 alpha: float = 0.25, beta: float = 0.001,
                 delta: float = 1e-5, epsilon: float = 0.002,
                 Du: float = 5.0e3, Dv: float = 1.0e2):
        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.dy = dx
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        self.epsilon = epsilon

        self.Du = Du
        self.Dv = Dv

        self.U = np.zeros((ny, nx))
        self.V = np.zeros((ny, nx))
        self._set_initial_condition()

    def _set_initial_condition(self):
        nx, ny = self.nx, self.ny

        cx, cy = nx // 3, ny // 2
        for j in range(ny):
            for i in range(nx):
                r2 = ((i - cx) / (nx / 8.0))**2 + ((j - cy) / (ny / 8.0))**2
                self.U[j, i] = 0.8 * np.exp(-r2)
                self.V[j, i] = 0.5 * self.alpha * (1.0 + 0.3 * np.sin(2.0 * np.pi * i / nx))

        self.U = np.clip(self.U, 0.0, 1.0)
        self.V = np.clip(self.V, 0.0, 1.0)

    def _laplacian_9point(self, A: np.ndarray) -> np.ndarray:
        ny, nx = A.shape
        L = np.zeros_like(A)
        coeff = 1.0 / (6.0 * self.dx * self.dy)
        for j in range(ny):
            for i in range(nx):
                jp = (j + 1) % ny
                jm = (j - 1) % ny
                ip = (i + 1) % nx
                im = (i - 1) % nx
                L[j, i] = coeff * (
                    A[jm, im] + 4.0 * A[jm, i] + A[jm, ip]
                    + 4.0 * A[j, im] - 20.0 * A[j, i] + 4.0 * A[j, ip]
                    + A[jp, im] + 4.0 * A[jp, i] + A[jp, ip]
                )
        return L

    def _rhs(self, U: np.ndarray, V: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        lapU = self._laplacian_9point(U)
        lapV = self._laplacian_9point(V)


        reaction_U = (1.0 / self.epsilon) * U * (1.0 - U) * (U - (V + self.beta) / self.alpha)
        reaction_V = U - V

        dUdt = self.Du * lapU + reaction_U
        dVdt = self.Dv * lapV + self.delta * lapV + reaction_V
        return dUdt, dVdt

    def step_rk4(self, dt: float):
        U0 = self.U.copy()
        V0 = self.V.copy()

        k1_U, k1_V = self._rhs(U0, V0)
        k2_U, k2_V = self._rhs(U0 + 0.5 * dt * k1_U, V0 + 0.5 * dt * k1_V)
        k3_U, k3_V = self._rhs(U0 + 0.5 * dt * k2_U, V0 + 0.5 * dt * k2_V)
        k4_U, k4_V = self._rhs(U0 + dt * k3_U, V0 + dt * k3_V)

        self.U = U0 + (dt / 6.0) * (k1_U + 2.0 * k2_U + 2.0 * k3_U + k4_U)
        self.V = V0 + (dt / 6.0) * (k1_V + 2.0 * k2_V + 2.0 * k3_V + k4_V)


        self.U = np.clip(self.U, 0.0, 1.0)
        self.V = np.clip(self.V, 0.0, 1.0)

        if np.any(~np.isfinite(self.U)) or np.any(~np.isfinite(self.V)):
            self.U = np.nan_to_num(self.U, nan=0.0, posinf=1.0, neginf=0.0)
            self.V = np.nan_to_num(self.V, nan=0.0, posinf=1.0, neginf=0.0)

    def integrate(self, dt: float, nsteps: int) -> Tuple[np.ndarray, np.ndarray]:
        for _ in range(nsteps):
            self.step_rk4(dt)
        return self.U.copy(), self.V.copy()

    def get_convective_intensity(self) -> np.ndarray:
        return self.U.copy()

    def get_moisture_accumulation(self) -> np.ndarray:
        return self.V.copy()

    def total_convective_energy(self) -> float:
        return float(np.sum(self.U**2) * self.dx * self.dy)
