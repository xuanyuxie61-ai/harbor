
import numpy as np
from typing import Tuple, Optional


def laplacian9_torus(field: np.ndarray, dx: float) -> np.ndarray:
    if field.ndim != 2:
        raise ValueError("Field must be 2D array")
    if dx <= 0:
        raise ValueError("dx must be positive")
    ny, nx = field.shape
    if nx < 3 or ny < 3:
        raise ValueError("Field dimensions must be at least 3x3")

    result = np.zeros_like(field)

    for i in range(ny):
        im1 = (i - 1) % ny
        ip1 = (i + 1) % ny
        for j in range(nx):
            jm1 = (j - 1) % nx
            jp1 = (j + 1) % nx
            result[i, j] = (
                1.0 * field[im1, jm1] + 4.0 * field[im1, j] + 1.0 * field[im1, jp1]
                + 4.0 * field[i, jm1] - 20.0 * field[i, j] + 4.0 * field[i, jp1]
                + 1.0 * field[ip1, jm1] + 4.0 * field[ip1, j] + 1.0 * field[ip1, jp1]
            ) / (6.0 * dx * dx)
    return result


def gray_scott_step(
    U: np.ndarray,
    V: np.ndarray,
    dt: float,
    dx: float,
    D_u: float,
    D_v: float,
    gamma: float,
    kappa: float
) -> Tuple[np.ndarray, np.ndarray]:
    if dt <= 0 or dx <= 0:
        raise ValueError("dt and dx must be positive")
    if D_u < 0 or D_v < 0:
        raise ValueError("Diffusion coefficients must be non-negative")


    stability_limit = dx * dx / (4.0 * max(D_u, D_v) + 1e-15)
    if dt > stability_limit:

        dt = 0.5 * stability_limit

    lapU = laplacian9_torus(U, dx)
    lapV = laplacian9_torus(V, dx)


    reaction = U * V * V

    dUdt = D_u * lapU - reaction + gamma * (1.0 - U)
    dVdt = D_v * lapV + reaction - (gamma + kappa) * V

    U_new = U + dt * dUdt
    V_new = V + dt * dVdt


    U_new = np.clip(U_new, 0.0, 1.0)
    V_new = np.clip(V_new, 0.0, 1.0)

    return U_new, V_new


def gray_scott_simulation(
    nx: int = 64,
    ny: int = 64,
    n_steps: int = 5000,
    D_u: float = 8.0e-5,
    D_v: float = 4.0e-5,
    gamma: float = 0.024,
    kappa: float = 0.06,
    dt: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray]:
    if nx < 3 or ny < 3:
        raise ValueError("Grid dimensions must be >= 3")
    if n_steps < 0:
        raise ValueError("n_steps must be non-negative")

    dx = 2.5 / (nx - 1)
    dy = 2.5 / (ny - 1)
    dx = min(dx, dy)

    if dt is None:
        dt = 0.5 * dx * dx / (4.0 * max(D_u, D_v) + 1e-15)


    x = np.linspace(0.0, 2.5, nx)
    y = np.linspace(0.0, 2.5, ny)
    X, Y = np.meshgrid(x, y)

    V = np.zeros((ny, nx))
    mask = (X >= 1.0) & (X <= 1.5) & (Y >= 1.0) & (Y <= 1.5)
    V[mask] = 0.25 * (np.sin(4.0 * np.pi * X[mask]) ** 2) * (np.sin(4.0 * np.pi * Y[mask]) ** 2)
    U = 1.0 - 2.0 * V


    for _ in range(n_steps):
        U, V = gray_scott_step(U, V, dt, dx, D_u, D_v, gamma, kappa)

    return U, V


def advection_ftcs_step(
    u: np.ndarray,
    c: float,
    dt: float,
    dx: float
) -> np.ndarray:





    pass


def pattern_to_quantum_parameters(
    pattern: np.ndarray,
    n_qubits: int,
    n_layers: int
) -> np.ndarray:
    if pattern.size == 0:
        raise ValueError("Pattern must not be empty")
    if n_qubits <= 0 or n_layers <= 0:
        raise ValueError("n_qubits and n_layers must be positive")


    flat = pattern.flatten()
    p_min, p_max = flat.min(), flat.max()
    if abs(p_max - p_min) < 1e-15:

        flat = flat + 1e-8 * np.sin(np.arange(len(flat)))
        p_min, p_max = flat.min(), flat.max()

    normalized = (flat - p_min) / (p_max - p_min)


    n_params = n_layers * n_qubits
    indices = np.linspace(0, len(normalized) - 1, n_params)
    idx_low = np.floor(indices).astype(int)
    idx_high = np.minimum(idx_low + 1, len(normalized) - 1)
    frac = indices - idx_low

    interpolated = normalized[idx_low] * (1.0 - frac) + normalized[idx_high] * frac


    params = np.pi * interpolated - np.pi / 2.0
    return params.reshape(n_layers, n_qubits)


class ReactionDiffusionFeatureMap:

    def __init__(
        self,
        n_qubits: int = 4,
        n_layers: int = 3,
        D_u: float = 8.0e-5,
        D_v: float = 4.0e-5,
        gamma: float = 0.024,
        kappa: float = 0.06
    ):
        if n_qubits <= 0 or n_layers <= 0:
            raise ValueError("n_qubits and n_layers must be positive")
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.D_u = D_u
        self.D_v = D_v
        self.gamma = gamma
        self.kappa = kappa
        self._cached_pattern: Optional[np.ndarray] = None

    def generate_pattern(self, grid_size: int = 32, n_steps: int = 3000) -> np.ndarray:
        if grid_size < 3:
            raise ValueError("grid_size must be >= 3")
        if n_steps < 0:
            raise ValueError("n_steps must be non-negative")

        U, V = gray_scott_simulation(
            nx=grid_size, ny=grid_size, n_steps=n_steps,
            D_u=self.D_u, D_v=self.D_v, gamma=self.gamma, kappa=self.kappa
        )

        self._cached_pattern = U
        return U

    def get_parameters(self, data_point: np.ndarray) -> np.ndarray:
        if self._cached_pattern is None:
            self.generate_pattern()

        if len(data_point) != self.n_qubits:
            raise ValueError(
                f"Data point dimension {len(data_point)} must match n_qubits {self.n_qubits}"
            )

        base_params = pattern_to_quantum_parameters(
            self._cached_pattern, self.n_qubits, self.n_layers
        )


        for l in range(self.n_layers):
            for q in range(self.n_qubits):
                base_params[l, q] += data_point[q] * np.pi / 2.0

                base_params[l, q] = ((base_params[l, q] + np.pi) % (2.0 * np.pi)) - np.pi

        return base_params
