
import numpy as np


def neural_activation_rd(U, V, dx, dy, dt, D_u, D_v, gamma, kappa,
                          stimulus, laplacian_func=None):
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    stimulus = np.asarray(stimulus, dtype=float)

    if U.shape != V.shape or U.shape != stimulus.shape:
        raise ValueError("U, V, stimulus 形状必须相同")

    if laplacian_func is None:
        from laplacian_operator import laplacian_5point
        laplacian_func = laplacian_5point

    Lu = laplacian_func(U, dx, dy)
    Lv = laplacian_func(V, dx, dy)


    reaction_u = -U * V**2 + gamma * (1.0 - U) + stimulus
    reaction_v = U * V**2 - (gamma + kappa) * V

    U_new = U + dt * (D_u * Lu + reaction_u)
    V_new = V + dt * (D_v * Lv + reaction_v)


    U_new = np.clip(U_new, 0.0, 1.0)
    V_new = np.clip(V_new, 0.0, 1.0)

    return U_new, V_new


class NeuralActivationPattern:

    def __init__(self, nx, ny, dx, dy, D_u=0.01, D_v=0.005,
                 gamma=0.024, kappa=0.06):
        self.nx = int(nx)
        self.ny = int(ny)
        self.dx = float(dx)
        self.dy = float(dy)
        self.D_u = float(D_u)
        self.D_v = float(D_v)
        self.gamma = float(gamma)
        self.kappa = float(kappa)


        dt_max = 0.25 * min(dx**2, dy**2) / max(D_u, D_v)
        self.dt_max = dt_max

        self.U = None
        self.V = None
        self._initialized = False

    def initialize(self, seed_pattern='gaussian'):
        if seed_pattern == 'gaussian':
            cx, cy = self.nx // 2, self.ny // 2
            X, Y = np.meshgrid(np.arange(self.nx), np.arange(self.ny), indexing='ij')
            sigma = min(self.nx, self.ny) / 8.0
            V_seed = 0.25 * np.exp(-((X - cx)**2 + (Y - cy)**2) / (2 * sigma**2))
            U = np.ones((self.nx, self.ny)) - 2.0 * V_seed
        elif seed_pattern == 'random':
            np.random.seed(42)
            U = np.random.rand(self.nx, self.ny) * 0.1 + 0.9
            V = np.random.rand(self.nx, self.ny) * 0.1
            self.U = np.clip(U, 0.0, 1.0)
            self.V = np.clip(V, 0.0, 1.0)
            self._initialized = True
            return
        elif seed_pattern == 'uniform':
            U = np.ones((self.nx, self.ny))
            V = np.zeros((self.nx, self.ny))
            self.U = U
            self.V = V
            self._initialized = True
            return
        else:
            raise ValueError(f"未知的 seed_pattern: {seed_pattern}")

        self.U = np.clip(U, 0.0, 1.0)
        self.V = np.clip(V_seed, 0.0, 1.0)
        self._initialized = True

    def evolve(self, n_steps, stimulus_history=None, dt=None):
        if not self._initialized:
            raise RuntimeError("必须先调用 initialize()")

        if dt is None:
            dt = self.dt_max * 0.5
        if dt > self.dt_max:
            raise ValueError(f"dt={dt} 超过 CFL 限制 {self.dt_max}")

        U_history = []
        V_history = []

        for step in range(n_steps):
            if stimulus_history is not None and step < len(stimulus_history):
                stim = stimulus_history[step]
            else:
                stim = np.zeros((self.nx, self.ny))

            self.U, self.V = neural_activation_rd(
                self.U, self.V, self.dx, self.dy, dt,
                self.D_u, self.D_v, self.gamma, self.kappa, stim
            )
            U_history.append(self.U.copy())
            V_history.append(self.V.copy())

        return U_history, V_history

    def compute_spread_metrics(self):
        if self.U is None:
            raise RuntimeError("未初始化")

        threshold = 0.5
        active_mask = self.U > threshold
        active_area = np.sum(active_mask)

        if active_area < 1:
            return 0.0, (0.0, 0.0), 0.0

        X, Y = np.meshgrid(np.arange(self.nx), np.arange(self.ny), indexing='ij')
        centroid_x = np.sum(X * active_mask) / active_area
        centroid_y = np.sum(Y * active_mask) / active_area

        spread_std = np.sqrt(
            np.sum(((X - centroid_x)**2 + (Y - centroid_y)**2) * active_mask)
            / active_area
        )

        return float(active_area), (float(centroid_x), float(centroid_y)), float(spread_std)
