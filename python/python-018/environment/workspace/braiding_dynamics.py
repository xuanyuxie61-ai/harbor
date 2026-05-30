
import numpy as np
from typing import Tuple, Optional


class MajoranaBraidingDynamics:

    def __init__(self, num_majorana: int = 4):
        if num_majorana < 4 or num_majorana % 2 != 0:
            raise ValueError("马约拉纳数目必须为≥4的偶数")
        self.N = num_majorana

    def _majorana_matrix(self, i: int, j: int) -> np.ndarray:

        mat = np.zeros((self.N, self.N))
        if 0 <= i < self.N and 0 <= j < self.N and i != j:
            mat[i, j] = 1.0
            mat[j, i] = -1.0
        return mat

    def braid_operator(self, i: int, j: int) -> np.ndarray:
        if i == j or not (0 <= i < self.N and 0 <= j < self.N):
            raise ValueError("编织指标必须不同且在有效范围内")

        A = self._majorana_matrix(i, j)

        I = np.eye(self.N)
        theta = np.pi / 4.0

        B = I + np.sin(theta) * A + (1.0 - np.cos(theta)) * (A @ A)
        return B

    def apply_braid_sequence(self, sequence: list) -> np.ndarray:
        U = np.eye(self.N)
        for i, j in sequence:
            B = self.braid_operator(i, j)
            U = B @ U
        return U

    def rigid_body_mapping_derivative(self, xyz: np.ndarray,
                                       I1: float, I2: float, I3: float
                                       ) -> np.ndarray:
        x, y, z = xyz[0], xyz[1], xyz[2]


        max_val = 1e6
        x = np.clip(x, -max_val, max_val)
        y = np.clip(y, -max_val, max_val)
        z = np.clip(z, -max_val, max_val)

        dxdt = (1.0 / I3 - 1.0 / I2) * z * y
        dydt = (1.0 / I1 - 1.0 / I3) * x * z
        dzdt = (1.0 / I2 - 1.0 / I1) * y * x

        return np.array([dxdt, dydt, dzdt])

    def integrate_rigid_body(self, xyz0: np.ndarray,
                              t_span: np.ndarray,
                              I1: float, I2: float, I3: float) -> np.ndarray:
        n_steps = len(t_span)
        xyz = np.zeros((n_steps, 3))
        xyz[0] = xyz0

        for i in range(n_steps - 1):
            dt = t_span[i + 1] - t_span[i]
            k1 = self.rigid_body_mapping_derivative(xyz[i], I1, I2, I3)
            k2 = self.rigid_body_mapping_derivative(
                xyz[i] + 0.5 * dt * k1, I1, I2, I3)
            k3 = self.rigid_body_mapping_derivative(
                xyz[i] + 0.5 * dt * k2, I1, I2, I3)
            k4 = self.rigid_body_mapping_derivative(
                xyz[i] + dt * k3, I1, I2, I3)

            xyz[i + 1] = xyz[i] + (dt / 6.0) * (k1 + 2.0 * k2
                                                  + 2.0 * k3 + k4)

            xyz[i + 1] = np.clip(xyz[i + 1], -1e6, 1e6)

        return xyz

    def compute_braid_group_relation(self, i: int, j: int, k: int
                                      ) -> float:
        if not all(0 <= idx < self.N for idx in [i, j, k]):
            return -1.0

        B_i = self.braid_operator(i, j)
        B_j = self.braid_operator(j, k)



        left = B_i @ B_j @ B_i
        right = B_j @ B_i @ B_j

        diff = np.linalg.norm(left - right, ord='fro')
        return float(diff)


class MajoranaImpurityDynamics:

    def __init__(self, alpha: float = 1.0, beta: float = 0.5,
                 gamma: float = 0.8, delta: float = 0.3):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta

    def derivatives(self, t: float, y: np.ndarray) -> np.ndarray:
        n_m, n_i = y[0], y[1]


        n_m = max(n_m, 0.0)
        n_i = max(n_i, 0.0)

        dn_m = self.alpha * n_m - self.beta * n_m * n_i
        dn_i = -self.gamma * n_i + self.delta * n_m * n_i

        return np.array([dn_m, dn_i])

    def integrate(self, y0: np.ndarray, t_span: np.ndarray) -> np.ndarray:
        n_steps = len(t_span)
        y = np.zeros((n_steps, 2))
        y[0] = np.maximum(y0, 0.0)

        for i in range(n_steps - 1):
            dt = t_span[i + 1] - t_span[i]
            k1 = self.derivatives(t_span[i], y[i])
            k2 = self.derivatives(t_span[i] + 0.5 * dt, y[i] + 0.5 * dt * k1)
            k3 = self.derivatives(t_span[i] + 0.5 * dt, y[i] + 0.5 * dt * k2)
            k4 = self.derivatives(t_span[i] + dt, y[i] + dt * k3)

            y[i + 1] = y[i] + (dt / 6.0) * (k1 + 2.0 * k2
                                             + 2.0 * k3 + k4)

            y[i + 1] = np.maximum(y[i + 1], 0.0)

        return y

    def conserved_quantity(self, y: np.ndarray) -> np.ndarray:
        n_m, n_i = y[:, 0], y[:, 1]

        n_m_safe = np.maximum(n_m, 1e-15)
        n_i_safe = np.maximum(n_i, 1e-15)

        H = (self.delta * n_m_safe + self.beta * n_i_safe
             - self.gamma * np.log(n_m_safe)
             - self.alpha * np.log(n_i_safe))
        return H


def demo():
    braid = MajoranaBraidingDynamics(num_majorana=4)


    diff = braid.compute_braid_group_relation(0, 1, 2)
    print("Yang-Baxter relation difference:", diff)


    seq = [(0, 1), (1, 2), (2, 3)]
    U = braid.apply_braid_sequence(seq)
    print("Braid sequence matrix trace:", np.trace(U))


    t_span = np.linspace(0, 10, 101)
    xyz = braid.integrate_rigid_body(
        xyz0=np.array([1.0, 0.5, 0.2]),
        t_span=t_span, I1=1.0, I2=0.8, I3=0.6
    )
    print("Rigid-body mapped dynamics final state:", xyz[-1])


    impurity = MajoranaImpurityDynamics(
        alpha=1.0, beta=0.5, gamma=0.8, delta=0.3
    )
    y = impurity.integrate(y0=np.array([2.0, 1.0]),
                           t_span=np.linspace(0, 20, 201))
    H = impurity.conserved_quantity(y)
    print("Conserved quantity variation:", np.max(H) - np.min(H))


if __name__ == "__main__":
    demo()
