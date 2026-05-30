
import numpy as np
from typing import Tuple
from utils import r8vec_bracket4






class FEM1DSolver:

    def __init__(self, nodes: np.ndarray, permittivity: np.ndarray, charge_density: np.ndarray):
        self.nodes = np.asarray(nodes, dtype=float)
        self.n = len(self.nodes)
        if self.n < 2:
            raise ValueError("FEM1DSolver: 节点数至少为 2")

        self.permittivity = np.asarray(permittivity, dtype=float)
        if len(self.permittivity) == self.n - 1:
            self.epsilon = self.permittivity
        elif len(self.permittivity) == self.n:
            self.epsilon = 0.5 * (self.permittivity[:-1] + self.permittivity[1:])
        else:
            raise ValueError("FEM1DSolver: permittivity 长度不匹配")

        self.charge_density = np.asarray(charge_density, dtype=float)
        if len(self.charge_density) == self.n - 1:
            self.rho = self.charge_density
        elif len(self.charge_density) == self.n:
            self.rho = 0.5 * (self.charge_density[:-1] + self.charge_density[1:])
        else:
            raise ValueError("FEM1DSolver: charge_density 长度不匹配")

    def _assemble(self) -> Tuple[np.ndarray, np.ndarray]:
        K = np.zeros((self.n, self.n))
        F = np.zeros(self.n)

        for e in range(self.n - 1):
            h_e = self.nodes[e + 1] - self.nodes[e]
            if h_e <= 0.0:
                raise ValueError(f"FEM1DSolver: 单元 {e} 长度非正")

            eps_e = self.epsilon[e]
            rho_e = self.rho[e]


            ke = (eps_e / h_e) * np.array([[1.0, -1.0], [-1.0, 1.0]])

            fe = (rho_e * h_e / 2.0) * np.array([1.0, 1.0])


            K[e, e] += ke[0, 0]
            K[e, e + 1] += ke[0, 1]
            K[e + 1, e] += ke[1, 0]
            K[e + 1, e + 1] += ke[1, 1]

            F[e] += fe[0]
            F[e + 1] += fe[1]

        return K, F

    def solve_dirichlet(self, phi_left: float, phi_right: float) -> np.ndarray:
        K, F = self._assemble()


        F[0] = phi_left
        F[1] -= K[1, 0] * phi_left
        K[0, :] = 0.0
        K[:, 0] = 0.0
        K[0, 0] = 1.0

        F[-1] = phi_right
        F[-2] -= K[-2, -1] * phi_right
        K[-1, :] = 0.0
        K[:, -1] = 0.0
        K[-1, -1] = 1.0


        phi = np.linalg.solve(K, F)
        return phi

    def compute_electric_field(self, phi: np.ndarray) -> np.ndarray:
        E = np.zeros(self.n - 1)
        for e in range(self.n - 1):
            h_e = self.nodes[e + 1] - self.nodes[e]
            E[e] = -(phi[e + 1] - phi[e]) / h_e
        return E

    def evaluate_at_points(self, phi: np.ndarray, query_points: np.ndarray) -> np.ndarray:
        query_points = np.asarray(query_points, dtype=float)
        nq = len(query_points)
        result = np.zeros(nq)
        for iq in range(nq):
            zq = query_points[iq]
            if zq <= self.nodes[0]:
                result[iq] = phi[0]
                continue
            if zq >= self.nodes[-1]:
                result[iq] = phi[-1]
                continue
            idx = r8vec_bracket4(self.n, self.nodes, zq)
            h_e = self.nodes[idx + 1] - self.nodes[idx]
            if h_e <= 0.0:
                result[iq] = phi[idx]
                continue
            xi = (zq - self.nodes[idx]) / h_e
            result[iq] = (1.0 - xi) * phi[idx] + xi * phi[idx + 1]
        return result






def r85_np_fs(n: int, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if n < 1:
        raise ValueError("r85_np_fs: n 必须 >= 1")
    a = np.copy(a)
    b = np.copy(b)
    x = np.zeros(n)


    for i in range(n):

        pivot = a[0, i]
        if abs(pivot) < 1.0e-30:

            pivot = 1.0e-30 if pivot >= 0 else -1.0e-30
            a[0, i] = pivot


        if i + 1 < n:
            factor1 = a[3, i + 1] / pivot
            a[3, i + 1] = factor1
            a[0, i + 1] -= factor1 * a[1, i]
            if i + 2 < n:
                a[1, i + 1] -= factor1 * a[2, i]
            b[i + 1] -= factor1 * b[i]


        if i + 2 < n:
            factor2 = a[4, i + 2] / pivot
            a[4, i + 2] = factor2
            a[3, i + 2] -= factor2 * a[1, i]
            a[0, i + 2] -= factor2 * a[2, i]
            b[i + 2] -= factor2 * b[i]


    x[-1] = b[-1] / a[0, -1]
    if n >= 2:
        x[-2] = (b[-2] - a[1, -2] * x[-1]) / a[0, -2]
    for i in range(n - 3, -1, -1):
        x[i] = (b[i] - a[1, i] * x[i + 1] - a[2, i] * x[i + 2]) / a[0, i]

    return x


def r85_dif2(n: int) -> np.ndarray:
    a = np.zeros((5, n))
    a[0, :] = 2.0
    if n > 1:
        a[1, :-1] = -1.0
        a[3, 1:] = -1.0
    return a


def solve_diffusion_1d(
    n: int,
    D: float,
    sigma_a: float,
    source: np.ndarray,
    dx: float,
    bc_left: float,
    bc_right: float,
) -> np.ndarray:
    if n < 3:
        raise ValueError("solve_diffusion_1d: n 必须 >= 3")
    if len(source) != n:
        raise ValueError("solve_diffusion_1d: source 长度必须等于 n")

    a = np.zeros((5, n))
    diag = 2.0 * D / (dx * dx) + sigma_a
    offdiag = -D / (dx * dx)

    a[0, :] = diag
    a[1, :-1] = offdiag
    a[3, 1:] = offdiag

    b = np.copy(source)

    b[0] = bc_left
    b[-1] = bc_right
    a[0, 0] = 1.0
    a[1, 0] = 0.0
    a[0, -1] = 1.0
    a[3, -1] = 0.0
    if n > 1:
        a[3, 1] = 0.0
        a[1, -2] = 0.0

    return r85_np_fs(n, a, b)






if __name__ == "__main__":

    nodes = np.linspace(0.0, 1.0, 11)
    eps = np.ones(10)
    rho = np.zeros(10)
    solver = FEM1DSolver(nodes, eps, rho)
    phi = solver.solve_dirichlet(0.0, 1.0)
    assert abs(phi[0]) < 1e-10, "左边界条件未满足"
    assert abs(phi[-1] - 1.0) < 1e-10, "右边界条件未满足"

    for i in range(len(nodes)):
        expected = nodes[i]
        assert abs(phi[i] - expected) < 1e-10, f"FEM 线性解偏差: {phi[i]} vs {expected}"


    a = r85_dif2(5)
    b = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    x = r85_np_fs(5, a, b)

    full = np.zeros((5, 5))
    for i in range(5):
        full[i, i] = a[0, i]
        if i + 1 < 5:
            full[i, i + 1] = a[1, i]
        if i - 1 >= 0:
            full[i, i - 1] = a[3, i]
    residual = np.linalg.norm(full @ x - b)
    assert residual < 1e-10, f"R85 残差过大: {residual}"


    n = 21
    dx = 0.05
    phi_diff = solve_diffusion_1d(n, D=1.0, sigma_a=0.1, source=np.zeros(n), dx=dx, bc_left=0.0, bc_right=1.0)
    assert abs(phi_diff[0]) < 1e-10
    assert abs(phi_diff[-1] - 1.0) < 1e-10
    assert np.all(np.isfinite(phi_diff)), "扩散方程解含非有限值"

    print("detector_field.py: 所有自测通过")
