
import numpy as np
from typing import Tuple, Optional


class KitaevBdGSolver:

    def __init__(self, n_sites: int, mu: float, t: float, delta: float,
                 periodic: bool = True):
        if n_sites < 3:
            raise ValueError("晶格格点数 N 必须至少为 3")
        if abs(t) < 1e-14:
            raise ValueError("跃迁强度 t 不能为零（会导致退耦）")
        self.n = n_sites
        self.mu = float(mu)
        self.t = float(t)
        self.delta = float(delta)
        self.periodic = periodic

    def _build_r83p_matrix(self) -> np.ndarray:
        n = self.n
        a = np.zeros((3, n))



        a[0, 0] = (self.t - self.delta) / 2.0
        for j in range(1, n):
            a[0, j] = (self.t - self.delta) / 2.0


        for j in range(n):
            a[1, j] = -self.mu / 2.0


        for j in range(n - 1):
            a[2, j] = -(self.t + self.delta) / 2.0

        a[2, n - 1] = -(self.t + self.delta) / 2.0

        return a

    def r83p_factorize(self, a: np.ndarray) -> Tuple[np.ndarray, np.ndarray,
                                                       np.ndarray, float, int]:
        n = a.shape[1]
        if n < 3:
            raise ValueError("R83P分解要求矩阵阶数 N >= 3")

        a_lu = np.zeros((3, n))


        a1 = np.copy(a[:, :n - 1])
        a1_lu, info = self._r83_np_factorize(n - 1, a1)
        if info != 0:
            return a_lu, np.zeros(n - 1), np.zeros(n - 1), 0.0, info

        a_lu[:, :n - 1] = a1_lu
        a_lu[0, 0] = a[0, 0]
        a_lu[2, n - 2] = a[2, n - 2]
        a_lu[:, n - 1] = a[:, n - 1]


        work2 = np.zeros(n - 1)
        work2[0] = a[2, n - 1]
        work2[n - 2] = a[0, n - 1]
        work2 = self._r83_np_solve(n - 1, a1_lu, work2, job=0)


        work3 = np.zeros(n - 1)
        work3[0] = a[0, 0]
        work3[n - 2] = a[2, n - 2]
        work3 = self._r83_np_solve(n - 1, a1_lu, work3, job=1)


        work4 = a[1, n - 1] - a[0, 0] * work2[0] - a[2, n - 2] * work2[n - 2]

        if abs(work4) < 1e-15:
            info = n
            return a_lu, work2, work3, work4, info

        return a_lu, work2, work3, work4, 0

    def _r83_np_factorize(self, n: int, a: np.ndarray) -> Tuple[np.ndarray, int]:
        info = 0
        a_lu = np.copy(a)

        for i in range(n):
            if i > 0:
                a_lu[1, i] -= a_lu[2, i - 1] * a_lu[0, i - 1]
            if abs(a_lu[1, i]) < 1e-15:
                info = i + 1
                return a_lu, info
            if i < n - 1:
                a_lu[0, i] /= a_lu[1, i]

        return a_lu, 0

    def _r83_np_solve(self, n: int, a_lu: np.ndarray, b: np.ndarray,
                      job: int = 0) -> np.ndarray:
        x = np.copy(b)

        if job == 0:

            for i in range(1, n):
                x[i] -= a_lu[2, i - 1] * x[i - 1]

            for i in range(n):
                x[i] /= a_lu[1, i]
            for i in range(n - 2, -1, -1):
                x[i] -= a_lu[0, i] * x[i + 1]
        else:

            for i in range(1, n):
                x[i] -= a_lu[0, i - 1] * x[i - 1]
            for i in range(n):
                x[i] /= a_lu[1, i]
            for i in range(n - 2, -1, -1):
                x[i] -= a_lu[2, i] * x[i + 1]

        return x

    def r83p_solve(self, a_lu: np.ndarray, work2: np.ndarray,
                   work3: np.ndarray, work4: float, b: np.ndarray) -> np.ndarray:
        n = len(b)
        if n < 3:
            raise ValueError("R83P求解要求向量长度 N >= 3")

        x = np.zeros(n)
        n1 = n - 1


        x1_temp = self._r83_np_solve(n1, a_lu[:, :n1], b[:n1], job=0)


        x[n - 1] = (b[n - 1] - a_lu[0, 0] * x1_temp[0]
                    - a_lu[2, n1 - 1] * x1_temp[n1 - 1]) / work4


        x[:n1] = x1_temp - work2 * x[n - 1]

        return x

    def build_full_bdg_hamiltonian(self) -> np.ndarray:
        n = self.n
        h_bdg = np.zeros((2 * n, 2 * n))


        for i in range(n):
            h_bdg[i, i] = -self.mu
            if i < n - 1:
                h_bdg[i, i + 1] = -self.t
                h_bdg[i + 1, i] = -self.t
            elif self.periodic:
                h_bdg[i, 0] = -self.t
                h_bdg[0, i] = -self.t


        for i in range(n):
            h_bdg[n + i, n + i] = self.mu
            if i < n - 1:
                h_bdg[n + i, n + i + 1] = self.t
                h_bdg[n + i + 1, n + i] = self.t
            elif self.periodic:
                h_bdg[n + i, n] = self.t
                h_bdg[n, n + i] = self.t


        for i in range(n):
            if i < n - 1:
                h_bdg[i, n + i + 1] = self.delta
                h_bdg[i + 1, n + i] = -self.delta
                h_bdg[n + i, i + 1] = -self.delta
                h_bdg[n + i + 1, i] = self.delta
            elif self.periodic:
                h_bdg[i, n] = self.delta
                h_bdg[0, n + i] = -self.delta
                h_bdg[n + i, 0] = -self.delta
                h_bdg[n, i] = self.delta

        return h_bdg

    def diagonalize(self) -> Tuple[np.ndarray, np.ndarray]:
        h_bdg = self.build_full_bdg_hamiltonian()


        eigvals, eigvecs = np.linalg.eigh(h_bdg)


        idx = np.argsort(eigvals)
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]


        tol = 1e-12 * max(abs(self.t), abs(self.delta), abs(self.mu), 1.0)
        eigvals[np.abs(eigvals) < tol] = 0.0

        return eigvals, eigvecs

    def identify_majorana_zero_modes(self, eigvals: np.ndarray,
                                      eigvecs: np.ndarray,
                                      energy_tol: float = 1e-10
                                      ) -> Tuple[np.ndarray, np.ndarray]:
        n = self.n
        mask = np.abs(eigvals) < energy_tol
        zero_indices = np.where(mask)[0]

        if len(zero_indices) == 0:
            return np.array([]), np.array([])

















        raise NotImplementedError("Hole 1: 请实现马约拉纳零能模识别逻辑")


    def compute_energy_gap(self, eigvals: np.ndarray) -> float:
        positive = eigvals[eigvals > 1e-12]
        if len(positive) == 0:
            return 0.0
        return float(np.min(positive))

    def topological_phase_diagram(self, mu_vals: np.ndarray) -> np.ndarray:
        gaps = np.zeros_like(mu_vals)
        original_mu = self.mu

        for i, mu in enumerate(mu_vals):
            self.mu = mu
            eigvals, _ = self.diagonalize()
            gaps[i] = self.compute_energy_gap(eigvals)

        self.mu = original_mu
        return gaps


def demo():
    solver = KitaevBdGSolver(n_sites=20, mu=0.5, t=1.0, delta=0.8,
                             periodic=False)
    eigvals, eigvecs = solver.diagonalize()
    print("Kitaev Chain BdG Spectrum (first 10):")
    print(eigvals[:10])
    gap = solver.compute_energy_gap(eigvals)
    print(f"Energy gap: {gap:.6e}")

    modes_u, modes_v = solver.identify_majorana_zero_modes(eigvals, eigvecs)
    print(f"Number of MZMs detected: {len(modes_u)}")


if __name__ == "__main__":
    demo()
