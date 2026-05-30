
import numpy as np
from typing import Tuple, Optional


class TightBindingSolver:

    def __init__(self, n_atoms: int, n_orbitals_per_atom: int = 1):
        self.n_atoms = n_atoms
        self.n_orbitals = n_orbitals_per_atom
        self.n_basis = n_atoms * n_orbitals_per_atom
        self.H = np.zeros((self.n_basis, self.n_basis))
        self.S = np.eye(self.n_basis)
        self.eigenvalues = None
        self.eigenvectors = None

    def build_hamiltonian_sk(self, positions: np.ndarray,
                             onsite_energies: np.ndarray,
                             v_ss_sigma: float = -1.0,
                             v_sp_sigma: float = 1.5,
                             r_cutoff: float = 3.5e-10,
                             decay_length: float = 0.5e-10):
        if positions.shape[0] != self.n_atoms:
            raise ValueError("positions 行数必须等于 n_atoms")
        if onsite_energies.shape[0] != self.n_atoms:
            raise ValueError("onsite_energies 长度必须等于 n_atoms")

        n = self.n_basis
        self.H = np.zeros((n, n))
        for i in range(self.n_atoms):
            self.H[i, i] = onsite_energies[i]
            for j in range(i + 1, self.n_atoms):
                r_vec = positions[i] - positions[j]
                r_ij = np.linalg.norm(r_vec)
                if r_ij > 1e-12 and r_ij <= r_cutoff:

                    hopping = v_ss_sigma * np.exp(-(r_ij - 2.5e-10) / decay_length)
                    self.H[i, j] = hopping
                    self.H[j, i] = hopping

    def apply_border_banded_factorization(self, n1: int, n2: int,
                                          ml: int, mu: int) -> Tuple[np.ndarray, int]:
        n = self.n_basis
        if n1 + n2 != n:
            raise ValueError("n1 + n2 必须等于矩阵阶数")
        if ml < 0 or mu < 0 or ml >= n1 or mu >= n1:
            raise ValueError("带宽参数非法")


        A_full = self.H.copy()
        A_lu = A_full.copy()


        if n1 > 0:
            A1 = A_lu[:n1, :n1]
            try:

                P, L, U = self._lu_decomposition(A1)
                A_lu[:n1, :n1] = U

                for i in range(n1):
                    for j in range(i):
                        A_lu[i, j] = L[i, j]
            except np.linalg.LinAlgError:
                return A_lu, 1


        if n1 > 0 and n2 > 0:
            A2 = A_lu[:n1, n1:]
            for j in range(n2):
                b = -A2[:, j]

                x = self._band_solve(A_lu[:n1, :n1], b, ml, mu)
                A_lu[:n1, n1 + j] = x


        if n1 > 0 and n2 > 0:
            A3 = A_full[n1:, :n1]
            A2_prime = A_lu[:n1, n1:]
            A4 = A_full[n1:, n1:]
            A4_schur = A4 + A3 @ A2_prime
            A_lu[n1:, n1:] = A4_schur


        if n2 > 0:
            try:
                P, L, U = self._lu_decomposition(A_lu[n1:, n1:])
                A_lu[n1:, n1:] = U
                for i in range(n2):
                    for j in range(i):
                        A_lu[n1 + i, n1 + j] = L[i, j]
            except np.linalg.LinAlgError:
                return A_lu, n1 + 1

        return A_lu, 0

    def _lu_decomposition(self, A: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = A.shape[0]
        L = np.eye(n)
        U = A.copy()
        for k in range(n - 1):
            if abs(U[k, k]) < 1e-15:
                raise np.linalg.LinAlgError("Zero pivot")
            for i in range(k + 1, n):
                L[i, k] = U[i, k] / U[k, k]
                U[i, k:] -= L[i, k] * U[k, k:]
        return np.eye(n), L, U

    def _band_solve(self, A_band: np.ndarray, b: np.ndarray,
                    ml: int, mu: int) -> np.ndarray:

        return np.linalg.solve(A_band, b)

    def solve_eigenvalues_dense(self) -> Tuple[np.ndarray, np.ndarray]:
        eigvals, eigvecs = np.linalg.eigh(self.H)
        self.eigenvalues = eigvals
        self.eigenvectors = eigvecs
        return eigvals, eigvecs

    def compute_dos(self, energies: np.ndarray, sigma: float = 0.05) -> np.ndarray:
        if self.eigenvalues is None:
            raise RuntimeError("必须先调用 solve_eigenvalues_dense()")
        energies = np.asarray(energies, dtype=float)
        dos = np.zeros_like(energies)
        prefactor = 1.0 / (np.sqrt(2.0 * np.pi) * sigma)
        for ei in self.eigenvalues:
            dos += prefactor * np.exp(-0.5 * ((energies - ei) / sigma) ** 2)
        return dos

    def compute_band_energy(self, n_electrons: int,
                           temperature_k: float = 300.0) -> float:
        if self.eigenvalues is None:
            raise RuntimeError("必须先调用 solve_eigenvalues_dense()")
        from utils import kb_t_ev
        kb_t = kb_t_ev(temperature_k)
        if kb_t < 1e-12:

            occ = np.zeros_like(self.eigenvalues)
            idx = np.argsort(self.eigenvalues)
            occ[idx[:n_electrons]] = 1.0
            return float(np.sum(occ * self.eigenvalues))


        e_min = np.min(self.eigenvalues) - 10.0
        e_max = np.max(self.eigenvalues) + 10.0
        for _ in range(100):
            e_fermi = 0.5 * (e_min + e_max)
            fd = 1.0 / (np.exp((self.eigenvalues - e_fermi) / kb_t) + 1.0)
            ne = np.sum(fd)
            if abs(ne - n_electrons) < 1e-10:
                break
            if ne > n_electrons:
                e_max = e_fermi
            else:
                e_min = e_fermi

        fd = 1.0 / (np.exp((self.eigenvalues - e_fermi) / kb_t) + 1.0)
        return float(np.sum(fd * self.eigenvalues))

    def compute_adsorption_energy(self, e_isolated: float,
                                  e_surface: float,
                                  e_complex: float) -> float:
        return e_complex - e_surface - e_isolated

    def write_slap_format(self, filename: str):
        n = self.n_basis

        rows, cols = np.nonzero(np.abs(self.H) > 1e-15)
        nelt = len(rows)
        isym = 1 if np.allclose(self.H, self.H.T, atol=1e-12) else 0
        irhs = 0
        isoln = 0

        with open(filename, 'w') as f:
            f.write(f"{n:10d}{nelt:10d}{isym:10d}{irhs:10d}{isoln:10d}\n")
            for idx in range(nelt):
                f.write(f" {rows[idx]:5d} {cols[idx]:5d} {self.H[rows[idx], cols[idx]]:16.7e}\n")

    @staticmethod
    def read_slap_format(filename: str) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray]:
        with open(filename, 'r') as f:
            header = f.readline().strip()
            n, nelt, isym, irhs, isoln = [int(x) for x in header.split()]
            ia = np.zeros(nelt, dtype=int)
            ja = np.zeros(nelt, dtype=int)
            a = np.zeros(nelt, dtype=float)
            for i in range(nelt):
                line = f.readline()
                parts = line.split()
                ia[i] = int(parts[0])
                ja[i] = int(parts[1])
                a[i] = float(parts[2])
        return n, ia, ja, a

    def solve_poisson_fd1d(self, charge_density: np.ndarray,
                           x_grid: np.ndarray,
                           epsilon_r: float = 1.0) -> np.ndarray:
        n = len(x_grid)
        if n < 3:
            raise ValueError("网格点数量必须 >= 3")
        if len(charge_density) != n:
            raise ValueError("charge_density 长度必须等于网格点数")

        eps0 = 8.854187817e-12
        A = np.zeros((n, n))
        rhs = np.zeros(n)


        A[0, 0] = 1.0
        rhs[0] = 0.0
        A[n - 1, n - 1] = 1.0
        rhs[n - 1] = 0.0

        for i in range(1, n - 1):
            xm = x_grid[i]
            dx_l = xm - x_grid[i - 1]
            dx_r = x_grid[i + 1] - xm
            dx = x_grid[i + 1] - x_grid[i - 1]

            eps_m = epsilon_r

            A[i, i - 1] = -2.0 * eps_m / (dx_l * dx)
            A[i, i] = 2.0 * eps_m / (dx_l * dx_r)
            A[i, i + 1] = -2.0 * eps_m / (dx_r * dx)
            rhs[i] = charge_density[i] / eps0

        phi = np.linalg.solve(A, rhs)
        return phi
