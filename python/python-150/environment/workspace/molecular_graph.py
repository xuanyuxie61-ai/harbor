
import numpy as np
from typing import Tuple, List, Dict


class MolecularGraph:

    def __init__(self, atoms: np.ndarray, bonds: List[Tuple[int, int, float]],
                 atom_features: np.ndarray):
        self.atoms = np.asarray(atoms, dtype=np.float64)
        self.bonds = bonds
        self.atom_features = np.asarray(atom_features, dtype=np.float64)
        self.n_atoms = self.atoms.shape[0]
        self.n_bonds = len(bonds)


        self.adj_coo = self._build_adj_coo()
        self.degree = self._build_degree()
        self.laplacian_coo = self._build_laplacian_coo()
        self.lmax = self._estimate_lmax_power()
        self.normalized_laplacian = self._build_normalized_laplacian()


        self.incidence = self._build_incidence()
        self.transition = self._incidence_to_transition(self.incidence)


        self.adj_transition = self._build_adjacency_transition()


        self.atom_importance = self._power_rank(self.adj_transition, max_iter=100, tol=1e-10)




    def _build_adj_coo(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        i_list, j_list, v_list = [], [], []
        for (a, b, order) in self.bonds:
            r = np.linalg.norm(self.atoms[a] - self.atoms[b])
            if r < 1e-6:
                r = 1e-6
            w = order / (r ** 2)

            i_list.extend([a, b])
            j_list.extend([b, a])
            v_list.extend([w, w])
        return (np.array(i_list, dtype=np.int32),
                np.array(j_list, dtype=np.int32),
                np.array(v_list, dtype=np.float64))

    def _build_degree(self) -> np.ndarray:
        deg = np.zeros(self.n_atoms, dtype=np.float64)
        i, j, v = self.adj_coo
        for idx in range(len(v)):
            deg[i[idx]] += v[idx]

        deg = np.where(deg < 1e-12, 1e-12, deg)
        return deg




    def _build_laplacian_coo(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        i, j, v = self.adj_coo

        diag_i = np.arange(self.n_atoms, dtype=np.int32)
        diag_j = diag_i.copy()
        diag_v = self.degree

        off_v = -v
        Li = np.concatenate([diag_i, i])
        Lj = np.concatenate([diag_j, j])
        Lv = np.concatenate([diag_v, off_v])
        return Li, Lj, Lv

    def _estimate_lmax_power(self) -> float:
        x = np.random.randn(self.n_atoms)
        x = x / np.linalg.norm(x)
        Li, Lj, Lv = self.laplacian_coo
        for _ in range(80):
            y = np.zeros(self.n_atoms, dtype=np.float64)
            for idx in range(len(Lv)):
                y[Li[idx]] += Lv[idx] * x[Lj[idx]]
            norm = np.linalg.norm(y)
            if norm < 1e-15:
                break
            x = y / norm

        y = np.zeros(self.n_atoms, dtype=np.float64)
        for idx in range(len(Lv)):
            y[Li[idx]] += Lv[idx] * x[Lj[idx]]
        lam = float(np.dot(x, y))

        return max(lam * 1.01, 1e-6)

    def _build_normalized_laplacian(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        Li, Lj, Lv = self.laplacian_coo
        Lv_norm = 2.0 * Lv / self.lmax

        mask_diag = (Li == Lj)
        Lv_norm[mask_diag] -= 1.0
        return Li, Lj, Lv_norm




    def _build_incidence(self) -> np.ndarray:
        B = np.zeros((self.n_atoms, self.n_bonds), dtype=np.float64)
        for eidx, (a, b, _) in enumerate(self.bonds):
            B[a, eidx] = 1.0
            B[b, eidx] = -1.0
        return B

    @staticmethod
    def _incidence_to_transition(B: np.ndarray) -> np.ndarray:
        row_sums = np.abs(B).sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums < 1e-12, 1.0, row_sums)
        B_norm = B / row_sums
        return B_norm.T

    def _build_adjacency_transition(self) -> np.ndarray:
        T = np.zeros((self.n_atoms, self.n_atoms), dtype=np.float64)
        for idx in range(len(self.adj_coo[0])):
            i = self.adj_coo[0][idx]
            j = self.adj_coo[1][idx]
            v = self.adj_coo[2][idx]
            T[j, i] = v / self.degree[i]
        return T

    @staticmethod
    def _power_rank(T: np.ndarray, max_iter: int = 100, tol: float = 1e-10) -> np.ndarray:
        n = T.shape[0]
        x = np.ones(n, dtype=np.float64) / n
        for _ in range(max_iter):
            x_next = T.dot(x)
            norm = np.linalg.norm(x_next)
            if norm > 0:
                x_next = x_next / norm
            if np.linalg.norm(x_next - x) < tol:
                break
            x = x_next

        x = np.abs(x)
        s = x.sum()
        return x / s if s > 0 else x




    def apply_normalized_laplacian(self, x: np.ndarray) -> np.ndarray:
        Li, Lj, Lv = self.normalized_laplacian
        out = np.zeros_like(x)
        for idx in range(len(Lv)):
            out[Li[idx]] += Lv[idx] * x[Lj[idx]]
        return out

    def adjacency_dense(self) -> np.ndarray:
        A = np.zeros((self.n_atoms, self.n_atoms), dtype=np.float64)
        i, j, v = self.adj_coo
        for idx in range(len(v)):
            A[i[idx], j[idx]] = v[idx]
        return A


def build_demo_molecules() -> List[MolecularGraph]:
    molecules = []



    atoms_h2o = np.array([
        [0.0, 0.0, 0.0],
        [0.96, 0.0, 0.0],
        [-0.24, 0.93, 0.0]
    ], dtype=np.float64)
    bonds_h2o = [(0, 1, 1.0), (0, 2, 1.0)]
    feats_h2o = np.array([
        [8.0, 3.44, 1.52],
        [1.0, 2.20, 1.20],
        [1.0, 2.20, 1.20]
    ], dtype=np.float64)
    molecules.append(MolecularGraph(atoms_h2o, bonds_h2o, feats_h2o))



    a = 1.09
    atoms_ch4 = np.array([
        [0.0, 0.0, 0.0],
        [a, a, a],
        [a, -a, -a],
        [-a, a, -a],
        [-a, -a, a]
    ], dtype=np.float64) / np.sqrt(3.0)
    bonds_ch4 = [(0, 1, 1.0), (0, 2, 1.0), (0, 3, 1.0), (0, 4, 1.0)]
    feats_ch4 = np.array([
        [6.0, 2.55, 1.70],
        [1.0, 2.20, 1.20],
        [1.0, 2.20, 1.20],
        [1.0, 2.20, 1.20],
        [1.0, 2.20, 1.20]
    ], dtype=np.float64)
    molecules.append(MolecularGraph(atoms_ch4, bonds_ch4, feats_ch4))


    R = 1.40
    angles = np.linspace(0, 2 * np.pi, 7)[:-1]
    carbons = np.stack([R * np.cos(angles), R * np.sin(angles), np.zeros(6)], axis=1)
    hydrogens = np.stack([2.48 * np.cos(angles), 2.48 * np.sin(angles), np.zeros(6)], axis=1)
    atoms_benzene = np.vstack([carbons, hydrogens]).astype(np.float64)
    bonds_benzene = []

    for i in range(6):
        bonds_benzene.append((i, (i + 1) % 6, 1.5))

    for i in range(6):
        bonds_benzene.append((i, 6 + i, 1.0))
    feats_benzene = np.vstack([
        np.tile([6.0, 2.55, 1.70], (6, 1)),
        np.tile([1.0, 2.20, 1.20], (6, 1))
    ]).astype(np.float64)
    molecules.append(MolecularGraph(atoms_benzene, bonds_benzene, feats_benzene))

    return molecules
