
import numpy as np
from typing import List, Dict, Tuple
from molecular_graph import MolecularGraph


class SyntheticMoleculeDataset:

    def __init__(self, n_samples: int = 100, seed: int = 42):
        np.random.seed(seed)
        self.molecules: List[MolecularGraph] = []
        self.targets: List[Dict[str, float]] = []
        self._generate_dataset(n_samples)

    def _generate_dataset(self, n_samples: int):
        templates = self._get_templates()
        for idx in range(n_samples):
            template = templates[idx % len(templates)]

            atoms = template["atoms"].copy()
            atoms += np.random.randn(*atoms.shape) * 0.05
            bonds = template["bonds"]
            feats = template["features"]
            graph = MolecularGraph(atoms, bonds, feats)
            target = self._compute_target(graph, template["Z"])
            self.molecules.append(graph)
            self.targets.append(target)

    def _get_templates(self) -> List[Dict]:
        templates = []


        templates.append({
            "atoms": np.array([[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]]),
            "bonds": [(0, 1, 1.0)],
            "features": np.array([[1.0, 2.20, 1.20], [1.0, 2.20, 1.20]]),
            "Z": np.array([1, 1])
        })


        templates.append({
            "atoms": np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]]),
            "bonds": [(0, 1, 1.0), (0, 2, 1.0)],
            "features": np.array([[8.0, 3.44, 1.52], [1.0, 2.20, 1.20], [1.0, 2.20, 1.20]]),
            "Z": np.array([8, 1, 1])
        })


        a = 1.09 / np.sqrt(3.0)
        templates.append({
            "atoms": np.array([[0.0, 0.0, 0.0], [a, a, a], [a, -a, -a], [-a, a, -a], [-a, -a, a]]),
            "bonds": [(0, 1, 1.0), (0, 2, 1.0), (0, 3, 1.0), (0, 4, 1.0)],
            "features": np.vstack([np.array([[6.0, 2.55, 1.70]]), np.tile([1.0, 2.20, 1.20], (4, 1))]),
            "Z": np.array([6, 1, 1, 1, 1])
        })


        templates.append({
            "atoms": np.array([
                [0.0, 0.0, 0.0],
                [1.01, 0.0, 0.0],
                [-0.505, 0.875, 0.0],
                [-0.505, -0.875, 0.0]
            ]),
            "bonds": [(0, 1, 1.0), (0, 2, 1.0), (0, 3, 1.0)],
            "features": np.vstack([np.array([[7.0, 3.04, 1.55]]), np.tile([1.0, 2.20, 1.20], (3, 1))]),
            "Z": np.array([7, 1, 1, 1])
        })


        templates.append({
            "atoms": np.array([[0.0, 0.0, 0.0], [1.16, 0.0, 0.0], [-1.16, 0.0, 0.0]]),
            "bonds": [(0, 1, 2.0), (0, 2, 2.0)],
            "features": np.vstack([
                np.array([[6.0, 2.55, 1.70]]),
                np.tile([8.0, 3.44, 1.52], (2, 1))
            ]),
            "Z": np.array([6, 8, 8])
        })


        templates.append({
            "atoms": np.array([
                [0.0, 0.0, 0.0],
                [1.54, 0.0, 0.0],
                [-0.51, 0.88, 0.0],
                [-0.51, -0.44, 0.76],
                [-0.51, -0.44, -0.76],
                [2.05, 0.88, 0.0],
                [2.05, -0.44, 0.76],
                [2.05, -0.44, -0.76]
            ]),
            "bonds": [(0, 1, 1.0), (0, 2, 1.0), (0, 3, 1.0), (0, 4, 1.0),
                      (1, 5, 1.0), (1, 6, 1.0), (1, 7, 1.0)],
            "features": np.vstack([
                np.tile([6.0, 2.55, 1.70], (2, 1)),
                np.tile([1.0, 2.20, 1.20], (6, 1))
            ]),
            "Z": np.array([6, 6, 1, 1, 1, 1, 1, 1])
        })

        return templates

    def _compute_target(self, graph: MolecularGraph, Z: np.ndarray) -> Dict[str, float]:
        atoms = graph.atoms
        n = atoms.shape[0]


        E_atoms = -0.5 * np.sum(Z.astype(np.float64) ** 2)


        E_bonds = 0.0
        for (a, b, order) in graph.bonds:
            r = np.linalg.norm(atoms[a] - atoms[b])
            r_eq = 1.0
            De = 4.0 * order
            a_morse = 2.0
            E_bonds += De * (1.0 - np.exp(-a_morse * (r - r_eq))) ** 2 - De


        E_angles = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    r1 = atoms[j] - atoms[i]
                    r2 = atoms[k] - atoms[i]
                    n1 = np.linalg.norm(r1)
                    n2 = np.linalg.norm(r2)
                    if n1 > 0.5 and n2 > 0.5:
                        cos_theta = np.dot(r1, r2) / (n1 * n2)
                        cos_theta = np.clip(cos_theta, -1.0, 1.0)
                        theta = np.arccos(cos_theta)
                        theta0 = 1.91
                        E_angles += 0.5 * 0.5 * (theta - theta0) ** 2


        E_coulomb = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                r = np.linalg.norm(atoms[i] - atoms[j])
                r = max(r, 0.5)
                E_coulomb += Z[i] * Z[j] / r

        atomization_energy = E_atoms + E_bonds + E_angles + 0.1 * E_coulomb


        n_electrons = np.sum(Z)
        homo_level = -0.5 * (n_electrons / max(n, 1)) ** 2
        lumo_level = -0.5 * ((n_electrons + 1) / max(n, 1)) ** 2
        homo_lumo_gap = abs(lumo_level - homo_level)


        dipole = np.zeros(3)
        for i in range(n):
            dipole += Z[i] * atoms[i]
        dipole_moment = np.linalg.norm(dipole)

        return {
            "atomization_energy": float(atomization_energy),
            "homo_lumo_gap": float(homo_lumo_gap),
            "dipole_moment": float(dipole_moment)
        }

    def __len__(self) -> int:
        return len(self.molecules)

    def __getitem__(self, idx: int) -> Tuple[MolecularGraph, Dict[str, float], np.ndarray]:
        return self.molecules[idx], self.targets[idx], self._get_Z(self.molecules[idx])

    def _get_Z(self, graph: MolecularGraph) -> np.ndarray:
        return graph.atom_features[:, 0].astype(np.int32)

    def train_test_split(self, ratio: float = 0.8) -> Tuple[List[int], List[int]]:
        n = len(self)
        indices = np.random.permutation(n)
        split = int(n * ratio)
        return indices[:split].tolist(), indices[split:].tolist()
