
import numpy as np
from typing import List, Tuple, Optional
from utils import (
    PT_LATTICE_CONSTANT, grid_uniform_nd, safe_divide,
    BOLTZMANN_KB, ELEMENTARY_CHARGE
)


class Pt111Surface:

    def __init__(self, nx: int = 6, ny: int = 6, n_layers: int = 3):
        if nx < 2 or ny < 2 or n_layers < 1:
            raise ValueError("nx, ny >= 2 且 n_layers >= 1")
        self.nx = nx
        self.ny = ny
        self.n_layers = n_layers
        self.a = PT_LATTICE_CONSTANT
        self.d_nn = self.a / np.sqrt(2.0)
        self.atoms = self._build_fcc_111()
        self.n_atoms = self.atoms.shape[0]

        self.sites = self._build_adsorption_sites()
        self.n_sites = self.sites.shape[0]

        self.site_occupancy = np.zeros(self.n_sites, dtype=int)

        self.site_energies = self._compute_site_energies()

    def _build_fcc_111(self) -> np.ndarray:
        d = self.d_nn
        dz = self.a / np.sqrt(3.0)
        atoms = []
        for layer in range(self.n_layers):
            z = -layer * dz
            offset = (layer % 3) * d / 3.0
            for i in range(self.nx):
                for j in range(self.ny):
                    x = i * d + (j % 2) * d * 0.5 + offset
                    y = j * d * np.sqrt(3.0) / 2.0
                    atoms.append([x, y, z])
        return np.array(atoms, dtype=float)

    def _build_adsorption_sites(self) -> np.ndarray:
        d = self.d_nn
        sites = []

        z_top = 1.5e-10
        for i in range(self.nx):
            for j in range(self.ny):
                x0 = i * d + (j % 2) * d * 0.5
                y0 = j * d * np.sqrt(3.0) / 2.0

                sites.append([x0, y0, z_top, 0])

                sites.append([x0 + d * 0.5, y0, z_top, 1])

                sites.append([x0 + d * 0.5, y0 + d * np.sqrt(3.0) / 6.0, z_top, 2])

                sites.append([x0 + d * 0.5, y0 - d * np.sqrt(3.0) / 6.0, z_top, 3])
        return np.array(sites, dtype=float)

    def _compute_site_energies(self) -> np.ndarray:
        energies = np.zeros(self.n_sites)
        site_types = self.sites[:, 3].astype(int)

        co_energy = {0: -1.3, 1: -1.0, 2: -0.8, 3: -0.7}
        for i, st in enumerate(site_types):
            energies[i] = co_energy.get(st, -0.5)
        return energies

    def get_surface_atoms(self) -> np.ndarray:
        return self.atoms[self.atoms[:, 2] == 0.0]

    def find_nearest_site(self, pos: np.ndarray) -> int:
        pos = np.asarray(pos, dtype=float)
        if pos.shape != (3,):
            raise ValueError("pos 必须为 3D 坐标")
        diffs = self.sites[:, :3] - pos
        dists = np.sqrt(np.sum(diffs ** 2, axis=1))
        return int(np.argmin(dists))

    def update_occupancy_ca(self, rule: int = 30, steps: int = 1):
        if steps < 0:
            raise ValueError("steps >= 0")
        occ = self.site_occupancy.copy()
        n = self.n_sites
        for _ in range(steps):
            new_occ = np.zeros(n, dtype=int)
            for j in range(1, n - 1):
                left = occ[j - 1]
                center = occ[j]
                right = occ[j + 1]

                pattern = (left << 2) | (center << 1) | right

                rule_table = [0, 1, 1, 1, 1, 0, 0, 0]
                new_occ[j] = rule_table[pattern]
            occ = new_occ
        self.site_occupancy = occ

    def cvt_optimize_sites(self, n_generators: int = 8,
                           it_num: int = 20,
                           density_exp: float = 5.0) -> np.ndarray:
        if n_generators < 4:
            raise ValueError("n_generators >= 4")
        if it_num < 1:
            raise ValueError("it_num >= 1")


        L = self.nx * self.d_nn * 0.5
        rng = np.random.default_rng(seed=42)
        g = rng.uniform(-L, L, size=(n_generators, 3))
        g[:, 2] = rng.uniform(0.3e-10, L, n_generators)


        s_num_1d = 20
        eps_margin = 1e-12
        s_1d = np.linspace(-L + eps_margin, L - eps_margin, s_num_1d)
        sx, sy, sz = np.meshgrid(s_1d, s_1d, s_1d, indexing='ij')
        s = np.vstack([sx.ravel(), sy.ravel(), sz.ravel()]).T
        n_samples = s.shape[0]


        def mu_density(xyz):
            dist_to_surface = np.abs(xyz[:, 2])
            mu = 1.0 / (dist_to_surface + 0.5e-10)
            return np.clip(mu, 0.0, 10.0)

        mu_vals = mu_density(s)
        rho_vals = mu_vals ** density_exp

        for it in range(it_num):

            k_idx = np.zeros(n_samples, dtype=int)
            for j in range(n_samples):
                dists = np.sum((s[j] - g) ** 2, axis=1)
                k_idx[j] = int(np.argmin(dists))


            g_new = np.zeros_like(g)
            for i in range(n_generators):
                mask = k_idx == i
                if np.any(mask):
                    mass = np.sum(rho_vals[mask])
                    if mass > 1e-300:
                        g_new[i] = np.sum(rho_vals[mask][:, None] * s[mask], axis=0) / mass
                    else:
                        g_new[i] = g[i]
                else:
                    g_new[i] = g[i]
            g = g_new

            g[:, 2] = np.maximum(g[:, 2], 0.2e-10)

        return g

    def surface_coverage(self, species: int = 1) -> float:
        if self.n_sites == 0:
            return 0.0
        count = np.sum(self.site_occupancy == species)
        return float(count) / self.n_sites

    def compute_lateral_interaction(self, interaction_strength: float = 0.1) -> np.ndarray:
        n = self.n_sites
        e_int = np.zeros(n)
        pos = self.sites[:, :3]
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                r_ij = np.linalg.norm(pos[i] - pos[j])
                if r_ij > 1e-12:
                    e_int[i] += interaction_strength * self.site_occupancy[j] / r_ij
        return e_int

    def dump_site_info(self):
        print("=" * 60)
        print("Pt(111) 表面结构参数")
        print("=" * 60)
        print(f"  晶格常数 a          = {self.a * 1e10:.4f} Å")
        print(f"  最近邻间距 d_nn     = {self.d_nn * 1e10:.4f} Å")
        print(f"  表面原子数          = {self.n_atoms}")
        print(f"  吸附位点数          = {self.n_sites}")
        print(f"  CO 覆盖率           = {self.surface_coverage(species=1):.4f}")
        print(f"  O 覆盖率            = {self.surface_coverage(species=2):.4f}")
        print("=" * 60)
