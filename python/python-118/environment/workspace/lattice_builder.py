
import numpy as np
from utils_numeric import RandomState


class LatticeBuilder:

    def __init__(self, lattice_type="fcc", a0=3.52, rng_seed=None):
        self.lattice_type = lattice_type.lower()
        self.a0 = float(a0)
        self.rng = RandomState(seed=rng_seed)

    def build_fcc_block(self, nx, ny, nz):
        basis = np.array([
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.5]
        ]) * self.a0

        positions = []
        for ix in range(nx):
            for iy in range(ny):
                for iz in range(nz):
                    origin = np.array([ix, iy, iz]) * self.a0
                    for b in basis:
                        positions.append(origin + b)
        return np.array(positions, dtype=np.float64)

    def build_bcc_block(self, nx, ny, nz):
        basis = np.array([
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5]
        ]) * self.a0
        positions = []
        for ix in range(nx):
            for iy in range(ny):
                for iz in range(nz):
                    origin = np.array([ix, iy, iz]) * self.a0
                    for b in basis:
                        positions.append(origin + b)
        return np.array(positions, dtype=np.float64)

    def build_liquid_block(self, n_atoms, box):
        positions = np.zeros((n_atoms, 3), dtype=np.float64)
        min_dist2 = (0.7 * self.a0) ** 2
        max_trials = 1000
        for i in range(n_atoms):
            for trial in range(max_trials):
                pos = self.rng.uniform_ab(3, 0.0, 1.0) * box
                if i == 0:
                    positions[i] = pos
                    break
                diff = positions[:i] - pos
                diff -= box * np.round(diff / box)
                d2 = np.sum(diff ** 2, axis=1)
                if np.min(d2) > min_dist2:
                    positions[i] = pos
                    break
            else:

                positions[i] = self.rng.uniform_ab(3, 0.0, 1.0) * box
        return positions

    def build_solid_liquid_interface(self, n_solid_x=4, n_solid_y=4, n_solid_z=4,
                                      n_liquid_z=4, y_span=None, z_span=None,
                                      concentration_b=0.15, randomize_solute=True):
        if self.lattice_type == "fcc":
            solid_pos = self.build_fcc_block(n_solid_x, n_solid_y, n_solid_z)
        elif self.lattice_type == "bcc":
            solid_pos = self.build_bcc_block(n_solid_x, n_solid_y, n_solid_z)
        else:
            solid_pos = self.build_fcc_block(n_solid_x, n_solid_y, n_solid_z)


        box_x = n_solid_x * self.a0
        if y_span is None:
            box_y = n_solid_y * self.a0
        else:
            box_y = y_span
        if z_span is None:
            box_z = (n_solid_z + n_liquid_z) * self.a0
        else:
            box_z = z_span
        box = np.array([box_x, box_y, box_z], dtype=np.float64)



        n_solid = solid_pos.shape[0]
        n_liquid = int(n_solid * (n_liquid_z / n_solid_z))
        liquid_pos = self.build_liquid_block(n_liquid, box)

        liquid_pos[:, 2] += n_solid_z * self.a0

        positions = np.vstack([solid_pos, liquid_pos])
        n_total = positions.shape[0]


        is_solid = np.zeros(n_total, dtype=bool)
        is_solid[:n_solid] = True


        species_idx = np.zeros(n_total, dtype=np.int32)
        if randomize_solute:

            n_solute_solid = int(concentration_b * n_solid)
            solid_indices = np.arange(n_solid)
            rng = np.random.default_rng(42)
            solute_solid = rng.choice(solid_indices, size=n_solute_solid, replace=False)
            species_idx[solute_solid] = 1


            liquid_conc = min(concentration_b * 1.5, 0.5)
            n_solute_liquid = int(liquid_conc * n_liquid)
            liquid_indices = np.arange(n_solid, n_total)
            solute_liquid = rng.choice(liquid_indices, size=n_solute_liquid, replace=False)
            species_idx[solute_liquid] = 1
        else:
            n_solute = int(concentration_b * n_total)
            rng = np.random.default_rng(42)
            solute_idx = rng.choice(np.arange(n_total), size=n_solute, replace=False)
            species_idx[solute_idx] = 1

        return positions, species_idx, box, is_solid

    def build_neighbor_list(self, positions, box, r_cut):
        n_atoms = positions.shape[0]
        r_cut2 = r_cut ** 2
        neighbors = []
        for i in range(n_atoms):
            neigh = []
            for j in range(n_atoms):
                if i == j:
                    continue
                rij = positions[j] - positions[i]
                rij -= box * np.round(rij / box)
                r2 = np.dot(rij, rij)
                if r2 < r_cut2:
                    neigh.append((j, np.sqrt(r2), rij))
            neighbors.append(neigh)
        return neighbors

    def apply_thermal_displacement(self, positions, T, masses_amu, species_idx):
        kb = 8.617333e-5

        k_spring = 20.0
        u_rms = np.sqrt(3.0 * kb * T / k_spring)
        n_atoms = positions.shape[0]
        disp = self.rng.maxwell_boltzmann(n_atoms, T=0.01, m=1.0) * u_rms / 0.1
        positions_new = positions + disp
        return positions_new
