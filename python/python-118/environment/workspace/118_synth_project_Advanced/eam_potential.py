
import numpy as np
from utils_numeric import check_bounds, safe_sqrt


class EAMPotential:

    def __init__(self, species_a="A", species_b="B",
                 D_aa=0.50, D_bb=0.35, D_ab=0.42,
                 alpha_aa=1.60, alpha_bb=1.45, alpha_ab=1.52,
                 r0_aa=2.56, r0_bb=2.78, r0_ab=2.67,
                 A_a=0.80, A_b=0.65, B_a=-0.02, B_b=-0.015,
                 f_a=1.0, f_b=0.85, beta_a=3.0, beta_b=2.8,
                 r_cut=5.5, mass_a=58.69, mass_b=63.55):
        self.species = [species_a, species_b]
        self.r_cut = float(r_cut)
        self.r_cut2 = self.r_cut ** 2
        self.mass = np.array([mass_a, mass_b], dtype=np.float64)


        self.D_mat = np.array([[D_aa, D_ab], [D_ab, D_bb]], dtype=np.float64)
        self.alpha_mat = np.array([[alpha_aa, alpha_ab], [alpha_ab, alpha_bb]], dtype=np.float64)
        self.r0_mat = np.array([[r0_aa, r0_ab], [r0_ab, r0_bb]], dtype=np.float64)


        self.A_emb = np.array([A_a, A_b], dtype=np.float64)
        self.B_emb = np.array([B_a, B_b], dtype=np.float64)


        self.f_rho = np.array([f_a, f_b], dtype=np.float64)
        self.beta_rho = np.array([beta_a, beta_b], dtype=np.float64)


        self._compute_cutoff_smoothing()

    def _compute_cutoff_smoothing(self):
        pass

    def pair_potential(self, r, si, sj):
        r = check_bounds(r, 1e-8, self.r_cut, name="pair_distance")
        D = self.D_mat[si, sj]
        alpha = self.alpha_mat[si, sj]
        r0 = self.r0_mat[si, sj]
        x = np.exp(-alpha * (r - r0))
        phi = D * (x * x - 2.0 * x)

        s = (1.0 - r / self.r_cut) ** 3
        s = np.where(r < self.r_cut, s, 0.0)
        return phi * s

    def pair_force_magnitude(self, r, si, sj):
        r = check_bounds(r, 1e-8, self.r_cut, name="pair_distance")
        D = self.D_mat[si, sj]
        alpha = self.alpha_mat[si, sj]
        r0 = self.r0_mat[si, sj]
        x = np.exp(-alpha * (r - r0))
        dphi_dr = D * (-2.0 * alpha * x * x + 2.0 * alpha * x)

        s = (1.0 - r / self.r_cut) ** 3
        ds_dr = -3.0 * (1.0 - r / self.r_cut) ** 2 / self.r_cut
        phi = D * (x * x - 2.0 * x)
        mask = r < self.r_cut
        total = np.where(mask, dphi_dr * s + phi * ds_dr, 0.0)
        return -total

    def electron_density(self, r, sj):
        r = check_bounds(r, 1e-8, self.r_cut, name="rho_distance")
        f = self.f_rho[sj]
        beta = self.beta_rho[sj]
        val = f * (r / self.r_cut) ** 6 * np.exp(-beta * (r - self.r_cut))
        mask = r < self.r_cut
        return np.where(mask, val, 0.0)

    def embedding_energy(self, rho_bar, si):
        rho_bar = np.maximum(rho_bar, 1e-12)
        return self.A_emb[si] * safe_sqrt(rho_bar) + self.B_emb[si] * rho_bar ** 2

    def embedding_derivative(self, rho_bar, si):
        rho_bar = np.maximum(rho_bar, 1e-12)
        return 0.5 * self.A_emb[si] / safe_sqrt(rho_bar) + 2.0 * self.B_emb[si] * rho_bar

    def compute_forces_and_energies(self, positions, species_idx, box):
        n_atoms = positions.shape[0]
        forces = np.zeros_like(positions)
        energies = np.zeros(n_atoms, dtype=np.float64)


        rho_bar = np.zeros(n_atoms, dtype=np.float64)


        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                rij = positions[j] - positions[i]
                rij -= box * np.round(rij / box)
                r2 = np.dot(rij, rij)
                if r2 < self.r_cut2:
                    r = safe_sqrt(r2)
                    sj = species_idx[j]
                    si = species_idx[i]
                    rho_bar[i] += self.electron_density(r, sj)
                    rho_bar[j] += self.electron_density(r, si)


        for i in range(n_atoms):
            si = species_idx[i]
            energies[i] += self.embedding_energy(rho_bar[i], si)


        virial = 0.0
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                rij = positions[j] - positions[i]
                rij -= box * np.round(rij / box)
                r2 = np.dot(rij, rij)
                if r2 < self.r_cut2:
                    r = safe_sqrt(r2)
                    si = species_idx[i]
                    sj = species_idx[j]

                    phi = self.pair_potential(r, si, sj)
                    energies[i] += 0.5 * phi
                    energies[j] += 0.5 * phi


                    f_pair = self.pair_force_magnitude(r, si, sj)
                    f_vec = f_pair * rij / r
                    forces[i] += f_vec
                    forces[j] -= f_vec
                    virial += np.dot(rij, f_vec)


                    dF_drho_i = self.embedding_derivative(rho_bar[i], si)
                    dF_drho_j = self.embedding_derivative(rho_bar[j], sj)

                    f_rho = self.f_rho[sj]
                    beta = self.beta_rho[sj]
                    s = (1.0 - r / self.r_cut) ** 3
                    ds = -3.0 * (1.0 - r / self.r_cut) ** 2 / self.r_cut
                    rho_val = f_rho * (r / self.r_cut) ** 6 * np.exp(-beta * (r - self.r_cut))
                    drho_j = (6.0 / r - beta) * rho_val * s + rho_val * ds
                    drho_j = np.where(r < self.r_cut, drho_j, 0.0)

                    f_rho_i = self.f_rho[si]
                    beta_i = self.beta_rho[si]
                    rho_val_i = f_rho_i * (r / self.r_cut) ** 6 * np.exp(-beta_i * (r - self.r_cut))
                    drho_i = (6.0 / r - beta_i) * rho_val_i * s + rho_val_i * ds
                    drho_i = np.where(r < self.r_cut, drho_i, 0.0)

                    f_emb = (dF_drho_i * drho_j + dF_drho_j * drho_i) * rij / r
                    forces[i] += f_emb
                    forces[j] -= f_emb
                    virial += np.dot(rij, f_emb)

        total_energy = np.sum(energies)
        return total_energy, forces, virial


def eam_parameterized_alloy(species_pair=("Ni", "Cu")):
    if species_pair == ("Ni", "Cu"):
        return EAMPotential(
            species_a="Ni", species_b="Cu",
            D_aa=0.4207, D_bb=0.3429, D_ab=0.3818,
            alpha_aa=1.5590, alpha_bb=1.3885, alpha_ab=1.4738,
            r0_aa=2.7800, r0_bb=2.5560, r0_ab=2.6180,
            A_a=1.100, A_b=0.900, B_a=-0.018, B_b=-0.012,
            f_a=1.00, f_b=0.82, beta_a=2.80, beta_b=2.60,
            r_cut=5.50, mass_a=58.69, mass_b=63.55
        )
    else:
        return EAMPotential()
