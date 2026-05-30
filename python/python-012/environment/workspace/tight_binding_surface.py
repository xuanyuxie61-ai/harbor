
import numpy as np
from scipy import linalg as sla


HBAR = 1.054571817e-34
EV_TO_J = 1.602176634e-19
NM_TO_M = 1e-9


class TightBindingSurface:

    def __init__(self, Nx=40, Ny=40, a=2.0, v_F=5.0e5, Delta=0.05,
                 boundary='periodic'):
        if Nx < 2 or Ny < 2:
            raise ValueError("Nx and Ny must be at least 2.")
        self.Nx = Nx
        self.Ny = Ny
        self.a = a * NM_TO_M
        self.v_F = v_F
        self.Delta = Delta * EV_TO_J
        self.boundary = boundary
        self.N_sites = Nx * Ny
        self.N_states = 2 * self.N_sites

    def _site_index(self, ix, iy):
        if self.boundary == 'periodic':
            ix = ix % self.Nx
            iy = iy % self.Ny
        else:
            if ix < 0 or ix >= self.Nx or iy < 0 or iy >= self.Ny:
                return -1
        return ix + iy * self.Nx

    def _pauli(self):
        sx = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
        sy = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
        sz = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
        i2 = np.eye(2, dtype=complex)
        return sx, sy, sz, i2

    def build_hamiltonian(self, disorder_potential=None):
        N = self.N_states
        H = np.zeros((N, N), dtype=complex)
        sx, sy, sz, i2 = self._pauli()

        t_x = 1.0j * HBAR * self.v_F / (2.0 * self.a) * sy
        t_y = -1.0j * HBAR * self.v_F / (2.0 * self.a) * sx

        for iy in range(self.Ny):
            for ix in range(self.Nx):
                i = self._site_index(ix, iy)
                if i < 0:
                    continue
                i_spin = 2 * i


                H[i_spin:i_spin + 2, i_spin:i_spin + 2] += self.Delta * sz


                if disorder_potential is not None:
                    V = disorder_potential[ix, iy] * EV_TO_J
                    H[i_spin:i_spin + 2, i_spin:i_spin + 2] += V * i2


                j = self._site_index(ix + 1, iy)
                if j >= 0:
                    j_spin = 2 * j
                    H[i_spin:i_spin + 2, j_spin:j_spin + 2] += t_x
                    H[j_spin:j_spin + 2, i_spin:i_spin + 2] += t_x.T.conj()


                j = self._site_index(ix, iy + 1)
                if j >= 0:
                    j_spin = 2 * j
                    H[i_spin:i_spin + 2, j_spin:j_spin + 2] += t_y
                    H[j_spin:j_spin + 2, i_spin:i_spin + 2] += t_y.T.conj()

        return H

    def build_neighbor_matrix(self):
        N = self.N_sites
        adj = np.zeros((N, N), dtype=int)
        for iy in range(self.Ny):
            for ix in range(self.Nx):
                i = self._site_index(ix, iy)
                if i < 0:
                    continue

                j = self._site_index(ix + 1, iy)
                if j >= 0:
                    adj[i, j] = 1
                    adj[j, i] = 1

                j = self._site_index(ix, iy + 1)
                if j >= 0:
                    adj[i, j] = 1
                    adj[j, i] = 1
        return adj

    def diagonalize(self, disorder_potential=None):
        H = self.build_hamiltonian(disorder_potential)
        energies, eigenvectors = np.linalg.eigh(H)
        return energies, eigenvectors

    def local_density_of_states(self, energies, eigenvectors, E, eta=1e-22,
                                site=None):
        delta = (1.0 / np.pi) * eta / ((energies - E) ** 2 + eta ** 2)
        if site is not None:
            weights = np.abs(eigenvectors[2 * site, :]) ** 2 \
                      + np.abs(eigenvectors[2 * site + 1, :]) ** 2
        else:
            weights = np.ones(energies.shape[0])
        ldos = np.sum(weights * delta)
        return ldos

    def edge_state_probability(self, eigenvectors, band_index):
        psi = eigenvectors[:, band_index]
        profile = np.zeros((self.Nx, self.Ny))
        for iy in range(self.Ny):
            for ix in range(self.Nx):
                i = self._site_index(ix, iy)
                prob = np.abs(psi[2 * i]) ** 2 + np.abs(psi[2 * i + 1]) ** 2
                profile[ix, iy] = prob


        edge_mask = (np.arange(self.Nx) < 2) | (np.arange(self.Nx) >= self.Nx - 2)
        edge_weight = np.sum(profile[edge_mask, :]) / np.sum(profile)
        return profile, edge_weight

    def current_operator(self, direction='x'):
        N = self.N_states
        J = np.zeros((N, N), dtype=complex)
        sx, sy, sz, i2 = self._pauli()
        e_charge = 1.602176634e-19

        if direction == 'x':
            t = 1.0j * HBAR * self.v_F / (2.0 * self.a) * sy
        else:
            t = -1.0j * HBAR * self.v_F / (2.0 * self.a) * sx

        for iy in range(self.Ny):
            for ix in range(self.Nx):
                i = self._site_index(ix, iy)
                if i < 0:
                    continue
                i_spin = 2 * i
                if direction == 'x':
                    j = self._site_index(ix + 1, iy)
                else:
                    j = self._site_index(ix, iy + 1)
                if j >= 0:
                    j_spin = 2 * j
                    J[i_spin:i_spin + 2, j_spin:j_spin + 2] += \
                        (1.0j * e_charge / HBAR) * t
                    J[j_spin:j_spin + 2, i_spin:i_spin + 2] += \
                        -(1.0j * e_charge / HBAR) * t.T.conj()
        return J

    def finite_size_conductivity(self, E_F, T=0.0, eta=1e-22):
        E_F_J = E_F * EV_TO_J
        energies, evecs = self.diagonalize()
        Jx = self.current_operator(direction='x')

        N = len(energies)
        sigma = 0.0
        area = self.Nx * self.Ny * self.a ** 2


        if T < 1e-6:
            f = np.where(energies < E_F_J, 1.0, 0.0)
        else:
            beta = 1.0 / (1.380649e-23 * T)
            f = 1.0 / (1.0 + np.exp(beta * (energies - E_F_J)))

        for n in range(N):
            for m in range(n + 1, N):
                dE = energies[m] - energies[n]
                if abs(dE) < 1e-30:
                    continue
                J_nm = np.vdot(evecs[:, n], Jx @ evecs[:, m])
                weight = (f[n] - f[m]) / dE
                delta = (1.0 / np.pi) * eta / (dE ** 2 + eta ** 2)
                sigma += abs(J_nm) ** 2 * weight * delta

        sigma_xx = (2.0 * np.pi / area) * sigma
        return sigma_xx
