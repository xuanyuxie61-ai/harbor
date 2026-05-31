"""
Tight-Binding Lattice Model for TI Surface States
=================================================
Constructs a 2D tight-binding Hamiltonian on a square lattice
that reproduces the low-energy Dirac physics of TI surface states.

The continuum Dirac Hamiltonian:
    H(k) = hbar v_F (k_x sigma_y - k_y sigma_x) + Delta sigma_z

is discretized on a 2D lattice with spacing 'a':
    k_x -> sin(k_x a) / a,    k_y -> sin(k_y a) / a

The tight-binding Hamiltonian on a square lattice:
    H_{ij} = sum_{<ij>} t_{ij} c_i^dagger c_j + Delta_i sigma_z + ...

Nearest-neighbor hopping:
    t_x = i * hbar v_F / (2a) * sigma_y
    t_y = -i * hbar v_F / (2a) * sigma_x

On-site term:
    H_0 = Delta * sigma_z

This model is useful for finite-size systems and studying edge states.
The neighbor-coupling structure is inspired by the Lights Out matrix (project 672).
"""

import numpy as np
from scipy import linalg as sla

# Physical constants
HBAR = 1.054571817e-34
EV_TO_J = 1.602176634e-19
NM_TO_M = 1e-9


class TightBindingSurface:
    """
    2D tight-binding model for TI surface states.
    """

    def __init__(self, Nx=40, Ny=40, a=2.0, v_F=5.0e5, Delta=0.05,
                 boundary='periodic'):
        """
        Parameters
        ----------
        Nx, Ny : int
            Lattice dimensions (must be >= 2).
        a : float
            Lattice spacing in nm.
        v_F : float
            Fermi velocity in m/s.
        Delta : float
            Exchange gap in eV.
        boundary : str
            'periodic' or 'open'.
        """
        if Nx < 2 or Ny < 2:
            raise ValueError("Nx and Ny must be at least 2.")
        self.Nx = Nx
        self.Ny = Ny
        self.a = a * NM_TO_M
        self.v_F = v_F
        self.Delta = Delta * EV_TO_J
        self.boundary = boundary
        self.N_sites = Nx * Ny
        self.N_states = 2 * self.N_sites  # spin-1/2 at each site

    def _site_index(self, ix, iy):
        """
        Map 2D lattice coordinates to 1D site index.

        Inspired by lo_matrix_index from project 672 (lights_out).
        Handles periodic or open boundary conditions.
        """
        if self.boundary == 'periodic':
            ix = ix % self.Nx
            iy = iy % self.Ny
        else:
            if ix < 0 or ix >= self.Nx or iy < 0 or iy >= self.Ny:
                return -1
        return ix + iy * self.Nx

    def _pauli(self):
        """Pauli matrices."""
        sx = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
        sy = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
        sz = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
        i2 = np.eye(2, dtype=complex)
        return sx, sy, sz, i2

    def build_hamiltonian(self, disorder_potential=None):
        """
        Build the full tight-binding Hamiltonian matrix.

        H = sum_i Delta * sigma_z ( onsite )
            + sum_{<i,j>_x} t_x * c_i^dagger c_j
            + sum_{<i,j>_y} t_y * c_i^dagger c_j

        where t_x = i * hbar v_F / (2a) * sigma_y
              t_y = -i * hbar v_F / (2a) * sigma_x

        Parameters
        ----------
        disorder_potential : ndarray, optional
            Shape (Nx, Ny) array of on-site disorder in eV.

        Returns
        -------
        H : ndarray
            Shape (2*Nx*Ny, 2*Nx*Ny) complex Hamiltonian.
        """
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

                # On-site gap
                H[i_spin:i_spin + 2, i_spin:i_spin + 2] += self.Delta * sz

                # Disorder
                if disorder_potential is not None:
                    V = disorder_potential[ix, iy] * EV_TO_J
                    H[i_spin:i_spin + 2, i_spin:i_spin + 2] += V * i2

                # Nearest-neighbor hopping in x direction
                j = self._site_index(ix + 1, iy)
                if j >= 0:
                    j_spin = 2 * j
                    H[i_spin:i_spin + 2, j_spin:j_spin + 2] += t_x
                    H[j_spin:j_spin + 2, i_spin:i_spin + 2] += t_x.T.conj()

                # Nearest-neighbor hopping in y direction
                j = self._site_index(ix, iy + 1)
                if j >= 0:
                    j_spin = 2 * j
                    H[i_spin:i_spin + 2, j_spin:j_spin + 2] += t_y
                    H[j_spin:j_spin + 2, i_spin:i_spin + 2] += t_y.T.conj()

        return H

    def build_neighbor_matrix(self):
        """
        Build the connectivity matrix (adjacency) of the lattice.

        Returns
        -------
        adj : ndarray
            Shape (Nx*Ny, Nx*Ny) integer adjacency matrix.
            adj[i,j] = 1 if sites i and j are neighbors.
        """
        N = self.N_sites
        adj = np.zeros((N, N), dtype=int)
        for iy in range(self.Ny):
            for ix in range(self.Nx):
                i = self._site_index(ix, iy)
                if i < 0:
                    continue
                # x-neighbor
                j = self._site_index(ix + 1, iy)
                if j >= 0:
                    adj[i, j] = 1
                    adj[j, i] = 1
                # y-neighbor
                j = self._site_index(ix, iy + 1)
                if j >= 0:
                    adj[i, j] = 1
                    adj[j, i] = 1
        return adj

    def diagonalize(self, disorder_potential=None):
        """
        Diagonalize the tight-binding Hamiltonian.

        Returns
        -------
        energies : ndarray
            Eigenvalues in Joules.
        eigenvectors : ndarray
            Eigenvectors as columns.
        """
        H = self.build_hamiltonian(disorder_potential)
        energies, eigenvectors = np.linalg.eigh(H)
        return energies, eigenvectors

    def local_density_of_states(self, energies, eigenvectors, E, eta=1e-22,
                                site=None):
        """
        Compute the local density of states at a given energy:

            rho_i(E) = sum_n |<i|n>|^2 * delta(E - E_n)

        Parameters
        ----------
        energies : ndarray
        eigenvectors : ndarray
        E : float
            Energy in Joules.
        eta : float
            Broadening.
        site : int, optional
            Site index. If None, sum over all sites.

        Returns
        -------
        ldos : float
        """
        delta = (1.0 / np.pi) * eta / ((energies - E) ** 2 + eta ** 2)
        if site is not None:
            weights = np.abs(eigenvectors[2 * site, :]) ** 2 \
                      + np.abs(eigenvectors[2 * site + 1, :]) ** 2
        else:
            weights = np.ones(energies.shape[0])
        ldos = np.sum(weights * delta)
        return ldos

    def edge_state_probability(self, eigenvectors, band_index):
        """
        Compute the spatial profile of an eigenstate to identify edge states.

        Edge states are localized near the boundaries x=0 or x=Nx-1.

        Parameters
        ----------
        eigenvectors : ndarray
        band_index : int

        Returns
        -------
        profile : ndarray
            Shape (Nx, Ny) probability density.
        edge_weight : float
            Fraction of weight within 2 lattice spacings of the boundary.
        """
        psi = eigenvectors[:, band_index]
        profile = np.zeros((self.Nx, self.Ny))
        for iy in range(self.Ny):
            for ix in range(self.Nx):
                i = self._site_index(ix, iy)
                prob = np.abs(psi[2 * i]) ** 2 + np.abs(psi[2 * i + 1]) ** 2
                profile[ix, iy] = prob

        # Edge weight
        edge_mask = (np.arange(self.Nx) < 2) | (np.arange(self.Nx) >= self.Nx - 2)
        edge_weight = np.sum(profile[edge_mask, :]) / np.sum(profile)
        return profile, edge_weight

    def current_operator(self, direction='x'):
        """
        Build the current operator J = (e / hbar) dH/dk.

        On the lattice, the current in direction 'x' is:
            J_x = (i*e / hbar) * sum_i (t_x c_{i+x}^dagger c_i - h.c.)

        Parameters
        ----------
        direction : str
            'x' or 'y'.

        Returns
        -------
        J : ndarray
            Current operator matrix.
        """
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
        """
        Compute the finite-size conductivity using the Kubo formula
        on the tight-binding lattice:

            sigma_{xx} = (2*pi / Omega) * sum_{nm} |
                         <n|J_x|m>|^2 * (f_n - f_m) / (E_n - E_m)
                         * delta(E_n - E_m)

        Parameters
        ----------
        E_F : float
            Fermi energy in eV.
        T : float
            Temperature in K.
        eta : float
            Broadening for delta function.

        Returns
        -------
        sigma_xx : float
            Conductivity in S.
        """
        E_F_J = E_F * EV_TO_J
        energies, evecs = self.diagonalize()
        Jx = self.current_operator(direction='x')

        N = len(energies)
        sigma = 0.0
        area = self.Nx * self.Ny * self.a ** 2

        # Fermi-Dirac
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
