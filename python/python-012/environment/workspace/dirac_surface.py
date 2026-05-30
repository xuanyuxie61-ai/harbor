
import numpy as np


HBAR = 1.054571817e-34
EV_TO_J = 1.602176634e-19
NM_TO_M = 1e-9


class DiracSurfaceHamiltonian:

    def __init__(self, v_F=5.0e5, Delta=0.05, lambda_w=0.0, hbar=HBAR):
        if v_F <= 0:
            raise ValueError("Fermi velocity v_F must be positive.")
        self.v_F = v_F
        self.Delta = Delta * EV_TO_J
        self.lambda_w = lambda_w * EV_TO_J * (NM_TO_M ** 3)
        self.hbar = hbar

    def _pauli_matrices(self):
        sigma_x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
        sigma_y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
        sigma_z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
        return sigma_x, sigma_y, sigma_z

    def hamiltonian(self, kx, ky):
        kx = np.asarray(kx, dtype=float)
        ky = np.asarray(ky, dtype=float)
        shape = kx.shape if kx.shape else ()

        sigma_x, sigma_y, sigma_z = self._pauli_matrices()


        H = (self.hbar * self.v_F * kx[..., None, None]) * sigma_y \
            - (self.hbar * self.v_F * ky[..., None, None]) * sigma_x \
            + self.Delta * sigma_z


        if abs(self.lambda_w) > 1e-40:
            k_plus = kx + 1.0j * ky
            k_minus = kx - 1.0j * ky
            warping = self.lambda_w * (k_plus ** 3 + k_minus ** 3)
            H = H + warping[..., None, None] * sigma_z

        return H

    def eigenvalues(self, kx, ky):
        kx = np.asarray(kx, dtype=float)
        ky = np.asarray(ky, dtype=float)
        k_sq = kx ** 2 + ky ** 2
        k = np.sqrt(k_sq)

        kinetic = (self.hbar * self.v_F * k) ** 2

        gap_term = self.Delta
        if abs(self.lambda_w) > 1e-40:
            k_plus = kx + 1.0j * ky
            k_minus = kx - 1.0j * ky
            gap_term = gap_term + self.lambda_w * (k_plus ** 3 + k_minus ** 3)

        E = np.sqrt(kinetic + np.abs(gap_term) ** 2)
        return E, -E

    def eigenvectors(self, kx, ky):
        H = self.hamiltonian(kx, ky)
        shape = H.shape[:-2]
        H_flat = H.reshape(-1, 2, 2)

        psi_plus = np.empty((H_flat.shape[0], 2), dtype=complex)
        psi_minus = np.empty((H_flat.shape[0], 2), dtype=complex)

        for i in range(H_flat.shape[0]):
            w, v = np.linalg.eigh(H_flat[i])
            psi_minus[i] = v[:, 0]
            psi_plus[i] = v[:, 1]

        psi_plus = psi_plus.reshape(shape + (2,))
        psi_minus = psi_minus.reshape(shape + (2,))
        return psi_plus, psi_minus

    def spin_texture(self, kx, ky, band='upper'):
        if band == 'upper':
            psi, _ = self.eigenvectors(kx, ky)
        else:
            _, psi = self.eigenvectors(kx, ky)

        sigma_x, sigma_y, sigma_z = self._pauli_matrices()

        Sx = 0.5 * self.hbar * np.real(
            np.einsum('...i,ij,...j->...', psi.conj(), sigma_x, psi)
        )
        Sy = 0.5 * self.hbar * np.real(
            np.einsum('...i,ij,...j->...', psi.conj(), sigma_y, psi)
        )
        Sz = 0.5 * self.hbar * np.real(
            np.einsum('...i,ij,...j->...', psi.conj(), sigma_z, psi)
        )
        return Sx, Sy, Sz

    def velocity_operator(self, kx, ky):
        _, sigma_y, sigma_z = self._pauli_matrices()
        sigma_x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)

        vx = self.v_F * sigma_y
        vy = -self.v_F * sigma_x

        if abs(self.lambda_w) > 1e-40:
            kx = np.asarray(kx, dtype=float)
            ky = np.asarray(ky, dtype=float)

            vx = vx + (3.0 * self.lambda_w / self.hbar) * (kx ** 2 - ky ** 2) * sigma_z
            vy = vy - (6.0 * self.lambda_w / self.hbar) * kx * ky * sigma_z

        return vx, vy


def effective_mass_tensor(kx, ky, v_F=5.0e5, Delta=0.05):
    Delta_J = Delta * EV_TO_J
    m_inv = np.zeros((2, 2), dtype=float)
    if abs(Delta_J) < 1e-30:
        return m_inv
    m_inv[0, 0] = v_F ** 2 / Delta_J
    m_inv[1, 1] = v_F ** 2 / Delta_J
    return m_inv
