"""
Dirac Surface Hamiltonian for 3D Topological Insulator
=======================================================
Implements the low-energy effective Hamiltonian for the surface states
of a 3D topological insulator (e.g., Bi2Se3, Bi2Te3, Sb2Te3).

The surface Hamiltonian in the continuum limit is:
    H_0(k) = hbar * v_F * (k_x * sigma_y - k_y * sigma_x) + Delta * sigma_z

where sigma_x, sigma_y, sigma_z are Pauli matrices, v_F is the Fermi velocity,
and Delta is the exchange gap induced by magnetic doping (time-reversal
symmetry breaking).

With hexagonal warping (relevant for Bi2Te3):
    H_w(k) = lambda * (k_+^3 + k_-^3) * sigma_z

where k_+ = k_x + i*k_y, k_- = k_x - i*k_y.

The total Hamiltonian:
    H(k) = H_0(k) + H_w(k)

Spin texture on the Fermi surface is computed via expectation values
of the spin operators.
"""

import numpy as np

# Physical constants
HBAR = 1.054571817e-34  # J·s
EV_TO_J = 1.602176634e-19
NM_TO_M = 1e-9


class DiracSurfaceHamiltonian:
    """
    Effective Hamiltonian for TI surface states.
    """

    def __init__(self, v_F=5.0e5, Delta=0.05, lambda_w=0.0, hbar=HBAR):
        """
        Parameters
        ----------
        v_F : float
            Fermi velocity in m/s (typical: 5e5 m/s for Bi2Se3).
        Delta : float
            Exchange gap in eV (magnetic doping).
        lambda_w : float
            Hexagonal warping amplitude in eV·nm^3.
        hbar : float
            Reduced Planck constant.
        """
        if v_F <= 0:
            raise ValueError("Fermi velocity v_F must be positive.")
        self.v_F = v_F
        self.Delta = Delta * EV_TO_J
        self.lambda_w = lambda_w * EV_TO_J * (NM_TO_M ** 3)
        self.hbar = hbar

    def _pauli_matrices(self):
        """Return Pauli matrices."""
        sigma_x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
        sigma_y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
        sigma_z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
        return sigma_x, sigma_y, sigma_z

    def hamiltonian(self, kx, ky):
        """
        Compute H(kx, ky).

        Parameters
        ----------
        kx, ky : float or ndarray
            Wavevector components in 1/m.

        Returns
        -------
        H : ndarray
            Shape (..., 2, 2) complex Hamiltonian matrix.
        """
        kx = np.asarray(kx, dtype=float)
        ky = np.asarray(ky, dtype=float)
        shape = kx.shape if kx.shape else ()

        sigma_x, sigma_y, sigma_z = self._pauli_matrices()

        # H_0 = hbar v_F (k_x sigma_y - k_y sigma_x) + Delta sigma_z
        H = (self.hbar * self.v_F * kx[..., None, None]) * sigma_y \
            - (self.hbar * self.v_F * ky[..., None, None]) * sigma_x \
            + self.Delta * sigma_z

        # Hexagonal warping term
        if abs(self.lambda_w) > 1e-40:
            k_plus = kx + 1.0j * ky
            k_minus = kx - 1.0j * ky
            warping = self.lambda_w * (k_plus ** 3 + k_minus ** 3)
            H = H + warping[..., None, None] * sigma_z

        return H

    def eigenvalues(self, kx, ky):
        """
        Compute eigenvalues analytically.

        For the Dirac cone:
            E_±(k) = ± sqrt((hbar v_F |k|)^2 + Delta^2)

        With warping, eigenvalues are:
            E_±(k) = ± sqrt((hbar v_F |k|)^2 + (Delta + lambda_w(k_+^3+k_-^3))^2)

        Parameters
        ----------
        kx, ky : float or ndarray

        Returns
        -------
        E_plus, E_minus : ndarray
            Upper and lower Dirac cone energies in Joules.
        """
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
        """
        Compute normalized eigenvectors of H(k).

        Returns
        -------
        psi_plus, psi_minus : ndarray
            Shape (..., 2) spinor wavefunctions.
        """
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
        """
        Compute spin expectation values on the Fermi surface.

        For the upper band:
            <S_x> = (hbar/2) * <psi|sigma_x|psi>
            <S_y> = (hbar/2) * <psi|sigma_y|psi>
            <S_z> = (hbar/2) * <psi|sigma_z|psi>

        Parameters
        ----------
        kx, ky : float or ndarray
        band : str
            'upper' or 'lower'.

        Returns
        -------
        Sx, Sy, Sz : ndarray
            Spin expectation values.
        """
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
        """
        Group velocity operator: v = (1/hbar) dH/dk.

        v_x = v_F * sigma_y + (3 lambda_w / hbar) * (k_x^2 - k_y^2) * sigma_z
        v_y = -v_F * sigma_x - (6 lambda_w / hbar) * k_x * k_y * sigma_z

        Returns
        -------
        vx, vy : ndarray
            Velocity operators (2x2 matrices) in m/s.
        """
        _, sigma_y, sigma_z = self._pauli_matrices()
        sigma_x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)

        vx = self.v_F * sigma_y
        vy = -self.v_F * sigma_x

        if abs(self.lambda_w) > 1e-40:
            kx = np.asarray(kx, dtype=float)
            ky = np.asarray(ky, dtype=float)
            # Derivative of warping term
            vx = vx + (3.0 * self.lambda_w / self.hbar) * (kx ** 2 - ky ** 2) * sigma_z
            vy = vy - (6.0 * self.lambda_w / self.hbar) * kx * ky * sigma_z

        return vx, vy


def effective_mass_tensor(kx, ky, v_F=5.0e5, Delta=0.05):
    """
    Compute the effective mass tensor at a given k-point from the
    curvature of the band dispersion:

        1/m*_ij = (1/hbar^2) d^2E/dk_i dk_j

    For the massive Dirac cone near k=0:
        1/m*_xx = 1/m*_yy = v_F^2 / Delta
        1/m*_xy = 0

    Parameters
    ----------
    kx, ky : float
    v_F : float
    Delta : float
        Gap in eV.

    Returns
    -------
    mass_inv : ndarray
        2x2 inverse effective mass tensor in 1/kg.
    """
    Delta_J = Delta * EV_TO_J
    m_inv = np.zeros((2, 2), dtype=float)
    if abs(Delta_J) < 1e-30:
        return m_inv
    m_inv[0, 0] = v_F ** 2 / Delta_J
    m_inv[1, 1] = v_F ** 2 / Delta_J
    return m_inv
