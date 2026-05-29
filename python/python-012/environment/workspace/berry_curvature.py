"""
Berry Curvature, Berry Connection and Chern Number
==================================================
For the gapped Dirac surface states of a topological insulator,
the Berry curvature is concentrated near the Dirac point and
gives rise to the quantum anomalous Hall effect when Delta != 0.

The Berry connection for band n is:
    A_n(k) = i <u_{n,k}| grad_k |u_{n,k}>

The Berry curvature is:
    Omega_n(k) = nabla_k x A_n(k)
               = i [ <d_kx u|d_ky u> - <d_ky u|d_kx u> ]

For the massive Dirac model, the Berry curvature of the upper band is:
    Omega_+(k) = - (1/2) * hbar^2 v_F^2 Delta / (E_k)^3

The Chern number (anomalous Hall conductivity in units of e^2/h):
    C = (1/2pi) sum_n int_BZ Omega_n(k) d^2k * f_n(k)

For the lower band of the massive Dirac cone, C = -1/2 * sign(Delta).
For the upper band, C = +1/2 * sign(Delta).
The total Chern number of the occupied bands gives the QAHE.
"""

import numpy as np
from dirac_surface import DiracSurfaceHamiltonian

# Physical constants
E_CHARGE = 1.602176634e-19  # C
H_PLANCK = 6.62607015e-34   # J·s


class BerryCurvatureCalculator:
    """
    Computes Berry curvature and Chern number for TI surface states.
    """

    def __init__(self, hamiltonian=None):
        if hamiltonian is None:
            hamiltonian = DiracSurfaceHamiltonian()
        self.H = hamiltonian

    def _finite_difference_derivative(self, kx, ky, band='upper', dk=1e6):
        """
        Compute derivative of eigenstate |u(k)> using central finite differences.

        d/dk_x |u> ≈ (|u(kx+dk, ky)> - |u(kx-dk, ky)>) / (2*dk)

        Parameters
        ----------
        kx, ky : float
        band : str
        dk : float
            Finite difference step in 1/m.

        Returns
        -------
        d_ukx, d_uky : ndarray
            Derivatives of the spinor.
        """
        if band == 'upper':
            psi_pxp, _ = self.H.eigenvectors(kx + dk, ky)
            psi_pxm, _ = self.H.eigenvectors(kx - dk, ky)
            psi_pyp, _ = self.H.eigenvectors(kx, ky + dk)
            psi_pym, _ = self.H.eigenvectors(kx, ky - dk)
        else:
            _, psi_pxp = self.H.eigenvectors(kx + dk, ky)
            _, psi_pxm = self.H.eigenvectors(kx - dk, ky)
            _, psi_pyp = self.H.eigenvectors(kx, ky + dk)
            _, psi_pym = self.H.eigenvectors(kx, ky - dk)

        d_ukx = (psi_pxp - psi_pxm) / (2.0 * dk)
        d_uky = (psi_pyp - psi_pym) / (2.0 * dk)
        return d_ukx, d_uky

    def berry_curvature_numerical(self, kx, ky, band='upper', dk=1e6):
        """
        Compute Berry curvature Omega(k) numerically.

        Omega(k) = i [ <d_kx u|d_ky u> - <d_ky u|d_kx u> ]

        Parameters
        ----------
        kx, ky : float or ndarray
        band : str
        dk : float

        Returns
        -------
        Omega : ndarray
            Berry curvature in m^2.
        """
        kx = np.asarray(kx, dtype=float)
        ky = np.asarray(ky, dtype=float)
        shape = kx.shape
        kx_flat = kx.ravel()
        ky_flat = ky.ravel()

        Omega = np.empty_like(kx_flat)
        for i in range(kx_flat.size):
            d_ukx, d_uky = self._finite_difference_derivative(
                kx_flat[i], ky_flat[i], band=band, dk=dk
            )
            # i * (<d_kx u | d_ky u> - <d_ky u | d_kx u>)
            val = 1.0j * (
                np.vdot(d_ukx, d_uky) - np.vdot(d_uky, d_ukx)
            )
            Omega[i] = np.real(val)

        return Omega.reshape(shape)

    def berry_curvature_analytical(self, kx, ky, band='upper'):
        """
        Analytical Berry curvature for the massive Dirac model without warping.

        For upper band (+):
            Omega_+(k) = - (1/2) * (hbar v_F)^2 * Delta / E_k^3
        For lower band (-):
            Omega_-(k) = + (1/2) * (hbar v_F)^2 * Delta / E_k^3

        Parameters
        ----------
        kx, ky : float or ndarray
        band : str

        Returns
        -------
        Omega : ndarray
            Berry curvature in m^2.
        """
        # TODO HOLE 1: Implement the analytical Berry curvature formula.
        # For the massive Dirac model without hexagonal warping:
        #   E_k = sqrt((hbar * v_F * |k|)^2 + Delta^2)
        #   Omega_+(k) = - (1/2) * (hbar * v_F)^2 * Delta / E_k^3
        #   Omega_-(k) = + (1/2) * (hbar * v_F)^2 * Delta / E_k^3
        # Hint: use self.H.hbar, self.H.v_F, self.H.Delta.
        raise NotImplementedError("HOLE 1: berry_curvature_analytical not implemented")

    def chern_number(self, k_max=1e10, n_k=400, method='analytical'):
        """
        Compute the Chern number by integrating Berry curvature over the BZ.

        C = (1 / 2*pi) * int d^2k Omega(k)

        Parameters
        ----------
        k_max : float
            Cutoff wavevector in 1/m (approximate BZ boundary).
        n_k : int
            Number of k-points in each direction.
        method : str
            'analytical' or 'numerical'.

        Returns
        -------
        C : float
            Chern number (dimensionless).
        """
        k_vals = np.linspace(-k_max, k_max, n_k)
        dkx = k_vals[1] - k_vals[0]
        dky = dkx
        KX, KY = np.meshgrid(k_vals, k_vals)

        if method == 'analytical':
            Omega = self.berry_curvature_analytical(KX, KY, band='lower')
        else:
            Omega = self.berry_curvature_numerical(KX, KY, band='lower')

        # Integrate over k-space
        integral = np.sum(Omega) * dkx * dky
        C = integral / (2.0 * np.pi)
        return C

    def anomalous_hall_conductivity(self, k_max=1e10, n_k=400, T=0.0, E_F=0.0):
        """
        Compute the intrinsic anomalous Hall conductivity from the Berry curvature.

        sigma_AH = (e^2 / h) * int_BZ [d^2k / (2*pi)^2] * Omega(k) * f(E_k)

        At zero temperature, for the lower band fully occupied and Fermi level
        in the gap:
            sigma_AH = (e^2 / h) * sign(Delta) / 2

        Parameters
        ----------
        k_max : float
        n_k : int
        T : float
            Temperature in K.
        E_F : float
            Fermi level in eV.

        Returns
        -------
        sigma_ah : float
            Anomalous Hall conductivity in S/m (or (e^2/h) units).
        """
        k_vals = np.linspace(-k_max, k_max, n_k)
        dkx = k_vals[1] - k_vals[0]
        dky = dkx
        KX, KY = np.meshgrid(k_vals, k_vals)

        E_plus, E_minus = self.H.eigenvalues(KX, KY)
        E_F_J = E_F * 1.602176634e-19

        # Fermi-Dirac distribution at temperature T
        if T < 1e-6:
            f_lower = np.where(E_minus < E_F_J, 1.0, 0.0)
            f_upper = np.where(E_plus < E_F_J, 1.0, 0.0)
        else:
            kB = 1.380649e-23
            beta = 1.0 / (kB * T)
            f_lower = 1.0 / (1.0 + np.exp(beta * (E_minus - E_F_J)))
            f_upper = 1.0 / (1.0 + np.exp(beta * (E_plus - E_F_J)))

        Omega_lower = self.berry_curvature_analytical(KX, KY, band='lower')
        Omega_upper = self.berry_curvature_analytical(KX, KY, band='upper')

        integrand = Omega_lower * f_lower + Omega_upper * f_upper
        integral = np.sum(integrand) * dkx * dky

        # sigma_AH in units of e^2/h per layer
        sigma_ah = integral / (2.0 * np.pi)

        # Convert to SI if desired
        # e2_over_h = E_CHARGE ** 2 / H_PLANCK
        return sigma_ah

    def berry_phase_1d(self, k_path):
        """
        Compute the Berry phase along a closed 1D path in k-space.

        gamma = i * sum_j <u_j | u_{j+1}> (discretized)

        Parameters
        ----------
        k_path : ndarray
            Shape (N, 2) array of (kx, ky) points forming a closed loop.

        Returns
        -------
        gamma : float
            Berry phase in radians.
        """
        if k_path.ndim != 2 or k_path.shape[1] != 2:
            raise ValueError("k_path must have shape (N, 2).")

        n_pts = k_path.shape[0]
        psi_list = []
        for i in range(n_pts):
            psi_p, _ = self.H.eigenvectors(k_path[i, 0], k_path[i, 1])
            psi_list.append(psi_p)

        gamma = 0.0
        for i in range(n_pts):
            j = (i + 1) % n_pts
            overlap = np.vdot(psi_list[i], psi_list[j])
            gamma += np.angle(overlap)

        return gamma
