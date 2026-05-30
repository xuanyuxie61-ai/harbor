
import numpy as np
from dirac_surface import DiracSurfaceHamiltonian


E_CHARGE = 1.602176634e-19
H_PLANCK = 6.62607015e-34


class BerryCurvatureCalculator:

    def __init__(self, hamiltonian=None):
        if hamiltonian is None:
            hamiltonian = DiracSurfaceHamiltonian()
        self.H = hamiltonian

    def _finite_difference_derivative(self, kx, ky, band='upper', dk=1e6):
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

            val = 1.0j * (
                np.vdot(d_ukx, d_uky) - np.vdot(d_uky, d_ukx)
            )
            Omega[i] = np.real(val)

        return Omega.reshape(shape)

    def berry_curvature_analytical(self, kx, ky, band='upper'):






        raise NotImplementedError("HOLE 1: berry_curvature_analytical not implemented")

    def chern_number(self, k_max=1e10, n_k=400, method='analytical'):
        k_vals = np.linspace(-k_max, k_max, n_k)
        dkx = k_vals[1] - k_vals[0]
        dky = dkx
        KX, KY = np.meshgrid(k_vals, k_vals)

        if method == 'analytical':
            Omega = self.berry_curvature_analytical(KX, KY, band='lower')
        else:
            Omega = self.berry_curvature_numerical(KX, KY, band='lower')


        integral = np.sum(Omega) * dkx * dky
        C = integral / (2.0 * np.pi)
        return C

    def anomalous_hall_conductivity(self, k_max=1e10, n_k=400, T=0.0, E_F=0.0):
        k_vals = np.linspace(-k_max, k_max, n_k)
        dkx = k_vals[1] - k_vals[0]
        dky = dkx
        KX, KY = np.meshgrid(k_vals, k_vals)

        E_plus, E_minus = self.H.eigenvalues(KX, KY)
        E_F_J = E_F * 1.602176634e-19


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


        sigma_ah = integral / (2.0 * np.pi)



        return sigma_ah

    def berry_phase_1d(self, k_path):
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
