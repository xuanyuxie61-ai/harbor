
import numpy as np
from numpy.fft import fft2, ifft2, fftfreq, fftshift, ifftshift

class QGBetaPlaneSolver:

    def __init__(self, Nx=256, Ny=256, Lx=2*np.pi*1e6, Ly=2*np.pi*1e6,
                 beta=2e-11, Ld=4e4, nu=1e4, r=1e-7, dt=3600.0):
        if Nx % 2 != 0 or Ny % 2 != 0:
            raise ValueError("Nx and Ny must be even for dealiasing.")
        self.Nx, self.Ny = Nx, Ny
        self.Lx, self.Ly = Lx, Ly
        self.dx, self.dy = Lx / Nx, Ly / Ny
        self.beta = beta
        self.Ld = Ld
        self.nu = nu
        self.r = r
        self.dt = dt


        self.kx = 2 * np.pi * fftshift(fftfreq(Nx, d=self.dx))
        self.ky = 2 * np.pi * fftshift(fftfreq(Ny, d=self.dy))
        self.KX, self.KY = np.meshgrid(self.kx, self.ky, indexing='ij')
        self.ksq = self.KX**2 + self.KY**2
        self.ksq[0, 0] = 1.0


        self.helmholtz = -(self.ksq + 1.0 / (Ld**2))
        self.invert_helmholtz = 1.0 / self.helmholtz
        self.invert_helmholtz[0, 0] = 0.0


        self.ab_coeff = np.array([23.0, -16.0, 5.0]) / 12.0


        kx_max = np.pi / self.dx
        ky_max = np.pi / self.dy
        self.dealias = ((np.abs(self.KX) < (2.0/3.0) * kx_max) &
                        (np.abs(self.KY) < (2.0/3.0) * ky_max)).astype(float)


        self.psi = np.zeros((Nx, Ny), dtype=np.float64)
        self.q = np.zeros((Nx, Ny), dtype=np.float64)
        self.rhs_hist = [None, None, None]

    def _physical_to_spectral(self, field):
        return fftshift(fft2(field))

    def _spectral_to_physical(self, spec):
        return np.real(ifft2(ifftshift(spec)))

    def _jacobian(self, psi, q):
        psihat = self._physical_to_spectral(psi) * self.dealias
        qhat = self._physical_to_spectral(q) * self.dealias

        dpsi_dx = self._spectral_to_physical(1j * self.KX * psihat)
        dpsi_dy = self._spectral_to_physical(1j * self.KY * psihat)
        dq_dx = self._spectral_to_physical(1j * self.KX * qhat)
        dq_dy = self._spectral_to_physical(1j * self.KY * qhat)

        jac = dpsi_dx * dq_dy - dpsi_dy * dq_dx
        return jac

    def _rhs(self, psi, q, stochastic_forcing=None):
        jac = self._jacobian(psi, q)

        psihat = self._physical_to_spectral(psi) * self.dealias
        dpsi_dx = self._spectral_to_physical(1j * self.KX * psihat)
        laplacian_psi = self._spectral_to_physical(-self.ksq * psihat)
        biharmonic_psi = self._spectral_to_physical(self.ksq**2 * psihat)

        rhs = -jac - self.beta * dpsi_dx + self.nu * biharmonic_psi - self.r * laplacian_psi
        if stochastic_forcing is not None:
            rhs += stochastic_forcing
        return rhs

    def step(self, stochastic_forcing=None):
        rhs = self._rhs(self.psi, self.q, stochastic_forcing)

        if self.rhs_hist[0] is None:

            self.q += self.dt * rhs
        elif self.rhs_hist[1] is None:

            self.q += self.dt * (1.5 * rhs - 0.5 * self.rhs_hist[0])
        else:

            self.q += self.dt * (
                self.ab_coeff[0] * rhs +
                self.ab_coeff[1] * self.rhs_hist[0] +
                self.ab_coeff[2] * self.rhs_hist[1]
            )


        self.rhs_hist[2] = self.rhs_hist[1]
        self.rhs_hist[1] = self.rhs_hist[0]
        self.rhs_hist[0] = rhs.copy()


        qhat = self._physical_to_spectral(self.q)
        psihat = qhat * self.invert_helmholtz
        self.psi = self._spectral_to_physical(psihat)

    def compute_energy(self):
        psihat = self._physical_to_spectral(self.psi)



        kinetic_spec = 0.5 * self.ksq * np.abs(psihat)**2
        potential_spec = 0.5 * (1.0 / self.Ld**2) * np.abs(psihat)**2
        energy_spec = kinetic_spec + potential_spec
        dA = self.dx * self.dy
        E = np.sum(energy_spec) * dA / (self.Nx * self.Ny)
        return float(E)

    def compute_enstrophy(self):
        qhat = self._physical_to_spectral(self.q)
        dA = self.dx * self.dy
        Z = 0.5 * np.sum(np.abs(qhat)**2) * dA / (self.Nx * self.Ny)
        return float(Z)

    def compute_vorticity(self):
        psihat = self._physical_to_spectral(self.psi)
        zeta_hat = -self.ksq * psihat
        return self._spectral_to_physical(zeta_hat)

    def compute_energy_flux_spectrum(self):
        psihat = self._physical_to_spectral(self.psi)
        jac = self._jacobian(self.psi, self.q)
        jachat = self._physical_to_spectral(jac)

        T_spec = -np.real(np.conj(psihat) * jachat)


        dk = 2 * np.pi / self.Lx
        k_max = np.max(np.abs(self.kx))
        k_bins = np.arange(0, k_max + dk, dk)
        T_radial = np.zeros_like(k_bins)

        for i in range(len(k_bins) - 1):
            mask = ((self.ksq >= k_bins[i]**2) & (self.ksq < k_bins[i+1]**2))
            if np.any(mask):
                T_radial[i] = np.sum(T_spec[mask])


        Pi = np.cumsum(T_radial)
        return k_bins, T_radial, Pi

    def initialize_gaussian_eddy(self, x0, y0, A, sigma):
        x = np.linspace(0, self.Lx, self.Nx, endpoint=False)
        y = np.linspace(0, self.Ly, self.Ny, endpoint=False)
        X, Y = np.meshgrid(x, y, indexing='ij')
        psi0 = A * np.exp(-((X - x0)**2 + (Y - y0)**2) / (2 * sigma**2))
        self.psi = psi0
        psihat = self._physical_to_spectral(self.psi)
        self.q = self._spectral_to_physical(self.helmholtz * psihat)
        self.rhs_hist = [None, None, None]

    def get_velocity(self):
        psihat = self._physical_to_spectral(self.psi)
        u = -self._spectral_to_physical(1j * self.KY * psihat)
        v =  self._spectral_to_physical(1j * self.KX * psihat)
        return u, v
