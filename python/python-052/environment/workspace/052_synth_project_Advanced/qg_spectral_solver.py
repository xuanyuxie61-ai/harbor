
import numpy as np
from numerics_core import next_fftfriendly_size, safe_divide, spd_inverse
from typing import Tuple, Optional, Callable


class QGSpectralSolver:

    def __init__(self, Nx: int = 128, Ny: int = 128,
                 Lx: float = 2.0 * np.pi, Ly: float = 2.0 * np.pi,
                 dt: float = 0.001, nu: float = 1e-6,
                 nu_lap: float = 0.0, beta: float = 1.0,
                 Ld: float = 1.0, dealias: bool = True):
        self.Nx = next_fftfriendly_size(Nx)
        self.Ny = next_fftfriendly_size(Ny)
        self.Lx = float(Lx)
        self.Ly = float(Ly)
        self.dx = self.Lx / self.Nx
        self.dy = self.Ly / self.Ny
        self.dt = float(dt)
        self.nu = float(nu)
        self.nu_lap = float(nu_lap)
        self.beta = float(beta)
        self.Ld = float(Ld)
        self.Ld2 = self.Ld ** 2
        self.dealias = dealias


        self._setup_wavenumbers()


        self._setup_dealias_mask()


        self._setup_linear_operators()


        self.t = 0.0
        self.qh = np.zeros((self.Ny, self.Nx // 2 + 1), dtype=complex)
        self.psih = np.zeros_like(self.qh)

    def _setup_wavenumbers(self):
        kx = 2.0 * np.pi * np.fft.fftfreq(self.Nx, self.dx)
        ky = 2.0 * np.pi * np.fft.fftfreq(self.Ny, self.dy)
        self.kx = kx[:self.Nx // 2 + 1]
        self.ky = ky
        self.KX, self.KY = np.meshgrid(self.kx, self.ky)
        self.K2 = self.KX ** 2 + self.KY ** 2
        self.K4 = self.K2 ** 2
        self.K = np.sqrt(self.K2)

        self.K2_safe = np.where(self.K2 < 1e-15, 1.0, self.K2)
        self.K_safe = np.where(self.K < 1e-15, 1.0, self.K)

    def _setup_dealias_mask(self):
        if self.dealias:
            kx_max = np.pi / self.dx
            ky_max = np.pi / self.dy
            self.dealias_mask = (
                (np.abs(self.KX) < (2.0 / 3.0) * kx_max) &
                (np.abs(self.KY) < (2.0 / 3.0) * ky_max)
            ).astype(float)
        else:
            self.dealias_mask = np.ones_like(self.K2)

    def _setup_linear_operators(self):
        self.helmholtz_denom = self.K2 + 1.0 / self.Ld2
        self.helmholtz_denom_safe = np.where(self.helmholtz_denom < 1e-15, 1.0, self.helmholtz_denom)


        self.inv_helmholtz = -1.0 / self.helmholtz_denom_safe
        self.inv_helmholtz[0, 0] = 0.0


        self.dissipation_coeff = (self.nu * self.K4 + self.nu_lap * self.K2) / self.helmholtz_denom_safe
        self.dissipation_coeff[0, 0] = 0.0


        self.beta_term_coeff = -1j * self.beta * self.KX / self.helmholtz_denom_safe
        self.beta_term_coeff[0, 0] = 0.0

    def q_to_psi(self, qh: np.ndarray) -> np.ndarray:
        return qh * self.inv_helmholtz

    def physical_to_spectral(self, field: np.ndarray) -> np.ndarray:
        return np.fft.rfft2(field)

    def spectral_to_physical(self, fieldh: np.ndarray) -> np.ndarray:
        return np.fft.irfft2(fieldh, s=(self.Ny, self.Nx))

    def _apply_dealias(self, fieldh: np.ndarray) -> np.ndarray:
        return fieldh * self.dealias_mask

    def compute_jacobian(self, psih: np.ndarray, qh: np.ndarray) -> np.ndarray:

        psi = self.spectral_to_physical(psih)
        q = self.spectral_to_physical(qh)

        dpsi_dx_h = 1j * self.KX * psih
        dpsi_dy_h = 1j * self.KY * psih
        dq_dx_h = 1j * self.KX * qh
        dq_dy_h = 1j * self.KY * qh

        dpsi_dx = self.spectral_to_physical(dpsi_dx_h)
        dpsi_dy = self.spectral_to_physical(dpsi_dy_h)
        dq_dx = self.spectral_to_physical(dq_dx_h)
        dq_dy = self.spectral_to_physical(dq_dy_h)


        jac_phys = dpsi_dx * dq_dy - dpsi_dy * dq_dx
        jac_h = self.physical_to_spectral(jac_phys)
        return self._apply_dealias(jac_h)

    def rhs(self, qh: np.ndarray, forcing_h: Optional[np.ndarray] = None) -> np.ndarray:
        psih = self.q_to_psi(qh)


        jac_h = self.compute_jacobian(psih, qh)


        beta_h = -1j * self.beta * self.KX * psih


        diss_h = self.dissipation_coeff * qh

        rhs_val = -jac_h + beta_h + diss_h

        if forcing_h is not None:
            rhs_val += forcing_h

        return self._apply_dealias(rhs_val)

    def step_rk4(self, forcing_h: Optional[np.ndarray] = None):
        k1 = self.dt * self.rhs(self.qh, forcing_h)
        k2 = self.dt * self.rhs(self.qh + 0.5 * k1, forcing_h)
        k3 = self.dt * self.rhs(self.qh + 0.5 * k2, forcing_h)
        k4 = self.dt * self.rhs(self.qh + k3, forcing_h)

        self.qh = self.qh + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        self.qh = self._apply_dealias(self.qh)
        self.t += self.dt

    def step_ab3lf(self, forcing_h: Optional[np.ndarray] = None):
        if not hasattr(self, '_rhs_history'):
            self._rhs_history = []

        rhs_current = self.rhs(self.qh, forcing_h)

        if len(self._rhs_history) < 2:

            self.step_rk4(forcing_h)
            self._rhs_history.append(rhs_current)
            return


        rhs_n = self._rhs_history[-1]
        rhs_n1 = self._rhs_history[-2]
        rhs_n2 = rhs_current


        self.qh = self.qh + self.dt * (
            (23.0 * rhs_n2 - 16.0 * rhs_n1 + 5.0 * rhs_n) / 12.0
        )
        self.qh = self._apply_dealias(self.qh)

        self._rhs_history.append(rhs_current)
        if len(self._rhs_history) > 3:
            self._rhs_history.pop(0)
        self.t += self.dt

    def compute_energy_spectrum_1d(self) -> Tuple[np.ndarray, np.ndarray]:
        psih = self.q_to_psi(self.qh)
        energy_density = 0.5 * self.helmholtz_denom * np.abs(psih) ** 2


        k_flat = self.K.flatten()
        e_flat = energy_density.flatten()
        k_max = min(np.pi / self.dx, np.pi / self.dy)
        nk = max(self.Nx, self.Ny) // 2
        dk = k_max / nk
        k_bins = np.arange(nk) * dk + 0.5 * dk
        E = np.zeros(nk)
        count = np.zeros(nk)

        for i in range(nk):
            k_low = i * dk
            k_high = (i + 1) * dk
            mask = (k_flat >= k_low) & (k_flat < k_high)
            if np.any(mask):
                E[i] = np.sum(e_flat[mask])
                count[i] = np.sum(mask)


        E = safe_divide(E, count) * (self.Lx * self.Ly)
        return k_bins, E

    def compute_enstrophy(self) -> float:
        return 0.5 * np.sum(np.abs(self.qh) ** 2) * self.dx * self.dy / (self.Nx * self.Ny)

    def compute_total_energy(self) -> float:
        psih = self.q_to_psi(self.qh)
        return 0.5 * np.sum(self.helmholtz_denom * np.abs(psih) ** 2) * self.dx * self.dy / (self.Nx * self.Ny)

    def set_initial_condition_gaussian_vortex(self, amplitude: float = 1.0,
                                              sigma: float = 0.2,
                                              x0: float = None, y0: float = None):
        x = np.linspace(0, self.Lx, self.Nx, endpoint=False)
        y = np.linspace(0, self.Ly, self.Ny, endpoint=False)
        X, Y = np.meshgrid(x, y)
        if x0 is None:
            x0 = self.Lx / 2.0
        if y0 is None:
            y0 = self.Ly / 2.0

        r2 = (X - x0) ** 2 + (Y - y0) ** 2
        psi = amplitude * np.exp(-r2 / (2.0 * sigma ** 2))


        psih = self.physical_to_spectral(psi)
        q_h = -(self.K2 + 1.0 / self.Ld2) * psih
        self.qh = q_h
        self.t = 0.0

    def set_initial_condition_double_vortex(self, A1: float = 1.0, A2: float = -0.8,
                                            sigma1: float = 0.15, sigma2: float = 0.15,
                                            x1: float = None, y1: float = None,
                                            x2: float = None, y2: float = None):
        x = np.linspace(0, self.Lx, self.Nx, endpoint=False)
        y = np.linspace(0, self.Ly, self.Ny, endpoint=False)
        X, Y = np.meshgrid(x, y)
        if x1 is None:
            x1 = self.Lx * 0.35
        if y1 is None:
            y1 = self.Ly * 0.5
        if x2 is None:
            x2 = self.Lx * 0.65
        if y2 is None:
            y2 = self.Ly * 0.5

        r2_1 = (X - x1) ** 2 + (Y - y1) ** 2
        r2_2 = (X - x2) ** 2 + (Y - y2) ** 2
        psi = A1 * np.exp(-r2_1 / (2.0 * sigma1 ** 2)) + A2 * np.exp(-r2_2 / (2.0 * sigma2 ** 2))

        psih = self.physical_to_spectral(psi)
        q_h = -(self.K2 + 1.0 / self.Ld2) * psih
        self.qh = q_h
        self.t = 0.0

    def get_physical_fields(self) -> dict:
        q = self.spectral_to_physical(self.qh)
        psih = self.q_to_psi(self.qh)
        psi = self.spectral_to_physical(psih)
        u = -self.spectral_to_physical(1j * self.KY * psih)
        v = self.spectral_to_physical(1j * self.KX * psih)
        return {"q": q, "psi": psi, "u": u, "v": v}






def helmholtz_solve_direct(q: np.ndarray, dx: float, dy: float, Ld: float) -> np.ndarray:
    Ny, Nx = q.shape
    qh = np.fft.rfft2(q)
    kx = 2.0 * np.pi * np.fft.fftfreq(Nx, dx)[:Nx // 2 + 1]
    ky = 2.0 * np.pi * np.fft.fftfreq(Ny, dy)
    KX, KY = np.meshgrid(kx, ky)
    K2 = KX ** 2 + KY ** 2
    denom = K2 + 1.0 / (Ld ** 2)
    denom[0, 0] = 1.0
    psih = -qh / denom
    psih[0, 0] = 0.0
    return np.fft.irfft2(psih, s=(Ny, Nx))


if __name__ == "__main__":
    solver = QGSpectralSolver(Nx=64, Ny=64, dt=0.005, nu=1e-5, beta=2.0, Ld=0.5)
    solver.set_initial_condition_double_vortex()
    E0 = solver.compute_total_energy()
    for _ in range(10):
        solver.step_rk4()
    E1 = solver.compute_total_energy()
    print(f"Energy change: {E0:.6f} -> {E1:.6f}, relative: {abs(E1-E0)/E0:.2e}")
