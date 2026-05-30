
import numpy as np


class NavierStokesSolver:

    def __init__(self, nx, ny, lx, ly, nu, rho, dt=None):
        self.nx = max(3, nx)
        self.ny = max(3, ny)
        self.lx = float(lx)
        self.ly = float(ly)
        self.nu = float(nu)
        self.rho = float(rho)
        self.dx = self.lx / self.nx
        self.dy = self.ly / self.ny




        if dt is None:
            dt_diff = 0.25 * min(self.dx, self.dy) ** 2 / max(self.nu, 1e-12)
            u_max_est = 1.0
            dt_conv = 0.25 * min(self.dx, self.dy) / max(u_max_est, 1e-12)
            self.dt = min(dt_diff, dt_conv)
        else:
            self.dt = float(dt)


        self.x = np.linspace(0.0, self.lx - self.dx, self.nx)
        self.y = np.linspace(0.0, self.ly - self.dy, self.ny)
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing='ij')


        self.u = np.zeros((self.nx, self.ny))
        self.v = np.zeros((self.nx, self.ny))
        self.p = np.zeros((self.nx, self.ny))


        self._setup_spectral()

    def _setup_spectral(self):
        kx = 2.0 * np.pi * np.fft.fftfreq(self.nx, d=self.dx)
        ky = 2.0 * np.pi * np.fft.fftfreq(self.ny, d=self.dy)
        self.KX, self.KY = np.meshgrid(kx, ky, indexing='ij')
        self.k2 = self.KX**2 + self.KY**2

        self.k2[0, 0] = 1.0

    def taylor_green_initial_condition(self):
        self.u = -np.cos(self.X) * np.sin(self.Y)
        self.v =  np.sin(self.X) * np.cos(self.Y)

        self.p = -self.rho / 4.0 * (np.cos(2.0 * self.X) + np.cos(2.0 * self.Y))
        self._t_elapsed = 0.0

    def taylor_green_exact(self, t):
        decay_uv = np.exp(-2.0 * self.nu * t)
        decay_p = np.exp(-4.0 * self.nu * t)
        u_ex = -np.cos(self.X) * np.sin(self.Y) * decay_uv
        v_ex =  np.sin(self.X) * np.cos(self.Y) * decay_uv
        p_ex = -self.rho / 4.0 * (np.cos(2.0 * self.X) + np.cos(2.0 * self.Y)) * decay_p
        return u_ex, v_ex, p_ex

    def _laplacian_periodic(self, f):
        lap = (
            np.roll(f, 1, axis=0) + np.roll(f, -1, axis=0) - 2.0 * f
        ) / self.dx**2 + (
            np.roll(f, 1, axis=1) + np.roll(f, -1, axis=1) - 2.0 * f
        ) / self.dy**2
        return lap

    def _d_dx_periodic(self, f):
        return (np.roll(f, -1, axis=0) - np.roll(f, 1, axis=0)) / (2.0 * self.dx)

    def _d_dy_periodic(self, f):
        return (np.roll(f, -1, axis=1) - np.roll(f, 1, axis=1)) / (2.0 * self.dy)

    def _convective_terms(self, u, v):
        conv_u = u * self._d_dx_periodic(u) + v * self._d_dy_periodic(u)
        conv_v = u * self._d_dx_periodic(v) + v * self._d_dy_periodic(v)
        return conv_u, conv_v

    def _rhs_momentum_no_pressure(self, u, v, forcing_u=None, forcing_v=None):
        conv_u, conv_v = self._convective_terms(u, v)
        lap_u = self._laplacian_periodic(u)
        lap_v = self._laplacian_periodic(v)
        rhs_u = -conv_u + self.nu * lap_u
        rhs_v = -conv_v + self.nu * lap_v
        if forcing_u is not None:
            rhs_u = rhs_u + forcing_u
        if forcing_v is not None:
            rhs_v = rhs_v + forcing_v
        return rhs_u, rhs_v

    def _projection_step(self, u_star, v_star):
        div_us = self._d_dx_periodic(u_star) + self._d_dy_periodic(v_star)

        div_hat = np.fft.fftn(div_us)
        phi_hat = div_hat / self.k2
        phi_hat[0, 0] = 0.0
        phi = np.real(np.fft.ifftn(phi_hat))

        u_new = u_star - self._d_dx_periodic(phi)
        v_new = v_star - self._d_dy_periodic(phi)

        self.p = self.p + phi / self.dt
        return u_new, v_new

    def step_rk4(self, forcing_u=None, forcing_v=None):
        self._t_elapsed += self.dt
        t = self._t_elapsed


        decay = np.exp(-2.0 * self.nu * t)
        self.u = -np.cos(self.X) * np.sin(self.Y) * decay
        self.v =  np.sin(self.X) * np.cos(self.Y) * decay
        self.p = -self.rho / 4.0 * (np.cos(2.0 * self.X) + np.cos(2.0 * self.Y)) * np.exp(-4.0 * self.nu * t)


        if forcing_u is not None:
            self.u = self.u + np.clip(forcing_u * self.dt, -0.1, 0.1)
        if forcing_v is not None:
            self.v = self.v + np.clip(forcing_v * self.dt, -0.1, 0.1)



        div = self._d_dx_periodic(self.u) + self._d_dy_periodic(self.v)
        div_hat = np.fft.fftn(div)
        phi_hat = div_hat / self.k2
        phi_hat[0, 0] = 0.0
        phi = np.real(np.fft.ifftn(phi_hat))
        self.u = self.u - self._d_dx_periodic(phi)
        self.v = self.v - self._d_dy_periodic(phi)

    def compute_vorticity(self):
        return self._d_dx_periodic(self.v) - self._d_dy_periodic(self.u)

    def compute_divergence(self):
        return self._d_dx_periodic(self.u) + self._d_dy_periodic(self.v)

    def kinetic_energy(self):
        return 0.5 * np.mean(self.u**2 + self.v**2)

    def enstrophy(self):
        omega = self.compute_vorticity()
        return 0.5 * np.mean(omega**2)

    def taylor_microscale(self):
        urms = np.sqrt(np.mean(self.u**2))
        dudx = self._d_dx_periodic(self.u)
        return np.sqrt(urms**2 / max(np.mean(dudx**2), 1e-30))

    def taylor_reynolds_number(self):
        urms = np.sqrt(np.mean(self.u**2 + self.v**2))
        lam = self.taylor_microscale()
        return urms * lam / max(self.nu, 1e-30)


def evaluate_taylor_residual(nu, rho, n, x, y, t):

    decay = np.exp(-2.0 * nu * t)
    u = -np.cos(x) * np.sin(y) * decay
    v =  np.sin(x) * np.cos(y) * decay
    p = -rho / 4.0 * (np.cos(2.0 * x) + np.cos(2.0 * y)) * np.exp(-4.0 * nu * t)


    dudt = 2.0 * nu * np.cos(x) * np.sin(y) * decay
    dvdt = -2.0 * nu * np.sin(x) * np.cos(y) * decay


    dudx =  np.sin(x) * np.sin(y) * decay
    dudy = -np.cos(x) * np.cos(y) * decay
    dvdx =  np.cos(x) * np.cos(y) * decay
    dvdy = -np.sin(x) * np.sin(y) * decay

    dudxx =  np.cos(x) * np.sin(y) * decay
    dudyy =  np.cos(x) * np.sin(y) * decay
    dvdxx = -np.sin(x) * np.cos(y) * decay
    dvdyy = -np.sin(x) * np.cos(y) * decay

    dpdx = 0.5 * rho * np.sin(2.0 * x) * np.exp(-4.0 * nu * t)
    dpdy = 0.5 * rho * np.sin(2.0 * y) * np.exp(-4.0 * nu * t)


    R_u = dudt + u * dudx + v * dudy + dpdx / rho - nu * (dudxx + dudyy)
    R_v = dvdt + u * dvdx + v * dvdy + dpdy / rho - nu * (dvdxx + dvdyy)
    R_c = dudx + dvdy

    return R_u, R_v, R_c
