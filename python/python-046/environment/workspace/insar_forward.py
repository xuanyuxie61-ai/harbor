
import numpy as np
from spectral_basis import legendre_polynomial_values
from utils import check_finite, normalize_vector


class OkadaGreenFunction:

    def __init__(self, nu=0.25):
        self.nu = nu

    def _chinnery(self, f, x, y, p, q, L, W):
        return (f(x, y, p, q) - f(x, y, p - W, q) -
                f(x - L, y, p, q) + f(x - L, y, p - W, q))

    def _f_strike_slip(self, x, y, p, q):

        R = np.sqrt(x ** 2 + y ** 2 + q ** 2)
        if R < 1e-10:
            R = 1e-10

        ux = (1.0 / (2.0 * np.pi)) * (x * y / (R * (R + q)) +
                                      np.arctan(x * y / (q * R)))
        uy = (1.0 / (2.0 * np.pi)) * (-q / R - y ** 2 / (R * (R + q)))
        uz = (1.0 / (2.0 * np.pi)) * (-y / R)
        return np.array([ux, uy, uz])

    def _f_dip_slip(self, x, y, p, q):
        R = np.sqrt(x ** 2 + y ** 2 + q ** 2)
        if R < 1e-10:
            R = 1e-10
        ux = (1.0 / (2.0 * np.pi)) * (q / R)
        uy = (1.0 / (2.0 * np.pi)) * (y * q / (R * (R + q)))
        uz = (1.0 / (2.0 * np.pi)) * (1.0 - y ** 2 / (R * (R + q)))
        return np.array([ux, uy, uz])

    def compute_displacement(self, obs_e, obs_n, obs_u,
                              fault_length, fault_width,
                              strike_deg, dip_deg, depth,
                              rake_deg, slip):
        strike = np.deg2rad(strike_deg)
        dip = np.deg2rad(dip_deg)
        rake = np.deg2rad(rake_deg)


        slip_strike = slip * np.cos(rake)
        slip_dip = slip * np.sin(rake)



        dx = obs_e
        dy = obs_n
        x1 = dx * np.cos(strike) + dy * np.sin(strike)
        x2 = -dx * np.sin(strike) + dy * np.cos(strike)


        p = x2 * np.cos(dip) + depth * np.sin(dip)
        q = x2 * np.sin(dip) - depth * np.cos(dip)
        if q < 0:
            q = -q


        u_strike = self._f_strike_slip(x1, x2, p, q)
        u_dip = self._f_dip_slip(x1, x2, p, q)

        u_local = slip_strike * u_strike + slip_dip * u_dip


        u_e = u_local[0] * np.cos(strike) - u_local[1] * np.sin(strike)
        u_n = u_local[0] * np.sin(strike) + u_local[1] * np.cos(strike)
        u_u = u_local[2]

        return np.array([u_e, u_n, u_u])

    def compute_displacements_vectorized(self, obs_points,
                                          fault_length, fault_width,
                                          strike_deg, dip_deg, depth,
                                          rake_deg, slip):
        N = obs_points.shape[0]
        displacements = np.zeros((N, 3))
        for i in range(N):
            displacements[i] = self.compute_displacement(
                obs_points[i, 0], obs_points[i, 1], obs_points[i, 2],
                fault_length, fault_width, strike_deg, dip_deg,
                depth, rake_deg, slip
            )
        return displacements


class InSARForwardModel:

    def __init__(self, los_vector=None, wavelength=0.056):
        if los_vector is None:

            theta_inc = np.deg2rad(34.0)
            alpha_az = np.deg2rad(190.0)
            self.los_vector = np.array([
                np.sin(theta_inc) * np.cos(alpha_az),
                -np.sin(theta_inc) * np.sin(alpha_az),
                np.cos(theta_inc)
            ])
        else:
            self.los_vector = normalize_vector(np.asarray(los_vector))
        self.wavelength = wavelength
        self.okada = OkadaGreenFunction(nu=0.25)

    def project_to_los(self, displacement_enu):
        return displacement_enu @ self.los_vector

    def forward(self, fault_mesh, slip_distribution, obs_points):









        raise NotImplementedError("forward: 待实现 InSAR 正演计算")

    def add_noise(self, d_los, sigma=0.01, atmospheric=False,
                  correlation_length=5000.0):
        N = len(d_los)
        noise = np.random.normal(0.0, sigma, N)

        if atmospheric:


            sigma_atm = 0.02

            atm_noise = np.random.normal(0.0, sigma_atm, N)

            if N > 3:
                atm_noise_smooth = np.convolve(
                    atm_noise, np.ones(3) / 3.0, mode='same')
            else:
                atm_noise_smooth = atm_noise
            noise += atm_noise_smooth

        return d_los + noise

    def los_vector_legendre_expansion(self, theta_min, theta_max, n_terms):
        theta_eval = np.linspace(theta_min, theta_max, 50)

        x_norm = 2.0 * (theta_eval - theta_min) / (theta_max - theta_min) - 1.0
        P = legendre_polynomial_values(len(x_norm), n_terms, x_norm)
        return P


class ElasticHalfspacePoisson1D:

    def __init__(self, H, mu, nx):
        self.H = H
        self.mu = mu
        self.nx = nx
        self.hz = H / (nx - 1)
        self.z = np.linspace(0, H, nx)

    def solve(self, f_func, u_bottom=0.0):
        nx = self.nx
        hz = self.hz
        A = np.zeros((nx, nx))
        rhs = np.zeros(nx)

        for i in range(nx):
            z_i = self.z[i]
            if i == 0:
                A[i, i] = 1.0
                rhs[i] = 0.0
            elif i == nx - 1:
                A[i, i] = 1.0
                rhs[i] = u_bottom
            else:
                A[i, i] = 2.0 / (hz * hz)
                A[i, i - 1] = -1.0 / (hz * hz)
                A[i, i + 1] = -1.0 / (hz * hz)
                rhs[i] = f_func(z_i) / self.mu

        u = np.linalg.solve(A, rhs)
        check_finite(u, "ElasticHalfspacePoisson1D solve")
        return self.z, u
