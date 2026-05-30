
import numpy as np


class RandomNumberGenerator:

    @staticmethod
    def rnorm_marsaglia():
        while True:
            u1 = np.random.uniform(-1.0, 1.0)
            u2 = np.random.uniform(-1.0, 1.0)
            s = u1 ** 2 + u2 ** 2
            if 0 < s <= 1.0:
                break

        factor = np.sqrt(-2.0 * np.log(s) / s)
        return u1 * factor, u2 * factor

    @staticmethod
    def standard_normal_array(shape):
        return np.random.standard_normal(shape)


class WishartSampler:

    @staticmethod
    def wishart_variate(Sigma, n, np_size):

        L = np.linalg.cholesky(Sigma)


        X = np.random.standard_normal((np_size, n))


        Y = L @ X


        W = Y @ Y.T

        return W


class ThermalNoise:

    def __init__(self, nx, ny, dx, dy, dt, kbt=0.01, mobility=1.0):
        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.dy = dy
        self.dt = dt
        self.kbt = kbt
        self.mobility = mobility



        self.noise_std = np.sqrt(
            2.0 * kbt * mobility / (dx * dy * dt)
        )

    def generate_white_noise(self):
        return self.noise_std * np.random.standard_normal((self.nx, self.ny))

    def generate_colored_noise(self, correlation_length=2.0):
        white = self.generate_white_noise()


        size = int(3 * correlation_length) + 1
        x = np.arange(-size, size + 1)
        y = np.arange(-size, size + 1)
        X, Y = np.meshgrid(x, y, indexing='ij')
        kernel = np.exp(-(X ** 2 + Y ** 2) / (2.0 * correlation_length ** 2))
        kernel = kernel / np.sum(kernel)


        from scipy.signal import convolve2d
        colored = convolve2d(white, kernel, mode='same', boundary='fill')

        return colored

    def apply_to_phase_field(self, phi_rhs, noise_type='white', **kwargs):
        if noise_type == 'white':
            noise = self.generate_white_noise()
        elif noise_type == 'colored':
            noise = self.generate_colored_noise(**kwargs)
        else:
            raise ValueError(f"不支持的噪声类型: {noise_type}")

        return phi_rhs + noise


class StochasticAllenCahn:

    def __init__(self, phase_field_model, thermal_noise):
        self.pf = phase_field_model
        self.noise = thermal_noise

    def rhs_with_noise(self, phi, T, C, velocity_x=None, velocity_y=None,
                       noise_type='white'):
        rhs = self.pf.phase_field_rhs(phi, T, C, velocity_x, velocity_y)
        rhs = self.noise.apply_to_phase_field(rhs, noise_type=noise_type)
        return rhs


class HodgkinHuxleyInspiredGating:

    def __init__(self, alpha0=1.0, beta0=1.0, Qa=1.0, Qd=1.0, kbt=0.1):
        self.alpha0 = alpha0
        self.beta0 = beta0
        self.Qa = Qa
        self.Qd = Qd
        self.kbt = kbt

    def activation_rate(self, undercooling):
        thermal_factor = np.exp(-self.Qa / max(self.kbt, 1e-14))
        driving_force = max(undercooling, 0.0)
        return self.alpha0 * thermal_factor * driving_force

    def deactivation_rate(self, undercooling):
        thermal_factor = np.exp(-self.Qd / max(self.kbt, 1e-14))
        driving_force = max(undercooling, 0.0)
        return self.beta0 * thermal_factor * (1.0 + driving_force)

    def gating_variable_derivative(self, g, undercooling):
        alpha = self.activation_rate(undercooling)
        beta = self.deactivation_rate(undercooling)
        return alpha * (1.0 - g) - beta * g
