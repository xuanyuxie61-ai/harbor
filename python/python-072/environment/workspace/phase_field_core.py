
import numpy as np


class PhaseFieldModel:

    def __init__(self, nx, ny, dx, dy, epsilon=0.01, tau=1.0,
                 lambda_thermal=1.0, lambda_solute=1.0,
                 T_m=1.0, C_e=0.5, mobility=1.0):
        if nx < 3 or ny < 3:
            raise ValueError("网格维度 nx, ny 必须至少为 3")
        if dx <= 0 or dy <= 0:
            raise ValueError("空间步长 dx, dy 必须为正")
        if epsilon <= 0:
            raise ValueError("界面宽度参数 epsilon 必须为正")

        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.dy = dy
        self.epsilon = epsilon
        self.tau = tau
        self.lambda_thermal = lambda_thermal
        self.lambda_solute = lambda_solute
        self.T_m = T_m
        self.C_e = C_e
        self.mobility = mobility


        self.cx = 1.0 / (dx * dx)
        self.cy = 1.0 / (dy * dy)

    def double_well_potential(self, phi):
        return 0.25 * (phi ** 2 - 1.0) ** 2

    def double_well_derivative(self, phi):
        return phi ** 3 - phi

    def interpolation_function(self, phi):
        return 0.5 * (1.0 + np.clip(phi, -1.0, 1.0))

    def laplacian_5point(self, field):
        lap = np.zeros_like(field)


        lap[1:-1, 1:-1] = (
            self.cx * (field[2:, 1:-1] - 2.0 * field[1:-1, 1:-1] + field[:-2, 1:-1]) +
            self.cy * (field[1:-1, 2:] - 2.0 * field[1:-1, 1:-1] + field[1:-1, :-2])
        )



        lap[0, 1:-1] = (
            self.cx * (field[1, 1:-1] - field[0, 1:-1]) +
            self.cy * (field[0, 2:] - 2.0 * field[0, 1:-1] + field[0, :-2])
        )

        lap[-1, 1:-1] = (
            self.cx * (field[-2, 1:-1] - field[-1, 1:-1]) +
            self.cy * (field[-1, 2:] - 2.0 * field[-1, 1:-1] + field[-1, :-2])
        )

        lap[1:-1, 0] = (
            self.cx * (field[2:, 0] - 2.0 * field[1:-1, 0] + field[:-2, 0]) +
            self.cy * (field[1:-1, 1] - field[1:-1, 0])
        )

        lap[1:-1, -1] = (
            self.cx * (field[2:, -1] - 2.0 * field[1:-1, -1] + field[:-2, -1]) +
            self.cy * (field[1:-1, -2] - field[1:-1, -1])
        )


        lap[0, 0] = self.cx * (field[1, 0] - field[0, 0]) + self.cy * (field[0, 1] - field[0, 0])
        lap[-1, 0] = self.cx * (field[-2, 0] - field[-1, 0]) + self.cy * (field[-1, 1] - field[-1, 0])
        lap[0, -1] = self.cx * (field[1, -1] - field[0, -1]) + self.cy * (field[0, -2] - field[0, -1])
        lap[-1, -1] = self.cx * (field[-2, -1] - field[-1, -1]) + self.cy * (field[-1, -2] - field[-1, -1])

        return lap

    def phase_field_rhs(self, phi, T, C, velocity_x=None, velocity_y=None):













        raise NotImplementedError("HOLE 1: 请实现 phase_field_rhs 方法")

    def interface_energy_density(self, phi):
        grad_x = np.zeros_like(phi)
        grad_y = np.zeros_like(phi)

        grad_x[1:-1, :] = (phi[2:, :] - phi[:-2, :]) / (2.0 * self.dx)
        grad_y[:, 1:-1] = (phi[:, 2:] - phi[:, :-2]) / (2.0 * self.dy)

        grad_sq = grad_x ** 2 + grad_y ** 2
        potential = self.double_well_potential(phi)

        return 0.5 * self.epsilon ** 2 * grad_sq + potential

    def initialize_circular_nucleus(self, center_x, center_y, radius,
                                    solid_value=1.0, liquid_value=-1.0):
        x = np.linspace(0, (self.nx - 1) * self.dx, self.nx)
        y = np.linspace(0, (self.ny - 1) * self.dy, self.ny)
        X, Y = np.meshgrid(x, y, indexing='ij')

        r = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)



        interface_width = np.sqrt(2.0) * self.epsilon
        phi = np.tanh((radius - r) / interface_width)

        return phi

    def compute_interface_normal(self, phi):
        grad_x = np.zeros_like(phi)
        grad_y = np.zeros_like(phi)

        grad_x[1:-1, :] = (phi[2:, :] - phi[:-2, :]) / (2.0 * self.dx)
        grad_y[:, 1:-1] = (phi[:, 2:] - phi[:, :-2]) / (2.0 * self.dy)

        grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)

        grad_mag = np.maximum(grad_mag, 1e-12)

        n_x = grad_x / grad_mag
        n_y = grad_y / grad_mag

        return n_x, n_y

    def compute_curvature(self, phi):
        n_x, n_y = self.compute_interface_normal(phi)


        dn_x_dx = np.zeros_like(n_x)
        dn_y_dy = np.zeros_like(n_y)

        dn_x_dx[1:-1, :] = (n_x[2:, :] - n_x[:-2, :]) / (2.0 * self.dx)
        dn_y_dy[:, 1:-1] = (n_y[:, 2:] - n_y[:, :-2]) / (2.0 * self.dy)

        curvature = dn_x_dx + dn_y_dy


        interface_mask = np.abs(phi) < 0.9
        curvature = curvature * interface_mask

        return curvature
