
import numpy as np


class InterfaceTracker:

    def __init__(self, nx, ny, dx, dy):
        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.dy = dy

    def compute_gradient(self, phi):
        grad_x = np.zeros_like(phi)
        grad_y = np.zeros_like(phi)

        grad_x[1:-1, :] = (phi[2:, :] - phi[:-2, :]) / (2.0 * self.dx)
        grad_y[:, 1:-1] = (phi[:, 2:] - phi[:, :-2]) / (2.0 * self.dy)


        grad_x[0, :] = (phi[1, :] - phi[0, :]) / self.dx
        grad_x[-1, :] = (phi[-1, :] - phi[-2, :]) / self.dx
        grad_y[:, 0] = (phi[:, 1] - phi[:, 0]) / self.dy
        grad_y[:, -1] = (phi[:, -1] - phi[:, -2]) / self.dy

        return grad_x, grad_y

    def compute_gradient_magnitude(self, phi):
        gx, gy = self.compute_gradient(phi)
        return np.sqrt(gx ** 2 + gy ** 2)

    def compute_normal(self, phi):
        gx, gy = self.compute_gradient(phi)
        grad_mag = np.sqrt(gx ** 2 + gy ** 2)
        grad_mag = np.maximum(grad_mag, 1e-12)
        return gx / grad_mag, gy / grad_mag

    def compute_curvature(self, phi):
        gx, gy = self.compute_gradient(phi)
        grad_mag = np.sqrt(gx ** 2 + gy ** 2)
        grad_mag = np.maximum(grad_mag, 1e-12)

        nx = gx / grad_mag
        ny = gy / grad_mag


        dnx_dx = np.zeros_like(nx)
        dny_dy = np.zeros_like(ny)

        dnx_dx[1:-1, :] = (nx[2:, :] - nx[:-2, :]) / (2.0 * self.dx)
        dny_dy[:, 1:-1] = (ny[:, 2:] - ny[:, :-2]) / (2.0 * self.dy)

        curvature = dnx_dx + dny_dy


        interface_mask = np.abs(phi) < 0.9
        return curvature * interface_mask

    def compute_interface_velocity(self, phi_old, phi_new, dt):
        dphi_dt = (phi_new - phi_old) / dt
        grad_mag = self.compute_gradient_magnitude(phi_new)
        grad_mag = np.maximum(grad_mag, 1e-12)
        V_n = -dphi_dt / grad_mag


        interface_mask = np.abs(phi_new) < 0.9
        return V_n * interface_mask

    def extract_interface_points(self, phi, threshold=0.0):
        points = []
        x_coords = np.linspace(0, (self.nx - 1) * self.dx, self.nx)
        y_coords = np.linspace(0, (self.ny - 1) * self.dy, self.ny)


        for j in range(self.ny):
            for i in range(self.nx - 1):
                val1 = phi[i, j] - threshold
                val2 = phi[i + 1, j] - threshold
                if val1 * val2 < 0:

                    t = abs(val1) / (abs(val1) + abs(val2))
                    x = x_coords[i] + t * (x_coords[i + 1] - x_coords[i])
                    y = y_coords[j]
                    points.append((x, y))


        for i in range(self.nx):
            for j in range(self.ny - 1):
                val1 = phi[i, j] - threshold
                val2 = phi[i, j + 1] - threshold
                if val1 * val2 < 0:
                    t = abs(val1) / (abs(val1) + abs(val2))
                    x = x_coords[i]
                    y = y_coords[j] + t * (y_coords[j + 1] - y_coords[j])
                    points.append((x, y))

        return points

    def compute_interface_area(self, phi):

        epsilon = 0.03
        delta_approx = (1.0 / (2.0 * epsilon)) * (1.0 - np.tanh(phi / epsilon) ** 2)
        grad_mag = self.compute_gradient_magnitude(phi)

        area = np.sum(delta_approx * grad_mag) * self.dx * self.dy
        return area

    def compute_morphology_number(self, phi):
        interface_length = self.compute_interface_area(phi)


        solid_area = np.sum(phi > 0) * self.dx * self.dy
        if solid_area < 1e-12:
            return float('inf')

        morphology = (interface_length ** 2) / (4.0 * np.pi * solid_area)
        return morphology

    def tip_velocity_dendrite(self, phi_old, phi_new, dt):
        curvature = self.compute_curvature(phi_new)
        V_n = self.compute_interface_velocity(phi_old, phi_new, dt)


        interface_mask = np.abs(phi_new) < 0.5
        if not np.any(interface_mask):
            return 0.0


        curv_interface = np.where(interface_mask, curvature, 0.0)
        min_curv = np.min(curv_interface)


        tip_mask = interface_mask & (curvature < min_curv * 0.8)
        if np.any(tip_mask):
            tip_velocity = np.mean(V_n[tip_mask])
            return tip_velocity
        return 0.0
