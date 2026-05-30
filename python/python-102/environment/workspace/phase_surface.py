
import numpy as np
from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import spsolve


class PhaseSurface:

    def __init__(self, x_grid, y_grid):
        self.x = np.array(x_grid, dtype=np.float64)
        self.y = np.array(y_grid, dtype=np.float64)
        self.nx = len(x_grid)
        self.ny = len(y_grid)
        self.dx = x_grid[1] - x_grid[0] if self.nx > 1 else 1.0
        self.dy = y_grid[1] - y_grid[0] if self.ny > 1 else 1.0
        self.k0 = 2.0 * np.pi / 1.55e-6




    def minimal_surface_smooth(self, phi_init, lambda_fidelity=0.1,
                                max_iter=100, dt=0.1):
        phi = phi_init.copy()
        for it in range(max_iter):

            dx_phi, dy_phi = np.gradient(phi, self.dx, self.dy)
            dxx_phi = np.gradient(dx_phi, self.dx, axis=0)
            dyy_phi = np.gradient(dy_phi, self.dy, axis=1)
            dxy_phi = np.gradient(dx_phi, self.dy, axis=1)


            grad_max = 1e6
            dx_phi = np.clip(dx_phi, -grad_max, grad_max)
            dy_phi = np.clip(dy_phi, -grad_max, grad_max)
            numerator = (1.0 + dy_phi ** 2) * dxx_phi \
                        - 2.0 * dx_phi * dy_phi * dxy_phi \
                        + (1.0 + dx_phi ** 2) * dyy_phi
            denom_sq = 1.0 + dx_phi ** 2 + dy_phi ** 2
            denom_sq = np.clip(denom_sq, 0.0, 1e12)
            denominator = 2.0 * denom_sq ** 1.5
            denominator = np.clip(denominator, 1e-15, 1e18)
            H = np.zeros_like(phi)
            mask = np.isfinite(denominator) & (np.abs(numerator) < 1e18)
            H[mask] = numerator[mask] / denominator[mask]


            phi_new = phi + dt * (H + lambda_fidelity * (phi_init - phi))


            phi_new[0, :] = phi_init[0, :]
            phi_new[-1, :] = phi_init[-1, :]
            phi_new[:, 0] = phi_init[:, 0]
            phi_new[:, -1] = phi_init[:, -1]

            diff = np.max(np.abs(phi_new - phi))
            phi = phi_new
            if diff < 1e-8:
                print(f"[phase_surface] 平均曲率流收敛于迭代 {it}")
                break

        return phi




    def catenoid_phase_profile(self, a_param, center=(0.0, 0.0)):
        X, Y = np.meshgrid(self.x, self.y, indexing='ij')
        cx, cy = center
        R = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)

        R = np.maximum(R, 1.01 / a_param)
        phi = self.k0 * np.arccosh(a_param * R) / a_param
        return phi

    def helicoid_phase_profile(self, a_param, center=(0.0, 0.0)):
        X, Y = np.meshgrid(self.x, self.y, indexing='ij')
        cx, cy = center
        phi = a_param * np.arctan2(Y - cy, X - cx)
        return phi

    def scherk_phase_profile(self, a_param):
        X, Y = np.meshgrid(self.x, self.y, indexing='ij')

        cx = np.cos(a_param * X)
        cy_ = np.cos(a_param * Y)
        cx = np.maximum(cx, 1e-6)
        cy_ = np.maximum(cy_, 1e-6)
        phi = self.k0 * np.log(cx / cy_) / a_param
        return phi




    def compute_mean_curvature(self, phi):
        dx_phi, dy_phi = np.gradient(phi, self.dx, self.dy)
        dxx_phi = np.gradient(dx_phi, self.dx, axis=0)
        dyy_phi = np.gradient(dy_phi, self.dy, axis=1)
        dxy_phi = np.gradient(dx_phi, self.dy, axis=1)

        dx_phi = np.clip(dx_phi, -1e6, 1e6)
        dy_phi = np.clip(dy_phi, -1e6, 1e6)
        numerator = (1.0 + dy_phi ** 2) * dxx_phi \
                    - 2.0 * dx_phi * dy_phi * dxy_phi \
                    + (1.0 + dx_phi ** 2) * dyy_phi
        denom_sq = np.clip(1.0 + dx_phi ** 2 + dy_phi ** 2, 0.0, 1e12)
        denominator = 2.0 * denom_sq ** 1.5
        denominator = np.clip(denominator, 1e-15, 1e18)
        H = np.zeros_like(phi)
        mask = np.isfinite(denominator) & (np.abs(numerator) < 1e18)
        H[mask] = numerator[mask] / denominator[mask]
        return H

    def compute_gaussian_curvature(self, phi):
        dx_phi, dy_phi = np.gradient(phi, self.dx, self.dy)
        dxx_phi = np.gradient(dx_phi, self.dx, axis=0)
        dyy_phi = np.gradient(dy_phi, self.dy, axis=1)
        dxy_phi = np.gradient(dx_phi, self.dy, axis=1)

        dx_phi = np.clip(dx_phi, -1e6, 1e6)
        dy_phi = np.clip(dy_phi, -1e6, 1e6)
        E = 1.0 + dx_phi ** 2
        F = dx_phi * dy_phi
        G = 1.0 + dy_phi ** 2
        denom_sqrt = np.sqrt(np.clip(1.0 + dx_phi ** 2 + dy_phi ** 2, 1e-15, 1e12))
        L = dxx_phi / denom_sqrt
        M = dxy_phi / denom_sqrt
        N = dyy_phi / denom_sqrt

        denom = E * G - F ** 2
        denom = np.clip(denom, 1e-15, 1e18)
        K = np.zeros_like(phi)
        mask = np.isfinite(denom)
        K[mask] = (L[mask] * N[mask] - M[mask] ** 2) / denom[mask]
        return K

    def surface_energy(self, phi):
        H = self.compute_mean_curvature(phi)
        dx_phi, dy_phi = np.gradient(phi, self.dx, self.dy)
        dA = np.sqrt(1.0 + dx_phi ** 2 + dy_phi ** 2) * self.dx * self.dy
        energy = np.sum(H ** 2 * dA)
        return energy

    def laplacian_smooth(self, phi_init, n_iter=50):
        phi = phi_init.copy()
        alpha = 0.1
        for _ in range(n_iter):
            lap = np.zeros_like(phi)
            lap[1:-1, 1:-1] = (
                phi[2:, 1:-1] + phi[:-2, 1:-1] +
                phi[1:-1, 2:] + phi[1:-1, :-2] - 4 * phi[1:-1, 1:-1]
            ) / (self.dx * self.dy)
            phi = phi + alpha * lap

            phi[0, :] = phi_init[0, :]
            phi[-1, :] = phi_init[-1, :]
            phi[:, 0] = phi_init[:, 0]
            phi[:, -1] = phi_init[:, -1]
        return phi


def demo():
    nx, ny = 65, 65
    x = np.linspace(-3e-6, 3e-6, nx)
    y = np.linspace(-3e-6, 3e-6, ny)

    ps = PhaseSurface(x, y)


    phi_cat = ps.catenoid_phase_profile(a_param=2.0e6, center=(0.0, 0.0))
    H_cat = ps.compute_mean_curvature(phi_cat)
    W_cat = ps.surface_energy(phi_cat)
    print(f"[phase_surface] 悬链面相位: max H = {np.max(np.abs(H_cat)):.3e}")
    print(f"[phase_surface] 悬链面 Willmore 能量: {W_cat:.4e}")


    phi_hel = ps.helicoid_phase_profile(a_param=2.0, center=(0.0, 0.0))
    H_hel = ps.compute_mean_curvature(phi_hel)
    W_hel = ps.surface_energy(phi_hel)
    print(f"[phase_surface] 螺旋面相位: max H = {np.max(np.abs(H_hel)):.3e}")
    print(f"[phase_surface] 螺旋面 Willmore 能量: {W_hel:.4e}")



    phi_disc = np.zeros((nx, ny))
    for i in range(nx):
        for j in range(ny):
            phi_disc[i, j] = np.floor((i + j) / 8) * (np.pi / 4)
    phi_smooth = ps.minimal_surface_smooth(phi_disc, lambda_fidelity=0.05,
                                            max_iter=80, dt=0.05)
    W_disc = ps.surface_energy(phi_disc)
    W_smooth = ps.surface_energy(phi_smooth)
    print(f"[phase_surface] 离散相位 Willmore 能量: {W_disc:.4e}")
    print(f"[phase_surface] 平滑后 Willmore 能量: {W_smooth:.4e}")
    print(f"[phase_surface] 能量降低比: {W_disc / W_smooth:.2f}x")

    return phi_cat, phi_hel, phi_smooth


if __name__ == "__main__":
    demo()
