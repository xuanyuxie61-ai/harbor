
import numpy as np
from utils_numeric import safe_sqrt


class SoluteField:

    def __init__(self, sigma=1.0, n_grid=(32, 32, 32)):
        self.sigma = float(sigma)
        self.n_grid = tuple(n_grid)

    def compute_concentration_on_grid(self, positions, species_idx, box):
        nx, ny, nz = self.n_grid
        C = np.zeros((nx, ny, nz), dtype=np.float64)
        density = np.zeros((nx, ny, nz), dtype=np.float64)
        n_atoms = positions.shape[0]

        dx, dy, dz = box[0] / nx, box[1] / ny, box[2] / nz
        sigma2 = self.sigma ** 2
        norm = (2.0 * np.pi * sigma2) ** 1.5

        for i in range(n_atoms):
            x, y, z = positions[i]
            ix0 = int(x / dx)
            iy0 = int(y / dy)
            iz0 = int(z / dz)

            for ix in range(max(0, ix0 - 1), min(nx, ix0 + 2)):
                for iy in range(max(0, iy0 - 1), min(ny, iy0 + 2)):
                    for iz in range(max(0, iz0 - 1), min(nz, iz0 + 2)):
                        rx = x - (ix + 0.5) * dx
                        ry = y - (iy + 0.5) * dy
                        rz = z - (iz + 0.5) * dz
                        rx -= box[0] * np.round(rx / box[0])
                        ry -= box[1] * np.round(ry / box[1])
                        rz -= box[2] * np.round(rz / box[2])
                        r2 = rx ** 2 + ry ** 2 + rz ** 2
                        w = np.exp(-r2 / (2.0 * sigma2)) / norm
                        C[ix, iy, iz] += species_idx[i] * w
                        density[ix, iy, iz] += w

        mask = density > 1e-12
        C[mask] /= density[mask]
        C[~mask] = 0.0
        return C, density

    def compute_concentration_gradient(self, C, box):
        nx, ny, nz = C.shape
        dx, dy, dz = box[0] / nx, box[1] / ny, box[2] / nz
        grad = np.zeros((nx, ny, nz, 3), dtype=np.float64)


        for ix in range(nx):
            ix_p = (ix + 1) % nx
            ix_m = (ix - 1) % nx
            for iy in range(ny):
                iy_p = (iy + 1) % ny
                iy_m = (iy - 1) % ny
                for iz in range(nz):
                    iz_p = (iz + 1) % nz
                    iz_m = (iz - 1) % nz
                    grad[ix, iy, iz, 0] = (C[ix_p, iy, iz] - C[ix_m, iy, iz]) / (2.0 * dx)
                    grad[ix, iy, iz, 1] = (C[ix, iy_p, iz] - C[ix, iy_m, iz]) / (2.0 * dy)
                    grad[ix, iy, iz, 2] = (C[ix, iy, iz_p] - C[ix, iy, iz_m]) / (2.0 * dz)
        return grad

    def compute_laplacian(self, C, box):
        nx, ny, nz = C.shape
        dx, dy, dz = box[0] / nx, box[1] / ny, box[2] / nz
        lap = np.zeros_like(C)
        for ix in range(nx):
            ix_p = (ix + 1) % nx
            ix_m = (ix - 1) % nx
            for iy in range(ny):
                iy_p = (iy + 1) % ny
                iy_m = (iy - 1) % ny
                for iz in range(nz):
                    iz_p = (iz + 1) % nz
                    iz_m = (iz - 1) % nz
                    lap[ix, iy, iz] = (
                        (C[ix_p, iy, iz] - 2 * C[ix, iy, iz] + C[ix_m, iy, iz]) / dx ** 2 +
                        (C[ix, iy_p, iz] - 2 * C[ix, iy, iz] + C[ix, iy_m, iz]) / dy ** 2 +
                        (C[ix, iy, iz_p] - 2 * C[ix, iy, iz] + C[ix, iy, iz_m]) / dz ** 2
                    )
        return lap

    def compute_segregation_coefficient(self, C, z_interface, box):
        nx, ny, nz = C.shape
        C_solid = []
        C_liquid = []
        for ix in range(nx):
            for iy in range(ny):
                z_int = z_interface[ix, iy] if z_interface.ndim == 2 else np.mean(z_interface)
                for iz in range(nz):
                    z = (iz + 0.5) * box[2] / nz
                    if z < z_int:
                        C_solid.append(C[ix, iy, iz])
                    else:
                        C_liquid.append(C[ix, iy, iz])
        k_eff = np.mean(C_solid) / (np.mean(C_liquid) + 1e-12)
        return k_eff

    def mean_squared_displacement(self, trajectories, species_idx, target_species=1):
        n_frames = len(trajectories)
        if n_frames < 2:
            return np.array([]), np.array([])
        mask = species_idx == target_species
        n_target = np.sum(mask)
        if n_target == 0:
            return np.array([]), np.array([])

        ref_pos = trajectories[0][mask]
        msd = np.zeros(n_frames, dtype=np.float64)
        for t in range(n_frames):
            disp = trajectories[t][mask] - ref_pos
            msd[t] = np.mean(np.sum(disp ** 2, axis=1))
        dt = np.arange(n_frames)
        return dt, msd

    def fit_diffusion_coefficient(self, dt, msd):
        if len(dt) < 3:
            return 0.0

        start = len(dt) // 3
        A = np.vstack([dt[start:], np.ones(len(dt) - start)]).T
        slope, _ = np.linalg.lstsq(A, msd[start:], rcond=None)[0]
        D = slope / 6.0
        return max(D, 0.0)


class LagrangeInterpolator3D:

    def __init__(self, order=3):
        self.order = int(order)

    def _lagrange_basis_1d(self, x_target, x_nodes):
        n = len(x_nodes)
        result = np.zeros(n, dtype=np.float64)
        for i in range(n):
            li = 1.0
            for j in range(n):
                if i != j:
                    li *= (x_target - x_nodes[j]) / (x_nodes[i] - x_nodes[j])
            result[i] = li
        return result

    def interpolate(self, C, box, points):
        nx, ny, nz = C.shape
        M = points.shape[0]
        result = np.zeros(M, dtype=np.float64)


        for p in range(M):
            x, y, z = points[p]
            ix = int(x / box[0] * nx)
            iy = int(y / box[1] * ny)
            iz = int(z / box[2] * nz)
            ix = min(max(ix, 0), nx - 2)
            iy = min(max(iy, 0), ny - 2)
            iz = min(max(iz, 0), nz - 2)

            fx = x / box[0] * nx - ix
            fy = y / box[1] * ny - iy
            fz = z / box[2] * nz - iz

            c000 = C[ix, iy, iz]
            c100 = C[ix + 1, iy, iz]
            c010 = C[ix, iy + 1, iz]
            c110 = C[ix + 1, iy + 1, iz]
            c001 = C[ix, iy, iz + 1]
            c101 = C[ix + 1, iy, iz + 1]
            c011 = C[ix, iy + 1, iz + 1]
            c111 = C[ix + 1, iy + 1, iz + 1]

            result[p] = (
                c000 * (1 - fx) * (1 - fy) * (1 - fz) +
                c100 * fx * (1 - fy) * (1 - fz) +
                c010 * (1 - fx) * fy * (1 - fz) +
                c110 * fx * fy * (1 - fz) +
                c001 * (1 - fx) * (1 - fy) * fz +
                c101 * fx * (1 - fy) * fz +
                c011 * (1 - fx) * fy * fz +
                c111 * fx * fy * fz
            )
        return result
