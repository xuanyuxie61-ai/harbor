
import numpy as np
from typing import Tuple, Optional


MU_0 = 4.0 * np.pi * 1e-7
K_B = 1.380649e-23
M_P = 1.6726219e-27


class HarrisEquilibrium:

    def __init__(self,
                 B0: float = 1.0e-2,
                 lambda_cs: float = 5.0e4,
                 B_guide: float = 2.0e-3,
                 p0: float = 2.0e-3,
                 T_plasma: float = 2.0e6,
                 rho_inf: float = 1.0e-12,
                 y_max: float = 3.0e5):
        if B0 <= 0.0:
            raise ValueError("B0 必须为正")
        if lambda_cs <= 0.0:
            raise ValueError("lambda_cs 必须为正")
        if T_plasma <= 0.0:
            raise ValueError("T_plasma 必须为正")
        if y_max <= 0.0:
            raise ValueError("y_max 必须为正")

        self.B0 = B0
        self.lambda_cs = lambda_cs
        self.B_guide = B_guide
        self.p0 = p0
        self.T_plasma = T_plasma
        self.rho_inf = rho_inf
        self.y_max = y_max


        self.n0 = p0 / (K_B * T_plasma)
        self.rho0 = self.n0 * M_P

    def B_field(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        if np.any(np.abs(y) > 1e10):
            raise ValueError("y 值过大，可能导致数值溢出")

        Bx = self.B0 * np.tanh(y / self.lambda_cs)
        By = np.zeros_like(y)
        Bz = np.full_like(y, self.B_guide)
        return np.stack([Bx, By, Bz], axis=-1)

    def pressure(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        sech2 = 1.0 / np.cosh(y / self.lambda_cs) ** 2
        return self.p0 + (self.B0 ** 2 / (2.0 * MU_0)) * sech2

    def current_density(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        sech2 = 1.0 / np.cosh(y / self.lambda_cs) ** 2
        Jz = (self.B0 / (MU_0 * self.lambda_cs)) * sech2
        Jx = np.zeros_like(y)
        Jy = np.zeros_like(y)
        return np.stack([Jx, Jy, Jz], axis=-1)

    def mass_density(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        sech2 = 1.0 / np.cosh(y / self.lambda_cs) ** 2
        return self.rho_inf + self.rho0 * sech2

    def alfven_speed(self, y: np.ndarray) -> np.ndarray:


        raise NotImplementedError("Hole 1: 请实现阿尔芬速度公式 v_A = |B| / sqrt(mu_0 * rho)")

    def plasma_beta(self, y: np.ndarray) -> np.ndarray:
        p = self.pressure(y)
        B = self.B_field(y)
        B2 = np.sum(B ** 2, axis=-1)
        B2_safe = np.where(B2 < 1e-30, 1e-30, B2)
        return 2.0 * MU_0 * p / B2_safe

    def generate_quadrilateral_mesh(self, nx: int = 32, ny: int = 64) -> Tuple[np.ndarray, np.ndarray]:
        if nx < 2 or ny < 2:
            raise ValueError("nx 和 ny 至少为 2")

        Lx = 2.0 * self.y_max
        x = np.linspace(0.0, Lx, nx)

        y_uniform = np.linspace(-1.0, 1.0, ny)

        y = self.y_max * np.tanh(1.5 * y_uniform) / np.tanh(1.5)

        X, Y = np.meshgrid(x, y, indexing='ij')
        nodes = np.column_stack([X.ravel(), Y.ravel()])

        nelems = (nx - 1) * (ny - 1)
        elements = np.zeros((nelems, 4), dtype=int)
        idx = 0
        for i in range(nx - 1):
            for j in range(ny - 1):
                n1 = i * ny + j
                n2 = (i + 1) * ny + j
                n3 = (i + 1) * ny + (j + 1)
                n4 = i * ny + (j + 1)
                elements[idx] = [n1, n2, n3, n4]
                idx += 1
        return nodes, elements

    def bilinear_interpolate_on_mesh(self, nodes: np.ndarray,
                                      elements: np.ndarray,
                                      field_1d: np.ndarray,
                                      y_coords: np.ndarray) -> np.ndarray:
        if len(field_1d) != len(y_coords):
            raise ValueError("field_1d 与 y_coords 长度不匹配")
        if nodes.shape[1] != 2:
            raise ValueError("nodes 必须是二维坐标")


        y_coords = np.asarray(y_coords)
        if not np.all(np.diff(y_coords) >= 0):

            sort_idx = np.argsort(y_coords)
            y_coords = y_coords[sort_idx]
            field_1d = field_1d[sort_idx]

        field_2d = np.zeros(len(nodes))
        for i, (xn, yn) in enumerate(nodes):

            yn_clip = np.clip(yn, y_coords[0], y_coords[-1])
            field_2d[i] = np.interp(yn_clip, y_coords, field_1d)
        return field_2d

    def compute_reconnection_rate(self, y: np.ndarray, eta: np.ndarray, v: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        eta = np.asarray(eta, dtype=float)
        v = np.asarray(v, dtype=float)
        if v.shape[-1] != 3:
            raise ValueError("v 必须是三维速度场")

        J = self.current_density(y)
        B = self.B_field(y)

        v_cross_B_z = v[..., 0] * B[..., 1] - v[..., 1] * B[..., 0]
        Jz = J[..., 2]
        E_rec = eta * Jz + v_cross_B_z
        return E_rec

    def magnetic_shear(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        sech2 = 1.0 / np.cosh(y / self.lambda_cs) ** 2
        return (self.B0 / self.lambda_cs) * sech2


def demo_harris():
    eq = HarrisEquilibrium()
    y = np.linspace(-eq.y_max, eq.y_max, 129)
    B = eq.B_field(y)
    p = eq.pressure(y)
    J = eq.current_density(y)
    rho = eq.mass_density(y)
    va = eq.alfven_speed(y)
    beta = eq.plasma_beta(y)
    shear = eq.magnetic_shear(y)

    print("[HarrisEquilibrium] 电流片中心 (y=0) 物理量:")
    print(f"  B_x(0) = {B[len(y)//2, 0]:.3e} T")
    print(f"  p(0)   = {p[len(y)//2]:.3e} Pa")
    print(f"  J_z(0) = {J[len(y)//2, 2]:.3e} A/m^2")
    print(f"  rho(0) = {rho[len(y)//2]:.3e} kg/m^3")
    print(f"  v_A(0) = {va[len(y)//2]:.3e} m/s")
    print(f"  beta(0)= {beta[len(y)//2]:.3f}")
    print(f"  shear(0)= {shear[len(y)//2]:.3e} T/m")
    return eq, y, B, p, J, rho, va, beta, shear


if __name__ == "__main__":
    demo_harris()
