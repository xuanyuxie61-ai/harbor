
import numpy as np
from typing import Tuple, Optional


class TrigonometricInterpolator:

    def __init__(self, phi_nodes: np.ndarray, values: np.ndarray):
        self.phi = np.asarray(phi_nodes, dtype=float)
        self.f = np.asarray(values, dtype=float)
        self.N = len(self.phi)
        if self.N < 2:
            raise ValueError("至少需要 2 个节点")
        self.h = 2.0 * np.pi / self.N

        diffs = np.diff(np.sort(self.phi))
        if not np.allclose(diffs, self.h, atol=1e-10):

            self.phi = np.linspace(0.0, 2.0 * np.pi, self.N, endpoint=False)

    def _cardinal(self, phi_eval: np.ndarray, j: int) -> np.ndarray:
        phi_eval = np.asarray(phi_eval, dtype=float)
        dphi = phi_eval - self.phi[j]

        dphi = np.mod(dphi + np.pi, 2.0 * np.pi) - np.pi

        eps = 1e-14
        if self.N % 2 == 1:
            denom = self.N * np.sin(dphi / 2.0 + eps)
        else:
            denom = self.N * np.tan(dphi / 2.0 + eps)
        num = np.sin(self.N * dphi / 2.0)
        C = num / denom

        C[np.abs(dphi) < 1e-12] = 1.0
        return C

    def interpolate(self, phi_eval: np.ndarray) -> np.ndarray:
        phi_eval = np.asarray(phi_eval, dtype=float)
        result = np.zeros_like(phi_eval, dtype=float)
        for j in range(self.N):
            result += self.f[j] * self._cardinal(phi_eval, j)
        return result

    def derivative(self, phi_eval: np.ndarray) -> np.ndarray:
        phi_eval = np.asarray(phi_eval, dtype=float)

        f_hat = np.fft.fft(self.f)

        k = np.fft.fftfreq(self.N, d=self.h) * 2.0 * np.pi * self.N
        df_hat = 1j * k * f_hat
        df = np.fft.ifft(df_hat).real

        interp = TrigonometricInterpolator(self.phi, df)
        return interp.interpolate(phi_eval)


class GridReshaper:

    @staticmethod
    def polar_to_cartesian_reshaper(r: np.ndarray,
                                     phi: np.ndarray,
                                     field_rphi: np.ndarray,
                                     nx: int = 64,
                                     ny: int = 64,
                                     x_range: Tuple[float, float] = (-1.0, 1.0),
                                     y_range: Tuple[float, float] = (-1.0, 1.0)) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        r = np.asarray(r)
        phi = np.asarray(phi)
        field = np.asarray(field_rphi)
        nr, nphi = field.shape
        if len(r) != nr or len(phi) != nphi:
            raise ValueError("field_rphi 维度与 r, phi 不匹配")

        x = np.linspace(x_range[0], x_range[1], nx)
        y = np.linspace(y_range[0], y_range[1], ny)
        X, Y = np.meshgrid(x, y, indexing='ij')


        Rq = np.sqrt(X ** 2 + Y ** 2)
        Phiq = np.arctan2(Y, X)
        Phiq = np.mod(Phiq + 2.0 * np.pi, 2.0 * np.pi)


        Z = np.zeros_like(X)
        dr = r[1] - r[0] if nr > 1 else 1.0
        dphi = phi[1] - phi[0] if nphi > 1 else 1.0

        for i in range(nx):
            for j in range(ny):
                ri = Rq[i, j]
                pi = Phiq[i, j]

                ir = int(np.floor((ri - r[0]) / dr))
                ir = np.clip(ir, 0, nr - 2)
                wr = (ri - r[ir]) / dr
                wr = np.clip(wr, 0.0, 1.0)

                ip = int(np.floor((pi - phi[0]) / dphi)) % nphi
                wp = (pi - phi[ip]) / dphi
                wp = np.clip(wp, 0.0, 1.0)
                ip2 = (ip + 1) % nphi


                f00 = field[ir, ip]
                f10 = field[ir + 1, ip] if ir + 1 < nr else field[ir, ip]
                f01 = field[ir, ip2]
                f11 = field[ir + 1, ip2] if ir + 1 < nr else field[ir, ip2]
                Z[i, j] = (1 - wr) * (1 - wp) * f00 + wr * (1 - wp) * f10 + \
                          (1 - wr) * wp * f01 + wr * wp * f11
        return X, Y, Z

    @staticmethod
    def vector_to_grid(values: np.ndarray, nx: int, ny: int,
                       orientation: str = 'row') -> np.ndarray:
        values = np.asarray(values)
        if len(values) != nx * ny:
            raise ValueError(f"values 长度 {len(values)} 不等于 nx*ny={nx*ny}")
        if orientation == 'row':
            return values.reshape((nx, ny), order='C')
        elif orientation == 'col':
            return values.reshape((ny, nx), order='F').T
        else:
            raise ValueError("orientation 必须是 'row' 或 'col'")


def demo_periodic():
    print("\n[PeriodicInterpolation] 演示: 三角插值")
    N = 16
    phi_nodes = np.linspace(0.0, 2.0 * np.pi, N, endpoint=False)

    f_exact = np.cos(2.0 * phi_nodes) + 0.5 * np.sin(3.0 * phi_nodes)
    interp = TrigonometricInterpolator(phi_nodes, f_exact)
    phi_fine = np.linspace(0.0, 2.0 * np.pi, 200)
    f_interp = interp.interpolate(phi_fine)
    f_exact_fine = np.cos(2.0 * phi_fine) + 0.5 * np.sin(3.0 * phi_fine)
    err = np.max(np.abs(f_interp - f_exact_fine))
    print(f"  N={N} 节点, 插值误差: {err:.3e}")


    df_exact_fine = -2.0 * np.sin(2.0 * phi_fine) + 1.5 * np.cos(3.0 * phi_fine)
    df_interp = interp.derivative(phi_fine)
    err_d = np.max(np.abs(df_interp - df_exact_fine))
    print(f"  导数插值误差: {err_d:.3e}")

    print("\n[PeriodicInterpolation] 演示: 极坐标重排")
    nr, nphi = 32, 32
    r = np.linspace(0.1, 1.0, nr)
    phi = np.linspace(0.0, 2.0 * np.pi, nphi, endpoint=False)

    from scipy.special import jv
    Rg, Pg = np.meshgrid(r, phi, indexing='ij')
    field = jv(0, 2.0 * np.pi * Rg) * np.cos(2.0 * Pg)
    X, Y, Z = GridReshaper.polar_to_cartesian_reshaper(r, phi, field, nx=64, ny=64)
    print(f"  重排后网格大小: {Z.shape}")
    print(f"  重排场值范围: [{np.min(Z):.3f}, {np.max(Z):.3f}]")


if __name__ == "__main__":
    demo_periodic()
