
import numpy as np
from signal_processor import shepard_interp_1d


def shepard_interp_2d(
    xd: np.ndarray,
    yd: np.ndarray,
    zd: np.ndarray,
    p: float,
    xi: np.ndarray,
    yi: np.ndarray,
    radius: float = None
) -> np.ndarray:
    xd = np.asarray(xd, dtype=np.float64)
    yd = np.asarray(yd, dtype=np.float64)
    zd = np.asarray(zd, dtype=np.float64)
    xi = np.asarray(xi, dtype=np.float64)
    yi = np.asarray(yi, dtype=np.float64)

    nx = len(xi)
    ny = len(yi)
    zi = np.zeros((ny, nx), dtype=np.float64)

    for iy in range(ny):
        for ix in range(nx):
            x = xi[ix]
            y = yi[iy]
            dx = x - xd
            dy = y - yd
            d = np.sqrt(dx ** 2 + dy ** 2)


            exact = np.where(d < 1e-12)[0]
            if len(exact) > 0:
                zi[iy, ix] = zd[exact[0]]
                continue


            if p == 0.0:
                w = np.ones_like(d) / len(d)
            else:
                if radius is not None:
                    mask = d <= radius
                    if not np.any(mask):

                        idx = np.argmin(d)
                        zi[iy, ix] = zd[idx]
                        continue
                    d = d[mask]
                    z_local = zd[mask]
                else:
                    z_local = zd

                w = 1.0 / (d ** p)
                s = np.sum(w)
                if s > 0:
                    w = w / s
                else:
                    w = np.ones_like(d) / len(d)

            zi[iy, ix] = np.dot(w, z_local)

    return zi


class BathymetryInterpolator:

    def __init__(self, x: np.ndarray, y: np.ndarray, z: np.ndarray, p: float = 2.5):
        self.x = np.asarray(x, dtype=np.float64).ravel()
        self.y = np.asarray(y, dtype=np.float64).ravel()
        self.z = np.asarray(z, dtype=np.float64).ravel()
        self.p = float(p)
        self.n_points = len(self.x)

        if self.n_points < 3:
            raise ValueError("至少需要 3 个数据点")

    def interpolate_grid(
        self,
        x_range: tuple,
        y_range: tuple,
        nx: int = 100,
        ny: int = 100,
        radius: float = None
    ) -> tuple:
        xi = np.linspace(x_range[0], x_range[1], nx)
        yi = np.linspace(y_range[0], y_range[1], ny)
        Z = shepard_interp_2d(self.x, self.y, self.z, self.p, xi, yi, radius)
        X, Y = np.meshgrid(xi, yi)
        return X, Y, Z

    def estimate_gradient(self, xq: float, yq: float, h: float = 1.0) -> np.ndarray:

        z_px = shepard_interp_2d(self.x, self.y, self.z, self.p,
                                 np.array([xq + h]), np.array([yq]))[0, 0]
        z_mx = shepard_interp_2d(self.x, self.y, self.z, self.p,
                                 np.array([xq - h]), np.array([yq]))[0, 0]
        z_py = shepard_interp_2d(self.x, self.y, self.z, self.p,
                                 np.array([xq]), np.array([yq + h]))[0, 0]
        z_my = shepard_interp_2d(self.x, self.y, self.z, self.p,
                                 np.array([xq]), np.array([yq - h]))[0, 0]

        dz_dx = (z_px - z_mx) / (2.0 * h)
        dz_dy = (z_py - z_my) / (2.0 * h)
        return np.array([dz_dx, dz_dy])

    def estimate_curvature(self, xq: float, yq: float, h: float = 5.0) -> float:
        z0 = shepard_interp_2d(self.x, self.y, self.z, self.p,
                               np.array([xq]), np.array([yq]))[0, 0]
        z_px = shepard_interp_2d(self.x, self.y, self.z, self.p,
                                 np.array([xq + h]), np.array([yq]))[0, 0]
        z_mx = shepard_interp_2d(self.x, self.y, self.z, self.p,
                                 np.array([xq - h]), np.array([yq]))[0, 0]
        z_py = shepard_interp_2d(self.x, self.y, self.z, self.p,
                                 np.array([xq]), np.array([yq + h]))[0, 0]
        z_my = shepard_interp_2d(self.x, self.y, self.z, self.p,
                                 np.array([xq]), np.array([yq - h]))[0, 0]

        curvature = (z_px + z_mx + z_py + z_my - 4.0 * z0) / (h ** 2)
        return float(curvature)

    def cross_section_profile(self, x_start: float, y_start: float,
                              x_end: float, y_end: float, n_samples: int = 200) -> tuple:
        t = np.linspace(0.0, 1.0, n_samples)
        x_line = x_start + t * (x_end - x_start)
        y_line = y_start + t * (y_end - y_start)
        z_line = shepard_interp_1d(self.x, self.z, self.p, x_line)

        dx = x_line[1:] - x_line[:-1]
        dy = y_line[1:] - y_line[:-1]
        ds = np.sqrt(dx ** 2 + dy ** 2)
        s = np.concatenate([[0.0], np.cumsum(ds)])
        return s, z_line
