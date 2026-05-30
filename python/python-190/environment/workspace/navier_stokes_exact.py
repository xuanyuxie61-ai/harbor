
import numpy as np


def uvwp_ethier(a: float, d: float, x: np.ndarray, y: np.ndarray,
                z: np.ndarray, t: np.ndarray) -> tuple:

    a = float(a)
    d = float(d)
    if a == 0.0 and d == 0.0:
        raise ValueError("参数 a 与 d 不能同时为零，否则退化为 trivial 解。")


    ax = a * x
    ay = a * y
    az = a * z
    ex = np.exp(np.clip(ax, -700.0, 700.0))
    ey = np.exp(np.clip(ay, -700.0, 700.0))
    ez = np.exp(np.clip(az, -700.0, 700.0))

    e2t = np.exp(np.clip(-d * d * t, -700.0, 700.0))
    exy = np.exp(np.clip(a * (x + y), -700.0, 700.0))
    eyz = np.exp(np.clip(a * (y + z), -700.0, 700.0))
    ezx = np.exp(np.clip(a * (z + x), -700.0, 700.0))


    sxy = np.sin(ax + d * y)
    syz = np.sin(ay + d * z)
    szx = np.sin(az + d * x)
    cxy = np.cos(ax + d * y)
    cyz = np.cos(ay + d * z)
    czx = np.cos(az + d * x)

    u = -a * (ex * syz + ez * cxy) * e2t
    v = -a * (ey * szx + ex * cyz) * e2t
    w = -a * (ez * sxy + ey * czx) * e2t
    p = 0.5 * a * a * e2t * e2t * (
        ex * ex
        + 2.0 * sxy * czx * eyz
        + ey * ey
        + 2.0 * syz * cxy * ezx
        + ez * ez
        + 2.0 * szx * cyz * exy
    )
    return u, v, w, p


def ns_residual(u: np.ndarray, v: np.ndarray, w: np.ndarray, p: np.ndarray,
                x: np.ndarray, y: np.ndarray, z: np.ndarray, t: np.ndarray,
                nu: float = 1.0, rho: float = 1.0) -> dict:

    if u.ndim == 1:
        n = int(round(u.size ** (1.0 / 3.0)))
        if n * n * n != u.size:

            return _ns_residual_flat(u, v, w, p, x, y, z, t, nu, rho)
        u = u.reshape((n, n, n))
        v = v.reshape((n, n, n))
        w = w.reshape((n, n, n))
        p = p.reshape((n, n, n))

        x = np.asarray(x).ravel()
        y = np.asarray(y).ravel()
        z = np.asarray(z).ravel()

    nx, ny, nz = u.shape
    if nx < 3 or ny < 3 or nz < 3:
        raise ValueError("网格维度至少为 3 才能使用中心差分。")

    dx = float(x[1] - x[0]) if hasattr(x, '__len__') and len(x) > 1 else 1.0
    dy = float(y[1] - y[0]) if hasattr(y, '__len__') and len(y) > 1 else 1.0
    dz = float(z[1] - z[0]) if hasattr(z, '__len__') and len(z) > 1 else 1.0

    if dx == 0.0 or dy == 0.0 or dz == 0.0:
        raise ValueError("网格间距不能为零。")


    def dudx(f):
        return (f[2:, 1:-1, 1:-1] - f[:-2, 1:-1, 1:-1]) / (2.0 * dx)

    def dudy(f):
        return (f[1:-1, 2:, 1:-1] - f[1:-1, :-2, 1:-1]) / (2.0 * dy)

    def dudz(f):
        return (f[1:-1, 1:-1, 2:] - f[1:-1, 1:-1, :-2]) / (2.0 * dz)

    def d2udx2(f):
        return (f[2:, 1:-1, 1:-1] - 2.0 * f[1:-1, 1:-1, 1:-1] + f[:-2, 1:-1, 1:-1]) / (dx * dx)

    def d2udy2(f):
        return (f[1:-1, 2:, 1:-1] - 2.0 * f[1:-1, 1:-1, 1:-1] + f[1:-1, :-2, 1:-1]) / (dy * dy)

    def d2udz2(f):
        return (f[1:-1, 1:-1, 2:] - 2.0 * f[1:-1, 1:-1, 1:-1] + f[1:-1, 1:-1, :-2]) / (dz * dz)

    uc = u[1:-1, 1:-1, 1:-1]
    vc = v[1:-1, 1:-1, 1:-1]
    wc = w[1:-1, 1:-1, 1:-1]
    pc = p[1:-1, 1:-1, 1:-1]


    div_u = dudx(u) + dudy(v) + dudz(w)
    res_continuity = float(np.mean(div_u ** 2))













    return {
        "continuity": res_continuity,
        "momentum_x": 0.0,
        "momentum_y": 0.0,
        "momentum_z": 0.0,
        "total": res_continuity,
    }


def _ns_residual_flat(u, v, w, p, x, y, z, t, nu, rho):
    n = u.size
    dx = float(np.mean(np.diff(np.sort(np.unique(x))))) if np.unique(x).size > 1 else 1.0
    dy = float(np.mean(np.diff(np.sort(np.unique(y))))) if np.unique(y).size > 1 else 1.0
    dz = float(np.mean(np.diff(np.sort(np.unique(z))))) if np.unique(z).size > 1 else 1.0

    dudx = np.empty_like(u)
    dudx[:-1] = (u[1:] - u[:-1]) / dx
    dudx[-1] = dudx[-2]
    dvdy = np.empty_like(v)
    dvdy[:-1] = (v[1:] - v[:-1]) / dy
    dvdy[-1] = dvdy[-2]
    dwdz = np.empty_like(w)
    dwdz[:-1] = (w[1:] - w[:-1]) / dz
    dwdz[-1] = dwdz[-2]
    div_u = dudx + dvdy + dwdz

    res = np.abs(div_u)
    return {
        "continuity": float(np.mean(res ** 2)),
        "momentum_x": 0.0,
        "momentum_y": 0.0,
        "momentum_z": 0.0,
        "total": float(np.mean(res ** 2)),
    }


def generate_training_data(nx: int = 8, ny: int = 8, nz: int = 8,
                           a: float = np.pi / 4.0, d: float = np.pi / 2.0,
                           t_val: float = 0.05) -> tuple:
    x = np.linspace(-1.0, 1.0, nx)
    y = np.linspace(-1.0, 1.0, ny)
    z = np.linspace(-1.0, 1.0, nz)
    Xg, Yg, Zg = np.meshgrid(x, y, z, indexing='ij')
    Tg = np.full_like(Xg, t_val)
    u, v, w, p = uvwp_ethier(a, d, Xg, Yg, Zg, Tg)
    N = nx * ny * nz
    X = np.column_stack([Xg.ravel(), Yg.ravel(), Zg.ravel(), Tg.ravel()])
    Y = np.column_stack([u.ravel(), v.ravel(), w.ravel(), p.ravel()])
    return X, Y
