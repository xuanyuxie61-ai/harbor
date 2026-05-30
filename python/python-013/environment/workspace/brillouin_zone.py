
import numpy as np
from typing import Tuple, List






LEBEDEV_ORDERS = [6, 14, 26, 38, 50, 74, 86, 110, 146, 170, 194, 230, 266, 302, 350, 434]


def lebedev_sphere_grid(order: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if order not in LEBEDEV_ORDERS:

        order = min(LEBEDEV_ORDERS, key=lambda o: abs(o - order))

    n = order
    phi = np.pi * (3.0 - np.sqrt(5.0))
    i = np.arange(n, dtype=np.float64)
    y = 1.0 - (i / (n - 1)) * 2.0
    radius = np.sqrt(1.0 - y * y)
    theta = phi * i
    x = np.cos(theta) * radius
    z = np.sin(theta) * radius

    w = np.full(n, 4.0 * np.pi / n)
    return x, y, z, w


def integrate_on_fermi_surface(fermi_energy: float, band_func, order: int = 110) -> float:
    x, y, z, w = lebedev_sphere_grid(order)
    kf = np.sqrt(max(fermi_energy, 0.0)) * 2.0
    result = 0.0
    for i in range(len(w)):
        kvec = kf * np.array([x[i], y[i], z[i]])
        result += w[i] * band_func(kvec)
    return result






def pyramid_rule_3d(f, n_legendre: int = 8, n_jacobi: int = 8) -> float:
    if n_legendre < 1 or n_jacobi < 1:
        raise ValueError("阶数必须 >= 1")

    xg, wg = np.polynomial.legendre.leggauss(n_legendre)

    zg, wj = np.polynomial.legendre.leggauss(n_jacobi)


    result = 0.0
    for k in range(n_jacobi):
        zk = (zg[k] + 1.0) * 0.5
        wk = wj[k] * 0.5
        scale_xy = 1.0 - zk
        for j in range(n_legendre):
            yj = xg[j] * scale_xy
            wj_y = wg[j] * scale_xy
            for i in range(n_legendre):
                xi = xg[i] * scale_xy
                wi_x = wg[i] * scale_xy
                result += wi_x * wj_y * wk * f(xi, yj, zk)

    volume = 4.0 / 3.0
    return result * volume






def hypersphere_surface_uniform(m: int, n: int) -> np.ndarray:
    if m < 1 or n < 1:
        raise ValueError("m, n >= 1 required")
    x = np.random.randn(m, n)
    norms = np.sqrt(np.sum(x ** 2, axis=0))
    norms = np.where(norms > 0, norms, 1.0)
    x = x / norms[np.newaxis, :]
    return x


def tetrahedron_method_integrate(k_points: np.ndarray, energies: np.ndarray,
                                 omega: float, eta: float = 0.05) -> complex:
    from scipy.spatial import Delaunay
    if len(k_points) < 3:
        raise ValueError("至少需要 3 个 k 点")
    tri = Delaunay(k_points)
    result = 0.0 + 0.0j
    dim = k_points.shape[1]
    for simplex in tri.simplices:
        pts = k_points[simplex]
        ens = energies[simplex]
        if dim == 2:

            if len(simplex) != 3:
                continue
            p0, p1, p2 = pts
            e0, e1, e2 = ens



        elif dim == 3:

            if len(simplex) != 4:
                continue
            k0, k1, k2, k3 = pts
            e0, e1, e2, e3 = ens
            M = np.vstack([k1 - k0, k2 - k0, k3 - k0])
            vol = abs(np.linalg.det(M)) / 6.0
            e_avg = (e0 + e1 + e2 + e3) / 4.0
            result += vol / (omega - e_avg + 1j * eta)
    return -result / np.pi


def compute_dos_tetrahedron(k_points: np.ndarray, band_energies: np.ndarray,
                            omega_grid: np.ndarray, eta: float = 0.05) -> np.ndarray:
    dos = np.zeros(len(omega_grid), dtype=np.float64)
    for i, w in enumerate(omega_grid):
        val = tetrahedron_method_integrate(k_points, band_energies, w, eta)
        dos[i] = -val.imag

    if np.max(dos) > 0:
        dos = dos / np.trapezoid(dos, omega_grid)
    return dos


def brillouin_zone_area(bz_vertices: np.ndarray) -> float:
    if len(bz_vertices) < 3:
        return 0.0
    x = bz_vertices[:, 0]
    y = bz_vertices[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


if __name__ == "__main__":
    x, y, z, w = lebedev_sphere_grid(50)
    print(f"Lebedev grid: {len(w)} points, weight sum = {np.sum(w):.6f} (expect {4*np.pi:.6f})")
    pts = hypersphere_surface_uniform(4, 100)
    print(f"Hypersphere norms: {np.mean(np.sqrt(np.sum(pts**2, axis=0))):.6f}")
