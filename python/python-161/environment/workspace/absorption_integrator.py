
import numpy as np
from typing import Tuple, Callable


def wedge01_volume(length_xy: float = 1.0, thickness_z: float = 2.0) -> float:
    if length_xy <= 0 or thickness_z <= 0:
        return 0.0
    return (length_xy ** 2) * thickness_z / 2.0


def wedge01_integral(exponents: np.ndarray, length_xy: float = 1.0, thickness_z: float = 2.0) -> float:
    e = np.asarray(exponents, dtype=int)
    if np.any(e < 0):
        raise ValueError("单项式指数必须非负")
    if length_xy <= 0 or thickness_z <= 0:
        return 0.0



    val_xy = 1.0
    k = e[0]
    for i in range(1, e[1] + 1):
        k += 1
        val_xy *= i / k
    k += 1
    val_xy /= k
    k += 1
    val_xy /= k

    val_xy *= (length_xy ** (e[0] + e[1] + 2))


    if e[2] % 2 == 1:
        val_z = 0.0
    else:
        val_z = 2.0 * ((thickness_z / 2.0) ** (e[2] + 1)) / (e[2] + 1)

    return val_xy * val_z


def generate_wedge_gauss_rule(
    order_xy: int = 4, order_z: int = 4,
    length_xy: float = 1.0, thickness_z: float = 2.0
) -> Tuple[np.ndarray, np.ndarray]:
    if order_xy < 1 or order_z < 1:
        raise ValueError("求积阶数必须为正")


    z_nodes, z_weights = np.polynomial.legendre.leggauss(order_z)
    z_nodes = z_nodes * (thickness_z / 2.0)
    z_weights = z_weights * (thickness_z / 2.0)


    tri_points, tri_weights = _dunavant_triangle_rule(order_xy)

    tri_points = tri_points * length_xy
    tri_weights = tri_weights * (length_xy ** 2 / 2.0)


    n_total = len(tri_weights) * len(z_weights)
    points = np.zeros((n_total, 3))
    weights = np.zeros(n_total)

    idx = 0
    for i in range(len(tri_weights)):
        for j in range(len(z_weights)):
            points[idx, 0] = tri_points[i, 0]
            points[idx, 1] = tri_points[i, 1]
            points[idx, 2] = z_nodes[j]
            weights[idx] = tri_weights[i] * z_weights[j]
            idx += 1

    return points, weights


def _dunavant_triangle_rule(order: int) -> Tuple[np.ndarray, np.ndarray]:
    if order == 1:
        pts = np.array([[1.0 / 3.0, 1.0 / 3.0]])
        w = np.array([1.0])
    elif order == 2:
        pts = np.array([[2.0 / 3.0, 1.0 / 6.0], [1.0 / 6.0, 2.0 / 3.0], [1.0 / 6.0, 1.0 / 6.0]])
        w = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
    elif order == 3:
        pts = np.array([
            [1.0 / 3.0, 1.0 / 3.0],
            [0.6, 0.2],
            [0.2, 0.6],
            [0.2, 0.2],
        ])
        w = np.array([-27.0 / 48.0, 25.0 / 48.0, 25.0 / 48.0, 25.0 / 48.0])
    elif order == 4:
        a = 0.445948490915965
        b = 0.091576213509771
        pts = np.array([
            [a, a], [1.0 - 2.0 * a, a], [a, 1.0 - 2.0 * a],
            [b, b], [1.0 - 2.0 * b, b], [b, 1.0 - 2.0 * b],
        ])
        w = np.array([0.111690794839005, 0.111690794839005, 0.111690794839005,
                      0.054975871827661, 0.054975871827661, 0.054975871827661])
    else:

        a = 0.470142064105115
        b = 0.101286507323456
        c = 0.333333333333333
        pts = np.array([
            [a, a], [1.0 - 2.0 * a, a], [a, 1.0 - 2.0 * a],
            [b, b], [1.0 - 2.0 * b, b], [b, 1.0 - 2.0 * b],
            [c, c],
        ])
        w = np.array([0.066197076394253, 0.066197076394253, 0.066197076394253,
                      0.062969590272413, 0.062969590272413, 0.062969590272413,
                      0.112500000000000])
    return pts, w


def evaluate_monomial(dim: int, npts: int, e: np.ndarray, x: np.ndarray) -> np.ndarray:
    v = np.ones(npts)
    for i in range(dim):
        if e[i] != 0:
            v *= x[:, i] ** e[i]
    return v


def compute_carrier_generation_rate(
    absorption_coeff: Callable[[np.ndarray], np.ndarray],
    irradiance_fn: Callable[[np.ndarray], np.ndarray],
    photon_energy_ev_fn: Callable[[np.ndarray], np.ndarray],
    length_xy: float = 1.0e-4,
    thickness_z: float = 5.0e-5,
    order_xy: int = 4,
    order_z: int = 4,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    points, weights = generate_wedge_gauss_rule(order_xy, order_z, length_xy, thickness_z)
    npts = points.shape[0]



    lambda_ref = 600.0
    alpha = absorption_coeff(np.array([lambda_ref]))[0]
    I0 = irradiance_fn(np.array([lambda_ref]))[0]
    Eph = photon_energy_ev_fn(np.array([lambda_ref]))[0]

    if alpha < 0 or I0 < 0 or Eph <= 0:
        raise ValueError("物理参数必须满足 α≥0, I0≥0, E_photon>0")





    z_surface = -thickness_z / 2.0
    depth = points[:, 2] - z_surface
    gen_density = np.zeros(npts)


    gen_density = np.where(np.isfinite(gen_density), gen_density, 0.0)
    gen_density = np.maximum(gen_density, 0.0)


    total_gen_rate = float(np.dot(weights, gen_density))

    return total_gen_rate, points, gen_density, weights


def test_exactness(degree_max: int = 3, length_xy: float = 1.0, thickness_z: float = 2.0) -> None:
    points, weights = generate_wedge_gauss_rule(order_xy=5, order_z=5,
                                                 length_xy=length_xy, thickness_z=thickness_z)
    npts = points.shape[0]
    dim = 3
    print("=== Wedge Quadrature Exactness Test ===")
    for degree in range(degree_max + 1):

        exponents_list = []
        for e1 in range(degree + 1):
            for e2 in range(degree - e1 + 1):
                e3 = degree - e1 - e2
                exponents_list.append([e1, e2, e3])
        for e in exponents_list:
            e_arr = np.array(e, dtype=int)
            v = evaluate_monomial(dim, npts, e_arr, points)
            quad = wedge01_volume(length_xy, thickness_z) * np.dot(weights / weights.sum(), v)
            exact = wedge01_integral(e_arr, length_xy, thickness_z)
            err = abs(quad - exact)
            print(f"  Degree {degree}, exponents {e}: quad={quad:.6e}, exact={exact:.6e}, err={err:.3e}")


if __name__ == "__main__":
    test_exactness(degree_max=3)
