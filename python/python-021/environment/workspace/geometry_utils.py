
import numpy as np
from parameters import R0, a_minor, KAPPA, DELTA, N_FEKETE


def point_in_flux_surface(R_test, Z_test, theta_poly=None, n_theta=128):
    if theta_poly is None:
        theta_poly = np.linspace(0, 2.0 * np.pi, n_theta, endpoint=False)


    R_poly = R0 + a_minor * np.cos(theta_poly + DELTA * np.sin(theta_poly))
    Z_poly = KAPPA * a_minor * np.sin(theta_poly)

    n = len(R_poly)
    inside = False

    for i in range(n):
        ip1 = (i + 1) % n
        y_i = Z_poly[i]
        y_ip1 = Z_poly[ip1]


        cond = (y_ip1 < Z_test) == (Z_test <= y_i)
        if cond:
            x_i = R_poly[i]
            x_ip1 = R_poly[ip1]
            t = R_test - x_i - (Z_test - y_i) * (x_ip1 - x_i) / (y_ip1 - y_i + 1e-20)
            if t < 0.0:
                inside = not inside

    return inside, R_poly, Z_poly


def compute_poloidal_area(theta_poly=None, n_theta=256):
    if theta_poly is None:
        theta_poly = np.linspace(0, 2.0 * np.pi, n_theta, endpoint=False)

    R_poly = R0 + a_minor * np.cos(theta_poly + DELTA * np.sin(theta_poly))
    Z_poly = KAPPA * a_minor * np.sin(theta_poly)

    n = len(R_poly)
    area = 0.0
    for i in range(n):
        ip1 = (i + 1) % n
        area += R_poly[i] * Z_poly[ip1] - R_poly[ip1] * Z_poly[i]

    return 0.5 * abs(area), R_poly, Z_poly


def compute_toroidal_volume(theta_poly=None, n_theta=256, n_radial=64):

    r_nodes = np.linspace(0, a_minor, n_radial)
    dr = r_nodes[1] - r_nodes[0] if n_radial > 1 else 0
    theta = np.linspace(0, 2.0 * np.pi, n_theta)
    dtheta = theta[1] - theta[0] if n_theta > 1 else 0

    V = 0.0
    for r in r_nodes:
        R_loc = R0 + r * np.cos(theta + DELTA * np.sin(theta))
        V += np.sum(R_loc) * dtheta * r * dr

    volume = 2.0 * np.pi * V
    volume_approx = 2.0 * (np.pi ** 2) * R0 * (a_minor ** 2) * KAPPA

    return volume, volume_approx


def fekete_points_on_flux_surface(m=N_FEKETE, n_sample=400):
    from quadrature_engine import chebyshev_vandermonde

    theta_sample = np.linspace(0, 2.0 * np.pi, n_sample)
    R_sample = R0 + a_minor * np.cos(theta_sample + DELTA * np.sin(theta_sample))
    Z_sample = KAPPA * a_minor * np.sin(theta_sample)


    ds = np.sqrt(np.gradient(R_sample) ** 2 + np.gradient(Z_sample) ** 2)
    s = np.cumsum(ds)
    s = np.concatenate(([0.0], s[:-1]))
    s_total = s[-1] + ds[-1]
    if s_total < 1e-15:
        raise ValueError("弧长计算失败")


    s_norm = s / s_total


    a, b = 0.0, 1.0
    V = chebyshev_vandermonde(m, a, b, s_norm)


    mom = np.zeros(m)
    mom[0] = s_total
    for k in range(1, m):

        integrand = np.cos(k * np.arccos(np.clip(2.0 * s_norm - 1.0, -1.0, 1.0)))
        mom[k] = np.trapezoid(integrand, s)


    w, _, _, _ = np.linalg.lstsq(V, mom, rcond=None)


    threshold = 1e-12 * np.max(np.abs(w))
    ind = np.where(np.abs(w) > threshold)[0]
    if len(ind) < m:
        ind = np.argsort(np.abs(w))[-m:]

    theta_fekete = theta_sample[ind]
    R_fekete = R0 + a_minor * np.cos(theta_fekete + DELTA * np.sin(theta_fekete))
    Z_fekete = KAPPA * a_minor * np.sin(theta_fekete)
    weights = w[ind]

    return theta_fekete, R_fekete, Z_fekete, weights


def compute_curvature_and_torsion(theta_points):
    theta = np.asarray(theta_points)
    dtheta = theta[1] - theta[0] if len(theta) > 1 else 1.0

    R = R0 + a_minor * np.cos(theta + DELTA * np.sin(theta))
    Z = KAPPA * a_minor * np.sin(theta)

    dR = np.gradient(R, dtheta)
    dZ = np.gradient(Z, dtheta)
    d2R = np.gradient(dR, dtheta)
    d2Z = np.gradient(dZ, dtheta)

    denom = (dR ** 2 + dZ ** 2) ** 1.5 + 1e-20
    kappa = np.abs(dR * d2Z - d2R * dZ) / denom
    radius_curvature = 1.0 / (kappa + 1e-20)

    return kappa, radius_curvature


def generate_triangular_mesh(n_r=8, n_theta=16):
    vertices = []
    for j in range(n_r + 1):
        r = a_minor * j / n_r
        for i in range(n_theta):
            theta = 2.0 * np.pi * i / n_theta
            R = R0 + r * np.cos(theta + DELTA * np.sin(theta))
            Z = KAPPA * r * np.sin(theta)
            vertices.append([R, Z])
    vertices = np.array(vertices)

    triangles = []
    for j in range(n_r):
        for i in range(n_theta):
            v0 = j * n_theta + i
            v1 = j * n_theta + ((i + 1) % n_theta)
            v2 = (j + 1) * n_theta + i
            v3 = (j + 1) * n_theta + ((i + 1) % n_theta)
            triangles.append([v0, v1, v2])
            triangles.append([v1, v3, v2])
    triangles = np.array(triangles)

    return vertices, triangles
