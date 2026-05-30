
import numpy as np


def hyperball01_sample(m, n):
    if m < 1 or n < 1:
        return np.empty((m, n))
    x = np.random.randn(m, n)
    norm = np.sqrt(np.sum(x ** 2, axis=0))
    norm = np.where(norm < 1e-15, 1.0, norm)
    x = x / norm
    r = np.random.rand(1, n) ** (1.0 / m)
    return x * r


def tp_to_xyz(theta, phi):
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    return np.array([x, y, z])


def sphere01_triangle_vertices_to_area(v1, v2, v3):
    v1 = np.asarray(v1, dtype=float)
    v2 = np.asarray(v2, dtype=float)
    v3 = np.asarray(v3, dtype=float)
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    v3 = v3 / np.linalg.norm(v3)


    a = np.arccos(np.clip(np.dot(v2, v3), -1.0, 1.0))
    b = np.arccos(np.clip(np.dot(v1, v3), -1.0, 1.0))
    c = np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))

    s = 0.5 * (a + b + c)

    tan_s2 = np.tan(s * 0.5)
    tan_sa2 = np.tan(max(0.0, (s - a) * 0.5))
    tan_sb2 = np.tan(max(0.0, (s - b) * 0.5))
    tan_sc2 = np.tan(max(0.0, (s - c) * 0.5))

    if tan_s2 <= 0 or tan_sa2 <= 0 or tan_sb2 <= 0 or tan_sc2 <= 0:
        return 0.0

    area = 4.0 * np.arctan(np.sqrt(tan_s2 * tan_sa2 * tan_sb2 * tan_sc2))
    return area


def sphere01_quad_llm(f, h):
    phi_num = max(1, int(np.floor(np.pi / h)))
    theta_num = max(1, int(np.floor(2.0 * np.pi / h)))

    result = 0.0
    n_eval = 0

    if phi_num == 1 and theta_num == 1:
        v = f(np.array([1.0, 0.0, 0.0]))
        return 4.0 * np.pi * v, 1


    phi1 = 0.0
    phi2 = np.pi / phi_num
    for j in range(theta_num):
        theta1 = j * 2.0 * np.pi / theta_num
        theta2 = (j + 1) * 2.0 * np.pi / theta_num
        x1 = tp_to_xyz(theta1, phi1)
        x12 = tp_to_xyz(theta1, phi2)
        x22 = tp_to_xyz(theta2, phi2)
        area = sphere01_triangle_vertices_to_area(x1, x12, x22)
        m1 = 0.5 * (x1 + x12)
        m2 = 0.5 * (x12 + x22)
        m3 = 0.5 * (x22 + x1)
        for m in [m1, m2, m3]:
            m = m / np.linalg.norm(m)
            result += area * f(m) / 3.0
            n_eval += 1


    for i in range(1, phi_num - 1):
        phi1 = i * np.pi / phi_num
        phi2 = (i + 1) * np.pi / phi_num
        for j in range(theta_num):
            theta1 = j * 2.0 * np.pi / theta_num
            theta2 = (j + 1) * 2.0 * np.pi / theta_num
            x11 = tp_to_xyz(theta1, phi1)
            x21 = tp_to_xyz(theta2, phi1)
            x12 = tp_to_xyz(theta1, phi2)
            x22 = tp_to_xyz(theta2, phi2)


            area = sphere01_triangle_vertices_to_area(x11, x12, x22)
            for m in [0.5 * (x11 + x12), 0.5 * (x12 + x22), 0.5 * (x22 + x11)]:
                m = m / np.linalg.norm(m)
                result += area * f(m) / 3.0
                n_eval += 1


            area = sphere01_triangle_vertices_to_area(x22, x21, x11)
            for m in [0.5 * (x22 + x21), 0.5 * (x21 + x11), 0.5 * (x11 + x22)]:
                m = m / np.linalg.norm(m)
                result += area * f(m) / 3.0
                n_eval += 1


    phi1 = (phi_num - 1) * np.pi / phi_num
    phi2 = np.pi
    for j in range(theta_num):
        theta1 = j * 2.0 * np.pi / theta_num
        theta2 = (j + 1) * 2.0 * np.pi / theta_num
        x11 = tp_to_xyz(theta1, phi1)
        x21 = tp_to_xyz(theta2, phi1)
        x2 = tp_to_xyz(theta2, phi2)
        area = sphere01_triangle_vertices_to_area(x11, x2, x21)
        for m in [0.5 * (x11 + x2), 0.5 * (x2 + x21), 0.5 * (x21 + x11)]:
            m = m / np.linalg.norm(m)
            result += area * f(m) / 3.0
            n_eval += 1

    return result, n_eval


def sphere_cvt_step(n_points, xyz):
    xyz = np.asarray(xyz, dtype=float)

    for i in range(n_points):
        norm = np.linalg.norm(xyz[:, i])
        if norm > 1e-15:
            xyz[:, i] = xyz[:, i] / norm



    centroid = np.zeros_like(xyz)
    n_samples = max(1000, n_points * 50)
    samples = np.random.randn(3, n_samples)
    samples = samples / np.linalg.norm(samples, axis=0)


    for s in range(n_samples):
        dists = np.sum((xyz - samples[:, s:s + 1]) ** 2, axis=0)
        idx = np.argmin(dists)
        centroid[:, idx] += samples[:, s]

    for i in range(n_points):
        norm = np.linalg.norm(centroid[:, i])
        if norm > 1e-15:
            centroid[:, i] = centroid[:, i] / norm
        else:
            centroid[:, i] = xyz[:, i]

    return centroid


def monte_carlo_uncertainty_quantification(param_center, param_std, n_samples=1000):
    m = param_center.size
    samples = np.random.randn(m, n_samples) * param_std[:, None] + param_center[:, None]
    return samples
