
import numpy as np
from math import sqrt, log, cos, sin, pi, acos


class MonteCarloError(Exception):
    pass


def sample_hg_scattering_angle(g):
    xi = np.random.rand()
    if abs(g) < 1e-8:
        return 2.0 * xi - 1.0


    denom = 1.0 - g + 2.0 * g * xi
    if abs(denom) < 1e-15:
        denom = 1e-15

    t = (1.0 - g ** 2) / denom
    cos_theta = (1.0 + g ** 2 - t ** 2) / (2.0 * g)

    return float(np.clip(cos_theta, -1.0, 1.0))


def rotate_direction(u, v, w, cos_theta):
    sin_theta = sqrt(max(0.0, 1.0 - cos_theta ** 2))
    phi = 2.0 * pi * np.random.rand()


    if abs(w) > 0.99999:
        u_new = sin_theta * cos(phi)
        v_new = sin_theta * sin(phi)
        w_new = np.sign(w) * cos_theta
    else:
        temp = sqrt(1.0 - w ** 2)
        u_new = sin_theta * (u * w * cos(phi) - v * sin(phi)) / temp + u * cos_theta
        v_new = sin_theta * (v * w * cos(phi) + u * sin(phi)) / temp + v * cos_theta
        w_new = -sin_theta * cos(phi) * temp + w * cos_theta


    norm = sqrt(u_new ** 2 + v_new ** 2 + w_new ** 2)
    if norm < 1e-15:
        return 0.0, 0.0, 1.0
    return u_new / norm, v_new / norm, w_new / norm


def photon_random_walk_3d(
    num_photons=1000,
    max_steps=200,
    extinction_coeff=1.0,
    layer_height=10.0,
    g_asymmetry=0.6,
    albedo=0.9,
    surface_albedo=0.2,
):
    if extinction_coeff <= 0 or layer_height <= 0:
        raise MonteCarloError("photon_random_walk_3d: 物理参数必须为正")

    escaped_up = 0
    absorbed_surface = 0
    absorbed_atm = 0
    path_lengths = []

    for _ in range(num_photons):

        x, y, z = 0.0, 0.0, layer_height

        u, v, w = 0.0, 0.0, -1.0
        path_len = 0.0

        for _ in range(max_steps):

            xi = np.random.rand()
            if xi < 1e-15:
                xi = 1e-15
            free_path = -log(xi) / extinction_coeff


            x_new = x + free_path * u
            y_new = y + free_path * v
            z_new = z + free_path * w

            path_len += free_path


            if z_new > layer_height:

                escaped_up += 1
                break
            if z_new < 0:

                if np.random.rand() < surface_albedo:

                    z_new = 0.0
                    w = abs(w)
                    x, y, z = x_new, y_new, z_new
                    continue
                else:
                    absorbed_surface += 1
                    break


            x, y, z = x_new, y_new, z_new

            if np.random.rand() > albedo:
                absorbed_atm += 1
                break


            cos_theta = sample_hg_scattering_angle(g_asymmetry)
            u, v, w = rotate_direction(u, v, w, cos_theta)
        else:

            absorbed_atm += 1

        path_lengths.append(path_len)

    return escaped_up, absorbed_surface, absorbed_atm, np.array(path_lengths)


def estimate_optical_depth_monte_carlo(path_lengths, layer_height):
    if len(path_lengths) == 0:
        return 0.0
    mean_path = np.mean(path_lengths)
    if mean_path <= 0:
        return 0.0
    return float(layer_height / mean_path)
