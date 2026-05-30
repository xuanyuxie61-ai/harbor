
import numpy as np


def chicken_egg_shape(B, L, w, x):
    x = np.clip(x, -L / 2.0 + 1.0e-9, L / 2.0 - 1.0e-9)
    numerator = L ** 2 - 4.0 * x ** 2
    denominator = L ** 2 + 8.0 * w * x + 4.0 * w ** 2


    numerator = np.maximum(numerator, 0.0)
    denominator = np.maximum(denominator, 1.0e-12)

    r = 0.5 * B * np.sqrt(numerator / denominator)
    return r


def pyriform_egg_shape(B, L, w, x):
    x = np.clip(x, -L / 2.0 + 1.0e-9, L / 2.0 - 1.0e-9)
    t1 = (L ** 2 - 4.0 * x ** 2) * L
    t2 = (2.0 * (L - 2.0 * w) * x ** 2 +
          (L ** 2 + 8.0 * L * w - 4.0 * w ** 2) * x +
          2.0 * L * w ** 2 + L ** 2 * w + L ** 3)

    t1 = np.maximum(t1, 0.0)
    t2 = np.maximum(t2, 1.0e-12)

    r = 0.5 * B * np.sqrt(t1 / t2)
    return r


def universal_egg_shape(B, L, w, D, x):
    r_chicken = chicken_egg_shape(B, L, w, x)

    s1 = np.sqrt(5.5 * L ** 2 + 11.0 * L * w + 4.0 * w ** 2)
    s2 = (np.sqrt(3.0) * B * L -
          2.0 * D * np.sqrt(L ** 2 + 2.0 * w * L + 4.0 * w ** 2))
    s3 = s1
    s4 = 2.0 * np.sqrt(L ** 2 + 2.0 * w * L + 4.0 * w ** 2)

    denom_t2 = np.sqrt(3.0) * (s3 - s4)
    denom_t2 = np.where(np.abs(denom_t2) < 1.0e-12, 1.0e-12, denom_t2)
    t2 = (s1 * s2) / denom_t2

    s5 = L * (L ** 2 + 8.0 * w * x + 4.0 * w ** 2)
    s6 = (2.0 * (L - 2.0 * w) * x ** 2 +
          (L ** 2 + 8.0 * L * w - 4.0 * w ** 2) * x +
          2.0 * L * w ** 2 + L ** 2 * w + L ** 3)

    s5 = np.maximum(s5, 0.0)
    s6 = np.maximum(s6, 1.0e-12)
    t3 = 1.0 - np.sqrt(s5 / s6)

    r = r_chicken * (1.0 - t2 * t3)
    r = np.maximum(r, 0.0)
    return r


def flame_front_surface_area(B, L, w, Ka=0.0, num_points=200):

    B_eff = B * (1.0 + 0.1 * Ka)
    w_eff = w * np.exp(-Ka / 10.0)
    L_eff = L * (1.0 + 0.05 * Ka)

    x = np.linspace(-L_eff / 2.0, L_eff / 2.0, num_points)
    dx = x[1] - x[0]

    r = chicken_egg_shape(B_eff, L_eff, w_eff, x)


    dr_dx = np.zeros_like(r)
    dr_dx[1:-1] = (r[2:] - r[:-2]) / (2.0 * dx)
    dr_dx[0] = (r[1] - r[0]) / dx
    dr_dx[-1] = (r[-1] - r[-2]) / dx

    integrand = 2.0 * np.pi * r * np.sqrt(1.0 + dr_dx ** 2)
    area = np.trapezoid(integrand, x)


    area_projected = np.pi * (B_eff / 2.0) ** 2
    area_ratio = area / area_projected if area_projected > 0 else 1.0

    return area, area_ratio
