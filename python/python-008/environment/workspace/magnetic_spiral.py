
import numpy as np


def spiral_array(thick, base=1):
    n = 2 * thick + 1
    S = np.zeros((n, n), dtype=int)

    row = thick
    col = thick - 1
    k = base - 1

    for t in range(thick + 1):
        col += 1
        k += 1
        S[row, col] = k
        done = False

        while not done:
            if row == t + thick and col == t + thick:
                done = True
                break
            elif col == t + thick and -t + thick < row:
                row -= 1
            elif row == -t + thick and -t + thick < col:
                col -= 1
            elif col == -t + thick and row < t + thick:
                row += 1
            elif row == t + thick and col < t + thick:
                col += 1
            k += 1
            S[row, col] = k

    return S


def prime_spiral_mask(thick):
    S = spiral_array(thick)

    def is_prime(n):
        if n < 2:
            return False
        if n % 2 == 0:
            return n == 2
        r = int(np.sqrt(n))
        for d in range(3, r + 1, 2):
            if n % d == 0:
                return False
        return True

    mask = np.vectorize(is_prime)(S)
    return mask


def magnetic_pitch_angle_grid(n_r=32, r_max=1e13, v_z=2.99e10, Omega=1e-3):
    r = np.linspace(0.0, r_max, n_r)
    r_safe = np.where(r > 1e-6, r, 1e-6)

    tan_psi = Omega * r_safe / v_z
    tan_psi = np.clip(tan_psi, 0.0, 10.0)
    psi = np.arctan(tan_psi)

    B_ratio = tan_psi
    return r, psi, B_ratio


def magnetization_parameter(rho, B, Gamma):
    c = 2.99792458e10
    sigma = B ** 2 / (4.0 * np.pi * Gamma * rho * c ** 2)
    sigma = np.clip(sigma, 0.0, 1e6)
    return sigma
