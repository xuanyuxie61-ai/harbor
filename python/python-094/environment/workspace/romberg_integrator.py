
import numpy as np


def tuple_next(m, base, tuple_vec):
    tuple_vec = np.asarray(tuple_vec, dtype=int)
    rank = -1
    for i in range(m - 1, -1, -1):
        if tuple_vec[i] < base:
            tuple_vec[i] += 1
            rank = 0
            return tuple_vec, rank
        tuple_vec[i] = 1
    return tuple_vec, rank


def monte_carlo_nd(func, a, b, dim_num, n_eval):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    volume = np.prod(b - a)
    total = 0.0
    total_sq = 0.0

    for _ in range(n_eval):
        x = a + np.random.rand(dim_num) * (b - a)
        fx = func(x)
        total += fx
        total_sq += fx ** 2

    mean = total / n_eval
    variance = (total_sq / n_eval) - mean ** 2
    std_err = volume * np.sqrt(variance / n_eval)
    result = volume * mean
    return result, std_err, n_eval


def romberg_nd(func, a, b, dim_num, sub_num, it_max, tol):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    sub_num = np.asarray(sub_num, dtype=int)

    if dim_num < 1:
        raise ValueError("dim_num must be >= 1.")
    if it_max < 1:
        raise ValueError("it_max must be >= 1.")
    if np.any(sub_num <= 0):
        raise ValueError("sub_num must be positive.")

    eval_num = 0
    ind = 0
    rnderr = np.finfo(float).eps
    iwork2 = np.zeros(it_max, dtype=int)
    iwork2[0] = 1
    if it_max > 1:
        iwork2[1] = 2

    sub_num2 = sub_num.copy()
    table = np.zeros(it_max, dtype=float)

    result = 0.0
    result_old = 0.0

    for it in range(it_max):
        weight = np.prod((b - a) / sub_num2)
        sum_val = 0.0


        iwork = np.ones(dim_num, dtype=int)
        while True:
            x = ((2.0 * sub_num2 - 2.0 * iwork + 1.0) * a +
                 (2.0 * iwork - 1.0) * b) / (2.0 * sub_num2)
            sum_val += func(x)
            eval_num += 1

            kdim = dim_num - 1
            while kdim >= 0:
                if iwork[kdim] < sub_num2[kdim]:
                    iwork[kdim] += 1
                    break
                iwork[kdim] = 1
                kdim -= 1
            if kdim < 0:
                break

        table[it] = weight * sum_val

        if it == 0:
            result = table[0]
            result_old = result
            if it_max <= 1:
                ind = 1
                break
            if it_max > 1:
                sub_num2 = iwork2[it + 1] * sub_num2 if (it + 1) < it_max else sub_num2
            continue


        for ll in range(2, it + 2):
            i = it + 1 - ll
            factor = (iwork2[i] ** 2) / (iwork2[it] ** 2 - iwork2[i] ** 2)
            table[i] = table[i + 1] + (table[i + 1] - table[i]) * factor

        result = table[0]

        if abs(result - result_old) <= abs(result * (tol + rnderr)):
            ind = 1
            break

        if it >= it_max - 1:
            ind = -1
            break

        result_old = result
        if it + 1 < it_max:
            iwork2[it + 1] = round(1.5 * iwork2[it])
            sub_num2 = iwork2[it + 1] * sub_num2

    return result, ind, eval_num


class AcousticEnergyIntegrator:

    def __init__(self, physics):
        self.physics = physics

    def beam_energy_3d(self, p_func, r_max, z_max, tau_max,
                       n_samples=10000, method='monte_carlo'):
        rho0 = self.physics.rho0
        c0 = self.physics.c0
        prefactor = 1.0 / (2.0 * rho0 * c0 ** 2)

        if method == 'monte_carlo':
            def integrand(x):
                r, z, tau = x
                if r < 0.0 or z < 0.0 or abs(tau) > tau_max:
                    return 0.0
                p_val = p_func(r, z, tau)
                if not np.isfinite(p_val):
                    return 0.0
                return prefactor * p_val ** 2 * 2.0 * np.pi * r

            a = np.array([0.0, 0.0, -tau_max])
            b = np.array([r_max, z_max, tau_max])
            result, _, _ = monte_carlo_nd(integrand, a, b, 3, n_samples)
            return result

        elif method == 'romberg':
            def integrand(x):
                r, z, tau = x
                if r < 0.0:
                    r = 0.0
                p_val = p_func(r, z, tau)
                if not np.isfinite(p_val):
                    return 0.0
                return prefactor * p_val ** 2 * 2.0 * np.pi * r

            a = np.array([0.0, 0.0, -tau_max])
            b = np.array([r_max, z_max, tau_max])
            sub_num = np.array([4, 4, 4])
            result, ind, evals = romberg_nd(integrand, a, b, 3, sub_num, it_max=4, tol=1e-3)
            if ind != 1:

                result, _, _ = monte_carlo_nd(integrand, a, b, 3, n_samples)
            return result
        else:
            raise ValueError(f"Unknown integration method: {method}")

    def spatial_average_pressure(self, p_field, r_grid, z_grid):
        p_field = np.asarray(p_field, dtype=float)
        r_grid = np.asarray(r_grid, dtype=float)
        if p_field.ndim != 2:
            raise ValueError("p_field must be 2D.")
        Nr, Nz = p_field.shape
        if r_grid.size != Nr:
            raise ValueError("r_grid size must match p_field first dimension.")

        p_avg = np.zeros(Nz, dtype=float)
        for j in range(Nz):
            integrand = p_field[:, j] * 2.0 * np.pi * r_grid

            p_avg[j] = np.trapezoid(integrand, r_grid) / (np.pi * r_grid[-1] ** 2)
        return p_avg
