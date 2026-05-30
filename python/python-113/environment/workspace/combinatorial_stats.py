
import numpy as np


def binomial_coefficient(n, k):
    if k < 0 or k > n:
        return 0
    if k == 0 or k == n:
        return 1

    k = min(k, n - k)
    result = 1
    for i in range(1, k + 1):
        result = result * (n - k + i) // i
    return result


def combination_lex_index(n, p, l):
    if p <= 0 or p > n:
        raise ValueError("p 必须在 (0, n] 范围内")
    if l < 1 or l > binomial_coefficient(n, p):
        raise ValueError("索引 l 超出范围")

    c = np.zeros(p, dtype=int)
    if p == 1:
        c[0] = l
        return c

    k = 0
    p1 = p - 1
    c[0] = 0

    for i in range(p1):
        if i > 0:
            c[i] = c[i - 1]
        while True:
            c[i] += 1
            r = binomial_coefficient(n - c[i], p - i - 1)
            k += r
            if l <= k:
                break
        k -= r

    c[p - 1] = c[p1 - 1] + l - k
    return c


def enumerate_occupations(n_sites, n_ions):
    total = binomial_coefficient(n_sites, n_ions)
    configs = []
    for l in range(1, total + 1):
        c = combination_lex_index(n_sites, n_ions, l)
        configs.append(c - 1)
    return configs


def multinomial_coefficient(n, groups):
    if sum(groups) != n:
        return 0
    result = np.math.factorial(n)
    for g in groups:
        result //= np.math.factorial(g)
    return result


def gamma_log_table(n_values):
    from special_functions import log_gamma_lanczos
    return np.array([log_gamma_lanczos(v) for v in n_values])


def canonical_partition_function(energies, T=300.0):
    kB = 1.380649e-23
    beta = 1.0 / (kB * T)
    e_min = np.min(energies)
    z = np.sum(np.exp(-beta * (energies - e_min)))
    return e_min + np.log(z)


def occupancy_probability(n_sites, n_ions, energy_func, T=300.0):
    configs = enumerate_occupations(n_sites, n_ions)
    kB = 1.380649e-23
    beta = 1.0 / (kB * T)

    weights = []
    for conf in configs:
        e = energy_func(conf)
        weights.append(np.exp(-beta * e))
    weights = np.array(weights)
    Z = np.sum(weights)

    probs = np.zeros(n_sites)
    for idx, conf in enumerate(configs):
        for site in conf:
            probs[site] += weights[idx]
    probs /= Z
    return probs
