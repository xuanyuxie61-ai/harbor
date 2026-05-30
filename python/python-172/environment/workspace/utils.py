# -*- coding: utf-8 -*-

import numpy as np


def enforce_dirichlet(u, val_left, val_right):
    u = np.asarray(u, dtype=np.float64).copy()
    u[-1] = val_left
    u[0] = val_right
    return u


def enforce_neumann(u, D, val_left=0.0, val_right=0.0):
    n = len(u)
    A_bc = np.zeros((2, n), dtype=np.float64)
    b_bc = np.array([val_right, val_left], dtype=np.float64)
    A_bc[0, :] = D[0, :]
    A_bc[1, :] = D[-1, :]
    return A_bc, b_bc


def check_solution_stability(u, max_val=1e6, min_val=-1e6):
    u = np.asarray(u)
    if np.any(np.isnan(u)) or np.any(np.isinf(u)):
        return False
    if np.any(u > max_val) or np.any(u < min_val):
        return False
    return True


def smooth_initial_condition(x, case="gaussian"):
    x = np.asarray(x, dtype=np.float64)
    if case == "gaussian":
        return np.exp(-10.0 * x ** 2)
    elif case == "sine":
        return np.sin(np.pi * (x + 1.0) / 2.0)
    elif case == "poly":
        return (1.0 - x ** 2) ** 2
    else:
        return np.exp(-10.0 * x ** 2)


def map_domain(x, a, b):
    return 0.5 * (b - a) * x + 0.5 * (a + b)


def relative_l2_error(u_num, u_ref, weights=None):
    u_num = np.asarray(u_num, dtype=np.float64)
    u_ref = np.asarray(u_ref, dtype=np.float64)
    diff = u_num - u_ref
    if weights is None:
        num = np.sqrt(np.mean(diff ** 2))
        den = np.sqrt(np.mean(u_ref ** 2))
    else:
        num = np.sqrt(np.sum(weights * diff ** 2))
        den = np.sqrt(np.sum(weights * u_ref ** 2))
    if den < 1e-15:
        return num
    return num / den


def print_banner(title, width=70):
    print("=" * width)
    print(title.center(width))
    print("=" * width)
