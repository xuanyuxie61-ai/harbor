"""
Utility functions for the membrane separation simulation.

Includes modular arithmetic helpers (adapted from doomsday calendar algorithm),
auxiliary numerical safeguards, and data-formatting routines.
"""

import numpy as np


def safe_sqrt(x):
    """
    Square root with strict non-negativity guard.
    """
    x = float(x)
    if x < 0.0:
        if x > -1e-14:
            return 0.0
        raise ValueError(f"safe_sqrt called with negative argument: {x}")
    return np.sqrt(x)


def safe_divide(a, b, default=0.0):
    """
    Division with zero-denominator protection.
    """
    b = float(b)
    if abs(b) < 1e-30:
        return float(default)
    return float(a) / b


def modular_wrap(value, lower, upper):
    """
    Wrap integer VALUE into the inclusive range [LOWER, UPPER].
    Adapted from the doomsday calendar i4_wrap routine.
    """
    if lower == upper:
        return lower
    value = int(value)
    lower = int(lower)
    upper = int(upper)
    if upper < lower:
        lower, upper = upper, lower
    width = upper - lower + 1
    if width == 1:
        return lower
    offset = (value - lower) % width
    return lower + offset


def cyclic_schedule_step(current, period, shift=1):
    """
    Compute the next step in a cyclic regeneration schedule using modular arithmetic.

    This maps the doomsday modular-calendar concept to membrane module
    regeneration scheduling.
    """
    return modular_wrap(current + shift, 1, period)


def compute_regeneration_calendar(year, month, day, cycle_period_days=30):
    """
    Determine the next regeneration date for a membrane module given a
    cyclic maintenance period.  Uses modular arithmetic adapted from the
    doomsday algorithm.
    """
    if year <= 0 or month < 1 or month > 12 or day < 1 or day > 31:
        raise ValueError("Invalid date.")
    # Simplified: next regeneration day = current day wrapped in cycle period
    next_day = modular_wrap(day + cycle_period_days, 1, 31)
    return {"next_regeneration_day": next_day, "cycle_period": cycle_period_days}


def bracket_interval(sorted_array, values):
    """
    For each value in VALUES, find the index LEFT such that
    sorted_array[left] <= value <= sorted_array[left+1].
    Adapted from the r8vec_bracket4 routine.
    """
    sorted_array = np.asarray(sorted_array, dtype=float)
    values = np.asarray(values, dtype=float)
    nt = len(sorted_array)
    if nt < 2:
        raise ValueError("sorted_array must contain at least 2 elements.")
    ns = len(values)
    left = np.empty(ns, dtype=int)
    for i in range(ns):
        s = values[i]
        lo = 0
        hi = nt - 2
        # If outside bounds, clamp
        if s <= sorted_array[0]:
            left[i] = 0
            continue
        if s >= sorted_array[-1]:
            left[i] = nt - 2
            continue
        # Binary search
        while lo <= hi:
            mid = (lo + hi) // 2
            if sorted_array[mid] <= s <= sorted_array[mid + 1]:
                left[i] = mid
                break
            elif s < sorted_array[mid]:
                hi = mid - 1
            else:
                lo = mid + 1
        else:
            left[i] = lo
    return left


def linear_interpolate(x_nodes, y_values, x_query):
    """
    Piecewise-linear interpolation on sorted nodes.
    """
    x_nodes = np.asarray(x_nodes, dtype=float)
    y_values = np.asarray(y_values, dtype=float)
    x_query = np.asarray(x_query, dtype=float)
    left = bracket_interval(x_nodes, x_query)
    result = np.empty_like(x_query, dtype=float)
    for i, idx in enumerate(left):
        x0 = x_nodes[idx]
        x1 = x_nodes[idx + 1]
        dx = x1 - x0
        if abs(dx) < 1e-30:
            result[i] = y_values[idx]
        else:
            t = (x_query[i] - x0) / dx
            result[i] = y_values[idx] * (1.0 - t) + y_values[idx + 1] * t
    return result


def generate_pore_coordinates(n_pores, mean_radius, std_radius, length):
    """
    Generate 3D cylindrical pore coordinates for a network model.
    Returns an (n_pores, 3) array of (r, theta, z) coordinates adapted
    from xyz_display 3D point handling.
    """
    rng = np.random.default_rng(seed=42)
    radii = rng.normal(loc=mean_radius, scale=std_radius, size=n_pores)
    radii = np.clip(radii, mean_radius * 0.1, mean_radius * 3.0)
    theta = rng.uniform(0.0, 2.0 * np.pi, size=n_pores)
    z = rng.uniform(0.0, length, size=n_pores)
    xyz = np.column_stack((radii * np.cos(theta), radii * np.sin(theta), z))
    return xyz


def van_t_hoff_correction(S0, dH_sorp, T, T0=298.15):
    """
    van't Hoff equation for temperature-dependent solubility:
        S(T) = S0 * exp[ -dH_sorp/R * (1/T - 1/T0) ]
    """
    R = 8.314
    return S0 * np.exp(-dH_sorp / R * (1.0 / T - 1.0 / T0))
