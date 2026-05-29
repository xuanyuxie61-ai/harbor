"""
Adaptive Mesh Refinement Module
===============================
Implements adaptive mesh refinement using greedy algorithms.

Incorporates:
- Greedy change-making algorithm (from 157_change_greedy)
- Ball grid generation concepts (from 067_ball_grid)

Scientific Background:
----------------------
For accurate climate simulation, mesh resolution should adapt to
regions of high solution gradient. The greedy strategy selects
cells for refinement based on error indicators:

    eta_K = h_K * ||R(u_h)||_{L^2(K)} + h_K^{1/2} * ||J(u_h)||_{L^2(dK)}

where R is element residual and J is jump across edges.

The refinement budget (total number of new cells) is allocated
using a greedy algorithm similar to change-making:
    1. Sort cells by error indicator (descending)
    2. While budget > 0:
       - Select cell with largest error
       - Subdivide it (consume budget)
       - Update indicators
"""

import numpy as np


def change_greedy(coin_values, target):
    """
    Greedy change-making algorithm.
    From 157_change_greedy.

    Given coin denominations, find minimum coins to make target amount.
    Used here for discretizing refinement budget into cell subdivisions.

    Parameters
    ----------
    coin_values : array_like
        Available coin denominations.
    target : int
        Target sum.

    Returns
    -------
    ndarray
        Number of each coin used.
    """
    coin_values = np.array(coin_values, dtype=int)
    target = int(target)

    # Sort descending
    idx = np.argsort(coin_values)[::-1]
    sorted_values = coin_values[idx]

    n_coins = len(coin_values)
    a = np.zeros(n_coins, dtype=int)
    remaining = target

    for i in range(n_coins):
        coin = idx[i]
        a[coin] = remaining // sorted_values[i]
        remaining -= a[coin] * sorted_values[i]

    return a


def compute_error_indicator(nodes, elements, solution, element_data=None):
    """
    Compute error indicator for each element based on solution gradient.

    Indicator: eta = h * |grad(u)|
    where h is element size.

    Parameters
    ----------
    nodes : ndarray, shape (N, 2)
        Node coordinates.
    elements : ndarray, shape (M, 3)
        Triangle connectivity.
    solution : ndarray
        Solution values at nodes.
    element_data : dict, optional
        Additional element-specific data.

    Returns
    -------
    indicators : ndarray
        Error indicator per element.
    """
    n_elem = len(elements)
    indicators = np.zeros(n_elem)

    for e in range(n_elem):
        elem = elements[e]
        x = nodes[elem, 0]
        y = nodes[elem, 1]

        # Element size (circumradius approximation)
        a = np.sqrt((x[1] - x[0]) ** 2 + (y[1] - y[0]) ** 2)
        b = np.sqrt((x[2] - x[1]) ** 2 + (y[2] - y[1]) ** 2)
        c = np.sqrt((x[0] - x[2]) ** 2 + (y[0] - y[2]) ** 2)
        s = 0.5 * (a + b + c)
        area = max(np.sqrt(max(s * (s - a) * (s - b) * (s - c), 1e-20)), 1e-20)
        h = a * b * c / (4.0 * area)  # Circumradius

        # Gradient magnitude
        u = solution[elem]
        # Area-weighted gradient
        det = (x[1] - x[0]) * (y[2] - y[0]) - (x[2] - x[0]) * (y[1] - y[0])
        if abs(det) < 1e-15:
            det = 1e-15

        dudx = ((u[1] - u[0]) * (y[2] - y[0]) - (u[2] - u[0]) * (y[1] - y[0])) / det
        dudy = ((u[2] - u[0]) * (x[1] - x[0]) - (u[1] - u[0]) * (x[2] - x[0])) / det
        grad_mag = np.sqrt(dudx ** 2 + dudy ** 2)

        indicators[e] = h * grad_mag

    return indicators


def greedy_adaptive_refine(nodes, elements, solution, max_new_elements=100,
                            min_element_size=0.01):
    """
    Adaptively refine mesh using greedy error-based selection.

    Parameters
    ----------
    nodes : ndarray
        Node coordinates.
    elements : ndarray
        Triangle connectivity.
    solution : ndarray
        Solution at nodes.
    max_new_elements : int
        Maximum new elements to add.
    min_element_size : float
        Minimum element size.

    Returns
    -------
    new_nodes : ndarray
        Refined node coordinates.
    new_elements : ndarray
        Refined triangle connectivity.
    refinement_stats : dict
        Statistics about refinement.
    """
    nodes = np.array(nodes, dtype=float)
    elements = np.array(elements, dtype=int)
    solution = np.array(solution, dtype=float)

    indicators = compute_error_indicator(nodes, elements, solution)
    n_elem = len(elements)

    # Greedy selection: sort by indicator and refine top candidates
    sorted_idx = np.argsort(indicators)[::-1]

    # Use change-making to discretize refinement budget
    # Coin values: possible subdivision depths
    coin_values = np.array([1, 2, 4, 8, 16])
    budget_allocation = change_greedy(coin_values, max_new_elements)

    new_nodes = nodes.tolist()
    new_elements = elements.tolist()
    n_original_nodes = len(nodes)
    refined_count = 0

    for depth_idx, count in enumerate(budget_allocation):
        subdivisions = coin_values[depth_idx]
        for _ in range(count):
            if refined_count >= n_elem:
                break
            e_idx = sorted_idx[refined_count % n_elem]
            refined_count += 1

            elem = elements[e_idx]
            x = nodes[elem, 0]
            y = nodes[elem, 1]

            # Check minimum size
            h = max(np.max(x) - np.min(x), np.max(y) - np.min(y))
            if h < min_element_size:
                continue

            # Add midpoint nodes
            mid1 = [(x[0] + x[1]) / 2.0, (y[0] + y[1]) / 2.0]
            mid2 = [(x[1] + x[2]) / 2.0, (y[1] + y[2]) / 2.0]
            mid3 = [(x[2] + x[0]) / 2.0, (y[2] + y[0]) / 2.0]

            n1 = n_original_nodes
            n2 = n_original_nodes + 1
            n3 = n_original_nodes + 2
            n_original_nodes += 3

            new_nodes.extend([mid1, mid2, mid3])

            # Replace element with 4 sub-triangles
            new_elements[e_idx] = [elem[0], n1, n3]
            new_elements.append([n1, elem[1], n2])
            new_elements.append([n3, n2, elem[2]])
            new_elements.append([n1, n2, n3])

    refinement_stats = {
        'original_elements': n_elem,
        'final_elements': len(new_elements),
        'original_nodes': len(nodes),
        'final_nodes': len(new_nodes),
        'refined_elements': refined_count
    }

    return np.array(new_nodes), np.array(new_elements), refinement_stats


def adaptive_time_step(error_estimate, dt_current, dt_min=0.1, dt_max=10.0,
                        safety_factor=0.9, order=2):
    """
    Adaptive time step selection based on error estimate.

    Formula:
    dt_new = safety_factor * dt * (tol / error)^{1/(order+1)}

    Parameters
    ----------
    error_estimate : float
        Estimated local truncation error.
    dt_current : float
        Current time step.
    dt_min, dt_max : float
        Bounds.
    safety_factor : float
        Safety factor.
    order : int
        Method order.

    Returns
    -------
    float
        New time step.
    bool
        Whether step is accepted.
    """
    if error_estimate < 1e-20:
        return dt_max, True

    tol = 1e-3
    ratio = (tol / error_estimate) ** (1.0 / (order + 1))
    dt_new = safety_factor * dt_current * ratio
    dt_new = np.clip(dt_new, dt_min, dt_max)

    accepted = error_estimate <= tol
    return dt_new, accepted
