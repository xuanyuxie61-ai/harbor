
import numpy as np


def change_greedy(coin_values, target):
    coin_values = np.array(coin_values, dtype=int)
    target = int(target)


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
    n_elem = len(elements)
    indicators = np.zeros(n_elem)

    for e in range(n_elem):
        elem = elements[e]
        x = nodes[elem, 0]
        y = nodes[elem, 1]


        a = np.sqrt((x[1] - x[0]) ** 2 + (y[1] - y[0]) ** 2)
        b = np.sqrt((x[2] - x[1]) ** 2 + (y[2] - y[1]) ** 2)
        c = np.sqrt((x[0] - x[2]) ** 2 + (y[0] - y[2]) ** 2)
        s = 0.5 * (a + b + c)
        area = max(np.sqrt(max(s * (s - a) * (s - b) * (s - c), 1e-20)), 1e-20)
        h = a * b * c / (4.0 * area)


        u = solution[elem]

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
    nodes = np.array(nodes, dtype=float)
    elements = np.array(elements, dtype=int)
    solution = np.array(solution, dtype=float)

    indicators = compute_error_indicator(nodes, elements, solution)
    n_elem = len(elements)


    sorted_idx = np.argsort(indicators)[::-1]



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


            h = max(np.max(x) - np.min(x), np.max(y) - np.min(y))
            if h < min_element_size:
                continue


            mid1 = [(x[0] + x[1]) / 2.0, (y[0] + y[1]) / 2.0]
            mid2 = [(x[1] + x[2]) / 2.0, (y[1] + y[2]) / 2.0]
            mid3 = [(x[2] + x[0]) / 2.0, (y[2] + y[0]) / 2.0]

            n1 = n_original_nodes
            n2 = n_original_nodes + 1
            n3 = n_original_nodes + 2
            n_original_nodes += 3

            new_nodes.extend([mid1, mid2, mid3])


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
    if error_estimate < 1e-20:
        return dt_max, True

    tol = 1e-3
    ratio = (tol / error_estimate) ** (1.0 / (order + 1))
    dt_new = safety_factor * dt_current * ratio
    dt_new = np.clip(dt_new, dt_min, dt_max)

    accepted = error_estimate <= tol
    return dt_new, accepted
