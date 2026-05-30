
import numpy as np


def polynomial_multiply(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    na, nb = len(a), len(b)
    c = np.zeros(na + nb - 1)

    for i in range(na):
        for j in range(nb):
            c[i + j] += a[i] * b[j]

    return c


def stoichiometric_paths(reaction_steps, target_change, max_steps):
    if target_change < 0 or max_steps <= 0:
        return 0, np.array([1.0])


    p = np.zeros(target_change + 1)
    p[0] = 1.0


    step_poly = np.zeros(max(reaction_steps) + 1)
    for s in reaction_steps:
        if 0 <= s <= max(reaction_steps):
            step_poly[s] += 1.0


    current = p.copy()
    for _ in range(max_steps):
        current = polynomial_multiply(current, step_poly)
        if len(current) > target_change:
            current = current[:target_change + 1]
        else:

            temp = np.zeros(target_change + 1)
            temp[:len(current)] = current
            current = temp

    count = int(round(current[target_change]))
    return count, current


def reaction_mechanism_complexity(reactions, max_depth=10):
    steps = [r['stoich_change'] for r in reactions]
    rates = [r['rate'] for r in reactions]


    max_change = max(steps) * max_depth
    total_paths = 0
    path_distribution = []

    for target in range(1, max_change + 1):
        count, _ = stoichiometric_paths(steps, target, max_depth)
        total_paths += count
        path_distribution.append((target, count))


    avg_rate = np.mean(rates) if rates else 1.0
    effective_complexity = total_paths * avg_rate

    complexity = {
        'num_reactions': len(reactions),
        'max_depth': max_depth,
        'total_paths': total_paths,
        'average_rate': avg_rate,
        'effective_complexity': effective_complexity,
        'path_distribution': path_distribution[:20],
    }

    return complexity
