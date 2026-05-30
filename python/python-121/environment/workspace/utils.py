
import numpy as np
from math import factorial, comb, exp, log, sqrt






def stirling_numbers_2(n, k):
    if n < 0 or k < 0 or k > n:
        return 0
    if n == 0 and k == 0:
        return 1
    result = 0
    for j in range(k + 1):
        sign = 1 if (k - j) % 2 == 0 else -1
        result += sign * comb(k, j) * (j ** n)
    return result // factorial(k)


def bell_numbers(n):
    if n < 0:
        return 0
    total = 0
    for k in range(n + 1):
        total += stirling_numbers_2(n, k)
    return total


def subset_lex_rank(n, a):
    if n <= 0:
        return 0
    rank = 0
    for i in range(n):
        if a[i]:
            rank += 2 ** (n - 1 - i)
    return rank


def subset_lex_unrank(n, rank):
    if n <= 0 or rank < 0 or rank >= 2 ** n:
        return [False] * n
    a = [False] * n
    for i in range(n - 1, -1, -1):
        if rank >= 2 ** i:
            a[n - 1 - i] = True
            rank -= 2 ** i
    return a


def perm_lex_rank(n, p):
    if n <= 0:
        return 0
    rank = 0
    used = [False] * (n + 1)
    for i in range(n):
        count = 0
        for j in range(1, p[i]):
            if not used[j]:
                count += 1
        rank += count * factorial(n - 1 - i)
        used[p[i]] = True
    return rank


def perm_lex_unrank(n, rank):
    if n <= 0 or rank < 0 or rank >= factorial(n):
        return list(range(1, n + 1))
    p = [0] * n
    used = [False] * (n + 1)
    for i in range(n):
        fi = factorial(n - 1 - i)
        c = rank // fi
        rank %= fi
        count = 0
        for j in range(1, n + 1):
            if not used[j]:
                if count == c:
                    p[i] = j
                    used[j] = True
                    break
                count += 1
    return p


def ion_channel_state_enumeration(n_states, n_open):
    if n_states < 0 or n_open < 0 or n_open > n_states:
        return 0, []
    total = comb(n_states, n_open)
    configs = []
    
    def backtrack(start, current, ones):
        if ones == n_open:
            configs.append(current[:])
            return
        if start >= n_states:
            return
        for i in range(start, n_states - (n_open - ones) + 1):
            current[i] = 1
            backtrack(i + 1, current, ones + 1)
            current[i] = 0
    
    backtrack(0, [0] * n_states, 0)
    return total, configs






def compute_relative_error(exact, approx):
    if exact == 0:
        return abs(approx) if approx != 0 else 0.0
    return abs(exact - approx) / abs(exact)


def compute_roundoff_error(x, y, operation):
    eps_mach = np.finfo(float).eps
    if operation == 'add':
        return eps_mach * abs(x + y)
    elif operation == 'mul':
        return eps_mach * abs(x * y)
    elif operation == 'div':
        if y == 0:
            return float('inf')
        return eps_mach * abs(x / y)
    return 0.0


def convergence_rate(errors, resolutions):
    if len(errors) < 2 or len(resolutions) < 2:
        return 0.0
    log_h = np.log(resolutions)
    log_e = np.log(errors)
    n = len(log_h)

    mean_x = np.mean(log_h)
    mean_y = np.mean(log_e)
    numerator = np.sum((log_h - mean_x) * (log_e - mean_y))
    denominator = np.sum((log_h - mean_x) ** 2)
    if denominator == 0:
        return 0.0
    p = numerator / denominator
    return p


def condition_number_analysis(A):
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        return float('inf')
    try:
        sigma = np.linalg.svd(A, compute_uv=False)
        if sigma[-1] == 0:
            return float('inf')
        return sigma[0] / sigma[-1]
    except Exception:
        return float('inf')


def estimate_truncation_error(f, x, h, order=2):
    if h <= 0:
        return float('inf')

    f3_approx = (f(x + 2 * h) - 2 * f(x + h) + 2 * f(x - h) - f(x - 2 * h)) / (2 * h ** 3)
    return abs(h ** 2 * f3_approx / 6.0)






def catastrophic_cancellation_test():
    p = 665857.0
    q = 470832.0
    exact = 1.0
    computed = p ** 2 - 2.0 * q ** 2
    rel_err = compute_relative_error(exact, computed)
    return {
        'p': p,
        'q': q,
        'exact': exact,
        'computed': computed,
        'relative_error': rel_err
    }


def generate_gray_code(n):
    if n < 0 or n > 20:
        return []
    return [i ^ (i >> 1) for i in range(2 ** n)]
