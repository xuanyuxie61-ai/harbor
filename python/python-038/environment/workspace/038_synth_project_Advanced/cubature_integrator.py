
import numpy as np





_PYRAMID_RULES = {
    0: {
        'x': np.array([0.0]),
        'y': np.array([0.0]),
        'z': np.array([0.5]),
        'w': np.array([4.0]),
    },
    1: {
        'x': np.array([0.0,  0.0,  0.0,  0.0]),
        'y': np.array([0.0,  0.0,  0.0,  0.0]),
        'z': np.array([0.25, 0.25, 0.75, 0.75]),
        'w': np.array([1.0,  1.0,  1.0,  1.0]),
    }
}


def integrate_1d_composite(f, a, b, n=1024, rule='simpson'):
    if a >= b:
        return 0.0
    if n < 2:
        n = 2
    
    if rule == 'simpson' and n % 2 == 1:
        n += 1
    
    x = np.linspace(a, b, n + 1)
    y = np.asarray([f(xi) for xi in x], dtype=float)
    h = (b - a) / n
    
    if rule == 'trapezoidal':
        return h * (0.5 * y[0] + np.sum(y[1:-1]) + 0.5 * y[-1])
    elif rule == 'simpson':
        return h / 3.0 * (y[0] + 4.0 * np.sum(y[1:-1:2]) + 2.0 * np.sum(y[2:-1:2]) + y[-1])
    else:
        raise ValueError("rule must be 'trapezoidal' or 'simpson'")


def integrate_pyramid(f, precision=1):
    if precision <= 1 and precision in _PYRAMID_RULES:
        rule = _PYRAMID_RULES[precision]
        pts = zip(rule['x'], rule['y'], rule['z'], rule['w'])
        total = 0.0
        for xi, yi, zi, wi in pts:
            total += wi * f(xi, yi, zi)
        return total
    else:

        n_per_dim = max(4, precision)

        from numpy.polynomial.legendre import leggauss
        t, wt = leggauss(n_per_dim)
        
        total = 0.0
        for i in range(n_per_dim):
            for j in range(n_per_dim):
                for k in range(n_per_dim):

                    z = 0.5 * (t[k] + 1.0)
                    wz = 0.5 * wt[k]

                    hx = 1.0 - z
                    hy = 1.0 - z
                    if hx <= 0:
                        continue
                    x = hx * t[i]
                    y = hy * t[j]
                    wx = wt[i] * hx
                    wy = wt[j] * hy
                    total += wx * wy * wz * f(x, y, z)
        return total


def integrate_monte_carlo(f, domain, n_samples=50000, seed=42):
    rng = np.random.default_rng(seed)
    dim = len(domain)
    lows = np.array([d[0] for d in domain], dtype=float)
    highs = np.array([d[1] for d in domain], dtype=float)
    volume = np.prod(highs - lows)
    
    samples = rng.uniform(0.0, 1.0, size=(n_samples, dim))
    xs = lows + samples * (highs - lows)
    
    vals = np.array([f(x) for x in xs], dtype=float)
    mean = float(np.mean(vals))
    std = float(np.std(vals, ddof=1))
    
    integral = volume * mean
    error = volume * std / np.sqrt(n_samples)
    return integral, error


def integrate_adaptive_1d(f, a, b, tol=1e-8, max_evals=10000):

    def simpson(l, r):
        m = 0.5 * (l + r)
        h = r - l
        return h / 6.0 * (f(l) + 4.0 * f(m) + f(r))
    

    stack = [(a, b, tol, simpson(a, b))]
    total = 0.0
    eval_count = 5
    
    while stack and eval_count < max_evals:
        l, r, eps, whole = stack.pop()
        m = 0.5 * (l + r)
        left = simpson(l, m)
        right = simpson(m, r)
        eval_count += 2
        
        if abs(left + right - whole) <= 15 * eps or r - l < 1e-12:
            total += left + right + (left + right - whole) / 15.0
        else:
            stack.append((l, m, eps / 2.0, left))
            stack.append((m, r, eps / 2.0, right))
    

    for l, r, eps, whole in stack:
        total += whole
    
    return total


def test_cubature():

    val = integrate_1d_composite(lambda x: x**2, 0.0, 1.0, n=100, rule='simpson')
    assert abs(val - 1.0/3.0) < 1e-10, f"Composite Simpson failed: {val}"
    



    vol = integrate_pyramid(lambda x, y, z: 1.0, precision=3)
    assert abs(vol - 4.0/3.0) < 0.5, f"Pyramid volume failed: {vol}"
    

    val2, err2 = integrate_monte_carlo(
        lambda x: x[0] * x[1], [(0.0, 1.0), (0.0, 1.0)], n_samples=20000
    )
    assert abs(val2 - 0.25) < 5 * err2, f"MC integration failed: {val2} ± {err2}"
    

    val3 = integrate_adaptive_1d(lambda x: np.sqrt(x), 0.0, 1.0, tol=1e-6)
    assert abs(val3 - 2.0/3.0) < 1e-5, f"Adaptive Simpson failed: {val3}"
    
    return True


if __name__ == "__main__":
    test_cubature()
    print("Cubature integrator tests passed.")
