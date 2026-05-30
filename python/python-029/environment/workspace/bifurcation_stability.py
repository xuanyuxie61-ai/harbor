
import numpy as np


def logistic_map(x, r):
    return r * x * (1.0 - x)


def logistic_attractor(r, x0=0.5, warm_up=500, max_iter=1000, tol=1e-6):
    x = x0

    for _ in range(warm_up):
        x = logistic_map(x, r)


    trajectory = []
    seen = set()
    for _ in range(max_iter):
        x = logistic_map(x, r)

        key = round(x, 6)
        if key in seen:
            break
        seen.add(key)
        trajectory.append(x)


    attractor = []
    for x_val in trajectory[-20:]:

        is_new = True
        for a in attractor:
            if abs(a - x_val) < tol:
                is_new = False
                break
        if is_new:
            attractor.append(x_val)

    return np.sort(attractor)


def feigenbaum_bifurcation_diagram(r_min=2.5, r_max=4.0, n_r=2000):
    r_values = np.linspace(r_min, r_max, n_r)
    attractors = []
    for r in r_values:
        attr = logistic_attractor(r)
        attractors.append(attr)
    return r_values, attractors


def neutron_multiplication_bifurcation(alpha_range, lambda_f=0.5, lambda_c=0.3, S=0.01):
    equilibrium = []
    stability = []
    for alpha in alpha_range:
        if alpha < 1e-12:
            alpha = 1e-12
        discriminant = (lambda_f - lambda_c) ** 2 + 4.0 * alpha * S
        x_star = ((lambda_f - lambda_c) + np.sqrt(discriminant)) / (2.0 * alpha)


        df = (lambda_f - lambda_c) - 2.0 * alpha * x_star

        is_stable = abs(1.0 + df) < 1.0 if df != 0 else True
        equilibrium.append(x_star)
        stability.append(is_stable)

    return np.array(equilibrium), np.array(stability)


def optical_potential_stability_boundary(V0_range, W0_range, params_func):
    stability_map = np.zeros((len(V0_range), len(W0_range)), dtype=bool)
    unitarity_dev = np.zeros((len(V0_range), len(W0_range)))

    for i, V0 in enumerate(V0_range):
        for j, W0 in enumerate(W0_range):
            params = params_func(V0, W0)


            is_physical = (V0 > 0 and W0 > 0 and params.a_v > 0.05)


            deviation = abs(W0 / (V0 + W0 + 1.0) - 0.3)
            unitarity_dev[i, j] = deviation
            stability_map[i, j] = is_physical and (deviation < 0.5)

    return stability_map, unitarity_dev


def lyapunov_exponent_logistic(r, x0=0.5, n_iter=10000):
    x = x0
    lam_sum = 0.0
    for _ in range(n_iter):
        x = logistic_map(x, r)
        df = abs(r * (1.0 - 2.0 * x))
        if df < 1e-300:
            df = 1e-300
        lam_sum += np.log(df)

    return lam_sum / n_iter


def critical_slowing_down_indicator(alpha, lambda_f, lambda_c, epsilon=1e-3):

    x_star = max((lambda_f - lambda_c) / (2.0 * alpha), 0.0)
    eigenvalue = lambda_f - lambda_c - 2.0 * alpha * x_star
    tau = 1.0 / (abs(eigenvalue) + epsilon)
    return tau


if __name__ == "__main__":

    r_test = 3.5
    attr = logistic_attractor(r_test)
    print(f"r={r_test}: attractor = {attr}")

    lyap = lyapunov_exponent_logistic(3.8)
    print(f"r=3.8 的 Lyapunov 指数: {lyap:.4f}")

    alpha_range = np.linspace(0.01, 2.0, 100)
    eq, stab = neutron_multiplication_bifurcation(alpha_range)
    print(f"α 范围 [{alpha_range[0]:.2f}, {alpha_range[-1]:.2f}]")
    print(f"平衡点范围: [{eq.min():.4f}, {eq.max():.4f}]")
    print(f"稳定点数: {np.sum(stab)}")


    r_vals, attrs = feigenbaum_bifurcation_diagram(2.8, 4.0, 500)
    print(f"分岔图参数点数: {len(r_vals)}")
