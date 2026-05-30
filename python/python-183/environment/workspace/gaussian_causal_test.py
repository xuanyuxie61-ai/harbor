
import numpy as np
from typing import Tuple


def owen_t_function(h: float, a: float) -> float:

    tv1 = 1.0e-35
    tv2 = 15.0
    tv3 = 15.0
    tv4 = 1.0e-5
    tp = 0.159154943091895

    if abs(h) < tv1:
        return tp * np.arctan(a)
    if abs(h) > tv2:
        return 0.0
    if abs(a) < tv1:
        return 0.0


    u = np.array([0.0744371695, 0.2166976971, 0.3397047841,
                  0.4325316833, 0.4869532649])
    r = np.array([0.1477621124, 0.1346333597, 0.1095431812,
                  0.0747256746, 0.0333356721])

    xs = -0.5 * h * h
    fxs = a * a


    if tv3 <= np.log(1.0 + fxs) - xs * fxs:
        x1 = 0.5 * a
        fxs = 0.25 * fxs
        while True:
            rt = fxs + 1.0
            x2 = x1 + (xs * fxs + tv3 - np.log(rt)) / (2.0 * x1 * (1.0 / rt - xs))
            fxs = x2 * x2
            if abs(x2 - x1) < tv4:
                break
            x1 = x2
        a_eff = x2
    else:
        a_eff = a

    rt_sum = 0.0
    for i in range(5):
        r1 = 1.0 + fxs * (0.5 + u[i]) ** 2
        r2 = 1.0 + fxs * (0.5 - u[i]) ** 2
        rt_sum += r[i] * (np.exp(xs * r1) / r1 + np.exp(xs * r2) / r2)

    value = rt_sum * a_eff * tp
    return float(value)


def bivariate_normal_cdf(x: float, y: float, rho: float) -> float:
    if rho <= -1.0 or rho >= 1.0:
        raise ValueError("相关系数 rho 必须在 (-1,1) 内。")

    def phi_cdf(z):
        return 0.5 * (1.0 + np.math.erf(z / np.sqrt(2.0)))


    if abs(rho) < 1e-6:
        return phi_cdf(x) * phi_cdf(y)






    n_quad = 20
    t_nodes, t_weights = np.polynomial.legendre.leggauss(n_quad)


    L = 6.0
    total = 0.0
    for i in range(n_quad):
        xi = L * t_nodes[i]
        wi = L * t_weights[i]
        if xi < x:
            continue
        for j in range(n_quad):
            yj = L * t_nodes[j]
            wj = L * t_weights[j]
            if yj < y:
                continue

            det = 1.0 - rho * rho
            if det <= 0.0:
                det = 1e-12
            z = (xi * xi - 2.0 * rho * xi * yj + yj * yj) / det
            dens = np.exp(-0.5 * z) / (2.0 * np.pi * np.sqrt(det))
            total += wi * wj * dens

    return max(0.0, min(1.0, total))


def partial_correlation_test(Theta: np.ndarray,
                              n_samples: int,
                              alpha_level: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    p = Theta.shape[0]
    if n_samples <= p + 2:
        raise ValueError("样本量必须大于 p+2 才能进行 t 检验。")

    pvals = np.ones((p, p))
    reject = np.zeros((p, p), dtype=bool)











    raise NotImplementedError("Hole 2: 偏相关系数检验循环待实现")

    return pvals, reject


def demo():

    val = owen_t_function(1.0, 0.5)
    print(f"[gaussian_causal_test] Owen T(1.0, 0.5) = {val:.8f}")


    np.random.seed(5)
    p = 8
    n = 300
    Theta = np.eye(p) * 2.0
    Theta[0, 1] = Theta[1, 0] = 0.5
    Theta[2, 3] = Theta[3, 2] = -0.4
    Sigma = np.linalg.inv(Theta)
    X = np.random.multivariate_normal(np.zeros(p), Sigma, size=n)
    S = np.cov(X, rowvar=False)
    Theta_est = np.linalg.inv(S + 0.1 * np.eye(p))
    pvals, reject = partial_correlation_test(Theta_est, n, alpha_level=0.05)
    n_edges = np.sum(reject) // 2
    print(f"[gaussian_causal_test] 显著边数 (alpha=0.05): {n_edges}")
    return val, n_edges


if __name__ == "__main__":
    demo()
