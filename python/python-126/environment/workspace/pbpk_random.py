
import numpy as np
from typing import Tuple





def r8vec_uniform_01_sorted_exponential(n: int) -> np.ndarray:
    if n < 0:
        raise ValueError("n must be non-negative")
    if n == 0:
        return np.array([], dtype=float)

    e = -np.log(np.random.rand(n + 1))
    cumsum = np.cumsum(e)
    total = cumsum[-1]
    if total == 0.0:

        return np.sort(np.random.rand(n))
    u = cumsum[:-1] / total

    return u


def r8vec_uniform_01_sorted_product(n: int) -> np.ndarray:
    if n < 0:
        raise ValueError("n must be non-negative")
    if n == 0:
        return np.array([], dtype=float)
    u = np.empty(n, dtype=float)
    curmax = 1.0
    for i in range(n, 0, -1):

        r = np.random.rand()
        if r <= 0.0:
            r = 1e-300
        u[i - 1] = curmax * np.exp(np.log(r) / i)
        curmax = u[i - 1]
    return u






def normal_01_cdf_inv(p: float) -> float:
    if p <= 0.0:
        return -np.inf
    if p >= 1.0:
        return np.inf



    q = p - 0.5
    if abs(q) <= 0.425:
        r = 0.180625 - q * q
        num = (((((((2.5090809287301226727e+03 * r
                     + 3.3430575583588128105e+04) * r
                    + 6.7265770927008700853e+04) * r
                   + 4.5921953931549871457e+04) * r
                  + 1.3731693765509461125e+04) * r
                 + 1.9715909503065514427e+03) * r
                + 1.3314136788658342429e+02) * r
               + 3.3871328727963666080e+00) * q
        den = (((((((5.2264952788528545610e+03 * r
                     + 2.8729085735721942674e+04) * r
                    + 3.9307895800092710610e+04) * r
                   + 2.1213794301586595867e+04) * r
                  + 5.3941960214247511077e+03) * r
                 + 6.8718700749205790830e+02) * r
                + 4.2313330701600911252e+01) * r
               + 1.0)
        return num / den
    else:
        if q < 0.0:
            r = p
        else:
            r = 1.0 - p
        if r <= 0.0:
            return np.inf if q > 0 else -np.inf
        r = np.sqrt(-np.log(r))
        if r <= 5.0:
            r = r - 1.6
            num = (((((((7.7454501427834140764e-04 * r
                         + 2.27238449892691845833e-02) * r
                        + 2.4178072517745061177e-01) * r
                       + 1.27045825245236838258e+00) * r
                      + 3.64784832476320460504e+00) * r
                     + 5.7694972214606914055e+00) * r
                    + 4.6303378461565452959e+00) * r
                   + 1.42343711074968357734e+00)
            den = (((((((1.05075007164441684324e-09 * r
                         + 5.475938084995344946e-04) * r
                        + 1.51986665636164571966e-02) * r
                       + 1.4810397642748007459e-01) * r
                      + 6.8976733498510000455e-01) * r
                     + 1.6763848301838038494e+00) * r
                    + 2.05319162663775882187e+00) * r
                   + 1.0)
        else:
            r = r - 5.0
            num = (((((((2.01033439929228813265e-07 * r
                         + 2.71155556874348757815e-05) * r
                        + 1.24266094738807843860e-03) * r
                       + 2.6532189526576123093e-02) * r
                      + 2.9656057182850489123e-01) * r
                     + 1.7848265399172913358e+00) * r
                    + 5.4637849111641143699e+00) * r
                   + 6.6579046435011037772e+00)
            den = (((((((2.04426310338993978564e-15 * r
                         + 1.4215117583164458887e-07) * r
                        + 1.8463183175100546818e-05) * r
                       + 7.868691311456132591e-04) * r
                      + 1.48753612908506148525e-02) * r
                     + 1.3692988092273580531e-01) * r
                    + 5.9983220655588793769e-01) * r
                   + 1.0)
        x = num / den
        if q < 0.0:
            x = -x
        return x


def r8vec_normal_01_sorted(n: int, method: str = "exponential") -> np.ndarray:
    if n < 0:
        raise ValueError("n must be non-negative")
    if n == 0:
        return np.array([], dtype=float)
    if method == "exponential":
        u_sorted = r8vec_uniform_01_sorted_exponential(n)
    elif method == "product":
        u_sorted = r8vec_uniform_01_sorted_product(n)
    else:
        raise ValueError("method must be 'exponential' or 'product'")


    x_sorted = np.array([normal_01_cdf_inv(ui) for ui in u_sorted])
    return x_sorted






def sample_drug_arrival_times(n_molecules: int, mean_interval: float,
                               cv: float = 0.2) -> np.ndarray:
    if n_molecules <= 0 or mean_interval <= 0.0 or cv < 0.0:
        raise ValueError("Invalid arrival time parameters")
    if cv == 0.0:
        return np.arange(1, n_molecules + 1) * mean_interval
    sigma = np.sqrt(np.log(1.0 + cv ** 2))
    mu = np.log(mean_interval) - sigma ** 2 / 2.0
    z_sorted = r8vec_normal_01_sorted(n_molecules)

    t_sorted = np.exp(mu + sigma * z_sorted)

    t_sorted = np.clip(t_sorted, mean_interval * 0.01, mean_interval * 100.0)

    arrival_times = np.cumsum(t_sorted)
    return arrival_times


def sample_physiological_params(n_samples: int, param_means: np.ndarray,
                                 param_cvs: np.ndarray) -> np.ndarray:
    n_params = len(param_means)
    if n_params != len(param_cvs):
        raise ValueError("param_means and param_cvs must have same length")
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    samples = np.empty((n_samples, n_params), dtype=float)
    for j in range(n_params):
        if param_cvs[j] < 0.0:
            raise ValueError("CV must be non-negative")
        if param_cvs[j] == 0.0:
            samples[:, j] = param_means[j]
        else:
            sigma = param_means[j] * param_cvs[j]

            z_sorted = r8vec_normal_01_sorted(n_samples)
            col = param_means[j] + sigma * z_sorted

            col = np.maximum(col, param_means[j] * 0.1)
            samples[:, j] = col
    return samples






if __name__ == "__main__":
    u1 = r8vec_uniform_01_sorted_exponential(10)
    print("Uniform sorted (exponential):", u1)
    u2 = r8vec_uniform_01_sorted_product(10)
    print("Uniform sorted (product):", u2)
    z = r8vec_normal_01_sorted(10)
    print("Normal sorted:", z)
    t = sample_drug_arrival_times(100, 10.0, 0.3)
    print("Arrival times (first 5):", t[:5])
    params = sample_physiological_params(50,
                                          np.array([70.0, 1.2, 0.05]),
                                          np.array([0.15, 0.20, 0.30]))
    print("Physiological param samples shape:", params.shape)
