
import numpy as np


def line01_sample_random(n, seed=None):
    rng = np.random.default_rng(seed)
    return rng.random(n)


def line01_monomial_integral(e):
    if e == -1:
        raise ValueError("e = -1 不允许")
    return 1.0 / (e + 1)


def monte_carlo_line_integral(fun, n, seed=None):
    x = line01_sample_random(n, seed)
    fx = fun(x)
    return float(np.mean(fx))


def subset_distance_hamming(t1, t2):
    t1 = np.asarray(t1)
    t2 = np.asarray(t2)
    if t1.shape != t2.shape:
        raise ValueError("t1 和 t2 形状必须相同")
    return int(np.sum(t1 != t2))


def subset_sample(m, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    return (rng.random(m) > 0.5).astype(np.int64)


def subset_distance_stats(m, n_samples=1000, seed=None):
    rng = np.random.default_rng(seed)
    distances = []
    for _ in range(n_samples):
        s1 = subset_sample(m, rng)
        s2 = subset_sample(m, rng)
        distances.append(subset_distance_hamming(s1, s2))
    distances = np.array(distances, dtype=np.float64)
    mu = float(np.mean(distances))
    if n_samples > 1:
        var = float(np.sum((distances - mu) ** 2) / (n_samples - 1))
    else:
        var = 0.0
    return mu, var


def generate_ensemble_perturbations(base_field, n_members, perturbation_scale=0.1,
                                     seed=None):
    rng = np.random.default_rng(seed)
    std = np.std(base_field)
    if std < 1e-14:
        std = 1.0
    ensemble = np.zeros((n_members,) + base_field.shape, dtype=np.float64)
    for i in range(n_members):
        noise = rng.normal(0.0, perturbation_scale * std, base_field.shape)
        ensemble[i] = base_field + noise
    return ensemble


def ensemble_attribution_distance(ensemble_binary, threshold_ratio=0.5):
    n_members = ensemble_binary.shape[0]
    flat = ensemble_binary.reshape(n_members, -1)
    m = flat.shape[1]


    total_dist = 0
    count = 0
    for i in range(n_members):
        for j in range(i + 1, n_members):
            total_dist += subset_distance_hamming(flat[i], flat[j])
            count += 1

    mean_dist = total_dist / count if count > 0 else 0.0


    consensus = np.mean(flat, axis=0)
    consensus_mask = (consensus >= threshold_ratio).astype(np.int64)

    return mean_dist, consensus_mask.reshape(ensemble_binary.shape[1:])


def test_monte_carlo():
    val = monte_carlo_line_integral(lambda x: x ** 2, 10000, seed=42)
    exact = line01_monomial_integral(2)
    assert abs(val - exact) < 0.05
    mu, var = subset_distance_stats(10, 500, seed=42)
    assert mu >= 0
    print("monte_carlo_ensemble 自测试通过")


if __name__ == "__main__":
    test_monte_carlo()
