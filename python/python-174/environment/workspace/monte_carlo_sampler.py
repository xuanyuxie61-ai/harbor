
import numpy as np


def walker_build(prob):
    prob = np.asarray(prob, dtype=float)
    if prob.size == 0:
        raise ValueError("概率分布不能为空")
    if np.any(prob < 0):
        raise ValueError("概率必须非负")
    n = prob.size
    s = np.sum(prob)
    if s < 1e-15:
        prob = np.ones(n) / n
    else:
        prob = prob / s


    y = prob * n
    a = np.arange(n)


    small = []
    large = []
    for i in range(n):
        if y[i] < 1.0:
            small.append(i)
        else:
            large.append(i)

    while small and large:
        l = small.pop()
        g = large.pop()
        a[l] = g
        y[g] = y[g] - (1.0 - y[l])
        if y[g] < 1.0:
            small.append(g)
        else:
            large.append(g)


    for i in large:
        y[i] = 1.0
    for i in small:
        y[i] = 1.0

    return y, a


def walker_sampler(y, a):
    n = y.size
    i = np.random.randint(0, n)
    r = np.random.rand()
    if r < y[i]:
        return i
    else:
        return int(a[i])


def disk01_positive_sample(n):
    if n <= 0:
        raise ValueError("n必须为正整数")
    p = np.random.normal(size=(n, 2))
    p = np.abs(p)
    norms = np.linalg.norm(p, axis=1, keepdims=True)
    norms = np.where(norms < 1e-15, 1.0, norms)
    p = p / norms
    r = np.sqrt(np.random.rand(n, 1))
    return r * p


def coin_biased(n, heads_prob):
    if n < 0:
        raise ValueError("n必须非负")
    heads_prob = float(np.clip(heads_prob, 0.0, 1.0))
    v = (np.random.rand(n) < heads_prob).astype(float)
    return 2.0 * v - 1.0


def alnorm(x, upper=True):
    a1 = 5.75885480458
    a2 = 2.62433121679
    a3 = 5.92885724438
    b1 = -29.8213557807
    b2 = 48.6959930692
    c1 = -0.000000038052
    c2 = 0.000398064794
    c3 = -0.151679116635
    c4 = 4.8385912808
    c5 = 0.742380924027
    c6 = 3.99019417011
    con = 1.28
    d1 = 1.00000615302
    d2 = 1.98615381364
    d3 = 5.29330324926
    d4 = -15.1508972451
    d5 = 30.789933034
    ltone = 7.0
    p = 0.39894228044
    q = 0.39990348504
    r_const = 0.398942280385
    utzero = 18.66

    up = upper
    z = float(x)

    if z < 0.0:
        up = not up
        z = -z

    if ltone < z and (not up or utzero < z):
        return 0.0 if up else 1.0

    y_val = 0.5 * z * z

    if z <= con:
        value = 0.5 - z * (p - q * y_val
                           / (y_val + a1 + b1
                              / (y_val + a2 + b2
                                 / (y_val + a3))))
    else:
        value = (r_const * np.exp(-y_val)
                 / (z + c1 + d1
                    / (z + c2 + d2
                       / (z + c3 + d3
                          / (z + c4 + d4
                             / (z + c5 + d5
                                / (z + c6)))))))

    if not up:
        value = 1.0 - value

    return value


def mc_estimate_integral(f_sampler, n_samples):
    if n_samples <= 1:
        raise ValueError("n_samples至少为2")
    samples = np.array([f_sampler() for _ in range(n_samples)])
    mean = np.mean(samples)
    std = np.std(samples, ddof=1)
    std_err = std / np.sqrt(n_samples)
    return mean, std_err


def monte_carlo_fmm_verification(particles, charges, fmm_potential, direct_potential,
                                  n_sample_pairs=None, confidence=0.95):
    N = particles.shape[0]
    if n_sample_pairs is None:
        n_sample_pairs = min(N, 1000)
    n_sample_pairs = min(n_sample_pairs, N)

    idx = np.random.choice(N, size=n_sample_pairs, replace=False)
    rel_err = np.abs((fmm_potential[idx] - direct_potential[idx])
                     / (np.abs(direct_potential[idx]) + 1e-15))

    mean_err = np.mean(rel_err)
    std_err = np.std(rel_err, ddof=1) / np.sqrt(n_sample_pairs)



    target = (1.0 + confidence) / 2.0

    z_low, z_high = 0.0, 5.0
    for _ in range(50):
        z_mid = (z_low + z_high) / 2.0
        if alnorm(z_mid, upper=False) < target:
            z_low = z_mid
        else:
            z_high = z_mid
    z_val = (z_low + z_high) / 2.0

    ci_low = max(0.0, mean_err - z_val * std_err)
    ci_high = mean_err + z_val * std_err

    return {
        "mean_relative_error": float(mean_err),
        "std_error": float(std_err),
        "confidence_interval": (float(ci_low), float(ci_high)),
        "confidence_level": confidence,
        "z_score": float(z_val),
        "n_samples": n_sample_pairs
    }


def nonuniform_particle_sample(prob_density, n_samples, domain="sphere"):
    if domain == "sphere":

        n_bins_theta = 20
        n_bins_phi = 40
        theta_edges = np.linspace(0, np.pi, n_bins_theta + 1)
        phi_edges = np.linspace(0, 2 * np.pi, n_bins_phi + 1)
        probs = []
        centers = []
        for i in range(n_bins_theta):
            for j in range(n_bins_phi):
                t = (theta_edges[i] + theta_edges[i + 1]) / 2.0
                p = (phi_edges[j] + phi_edges[j + 1]) / 2.0
                st = np.sin(t)

                domega = st * (theta_edges[i + 1] - theta_edges[i]) * (phi_edges[j + 1] - phi_edges[j])
                val = prob_density(np.array([np.sin(t) * np.cos(p),
                                              np.sin(t) * np.sin(p),
                                              np.cos(t)]))
                probs.append(max(0.0, val) * domega)
                centers.append((t, p))
        probs = np.array(probs)
        y, a = walker_build(probs)
        samples = []
        for _ in range(n_samples):
            idx = walker_sampler(y, a)
            t, p = centers[idx]

            dt = theta_edges[1] - theta_edges[0]
            dp = phi_edges[1] - phi_edges[0]
            t += (np.random.rand() - 0.5) * dt
            p += (np.random.rand() - 0.5) * dp
            t = np.clip(t, 0, np.pi)
            p = np.clip(p, 0, 2 * np.pi)
            samples.append(np.array([np.sin(t) * np.cos(p),
                                      np.sin(t) * np.sin(p),
                                      np.cos(t)]))
        return np.array(samples)
    elif domain == "disk_positive":
        pts = disk01_positive_sample(n_samples)

        return np.column_stack([pts, np.zeros(n_samples)])
    else:
        raise ValueError(f"未知domain: {domain}")
