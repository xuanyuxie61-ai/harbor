
import numpy as np


def ksub_random2(n, k, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    if k < 0 or k > n:
        raise ValueError("k must satisfy 0 <= k <= n")
    if k == 0:
        return np.array([], dtype=int)

    y = rng.choice(n, size=k, replace=False)
    return np.sort(y + 1)


def urn_sample(marble_num, draw_num, color_num, color_count, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    color_count = np.asarray(color_count, dtype=int)
    if np.sum(color_count) != marble_num:
        raise ValueError("sum(color_count) must equal marble_num")
    y = ksub_random2(marble_num, draw_num, rng=rng)
    draw_color = np.zeros(color_num, dtype=int)
    t = 0
    for i in range(color_num):
        b = t
        t = t + color_count[i]

        draw_color[i] = np.sum((b < y) & (y <= t))
    return draw_color


def urn_two_color_pdf(w, draw_num, color_count):
    from scipy.special import comb
    marble_num = np.sum(color_count)
    w = np.asarray(w, dtype=int)
    pw = np.zeros_like(w, dtype=float)
    for i in range(len(w)):
        if w[i] < 0 or w[i] > color_count[0] or (draw_num - w[i]) > color_count[1]:
            pw[i] = 0.0
        else:
            pw[i] = (
                comb(color_count[0], w[i], exact=True) *
                comb(color_count[1], draw_num - w[i], exact=True) /
                comb(marble_num, draw_num, exact=True)
            )
    return pw


def bayesian_posterior_sample(likelihood_func, prior_sampler, n_samples=1000, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    samples = []
    trials = 0
    max_trials = n_samples * 100

    pre_samples = [prior_sampler() for _ in range(200)]
    pre_likes = [likelihood_func(s) for s in pre_samples]
    max_like = np.max(pre_likes)
    if max_like <= 0:
        max_like = 1.0
    while len(samples) < n_samples and trials < max_trials:
        m = prior_sampler()
        like = likelihood_func(m)
        trials += 1
        u = rng.random()
        if u < like / max_like:
            samples.append(m)
    acceptance_rate = len(samples) / max(trials, 1)
    return samples, acceptance_rate
