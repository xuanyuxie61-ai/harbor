
import numpy as np


class MonteCarloPoreError(Exception):
    pass


def random_triangle_area_in_disk(n_trials, rng=None):
    if n_trials < 1:
        raise MonteCarloPoreError("n_trials 必须 ≥ 1")
    if rng is None:
        rng = np.random.default_rng()

    areas = np.empty(n_trials, dtype=float)
    for k in range(n_trials):
        theta = 2.0 * np.pi * rng.random(3)
        r = np.sqrt(rng.random(3))
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        s1 = np.hypot(x[0] - x[1], y[0] - y[1])
        s2 = np.hypot(x[1] - x[2], y[1] - y[2])
        s3 = np.hypot(x[2] - x[0], y[2] - y[0])
        s = 0.5 * (s1 + s2 + s3)

        area_sq = s * (s - s1) * (s - s2) * (s - s3)
        if area_sq < 0:
            area_sq = 0.0
        areas[k] = np.sqrt(area_sq)

    return float(np.mean(areas)), float(np.std(areas))


def pore_accessibility_simulation(n_pores, hit_probs, n_trials=100000, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    hit_probs = np.asarray(hit_probs, dtype=float)
    if hit_probs.ndim != 2 or hit_probs.shape[1] != 2:
        raise MonteCarloPoreError("hit_probs 形状必须为 (n, 2)")
    if np.any((hit_probs < 0) | (hit_probs > 1)):
        raise MonteCarLOPoreError("概率必须在 [0, 1] 之间")

    total_arrivals = 0
    total_steps = 0

    for _ in range(n_trials):
        steps = 0
        arrived = False
        for layer in range(n_pores):
            p_pass = hit_probs[layer, 0]
            p_ads = hit_probs[layer, 1]

            steps += 1
            if rng.random() > p_pass:

                break

            if layer == n_pores - 1:
                arrived = True
        if arrived:
            total_arrivals += 1
            total_steps += steps

    arrival_prob = total_arrivals / n_trials
    mean_steps = total_steps / max(total_arrivals, 1)
    return arrival_prob, mean_steps


def estimate_effective_diffusivity_mc(pore_network, temperature, molecular_weight,
                                      n_walks=50000, n_steps=200, step_size=1e-9,
                                      rng=None):
    if rng is None:
        rng = np.random.default_rng()

    R = 8.314462618

    v_thermal = np.sqrt(3.0 * R * temperature / molecular_weight)
    dt = step_size / v_thermal

    msd_sum = 0.0
    valid_walks = 0

    for _ in range(n_walks):

        pos = np.zeros(3, dtype=float)
        if not pore_network(*pos):
            continue
        valid_walks += 1
        start_pos = pos.copy()

        for _ in range(n_steps):

            theta = np.arccos(2.0 * rng.random() - 1.0)
            phi = 2.0 * np.pi * rng.random()
            direction = np.array([
                np.sin(theta) * np.cos(phi),
                np.sin(theta) * np.sin(phi),
                np.cos(theta)
            ])
            new_pos = pos + step_size * direction
            if pore_network(*new_pos):
                pos = new_pos


        msd = np.sum((pos - start_pos) ** 2)
        msd_sum += msd

    if valid_walks == 0:
        raise MonteCarloPoreError("没有有效的随机行走轨迹")

    msd_mean = msd_sum / valid_walks
    D_eff = msd_mean / (6.0 * n_steps * dt)
    return D_eff


def pore_tortuosity_from_mc(n_trials=100000, rng=None):
    mean_area, std_area = random_triangle_area_in_disk(n_trials, rng)
    disk_area = np.pi


    fill_ratio = mean_area / disk_area
    tau_estimate = 1.0 + 2.0 * (1.0 - fill_ratio)

    tau_estimate = max(1.0, min(tau_estimate, 10.0))
    return tau_estimate
