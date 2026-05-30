import numpy as np
from constants import TINY




def nearest_neighbor_distances(points):
    n = points.shape[0]
    nn_dist = np.full(n, np.inf)
    for i in range(n):
        for j in range(n):
            if i != j:
                dist = np.linalg.norm(points[i] - points[j])
                if dist < nn_dist[i]:
                    nn_dist[i] = dist
    return nn_dist





def gamma_measure(points):
    nn = nearest_neighbor_distances(points)
    d_min = np.min(nn)
    d_max = np.max(nn)
    if d_min < TINY:
        return np.inf
    return d_max / d_min





def beta_measure(points):
    nn = nearest_neighbor_distances(points)
    mu = np.mean(nn)
    if mu < TINY:
        return np.inf
    return np.std(nn) / mu





def r0_measure(points):
    n = points.shape[0]
    if n < 2:
        return 0.0
    total = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(points[i] - points[j])
            dist = max(dist, TINY)
            total += np.log(dist)
            count += 1
    return -total / count if count > 0 else 0.0





def chi_measure(points, n_samples=5000):
    n = points.shape[0]
    dim = points.shape[1]
    

    lows = np.min(points, axis=0)
    highs = np.max(points, axis=0)

    margin = 0.05 * (highs - lows)
    margin[margin < TINY] = TINY
    lows -= margin
    highs += margin
    

    samples = np.random.uniform(0.0, 1.0, (n_samples, dim))
    for d in range(dim):
        span = highs[d] - lows[d]
        if span > TINY:
            samples[:, d] = lows[d] + samples[:, d] * span
    

    counts = np.zeros(n)
    for s in samples:
        dists = np.linalg.norm(points - s, axis=1)
        idx = np.argmin(dists)
        counts[idx] += 1.0
    
    mean_count = np.mean(counts)
    if mean_count < TINY:
        return np.inf
    return np.std(counts) / mean_count





def q_measure_2d(points):
    if points.shape[1] != 2:
        return None
    n = points.shape[0]
    if n < 3:
        return 0.0
    
    q_values = []
    for i in range(n):

        dists = np.linalg.norm(points - points[i], axis=1)
        dists[i] = np.inf
        idx = np.argsort(dists)[:2]
        if dists[idx[1]] == np.inf:
            continue
        p0, p1, p2 = points[i], points[idx[0]], points[idx[1]]
        

        a = np.linalg.norm(p1 - p2)
        b = np.linalg.norm(p0 - p2)
        c = np.linalg.norm(p0 - p1)
        if a < TINY or b < TINY or c < TINY:
            continue
        

        s = 0.5 * (a + b + c)
        area_sq = s * (s - a) * (s - b) * (s - c)
        area = np.sqrt(max(area_sq, 0.0))
        
        denom = a * a + b * b + c * c
        if denom > TINY:
            q = 4.0 * np.sqrt(3.0) * area / denom
            q_values.append(q)
    
    return np.min(q_values) if q_values else 0.0





def sampling_quality_report(points):
    report = {
        "n_points": points.shape[0],
        "dimension": points.shape[1],
        "gamma": gamma_measure(points),
        "beta": beta_measure(points),
        "r0_energy": r0_measure(points),
        "chi": chi_measure(points),
    }
    if points.shape[1] == 2:
        report["q_2d"] = q_measure_2d(points)
    return report


def evaluate_phase_space_sampling(events):
    features = []
    for e in events:

        pz1_mag = np.linalg.norm(e["pz1"][1:])
        pz2_mag = np.linalg.norm(e["pz2"][1:])
        features.append([e["m_z1"], e["m_z2"], pz1_mag, pz2_mag])
    
    points = np.array(features)

    mins = np.min(points, axis=0)
    maxs = np.max(points, axis=0)
    spans = maxs - mins
    spans[spans < TINY] = 1.0
    normalized = (points - mins) / spans
    
    return sampling_quality_report(normalized)
