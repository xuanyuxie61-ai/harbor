
import numpy as np


def enumerate_triads(k_max, dim=2):
    triads = []
    rng = range(-k_max, k_max + 1)

    if dim == 2:
        for k1 in rng:
            for k2 in rng:
                if k1 == 0 and k2 == 0:
                    continue
                for p1 in rng:
                    for p2 in rng:
                        if p1 == 0 and p2 == 0:
                            continue
                        q1 = -(k1 + p1)
                        q2 = -(k2 + p2)
                        if abs(q1) > k_max or abs(q2) > k_max:
                            continue
                        if q1 == 0 and q2 == 0:
                            continue

                        k_vec = (k1, k2)
                        p_vec = (p1, p2)
                        q_vec = (q1, q2)
                        triads.append((k_vec, p_vec, q_vec))
    else:
        for k1 in rng:
            for k2 in rng:
                for k3 in rng:
                    if k1 == 0 and k2 == 0 and k3 == 0:
                        continue
                    for p1 in rng:
                        for p2 in rng:
                            for p3 in rng:
                                if p1 == 0 and p2 == 0 and p3 == 0:
                                    continue
                                q1 = -(k1 + p1)
                                q2 = -(k2 + p2)
                                q3 = -(k3 + p3)
                                if max(abs(q1), abs(q2), abs(q3)) > k_max:
                                    continue
                                if q1 == 0 and q2 == 0 and q3 == 0:
                                    continue
                                triads.append(((k1, k2, k3), (p1, p2, p3), (q1, q2, q3)))

    return triads


def projection_operator(k_vec):
    k = np.array(k_vec, dtype=np.float64)
    d = len(k)
    k2 = np.dot(k, k)

    if k2 < 1e-15:
        return np.eye(d)

    P = np.eye(d) - np.outer(k, k) / k2
    return P


def energy_transfer_rate_triad(uk, up, uq, k_vec, p_vec, q_vec):
    k = np.array(k_vec, dtype=np.float64)
    p = np.array(p_vec, dtype=np.float64)



    uk_conj = np.conj(uk)

    if len(k_vec) == 3:
        cross = np.cross(up, uq)
        T = np.imag(np.dot(uk_conj, cross))
    else:

        T = np.dot(k, np.imag(uk_conj * np.dot(up, uq)))

    return float(T)


def shell_energy_flux(triads, velocities, k_bins):
    n_bins = len(k_bins) - 1
    Pi = np.zeros(n_bins, dtype=np.float64)

    for k_vec, p_vec, q_vec in triads:
        k_mag = np.linalg.norm(k_vec)
        p_mag = np.linalg.norm(p_vec)

        uk = velocities.get(k_vec)
        up = velocities.get(p_vec)
        uq = velocities.get(q_vec)

        if uk is None or up is None or uq is None:
            continue

        T = energy_transfer_rate_triad(uk, up, uq, k_vec, p_vec, q_vec)


        for b in range(n_bins):
            if k_bins[b] <= k_mag < k_bins[b + 1]:
                Pi[b] += T
                break

    return Pi
