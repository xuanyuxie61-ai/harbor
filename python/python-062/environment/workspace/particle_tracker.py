
import numpy as np


def closest_point_brute(points, target):
    points = np.atleast_2d(points)
    target = np.atleast_1d(target)

    diffs = points - target
    dists_sq = np.sum(diffs**2, axis=1)
    idx = int(np.argmin(dists_sq))
    dist = np.sqrt(dists_sq[idx])

    return idx, dist


def find_containing_tetrahedron(p, nodes, element_nodes):
    from fem_basis import tetrahedron_volume

    for e in range(element_nodes.shape[0]):
        en = element_nodes[e]
        elem_nodes = nodes[en]

        try:
            vol = tetrahedron_volume(elem_nodes)
        except ValueError:
            continue


        sub_vols = np.zeros(4)
        valid = True
        for i in range(4):
            sub_nodes = np.copy(elem_nodes)
            sub_nodes[i] = p
            try:
                sub_vols[i] = tetrahedron_volume(sub_nodes)
            except ValueError:
                valid = False
                break

        if not valid:
            continue

        bary = sub_vols / vol


        if np.all(bary >= -1e-8) and np.all(bary <= 1 + 1e-8):
            return e, np.clip(bary, 0.0, 1.0)

    return -1, None


def track_particles_rk2(particles, velocity_func, dt, n_steps):
    n_particle = particles.shape[0]
    trajectories = np.zeros((n_steps + 1, n_particle, 3), dtype=np.float64)
    trajectories[0] = particles

    for step in range(n_steps):
        pos = trajectories[step]


        k1 = np.zeros((n_particle, 3))
        for i in range(n_particle):
            k1[i] = velocity_func(pos[i])

        pos_mid = pos + 0.5 * dt * k1

        k2 = np.zeros((n_particle, 3))
        for i in range(n_particle):
            k2[i] = velocity_func(pos_mid[i])

        new_pos = pos + dt * k2


        for i in range(n_particle):
            if new_pos[i, 2] < 0:
                new_pos[i, 2] = abs(new_pos[i, 2])

        trajectories[step + 1] = new_pos

    return trajectories
