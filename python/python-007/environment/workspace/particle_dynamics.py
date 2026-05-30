import numpy as np


def sinsq_potential(r, cutoff=np.pi / 2):
    r = np.asarray(r, dtype=np.float64)
    r_eff = np.minimum(r, cutoff)
    return np.sin(r_eff) ** 2


def sinsq_force(r_vec, cutoff=np.pi / 2):
    r = np.linalg.norm(r_vec)
    if r < 1e-15:
        return np.zeros_like(r_vec)

    r_eff = min(r, cutoff)


    force_magnitude = np.sin(2.0 * r_eff)
    return force_magnitude * r_vec / r


def lennard_jones_potential(r, epsilon=1.0, sigma=1.0):
    r = np.asarray(r, dtype=np.float64)
    sr = sigma / r
    sr6 = sr ** 6
    sr12 = sr6 ** 2
    return 4.0 * epsilon * (sr12 - sr6)


def velocity_verlet_step(positions, velocities, forces, dt, mass=1.0,
                          force_func=None, box_size=None):
    positions = np.asarray(positions, dtype=np.float64)
    velocities = np.asarray(velocities, dtype=np.float64)
    forces = np.asarray(forces, dtype=np.float64)


    new_positions = positions + velocities * dt + 0.5 * forces * dt * dt / mass


    if box_size is not None:
        new_positions = new_positions % box_size


    if force_func is not None:
        new_forces = force_func(new_positions)
    else:
        new_forces = np.zeros_like(forces)


    new_velocities = velocities + 0.5 * (forces + new_forces) * dt / mass

    return new_positions, new_velocities, new_forces


def compute_pair_forces(positions, potential='sinsq', cutoff=None, box_size=None):
    positions = np.asarray(positions, dtype=np.float64)
    n, dim = positions.shape

    forces = np.zeros_like(positions)
    pe = 0.0

    if cutoff is None:
        cutoff = np.pi / 2 if potential == 'sinsq' else 2.5

    for i in range(n):
        for j in range(i + 1, n):
            r_vec = positions[j] - positions[i]


            if box_size is not None:
                r_vec = r_vec - box_size * np.rint(r_vec / box_size)

            r = np.linalg.norm(r_vec)
            if r < 1e-15 or r > cutoff:
                continue

            if potential == 'sinsq':
                r_eff = min(r, cutoff)
                f_mag = np.sin(2.0 * r_eff)
                f_vec = f_mag * r_vec / r
                pe += np.sin(r_eff) ** 2
            elif potential == 'lj':
                sr = 1.0 / r
                sr6 = sr ** 6
                sr12 = sr6 ** 2
                f_mag = 24.0 * (2.0 * sr12 - sr6) / r
                f_vec = f_mag * r_vec / r
                pe += 4.0 * (sr12 - sr6)
            else:
                continue

            forces[i] += f_vec
            forces[j] -= f_vec

    return forces, pe


def compute_kinetic_energy(velocities, mass=1.0):
    velocities = np.asarray(velocities, dtype=np.float64)
    return 0.5 * mass * np.sum(velocities ** 2)


def run_particle_simulation(n_particles, dim, n_steps, dt, box_size=10.0,
                             mass=1.0, potential='sinsq', seed=None):
    if seed is not None:
        np.random.seed(seed)


    positions = np.random.rand(n_particles, dim) * box_size
    velocities = np.random.randn(n_particles, dim)

    velocities -= np.mean(velocities, axis=0)


    forces, pe = compute_pair_forces(positions, potential=potential,
                                      cutoff=np.pi / 2 if potential == 'sinsq' else 2.5,
                                      box_size=box_size)

    ke_history = []
    pe_history = []
    pos_history = []

    for step in range(n_steps):
        ke = compute_kinetic_energy(velocities, mass)
        ke_history.append(ke)
        pe_history.append(pe)
        pos_history.append(positions.copy())

        positions, velocities, forces = velocity_verlet_step(
            positions, velocities, forces, dt, mass=mass,
            force_func=lambda p: compute_pair_forces(p, potential=potential,
                                                      cutoff=np.pi / 2 if potential == 'sinsq' else 2.5,
                                                      box_size=box_size)[0],
            box_size=box_size
        )


        _, pe = compute_pair_forces(positions, potential=potential,
                                     cutoff=np.pi / 2 if potential == 'sinsq' else 2.5,
                                     box_size=box_size)

    return {
        'positions': pos_history,
        'kinetic_energy': np.array(ke_history),
        'potential_energy': np.array(pe_history),
        'total_energy': np.array(ke_history) + np.array(pe_history)
    }


def accretion_disk_particle_model(n_dust, r_in, r_out, z_scale,
                                   n_steps, dt, seed=None):
    if seed is not None:
        np.random.seed(seed)

    dim = 3

    phi = np.random.uniform(0, 2 * np.pi, n_dust)
    r = np.random.uniform(r_in, r_out, n_dust)
    z = np.random.normal(0, z_scale, n_dust)

    positions = np.zeros((n_dust, dim))
    positions[:, 0] = r * np.cos(phi)
    positions[:, 1] = r * np.sin(phi)
    positions[:, 2] = z


    G = 1.0
    M_bh = 1.0
    v_kep = np.sqrt(G * M_bh / r)
    velocities = np.zeros((n_dust, dim))
    velocities[:, 0] = -v_kep * np.sin(phi)
    velocities[:, 1] = v_kep * np.cos(phi)


    velocities += 0.01 * np.random.randn(n_dust, dim)


    def disk_force_func(pos):
        n = pos.shape[0]
        forces = np.zeros_like(pos)


        dists = np.linalg.norm(pos, axis=1)
        dists = np.where(dists < 0.01, 0.01, dists)
        forces = -G * M_bh * pos / (dists.reshape(-1, 1) ** 3)


        cutoff = 0.05
        for i in range(n):
            for j in range(i + 1, n):
                r_vec = pos[j] - pos[i]
                r = np.linalg.norm(r_vec)
                if r < cutoff and r > 1e-15:
                    f_rep = -0.01 * (1.0 / r - 1.0 / cutoff) * r_vec / r
                    forces[i] += f_rep
                    forces[j] -= f_rep

        return forces

    forces = disk_force_func(positions)

    pos_history = []
    for step in range(n_steps):
        pos_history.append(positions.copy())
        positions, velocities, forces = velocity_verlet_step(
            positions, velocities, forces, dt,
            force_func=disk_force_func
        )

    return {
        'positions': pos_history,
        'final_positions': positions,
        'final_velocities': velocities
    }
