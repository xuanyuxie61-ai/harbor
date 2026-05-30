
import numpy as np


def random_walk_1d(step_num, walk_num, step_length=1.0):
    if step_num < 1 or walk_num < 1:
        raise ValueError("step_num and walk_num must be positive integers.")
    x2_ave = np.zeros(step_num + 1)
    x2_max = np.zeros(step_num + 1)

    for _ in range(walk_num):
        x = 0.0
        x2_ave[0] += 0.0
        for step in range(1, step_num + 1):
            r = np.random.rand()
            if r <= 0.5:
                x -= step_length
            else:
                x += step_length
            x2 = x ** 2
            x2_ave[step] += x2
            x2_max[step] = max(x2_max[step], x2)

    x2_ave /= walk_num
    return x2_ave, x2_max


def random_walk_3d(step_num, walk_num, step_length=1.0):
    if step_num < 1 or walk_num < 1:
        raise ValueError("step_num and walk_num must be positive integers.")
    r2_ave = np.zeros(step_num + 1)

    for _ in range(walk_num):
        pos = np.zeros(3)
        for step in range(1, step_num + 1):
            axis = np.random.randint(0, 3)
            direction = 1 if np.random.rand() > 0.5 else -1
            pos[axis] += direction * step_length
            r2_ave[step] += np.dot(pos, pos)

    r2_ave /= walk_num
    return r2_ave


def hot_electron_escape_probability(energy, theta, barrier_height):
    if energy <= 0 or barrier_height < 0:
        raise ValueError("energy must be positive and barrier_height non-negative.")
    normal_energy = energy * (np.cos(theta) ** 2)
    return 1.0 if normal_energy >= barrier_height else 0.0


def effective_diffusion_coefficient(mfp, tau):
    if tau <= 0:
        raise ValueError("tau must be positive.")
    return (mfp ** 2) / (3.0 * tau)


def calculate_collection_efficiency(particle_radius, mfp, tau, barrier_height,
                                     plasmon_energy, num_walkers=1000):
    if particle_radius <= 0 or mfp <= 0 or tau <= 0:
        raise ValueError("Physical parameters must be positive.")

    collected = 0
    max_steps = int(1e5)
    thermalization_steps = int(10.0 * tau / (mfp / 1.0e6))

    for _ in range(num_walkers):

        r = particle_radius * (np.random.rand() ** (1.0 / 3.0))
        theta = np.arccos(2.0 * np.random.rand() - 1.0)
        phi = 2.0 * np.pi * np.random.rand()
        pos = np.array([
            r * np.sin(theta) * np.cos(phi),
            r * np.sin(theta) * np.sin(phi),
            r * np.cos(theta)
        ])

        alive = True
        for step in range(min(max_steps, thermalization_steps)):

            step_theta = np.arccos(2.0 * np.random.rand() - 1.0)
            step_phi = 2.0 * np.pi * np.random.rand()
            step_vec = mfp * np.array([
                np.sin(step_theta) * np.cos(step_phi),
                np.sin(step_theta) * np.sin(step_phi),
                np.cos(step_theta)
            ])
            pos += step_vec

            dist = np.linalg.norm(pos)
            if dist >= particle_radius:

                normal_angle = np.arccos(np.clip(
                    np.dot(pos / dist, step_vec / (mfp + 1e-20)), -1.0, 1.0))
                if hot_electron_escape_probability(plasmon_energy, normal_angle, barrier_height):
                    collected += 1
                alive = False
                break

        if not alive:
            continue

    efficiency = collected / num_walkers if num_walkers > 0 else 0.0
    return efficiency
