
import numpy as np
from utils import check_bounds


def sinsq_potential(r, r0=1.0):
    r = np.asarray(r, dtype=np.float64)
    pi2 = np.pi / 2.0
    r_trunc = np.where(r <= pi2, r, pi2)
    return np.sin(r_trunc) ** 2


def sinsq_force(r, r0=1.0):
    r = np.asarray(r, dtype=np.float64)
    pi2 = np.pi / 2.0
    r_trunc = np.where(r <= pi2, r, pi2)
    return -np.sin(2.0 * r_trunc)


def compute_forces_and_energies(pos, vel, mass, box, interaction_type='sinsq'):
    nd, np_particles = pos.shape
    force = np.zeros_like(pos, dtype=np.float64)
    potential = 0.0

    for i in range(np_particles):

        Ri = pos - pos[:, i:i + 1]

        for d in range(nd):
            Ri[d, :] -= box[d] * np.round(Ri[d, :] / box[d])

        D = np.sqrt(np.sum(Ri ** 2, axis=0))
        mask = D > 1e-10
        Ri_valid = Ri[:, mask]
        D_valid = D[mask]

        if interaction_type == 'sinsq':
            pi2 = np.pi / 2.0
            D2 = np.where(D_valid <= pi2, D_valid, pi2)
            potential += 0.5 * np.sum(np.sin(D2) ** 2)

            force_factor = np.sin(2.0 * D2) / D_valid
            force[:, i] += np.sum(Ri_valid * force_factor, axis=1)
        elif interaction_type == 'lennard_jones':

            sigma = 0.3
            epsilon = 1.0
            sr = sigma / D_valid
            sr6 = sr ** 6
            sr12 = sr6 ** 2
            potential += 0.5 * np.sum(4.0 * epsilon * (sr12 - sr6))
            force_factor = 24.0 * epsilon * (2.0 * sr12 - sr6) / D_valid
            force[:, i] += np.sum(Ri_valid * force_factor, axis=1)

    kinetic = 0.5 * mass * np.sum(vel ** 2)
    return force, potential, kinetic


def initialize_particles(np_particles, nd, box, temperature=300.0, mass=1.0):
    pos = np.random.rand(nd, np_particles).astype(np.float64)
    for d in range(nd):
        pos[d, :] *= box[d]

    kB = 1.380649e-23
    sigma_v = np.sqrt(kB * temperature / mass)
    vel = np.random.randn(nd, np_particles).astype(np.float64) * sigma_v
    acc = np.zeros((nd, np_particles), dtype=np.float64)
    return pos, vel, acc


def velocity_verlet_step(pos, vel, acc, force, mass, dt, box):
    rmass = 1.0 / mass
    pos_new = pos + vel * dt + 0.5 * acc * dt * dt

    for d in range(pos_new.shape[0]):
        pos_new[d, :] -= box[d] * np.floor(pos_new[d, :] / box[d])

    acc_new = force * rmass
    vel_new = vel + 0.5 * dt * (acc_new + acc)

    return pos_new, vel_new, acc_new


def simulate_particle_transport(np_particles, nd, box, dt, n_steps, mass=1.0,
                                 temperature=300.0, interaction_type='sinsq'):
    pos, vel, acc = initialize_particles(np_particles, nd, box, temperature, mass)
    trajectory = np.zeros((n_steps + 1, nd, np_particles), dtype=np.float64)
    energy_history = np.zeros((n_steps + 1, 3), dtype=np.float64)

    force, pot, kin = compute_forces_and_energies(pos, vel, mass, box, interaction_type)
    e0 = pot + kin
    trajectory[0, :, :] = pos
    energy_history[0, :] = [pot, kin, e0]

    for step in range(n_steps):
        pos, vel, acc = velocity_verlet_step(pos, vel, acc, force, mass, dt, box)
        force, pot, kin = compute_forces_and_energies(pos, vel, mass, box, interaction_type)
        trajectory[step + 1, :, :] = pos
        energy_history[step + 1, :] = [pot, kin, pot + kin]

    return trajectory, energy_history


def compute_local_temperature_from_kinetic(vel, mass, kB=1.380649e-23):
    nd, np_particles = vel.shape
    v2_mean = np.mean(np.sum(vel ** 2, axis=0))
    T_local = mass * v2_mean / (nd * kB)
    return T_local
