
import numpy as np


def langevin_step_euler(particles, velocities, sigma_w, epsilon, dt, C0=2.1):
    n_particle = particles.shape[0]


    w = velocities[:, 2]
    sigma_w_safe = np.maximum(sigma_w, 1e-6)


    a_w = - (C0 * epsilon / sigma_w_safe**2) * w


    b = np.sqrt(C0 * epsilon)


    dW = np.random.randn(n_particle) * np.sqrt(dt)


    new_velocities = np.copy(velocities)
    new_velocities[:, 2] = w + a_w * dt + b * dW


    new_particles = particles + new_velocities * dt


    for i in range(n_particle):
        if new_particles[i, 2] < 0:
            new_particles[i, 2] = -new_particles[i, 2]
            new_velocities[i, 2] = -abs(new_velocities[i, 2])

    return new_particles, new_velocities


def initialize_particles(n_particles, domain_x, domain_y, domain_z, release_height=10.0):
    np.random.seed(123)

    particles = np.zeros((n_particles, 3), dtype=np.float64)
    particles[:, 0] = np.random.uniform(domain_x[0], domain_x[1], n_particles)
    particles[:, 1] = np.random.uniform(domain_y[0], domain_y[1], n_particles)
    particles[:, 2] = release_height + np.random.exponential(2.0, n_particles)
    particles[:, 2] = np.clip(particles[:, 2], 0.1, domain_z[1])

    velocities = np.zeros((n_particles, 3), dtype=np.float64)
    velocities[:, 0] = 5.0 + np.random.randn(n_particles) * 0.5
    velocities[:, 2] = np.random.randn(n_particles) * 0.3

    return particles, velocities


def ensemble_concentration(particles, grid_x, grid_y, grid_z):
    nx, ny, nz = len(grid_x) - 1, len(grid_y) - 1, len(grid_z) - 1
    conc = np.zeros((nx, ny, nz), dtype=np.float64)

    dx = grid_x[1] - grid_x[0]
    dy = grid_y[1] - grid_y[0]
    dz = grid_z[1] - grid_z[0]
    vol = dx * dy * dz

    for p in particles:
        ix = int((p[0] - grid_x[0]) / dx)
        iy = int((p[1] - grid_y[0]) / dy)
        iz = int((p[2] - grid_z[0]) / dz)

        if 0 <= ix < nx and 0 <= iy < ny and 0 <= iz < nz:
            conc[ix, iy, iz] += 1.0 / vol

    return conc
