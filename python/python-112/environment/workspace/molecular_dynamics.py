
import numpy as np
from typing import Callable, Tuple, Optional





def sawtooth_driver(t: float, omega: float = 1.0) -> float:
    if omega <= 0:
        raise ValueError("sawtooth_driver: omega must be > 0.")
    return np.mod(t + omega * np.pi, 2.0 * omega * np.pi) - omega * np.pi





def lennard_jones_potential(r: np.ndarray, sigma: float = 3.4, epsilon: float = 0.1) -> np.ndarray:
    if sigma <= 0 or epsilon <= 0:
        raise ValueError("lennard_jones_potential: sigma and epsilon must be > 0.")
    r = np.asarray(r, dtype=float)
    if np.any(r <= 0):
        raise ValueError("lennard_jones_potential: r must be > 0.")

    sr6 = (sigma / r) ** 6
    u = 4.0 * epsilon * sr6 * (sr6 - 1.0)
    return u


def lennard_jones_force(r: np.ndarray, sigma: float = 3.4, epsilon: float = 0.1) -> np.ndarray:
    sr6 = (sigma / r) ** 6
    f = 24.0 * epsilon * (2.0 * sr6 ** 2 - sr6) / r
    return f


def debye_huckel_potential(
    r: np.ndarray,
    q_i: float = 1.0,
    q_j: float = -1.0,
    epsilon_r: float = 80.0,
    kappa: float = 0.1,
) -> np.ndarray:
    if epsilon_r <= 0 or kappa < 0:
        raise ValueError("debye_huckel_potential: epsilon_r > 0 and kappa >= 0.")
    r = np.asarray(r, dtype=float)
    if np.any(r <= 0):
        raise ValueError("debye_huckel_potential: r must be > 0.")

    coulomb_const = 332.0637
    return coulomb_const * q_i * q_j / epsilon_r * np.exp(-kappa * r) / r





def velocity_verlet_step(
    positions: np.ndarray,
    velocities: np.ndarray,
    forces: np.ndarray,
    masses: np.ndarray,
    dt: float,
) -> Tuple[np.ndarray, np.ndarray]:
    if dt <= 0:
        raise ValueError("velocity_verlet_step: dt must be > 0.")
    positions = np.asarray(positions, dtype=float)
    velocities = np.asarray(velocities, dtype=float)
    forces = np.asarray(forces, dtype=float)
    masses = np.asarray(masses, dtype=float)

    if masses.ndim == 0:
        masses = np.full(positions.shape[0], float(masses))
    if masses.shape[0] != positions.shape[0]:
        raise ValueError("velocity_verlet_step: masses length must match positions.")
    if np.any(masses <= 0):
        raise ValueError("velocity_verlet_step: all masses must be > 0.")


    positions_new = positions + velocities * dt + 0.5 * (forces / masses[:, None]) * dt ** 2


    velocities_half = velocities + 0.5 * (forces / masses[:, None]) * dt

    return positions_new, velocities_half





def coarse_grained_md_simulation(
    n_steps: int = 5000,
    dt: float = 0.001,
    temperature: float = 300.0,
    n_protein_atoms: int = 50,
    n_drug_atoms: int = 10,
    n_lipid_atoms: int = 100,
    box_size: np.ndarray = np.array([60.0, 60.0, 60.0]),
    sigma_protein: float = 3.5,
    sigma_drug: float = 3.0,
    sigma_lipid: float = 4.0,
    epsilon_lj: float = 0.1,
    kappa: float = 0.1,
    random_seed: int = 42,
) -> dict:
    if n_steps < 1:
        raise ValueError("coarse_grained_md_simulation: n_steps >= 1.")
    if dt <= 0:
        raise ValueError("coarse_grained_md_simulation: dt > 0.")
    if temperature <= 0:
        raise ValueError("coarse_grained_md_simulation: temperature > 0.")

    np.random.seed(random_seed)

    n_total = n_protein_atoms + n_drug_atoms + n_lipid_atoms
    kB = 0.0019872041


    amu_to_kcal = 0.00239006


    positions = np.zeros((n_total, 3), dtype=float)

    positions[:n_protein_atoms] = np.random.randn(n_protein_atoms, 3) * 1.5

    positions[n_protein_atoms:n_protein_atoms + n_drug_atoms] = np.array([8.0, 0.0, 0.0]) + np.random.randn(n_drug_atoms, 3) * 0.8

    n_lipid_half = n_lipid_atoms // 2
    angles = np.linspace(0, 2 * np.pi, n_lipid_half, endpoint=False)
    r_lipid = 20.0 + 2.0 * np.random.randn(n_lipid_half)
    for i in range(n_lipid_half):
        positions[n_protein_atoms + n_drug_atoms + i] = [
            r_lipid[i] * np.cos(angles[i]),
            r_lipid[i] * np.sin(angles[i]),
            15.0 + np.random.randn() * 0.5,
        ]
    for i in range(n_lipid_half, n_lipid_atoms):
        idx = i - n_lipid_half
        positions[n_protein_atoms + n_drug_atoms + i] = [
            r_lipid[idx] * np.cos(angles[idx]),
            r_lipid[idx] * np.sin(angles[idx]),
            -15.0 + np.random.randn() * 0.5,
        ]


    masses = np.ones(n_total, dtype=float)
    masses[:n_protein_atoms] = 110.0
    masses[n_protein_atoms:n_protein_atoms + n_drug_atoms] = 30.0
    masses[n_protein_atoms + n_drug_atoms:] = 60.0


    m_eff = masses * amu_to_kcal
    v_scale = np.sqrt(kB * temperature / m_eff)
    velocities = np.random.randn(n_total, 3) * v_scale[:, None]


    velocities -= np.sum(velocities * m_eff[:, None], axis=0) / np.sum(m_eff)


    traj_positions = np.zeros((n_steps // 10 + 1, n_total, 3), dtype=float)
    traj_positions[0] = positions.copy()


    potential_energy = np.zeros(n_steps, dtype=float)
    kinetic_energy = np.zeros(n_steps, dtype=float)
    temperature_trace = np.zeros(n_steps, dtype=float)

    def compute_forces(pos: np.ndarray) -> Tuple[np.ndarray, float]:
        f = np.zeros_like(pos)
        u = 0.0


        cutoff = 12.0
        cutoff2 = cutoff ** 2

        for i in range(n_total):
            for j in range(i + 1, n_total):
                dr = pos[j] - pos[i]

                for dim in range(3):
                    if dr[dim] > box_size[dim] * 0.5:
                        dr[dim] -= box_size[dim]
                    elif dr[dim] < -box_size[dim] * 0.5:
                        dr[dim] += box_size[dim]

                r2 = np.dot(dr, dr)
                if r2 > cutoff2:
                    continue

                r = np.sqrt(max(r2, 0.25))
                r2_safe = max(r2, 0.25)


                if i < n_protein_atoms and j < n_protein_atoms:
                    sig, eps = sigma_protein, epsilon_lj
                elif i < n_protein_atoms and j < n_protein_atoms + n_drug_atoms:
                    sig, eps = 0.5 * (sigma_protein + sigma_drug), epsilon_lj
                elif i < n_protein_atoms + n_drug_atoms and j < n_protein_atoms + n_drug_atoms:
                    sig, eps = sigma_drug, epsilon_lj
                else:
                    sig, eps = sigma_lipid, epsilon_lj


                r_lj = max(r, 0.8 * sig)
                f_lj = lennard_jones_force(r_lj, sig, eps)

                f_lj = np.clip(f_lj, -1000.0, 1000.0)
                f_ij = f_lj * dr / r
                f[i] -= f_ij
                f[j] += f_ij
                u += lennard_jones_potential(r_lj, sig, eps)


                if i < n_protein_atoms:
                    q_i = 0.2 if i % 2 == 0 else -0.2
                elif i < n_protein_atoms + n_drug_atoms:
                    q_i = 0.5 if i % 3 == 0 else -0.25
                else:
                    q_i = 0.0

                if j < n_protein_atoms:
                    q_j = 0.2 if j % 2 == 0 else -0.2
                elif j < n_protein_atoms + n_drug_atoms:
                    q_j = 0.5 if j % 3 == 0 else -0.25
                else:
                    q_j = 0.0

                if abs(q_i * q_j) > 1.0e-6:
                    u_elec = debye_huckel_potential(r, q_i, q_j, 80.0, kappa)
                    u += u_elec
                    f_elec = -u_elec * (kappa + 1.0 / r) * dr / r
                    f_elec = np.clip(f_elec, -100.0, 100.0)
                    f[i] += f_elec
                    f[j] -= f_elec


        for i in range(n_protein_atoms):
            k_spring = 0.5
            f[i] -= k_spring * pos[i]
            u += 0.5 * k_spring * np.dot(pos[i], pos[i])


        f_norm = np.linalg.norm(f, axis=1)
        max_force = 500.0
        scale = np.where(f_norm > max_force, max_force / f_norm, 1.0)
        f *= scale[:, None]

        return f, u


    current_pos = positions.copy()
    current_vel = velocities.copy()

    for step in range(n_steps):
        forces, u = compute_forces(current_pos)
        new_pos, vel_half = velocity_verlet_step(current_pos, current_vel, forces, masses, dt)


        new_forces, u_new = compute_forces(new_pos)


        new_vel = vel_half + 0.5 * (new_forces / masses[:, None]) * dt


        for dim in range(3):
            mask_high = new_pos[:, dim] > box_size[dim] * 0.5
            mask_low = new_pos[:, dim] < -box_size[dim] * 0.5
            new_pos[mask_high, dim] = box_size[dim] * 0.5 - (new_pos[mask_high, dim] - box_size[dim] * 0.5)
            new_vel[mask_high, dim] = -abs(new_vel[mask_high, dim])
            new_pos[mask_low, dim] = -box_size[dim] * 0.5 + (-box_size[dim] * 0.5 - new_pos[mask_low, dim])
            new_vel[mask_low, dim] = abs(new_vel[mask_low, dim])


        new_vel *= 0.9995


        if step % 50 == 0 and step > 0:
            ke_inst = 0.5 * np.sum(m_eff * np.sum(new_vel ** 2, axis=1))
            current_T = 2.0 * ke_inst / (3.0 * kB * n_total)
            if current_T > 1.0e-6:

                tau = 0.1
                lambda_b = np.sqrt(1.0 + (dt * 50.0 / tau) * (temperature / current_T - 1.0))
                lambda_b = np.clip(lambda_b, 0.95, 1.05)
                saw = 1.0 + 0.005 * sawtooth_driver(step * dt, omega=0.1)
                new_vel *= lambda_b * saw

        current_pos = new_pos
        current_vel = new_vel

        potential_energy[step] = u_new
        ke = 0.5 * np.sum(m_eff * np.sum(current_vel ** 2, axis=1))
        kinetic_energy[step] = ke
        temperature_trace[step] = 2.0 * ke / (3.0 * kB * n_total)

        if step % 10 == 0:
            traj_positions[step // 10] = current_pos.copy()

    return {
        "positions": current_pos,
        "velocities": current_vel,
        "trajectory": traj_positions,
        "potential_energy": potential_energy,
        "kinetic_energy": kinetic_energy,
        "temperature_trace": temperature_trace,
        "avg_temperature": float(np.mean(temperature_trace)),
        "avg_potential": float(np.mean(potential_energy)),
        "box_size": box_size,
    }
