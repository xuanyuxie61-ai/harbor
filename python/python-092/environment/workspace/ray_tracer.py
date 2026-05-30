
import numpy as np
from quadrature_rules import line01_sample_ergodic


C_AIR = 343.0


def sobol_generate(m, n, skip=0):

    phi = (1.0 + np.sqrt(5.0)) / 2.0
    result = np.zeros((m, n), dtype=float)
    for dim in range(m):
        alpha = (phi ** (dim + 1)) % 1.0
        for i in range(n):
            result[dim, i] = ((skip + i + 1) * alpha) % 1.0
    return result


def sample_directions_sobol(n_rays, dim=3):

    sobol = sobol_generate(dim, n_rays)


    u1 = sobol[0, :]
    u2 = sobol[1, :] if dim > 1 else np.random.rand(n_rays)
    u3 = sobol[2, :] if dim > 2 else np.random.rand(n_rays)


    u1 = np.clip(u1, 1e-10, 1.0 - 1e-10)
    u2 = np.clip(u2, 1e-10, 1.0 - 1e-10)
    u3 = np.clip(u3, 1e-10, 1.0 - 1e-10)


    r = np.sqrt(-2.0 * np.log(u1))
    theta = 2.0 * np.pi * u2
    x = r * np.cos(theta)
    y = r * np.sin(theta)

    r2 = np.sqrt(-2.0 * np.log(u3))
    phi_angle = 2.0 * np.pi * u1
    z = r2 * np.cos(phi_angle)

    dirs = np.column_stack((x, y, z))
    norms = np.linalg.norm(dirs, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-14)
    return dirs / norms


def ray_plane_intersection(ray_origin, ray_dir, plane_point, plane_normal):
    denom = np.dot(ray_dir, plane_normal)
    if abs(denom) < 1e-14:
        return np.inf
    t = np.dot(plane_point - ray_origin, plane_normal) / denom
    return t


def reflect_direction(dir_vec, normal):
    dn = np.dot(dir_vec, normal)
    return dir_vec - 2.0 * dn * normal


def scatter_direction(dir_vec, normal, scattering_coeff=0.1):
    d_ref = reflect_direction(dir_vec, normal)
    noise = np.random.randn(3) * scattering_coeff
    d_scat = d_ref + noise
    norm = np.linalg.norm(d_scat)
    if norm < 1e-14:
        return d_ref
    return d_scat / norm


def trace_ray(room_bounds, surfaces, normals, absorption,
              ray_origin, ray_dir, max_reflections=50,
              energy_threshold=1e-6, scattering_coeff=0.05):
    path_lengths = []
    energies = []
    hit_surfaces = []
    positions = [ray_origin.copy()]

    energy = 1.0
    pos = ray_origin.copy()
    direction = ray_dir.copy()

    for refl in range(max_reflections):
        if energy < energy_threshold:
            break


        t_min = np.inf
        hit_surf = None
        hit_normal = None

        for name, normal in normals.items():

            surf_tris = surfaces[name]
            rep_point = surf_tris[0]
            t = ray_plane_intersection(pos, direction, rep_point, normal)
            if t > 1e-6 and t < t_min:

                hit_point = pos + t * direction

                if name == 'floor' or name == 'ceiling':
                    if 0.0 <= hit_point[0] <= 10.0 and 0.0 <= hit_point[1] <= 8.0:
                        t_min = t
                        hit_surf = name
                        hit_normal = normal
                elif name == 'front_wall' or name == 'back_wall':
                    if 0.0 <= hit_point[0] <= 10.0 and 0.0 <= hit_point[2] <= 5.0:
                        t_min = t
                        hit_surf = name
                        hit_normal = normal
                elif name == 'left_wall' or name == 'right_wall':
                    if 0.0 <= hit_point[1] <= 8.0 and 0.0 <= hit_point[2] <= 5.0:
                        t_min = t
                        hit_surf = name
                        hit_normal = normal

        if hit_surf is None or t_min == np.inf:
            break


        pos = pos + t_min * direction
        path_lengths.append(t_min)
        energies.append(energy)
        hit_surfaces.append(hit_surf)
        positions.append(pos.copy())


        alpha = absorption.get(hit_surf, 0.05)
        energy *= (1.0 - alpha)


        if np.random.rand() < scattering_coeff * 5.0:
            direction = scatter_direction(direction, hit_normal, scattering_coeff)
        else:
            direction = reflect_direction(direction, hit_normal)

    return {
        'reflections': len(hit_surfaces),
        'path_lengths': path_lengths,
        'energies': energies,
        'surfaces': hit_surfaces,
        'positions': positions
    }


def monte_carlo_ray_tracing(surfaces, normals, absorption,
                            source_pos, n_rays=5000,
                            max_reflections=50, scattering_coeff=0.05):
    directions = sample_directions_sobol(n_rays)
    all_energies = []
    all_times = []

    for i in range(n_rays):
        result = trace_ray(
            None, surfaces, normals, absorption,
            source_pos, directions[i],
            max_reflections=max_reflections,
            scattering_coeff=scattering_coeff
        )

        cum_time = 0.0
        for j, length in enumerate(result['path_lengths']):
            cum_time += length / C_AIR
            all_times.append(cum_time)
            all_energies.append(result['energies'][j])


    if len(all_times) == 0:
        return np.array([]), np.array([]), 0.0

    idx = np.argsort(all_times)
    all_times = np.array(all_times)[idx]
    all_energies = np.array(all_energies)[idx]


    edc = np.zeros_like(all_energies)
    edc[-1] = all_energies[-1]
    for i in range(len(all_energies) - 2, -1, -1):
        edc[i] = edc[i + 1] + all_energies[i]



    if edc[0] > 1e-14:
        edc_db = 10.0 * np.log10(edc / edc[0])
    else:
        edc_db = np.zeros_like(edc)


    mask = (edc_db <= -5.0) & (edc_db >= -35.0)
    if np.sum(mask) > 5:
        t_fit = all_times[mask]
        e_fit = edc_db[mask]

        A_mat = np.vstack([t_fit, np.ones(len(t_fit))]).T
        slope, intercept = np.linalg.lstsq(A_mat, e_fit, rcond=None)[0]
        if slope < 0:
            T60 = -60.0 / slope
            EDT = -10.0 / slope
        else:
            T60 = 0.0
            EDT = 0.0
    else:
        T60 = 0.0
        EDT = 0.0

    return all_times, edc, T60, EDT


def build_reflection_graph(surfaces, normals, absorption, n_rays=2000):
    source_pos = np.array([5.0, 4.0, 2.5])
    directions = sample_directions_sobol(n_rays)
    surf_names = list(surfaces.keys())
    n_surf = len(surf_names)
    trans_counts = np.zeros((n_surf, n_surf), dtype=float)
    surf_to_idx = {name: i for i, name in enumerate(surf_names)}

    for i in range(n_rays):
        result = trace_ray(
            None, surfaces, normals, absorption,
            source_pos, directions[i], max_reflections=20
        )
        surfaces_hit = result['surfaces']
        for j in range(len(surfaces_hit) - 1):
            s1 = surf_to_idx[surfaces_hit[j]]
            s2 = surf_to_idx[surfaces_hit[j + 1]]
            trans_counts[s1, s2] += 1.0


    row_sums = trans_counts.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums < 1e-14, 1.0, row_sums)
    trans_prob = trans_counts / row_sums

    return trans_prob, surf_names


def compute_room_response_stats(surfaces, normals, absorption, source_pos,
                                 n_rays=1000, max_reflections=30):
    directions = sample_directions_sobol(n_rays)
    free_paths = []
    reflection_counts = []
    final_energies = []

    for i in range(n_rays):
        result = trace_ray(
            None, surfaces, normals, absorption,
            source_pos, directions[i], max_reflections=max_reflections
        )
        if len(result['path_lengths']) > 0:
            free_paths.extend(result['path_lengths'])
            reflection_counts.append(result['reflections'])
            if len(result['energies']) > 0:
                final_energies.append(result['energies'][-1])

    stats = {
        'mean_free_path': float(np.mean(free_paths)) if free_paths else 0.0,
        'std_free_path': float(np.std(free_paths)) if free_paths else 0.0,
        'mean_reflections': float(np.mean(reflection_counts)) if reflection_counts else 0.0,
        'mean_final_energy': float(np.mean(final_energies)) if final_energies else 0.0,
    }
    return stats
