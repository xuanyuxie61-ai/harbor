
import numpy as np


def compute_distance_matrix(points):
    n = points.shape[0]
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(points[i] - points[j])
            D[i, j] = d
            D[j, i] = d
    return D


def tsp_greedy_path(n, distance, start=0):
    visited = np.zeros(n, dtype=bool)
    path = []
    current = start
    
    for _ in range(n):
        path.append(current)
        visited[current] = True

        min_dist = np.inf
        next_node = current
        for j in range(n):
            if not visited[j] and distance[current, j] < min_dist:
                min_dist = distance[current, j]
                next_node = j
        current = next_node
    
    return path


def tsp_multi_start(n, distance):
    best_cost = np.inf
    best_path = list(range(n))
    
    for start in range(n):
        path = tsp_greedy_path(n, distance, start)
        cost = sum(distance[path[i], path[(i + 1) % n]] for i in range(n))
        if cost < best_cost:
            best_cost = cost
            best_path = path
    
    return best_path, best_cost


def model_communication_latency(msg_size, alpha=1e-4, beta=1e-8, noise_level=0.05):
    base = alpha + beta * msg_size
    noise = np.random.normal(0, noise_level * base)
    noise = np.clip(noise, -3 * noise_level * base, 3 * noise_level * base)
    return base + noise


def estimate_disk_distance_mean(n_samples=10000):

    theta1 = np.random.uniform(0, 2 * np.pi, n_samples)
    r1 = np.sqrt(np.random.uniform(0, 1, n_samples))
    theta2 = np.random.uniform(0, 2 * np.pi, n_samples)
    r2 = np.sqrt(np.random.uniform(0, 1, n_samples))
    
    p1 = np.column_stack((r1 * np.cos(theta1), r1 * np.sin(theta1)))
    p2 = np.column_stack((r2 * np.cos(theta2), r2 * np.sin(theta2)))
    
    distances = np.linalg.norm(p1 - p2, axis=1)
    return float(np.mean(distances)), float(np.var(distances))


def ca_speedup_theory(s, t_comp_per_step, t_comm_per_step):
    if s < 1:
        return 1.0
    t_std = t_comp_per_step + t_comm_per_step
    t_ca = t_comp_per_step + t_comm_per_step / (s ** 2)
    if t_ca < 1e-15:
        return 1.0
    return t_std / t_ca


def optimize_s_parameter(t_comp, t_comm, s_max=20):
    best_s = 1
    best_speedup = 1.0
    for s in range(1, s_max + 1):
        sp = ca_speedup_theory(s, t_comp, t_comm)
        if sp > best_speedup:
            best_speedup = sp
            best_s = s
    return best_s, best_speedup


def processor_communication_schedule(n_procs, topology='ring'):
    if topology == 'ring':
        schedule = []
        for i in range(n_procs):
            src = i
            dst = (i + 1) % n_procs
            schedule.append((src, dst, 1.0))
        return schedule
    elif topology == 'tsp_optimized':

        angles = np.linspace(0, 2 * np.pi, n_procs, endpoint=False)
        points = np.column_stack((np.cos(angles), np.sin(angles)))
        D = compute_distance_matrix(points)
        path, _ = tsp_multi_start(n_procs, D)
        schedule = []
        for i in range(n_procs):
            src = path[i]
            dst = path[(i + 1) % n_procs]
            schedule.append((src, dst, 1.0))
        return schedule
    else:

        schedule = []
        for i in range(n_procs):
            for j in range(n_procs):
                if i != j:
                    schedule.append((i, j, 1.0))
        return schedule
