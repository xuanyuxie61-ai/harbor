
import numpy as np


def dijkstra_shortest_path(graph, source):
    n = len(graph)
    visited = np.zeros(n, dtype=bool)
    distances = np.full(n, np.inf)
    previous = np.full(n, -1, dtype=int)
    
    distances[source] = 0.0
    
    for _ in range(n):

        min_dist = np.inf
        u = -1
        for i in range(n):
            if not visited[i] and distances[i] < min_dist:
                min_dist = distances[i]
                u = i
        
        if u == -1:
            break
        
        visited[u] = True
        

        for v in range(n):
            if not visited[v] and graph[u, v] > 0:
                alt = distances[u] + graph[u, v]
                if alt < distances[v]:
                    distances[v] = alt
                    previous[v] = u
    
    return distances, previous


def reconstruct_path(previous, target):
    path = []
    u = target
    
    while u != -1:
        path.append(u)
        u = previous[u]
    
    path.reverse()
    return path


def build_energy_propagation_graph(depths, horizontal_positions,
                                   N_profile, f=1.0e-4):
    n_z = len(depths)
    n_x = len(horizontal_positions)
    n_nodes = n_z * n_x
    

    node_coords = []
    for i in range(n_z):
        for j in range(n_x):
            node_coords.append((depths[i], horizontal_positions[j]))
    
    graph = np.full((n_nodes, n_nodes), np.inf)
    np.fill_diagonal(graph, 0.0)
    

    dz = np.abs(depths[1] - depths[0]) if n_z > 1 else 1.0
    dx = np.abs(horizontal_positions[1] - horizontal_positions[0]) if n_x > 1 else 1.0
    
    for i in range(n_z):
        for j in range(n_x):
            node_idx = i * n_x + j
            N = N_profile[i] if i < len(N_profile) else 0.01
            

            if j + 1 < n_x:
                neighbor_idx = i * n_x + (j + 1)

                kh = 2.0 * np.pi / 1000.0
                m = np.pi / max(np.abs(depths[i]), 1.0)
                denom = kh**2 + m**2
                cgx = kh * (N**2 - f**2) * m**2 / (denom**2 + 1.0e-12)
                cgx = max(abs(cgx), 0.01)
                weight = dx / cgx
                graph[node_idx, neighbor_idx] = weight
            

            if i + 1 < n_z:
                neighbor_idx = (i + 1) * n_x + j
                kh = 2.0 * np.pi / 1000.0
                m = np.pi / max(np.abs(depths[i]), 1.0)
                denom = kh**2 + m**2
                cgz = m * (N**2 - f**2) * kh**2 / (denom**2 + 1.0e-12)
                cgz = max(abs(cgz), 0.001)
                weight = dz / cgz
                graph[node_idx, neighbor_idx] = weight
            

            if i + 1 < n_z and j + 1 < n_x:
                neighbor_idx = (i + 1) * n_x + (j + 1)
                weight = np.sqrt(dx**2 + dz**2) / max(cgx, cgz)
                graph[node_idx, neighbor_idx] = weight
    

    graph = np.minimum(graph, graph.T)
    
    return graph, node_coords


def permutation_cycle_analysis(n_lockers=100, n_tries=50):

    permutation = np.random.permutation(n_lockers)
    

    visited = np.zeros(n_lockers, dtype=bool)
    cycles = []
    cycle_lengths = []
    
    for start in range(n_lockers):
        if visited[start]:
            continue
        
        cycle = []
        current = start
        
        while not visited[current]:
            visited[current] = True
            cycle.append(current)
            current = permutation[current]
        
        if len(cycle) > 0:
            cycles.append(cycle)
            cycle_lengths.append(len(cycle))
    

    n_success = sum(1 for cl in cycle_lengths if cl <= n_tries)
    success_rate = n_success / len(cycles) if len(cycles) > 0 else 0.0
    
    return cycles, np.array(cycle_lengths), success_rate


def ray_tracing_cycle(wave_frequency, N_profile, z, x0=0.0,
                       theta0=np.pi/4, max_steps=500):
    dz = np.abs(z[1] - z[0]) if len(z) > 1 else 1.0
    dt = dz / 0.5
    
    x_path = np.zeros(max_steps)
    z_path = np.zeros(max_steps)
    theta_path = np.zeros(max_steps)
    
    x = x0
    z_current = np.mean(z)
    theta = theta0
    
    for step in range(max_steps):
        x_path[step] = x
        z_path[step] = z_current
        theta_path[step] = theta
        

        N = np.interp(z_current, z, N_profile)
        N = max(N, 1.0e-6)
        

        c_g = 0.5 * N * np.cos(theta) * np.sin(theta)
        c_gx = c_g * np.cos(theta)
        c_gz = c_g * np.sin(theta)
        

        x += c_gx * dt
        z_current += c_gz * dt
        

        if z_current > np.max(z):
            z_current = 2 * np.max(z) - z_current
            theta = -theta
        elif z_current < np.min(z):
            z_current = 2 * np.min(z) - z_current
            theta = -theta
        

        dN_dz = 0.0
        if step > 0:
            idx = np.argmin(np.abs(z - z_current))
            idx = max(1, min(idx, len(N_profile) - 2))
            dN_dz = (N_profile[idx+1] - N_profile[idx-1]) / (z[idx+1] - z[idx-1])
        
        theta += dN_dz * np.sin(theta) * dt
        

        theta = np.clip(theta, -np.pi/2 + 0.01, np.pi/2 - 0.01)
    
    return x_path, z_path, theta_path
