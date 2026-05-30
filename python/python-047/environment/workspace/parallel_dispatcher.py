
import numpy as np
import multiprocessing as mp


def simulate_cuda_indexing(grid_dim, block_dim, n_tasks):
    gx, gy, gz = grid_dim
    bx, by, bz = block_dim
    
    total_threads = gx * gy * gz * bx * by * bz
    task_map = []
    
    task_id = 0
    for block_z in range(gz):
        for block_y in range(gy):
            for block_x in range(gx):
                for thread_z in range(bz):
                    for thread_y in range(by):
                        for thread_x in range(bx):
                            global_idx = (thread_x
                                          + bx * thread_y
                                          + bx * by * thread_z
                                          + bx * by * bz * block_x
                                          + bx * by * bz * gx * block_y
                                          + bx * by * bz * gx * gy * block_z)
                            

                            assigned_tasks = []
                            t = global_idx
                            while t < n_tasks and len(assigned_tasks) < 100:
                                assigned_tasks.append(t)
                                t += total_threads
                            
                            task_map.append({
                                'global_idx': global_idx,
                                'block': (block_x, block_y, block_z),
                                'thread': (thread_x, thread_y, thread_z),
                                'tasks': assigned_tasks
                            })
                            task_id += 1
    
    return task_map


def parallel_forward_compute(obs_chunks, grid_params, density_model, n_workers=None):
    if n_workers is None:
        n_workers = max(1, mp.cpu_count() - 1)
    
    n_workers = min(n_workers, len(obs_chunks))
    
    if n_workers <= 1:

        all_obs = np.vstack(obs_chunks)
        return _compute_chunk(all_obs, grid_params, density_model)
    

    with mp.Pool(n_workers) as pool:
        args = [(chunk, grid_params, density_model) for chunk in obs_chunks]
        results = pool.starmap(_compute_chunk, args)
    
    dg_total = np.concatenate(results)
    return dg_total


def _compute_chunk(obs_chunk, grid_params, density_model):
    from forward_model import composite_forward_model
    

    prisms = []
    centers = grid_params.get('grid_centers', None)
    volumes = grid_params.get('grid_volumes', None)
    
    if centers is not None and volumes is not None:

        for j in range(len(centers)):
            c = centers[j]
            dv = volumes[j] ** (1.0 / 3.0)
            half = dv / 2.0
            bounds = (c[0] - half, c[0] + half,
                      c[1] - half, c[1] + half,
                      c[2] - half, c[2] + half)
            prisms.append(bounds + (density_model[j],))
    
    dg = composite_forward_model(prisms, [], None, obs_chunk, qmc_samples=500)
    return dg


def parallel_sensitivity_matrix(obs_points, grid_centers, grid_volumes, n_workers=None):
    if n_workers is None:
        n_workers = max(1, mp.cpu_count() - 1)
    
    obs_points = np.asarray(obs_points, dtype=float)
    N_obs = obs_points.shape[0]
    

    chunk_size = max(1, N_obs // n_workers)
    chunks = []
    for i in range(0, N_obs, chunk_size):
        chunks.append(obs_points[i:i + chunk_size])
    
    from inverse_solver import build_sensitivity_matrix
    
    def build_rows(chunk):
        return build_sensitivity_matrix(chunk, grid_centers, grid_volumes, use_toeplitz=False)
    
    if n_workers <= 1 or len(chunks) <= 1:
        G = build_sensitivity_matrix(obs_points, grid_centers, grid_volumes)
        return G
    
    with mp.Pool(min(n_workers, len(chunks))) as pool:
        results = pool.map(build_rows, chunks)
    
    G = np.vstack(results)
    return G


def task_scheduler_static(n_tasks, n_workers):
    chunk_size = n_tasks // n_workers
    remainder = n_tasks % n_workers
    
    ranges = []
    start = 0
    for w in range(n_workers):
        end = start + chunk_size + (1 if w < remainder else 0)
        ranges.append((start, end))
        start = end
    
    return ranges


def task_scheduler_dynamic(n_tasks, n_workers, chunk_size=1):
    chunks = []
    for i in range(0, n_tasks, chunk_size):
        chunks.append((i, min(i + chunk_size, n_tasks)))
    return chunks


def parallel_matrix_vector_product(G, v, n_workers=None):
    if n_workers is None:
        n_workers = max(1, mp.cpu_count() - 1)
    
    G = np.asarray(G, dtype=float)
    v = np.asarray(v, dtype=float)
    N_obs = G.shape[0]
    
    ranges = task_scheduler_static(N_obs, n_workers)
    
    def compute_range(rng):
        start, end = rng
        return G[start:end, :] @ v
    
    if n_workers <= 1:
        return G @ v
    
    with mp.Pool(n_workers) as pool:
        results = pool.map(compute_range, ranges)
    
    y = np.concatenate(results)
    return y


def parallel_ensemble_inversion(ensemble_G, ensemble_d, alpha, order=1, n_workers=None):
    if n_workers is None:
        n_workers = max(1, mp.cpu_count() - 1)
    
    from inverse_solver import tikhonov_solve_dense
    
    n_ensemble = len(ensemble_G)
    
    def invert_single(idx):
        G_i = ensemble_G[idx]
        d_i = ensemble_d[idx]
        m_i, _, _ = tikhonov_solve_dense(G_i, d_i, alpha, order)
        return m_i
    
    if n_workers <= 1 or n_ensemble <= 1:
        results = [invert_single(i) for i in range(n_ensemble)]
    else:
        with mp.Pool(min(n_workers, n_ensemble)) as pool:
            results = pool.map(invert_single, range(n_ensemble))
    
    return np.array(results)
