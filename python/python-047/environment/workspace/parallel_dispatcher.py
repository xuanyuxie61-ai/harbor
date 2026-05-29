"""
parallel_dispatcher.py
并行任务调度与分块计算模块

融合以下种子项目的核心算法：
  - 237_cuda_loop：CUDA并行循环的模拟与任务分配策略

物理背景：
  深部密度结构反演涉及大规模矩阵运算和大量正演积分计算，
  需要通过并行化加速。本模块模拟CUDA风格的网格-线程块任务分配策略，
  将大规模重力正演和反演计算分解为可并行执行的任务单元。
  
  任务索引映射（仿CUDA）：
      global_id = threadIdx.x 
                  + blockDim.x * threadIdx.y
                  + blockDim.x * blockDim.y * blockIdx.x
                  + blockDim.x * blockDim.y * blockDim.z * blockIdx.y * gridDim.x
                  + ...
"""

import numpy as np
import multiprocessing as mp


def simulate_cuda_indexing(grid_dim, block_dim, n_tasks):
    """
    模拟CUDA线程索引到任务ID的映射。
    
    融合 237_cuda_loop 的核心算法。
    
    参数：
        grid_dim: (gx, gy, gz) 网格维度
        block_dim: (bx, by, bz) 线程块维度
        n_tasks: 总任务数
    返回：
        task_map: list of (task_id, block_idx, thread_idx) 的列表
    """
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
                            
                            # 每个线程负责多个任务（stride = total_threads）
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
    """
    并行化重力正演计算。
    
    将观测点分块，每块由一个工作进程计算正演重力异常。
    
    参数：
        obs_chunks: list of (N_i, 3) 观测点分块
        grid_params: dict 包含 grid_centers, grid_volumes, green_row 等
        density_model: (N_param,) 密度模型
        n_workers: 工作进程数（None表示使用CPU核心数）
    返回：
        dg_total: (N_total,) 总重力异常
    """
    if n_workers is None:
        n_workers = max(1, mp.cpu_count() - 1)
    
    n_workers = min(n_workers, len(obs_chunks))
    
    if n_workers <= 1:
        # 串行计算
        all_obs = np.vstack(obs_chunks)
        return _compute_chunk(all_obs, grid_params, density_model)
    
    # 多进程并行
    with mp.Pool(n_workers) as pool:
        args = [(chunk, grid_params, density_model) for chunk in obs_chunks]
        results = pool.starmap(_compute_chunk, args)
    
    dg_total = np.concatenate(results)
    return dg_total


def _compute_chunk(obs_chunk, grid_params, density_model):
    """计算单个观测点块的重力异常。"""
    from forward_model import composite_forward_model
    
    # 构建简化的棱柱体列表
    prisms = []
    centers = grid_params.get('grid_centers', None)
    volumes = grid_params.get('grid_volumes', None)
    
    if centers is not None and volumes is not None:
        # 将网格单元近似为等体积立方体
        for j in range(len(centers)):
            c = centers[j]
            dv = volumes[j] ** (1.0 / 3.0)  # 等效边长
            half = dv / 2.0
            bounds = (c[0] - half, c[0] + half,
                      c[1] - half, c[1] + half,
                      c[2] - half, c[2] + half)
            prisms.append(bounds + (density_model[j],))
    
    dg = composite_forward_model(prisms, [], None, obs_chunk, qmc_samples=500)
    return dg


def parallel_sensitivity_matrix(obs_points, grid_centers, grid_volumes, n_workers=None):
    """
    并行构造灵敏度矩阵。
    
    将观测点分块，每块计算对应的 G 矩阵行。
    """
    if n_workers is None:
        n_workers = max(1, mp.cpu_count() - 1)
    
    obs_points = np.asarray(obs_points, dtype=float)
    N_obs = obs_points.shape[0]
    
    # 分块
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
    """
    静态任务调度：将 n_tasks 均匀分配给 n_workers。
    
    返回每个worker的任务索引范围。
    """
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
    """
    动态任务调度：将任务分成小块，由worker动态获取。
    
    返回所有task chunks的列表。
    """
    chunks = []
    for i in range(0, n_tasks, chunk_size):
        chunks.append((i, min(i + chunk_size, n_tasks)))
    return chunks


def parallel_matrix_vector_product(G, v, n_workers=None):
    """
    并行矩阵-向量乘法 y = G @ v。
    
    按行分块并行计算。
    """
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
    """
    对多个数据子集进行并行反演（集成反演）。
    
    用于Bootstrap不确定性估计。
    """
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
