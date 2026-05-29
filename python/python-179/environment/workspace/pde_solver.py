"""
pde_solver.py
参数化反应-扩散方程求解器模块
=============================
整合 FEM 离散化（fem_discretization）、反应动力学（reaction_kinetics）、
三对角求解器（tridiagonal_solver）与 TT 分解（tensor_train_decomposition），
实现从参数空间采样到解张量构建的完整流程。

控制方程（1D 轴对称参数化反应-扩散）：
    ∂u/∂t = D * ∂²u/∂x² + R(u; k1,k2,μ,α)   on x∈[-L/2, L/2]
    ∂u/∂x = 0                                 at x = ±L/2   (Neumann)
    u(x,0) = u0(x)

参数张量：
    解 u 依赖 (x, t, k1, k2, μ, α, B, L, w, D_egg)
    构成 10 维张量（空间 × 时间 × 7 个参数），通过 TT 分解压缩。
"""

import numpy as np
from typing import Dict, Tuple, List
from system_utils import EPS, TOL_RANK, check_finite, Logger

from domain_generator import generate_parametric_radial_domain, compute_radial_cross_section
from fem_discretization import assemble_fem_matrices_1d, extract_tridiagonal, fem_l2_norm
from reaction_kinetics import parametric_reaction_source, reaction_jacobian_diagonal
from tridiagonal_solver import r83_cg, r83_cr_fa, r83_cr_sl
from tensor_train_decomposition import tt_svd


# ---------------------------------------------------------------------------
# 隐式-显式（IMEX）时间步进
# ---------------------------------------------------------------------------

def imex_euler_step(u: np.ndarray, M: np.ndarray, K: np.ndarray,
                    dt: float, diffusion_coeff: float,
                    k1: float, k2: float, mu: float, mix_ratio: float) -> np.ndarray:
    """
    一阶 IMEX Euler 时间步：
        M (u^{n+1} - u^n) / dt + D*K*u^{n+1} = M*R(u^n)
    等价于
        (M + dt*D*K) u^{n+1} = M*u^n + dt*M*R(u^n)

    左侧为 SPD 三对角矩阵，使用 CG 求解。
    """
    # TODO: 实现 IMEX Euler 时间步进的核心计算逻辑
    # 提示：
    #   1. 构造左侧矩阵 A = M + dt * diffusion_coeff * K
    #   2. 构造右侧向量 b = M @ u + dt * M @ R(u^n)
    #   3. 将 A 提取为三对角紧凑格式 R83
    #   4. 调用 r83_cg 求解线性系统 A * u_new = b
    # 注意：此处需与 fem_discretization.assemble_fem_matrices_1d 产生的 M/K 格式兼容，
    #       并与 tridiagonal_solver.r83_cg 的输入格式保持一致。
    raise NotImplementedError("Hole 1: IMEX Euler step 待实现")


def crank_nicolson_step(u: np.ndarray, M: np.ndarray, K: np.ndarray,
                        dt: float, diffusion_coeff: float,
                        k1: float, k2: float, mu: float, mix_ratio: float) -> np.ndarray:
    """
    Crank-Nicolson 时间步（对扩散项隐式，对反应项显式修正）：
        (M + 0.5*dt*D*K) u^{n+1} = (M - 0.5*dt*D*K) u^n + dt*M*R(u^n)

    无条件稳定，时间精度 O(Δt²)。
    """
    A = M + 0.5 * dt * diffusion_coeff * K
    b_rhs = M @ u - 0.5 * dt * diffusion_coeff * K @ u
    b_rhs += dt * M @ parametric_reaction_source(u, k1, k2, mu, mix_ratio)
    A_r83 = extract_tridiagonal(A)
    u_new = r83_cg(A_r83, b_rhs, x0=u.copy(), tol=1e-12)
    return u_new


# ---------------------------------------------------------------------------
# 完整 PDE 求解
# ---------------------------------------------------------------------------

def solve_reaction_diffusion(params: Dict[str, float],
                              n_space: int = 64,
                              n_time: int = 50,
                              t_final: float = 2.0,
                              diffusion_coeff: float = 0.05,
                              logger: Logger = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    求解参数化反应-扩散方程，返回 (x_nodes, t_array, solution)。

    solution shape: (n_space, n_time)
    solution[:, j] 为时刻 t_j 的空间分布。
    """
    if logger is None:
        logger = Logger()
    B = params.get('B', 1.0)
    L = params.get('L', 2.0)
    w = params.get('w', 0.1)
    D_egg = params.get('D_egg', 0.6)
    k1 = params.get('k1', 1.0)
    k2 = params.get('k2', 10.0)
    mu = params.get('mu', 1.5)
    mix_ratio = params.get('mix_ratio', 0.5)

    # 生成空间网格（Chebyshev 节点）
    x = generate_parametric_radial_domain(n_space, B, L, w, D_egg)
    # 按升序排列（FEM 需要）
    sort_idx = np.argsort(x)
    x = x[sort_idx]

    # 初始条件：以截面半径的归一化分布作为初始浓度
    r0 = compute_radial_cross_section(x, B, L, w, D_egg)
    u0 = r0 / (np.max(r0) + EPS)

    # 组装 FEM 矩阵
    M, K = assemble_fem_matrices_1d(x, diffusion_coeff=diffusion_coeff)

    # 时间步进
    t_array = np.linspace(0.0, t_final, n_time)
    dt = t_array[1] - t_array[0] if n_time > 1 else t_final
    solution = np.zeros((n_space, n_time), dtype=float)
    solution[:, 0] = u0
    u = u0.copy()

    for j in range(1, n_time):
        # 使用 IMEX Euler（简单稳定）
        u = imex_euler_step(u, M, K, dt, diffusion_coeff, k1, k2, mu, mix_ratio)
        # 物理边界：浓度非负且不超过 1
        u = np.clip(u, 0.0, 1.0)
        solution[:, j] = u

    if logger:
        logger.info(f"PDE solved: n_space={n_space}, n_time={n_time}, params={params}")
    return x, t_array, solution


# ---------------------------------------------------------------------------
# 参数张量构建
# ---------------------------------------------------------------------------

def build_solution_tensor(param_grid: Dict[str, np.ndarray],
                           n_space: int = 32,
                           n_time: int = 20,
                           t_final: float = 1.0,
                           diffusion_coeff: float = 0.05,
                           logger: Logger = None) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    在参数网格上批量求解 PDE，构建解张量。

    参数网格示例
    ------------
    param_grid = {
        'k1': np.linspace(0.5, 2.0, 4),
        'k2': np.linspace(5.0, 15.0, 4),
        'mu': np.linspace(0.5, 3.0, 3),
        'mix_ratio': np.linspace(0.0, 1.0, 3),
        'B': np.array([0.8, 1.0, 1.2]),
        'w': np.array([0.0, 0.1, 0.2]),
        'D_egg': np.array([0.5, 0.6, 0.7]),
    }

    返回
    ----
    tensor : np.ndarray
        解张量，shape = (n_space, n_time, len(k1), len(k2), ..., len(D_egg))
    param_values : dict
        参数取值字典。
    """
    if logger is None:
        logger = Logger()
    param_names = list(param_grid.keys())
    param_shapes = [len(param_grid[name]) for name in param_names]
    tensor_shape = (n_space, n_time) + tuple(param_shapes)
    tensor = np.zeros(tensor_shape, dtype=float)

    # 遍历所有参数组合
    total = int(np.prod(param_shapes))
    count = 0
    indices = [range(s) for s in param_shapes]
    from itertools import product
    for idx_tuple in product(*indices):
        params = {name: param_grid[name][idx_tuple[i]] for i, name in enumerate(param_names)}
        _, _, sol = solve_reaction_diffusion(params,
                                              n_space=n_space,
                                              n_time=n_time,
                                              t_final=t_final,
                                              diffusion_coeff=diffusion_coeff,
                                              logger=None)
        # 放入张量
        target_idx = (slice(None), slice(None)) + idx_tuple
        tensor[target_idx] = sol
        count += 1
        if count % max(1, total // 10) == 0 and logger:
            logger.info(f"Tensor build progress: {count}/{total}")

    if logger:
        logger.info(f"Solution tensor built, shape={tensor.shape}, total_params={total}")
    return tensor, param_grid


# ---------------------------------------------------------------------------
# 张量压缩与误差分析
# ---------------------------------------------------------------------------

def compress_and_analyze(tensor: np.ndarray, max_tt_rank: int = 8,
                         logger: Logger = None) -> Dict:
    """
    对解张量执行 TT-SVD 压缩，并返回误差分析结果。
    """
    if logger is None:
        logger = Logger()
    # TT 分解
    cores = tt_svd(tensor, max_rank=max_tt_rank, tol=1e-8)
    tt_ranks = [cores[k].shape[2] for k in range(len(cores) - 1)]
    tt_ranks = [1] + tt_ranks + [1]

    # 还原近似并计算误差
    from itertools import product
    approx = tt_cores_to_full(cores)
    err_rel = np.linalg.norm(tensor - approx) / (np.linalg.norm(tensor) + EPS)
    err_max = np.max(np.abs(tensor - approx))

    # 压缩比
    orig_size = tensor.size
    tt_size = sum(c.size for c in cores)
    compression_ratio = orig_size / max(tt_size, 1)

    if logger:
        logger.info(f"TT compression: ranks={tt_ranks}, rel_err={err_rel:.3e}, ratio={compression_ratio:.2f}x")

    return {
        'tt_cores': cores,
        'tt_ranks': tt_ranks,
        'compression_ratio': compression_ratio,
        'relative_error': err_rel,
        'max_error': err_max,
        'original_size': orig_size,
        'tt_size': tt_size,
    }


from tensor_train_decomposition import tt_cores_to_full
