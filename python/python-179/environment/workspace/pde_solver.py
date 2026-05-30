
import numpy as np
from typing import Dict, Tuple, List
from system_utils import EPS, TOL_RANK, check_finite, Logger

from domain_generator import generate_parametric_radial_domain, compute_radial_cross_section
from fem_discretization import assemble_fem_matrices_1d, extract_tridiagonal, fem_l2_norm
from reaction_kinetics import parametric_reaction_source, reaction_jacobian_diagonal
from tridiagonal_solver import r83_cg, r83_cr_fa, r83_cr_sl
from tensor_train_decomposition import tt_svd






def imex_euler_step(u: np.ndarray, M: np.ndarray, K: np.ndarray,
                    dt: float, diffusion_coeff: float,
                    k1: float, k2: float, mu: float, mix_ratio: float) -> np.ndarray:








    raise NotImplementedError("Hole 1: IMEX Euler step 待实现")


def crank_nicolson_step(u: np.ndarray, M: np.ndarray, K: np.ndarray,
                        dt: float, diffusion_coeff: float,
                        k1: float, k2: float, mu: float, mix_ratio: float) -> np.ndarray:
    A = M + 0.5 * dt * diffusion_coeff * K
    b_rhs = M @ u - 0.5 * dt * diffusion_coeff * K @ u
    b_rhs += dt * M @ parametric_reaction_source(u, k1, k2, mu, mix_ratio)
    A_r83 = extract_tridiagonal(A)
    u_new = r83_cg(A_r83, b_rhs, x0=u.copy(), tol=1e-12)
    return u_new






def solve_reaction_diffusion(params: Dict[str, float],
                              n_space: int = 64,
                              n_time: int = 50,
                              t_final: float = 2.0,
                              diffusion_coeff: float = 0.05,
                              logger: Logger = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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


    x = generate_parametric_radial_domain(n_space, B, L, w, D_egg)

    sort_idx = np.argsort(x)
    x = x[sort_idx]


    r0 = compute_radial_cross_section(x, B, L, w, D_egg)
    u0 = r0 / (np.max(r0) + EPS)


    M, K = assemble_fem_matrices_1d(x, diffusion_coeff=diffusion_coeff)


    t_array = np.linspace(0.0, t_final, n_time)
    dt = t_array[1] - t_array[0] if n_time > 1 else t_final
    solution = np.zeros((n_space, n_time), dtype=float)
    solution[:, 0] = u0
    u = u0.copy()

    for j in range(1, n_time):

        u = imex_euler_step(u, M, K, dt, diffusion_coeff, k1, k2, mu, mix_ratio)

        u = np.clip(u, 0.0, 1.0)
        solution[:, j] = u

    if logger:
        logger.info(f"PDE solved: n_space={n_space}, n_time={n_time}, params={params}")
    return x, t_array, solution






def build_solution_tensor(param_grid: Dict[str, np.ndarray],
                           n_space: int = 32,
                           n_time: int = 20,
                           t_final: float = 1.0,
                           diffusion_coeff: float = 0.05,
                           logger: Logger = None) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    if logger is None:
        logger = Logger()
    param_names = list(param_grid.keys())
    param_shapes = [len(param_grid[name]) for name in param_names]
    tensor_shape = (n_space, n_time) + tuple(param_shapes)
    tensor = np.zeros(tensor_shape, dtype=float)


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

        target_idx = (slice(None), slice(None)) + idx_tuple
        tensor[target_idx] = sol
        count += 1
        if count % max(1, total // 10) == 0 and logger:
            logger.info(f"Tensor build progress: {count}/{total}")

    if logger:
        logger.info(f"Solution tensor built, shape={tensor.shape}, total_params={total}")
    return tensor, param_grid






def compress_and_analyze(tensor: np.ndarray, max_tt_rank: int = 8,
                         logger: Logger = None) -> Dict:
    if logger is None:
        logger = Logger()

    cores = tt_svd(tensor, max_rank=max_tt_rank, tol=1e-8)
    tt_ranks = [cores[k].shape[2] for k in range(len(cores) - 1)]
    tt_ranks = [1] + tt_ranks + [1]


    from itertools import product
    approx = tt_cores_to_full(cores)
    err_rel = np.linalg.norm(tensor - approx) / (np.linalg.norm(tensor) + EPS)
    err_max = np.max(np.abs(tensor - approx))


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
