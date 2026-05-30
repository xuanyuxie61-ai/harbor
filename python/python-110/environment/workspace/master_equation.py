
import numpy as np
from typing import Callable, Tuple, Dict
from utils import validate_array_1d, validate_array_2d



H_BAR = 1.054571817e-34


def jaynes_cummings_hamiltonian(
    omega_c: float,
    omega_dot: float,
    g_coupling: float,
    n_photon_cutoff: int = 5,
) -> np.ndarray:
    if n_photon_cutoff < 1:
        raise ValueError("n_photon_cutoff must be >= 1")





    raise NotImplementedError("Hole 1: 请实现 jaynes_cummings_hamiltonian 函数体")


def lindblad_dissipator(
    rho: np.ndarray,
    L: np.ndarray,
) -> np.ndarray:
    rho = validate_array_2d(rho, "rho")
    L = validate_array_2d(L, "L")
    if rho.shape != L.shape:
        raise ValueError("rho and L must have same shape")
    Ld = L.conj().T
    term = L @ rho @ Ld
    anti = Ld @ L @ rho + rho @ Ld @ L
    return term - 0.5 * anti


def lindblad_master_equation_rhs(
    rho: np.ndarray,
    H: np.ndarray,
    jump_operators: list,
    gamma_rates: np.ndarray,
) -> np.ndarray:
    rho = validate_array_2d(rho, "rho")
    H = validate_array_2d(H, "H")
    if rho.shape != H.shape:
        raise ValueError("rho and H must have same shape")
    comm = H @ rho - rho @ H
    drho = -1j / H_BAR * comm
    for Lk, gk in zip(jump_operators, gamma_rates):
        drho += gk * lindblad_dissipator(rho, Lk)
    return drho


def vectorize_density_matrix(rho: np.ndarray) -> np.ndarray:
    rho = validate_array_2d(rho, "rho")
    return rho.T.ravel()


def unvectorize_density_matrix(vec: np.ndarray, dim: int) -> np.ndarray:
    vec = validate_array_1d(vec, "vec")
    if vec.size != dim * dim:
        raise ValueError("Vector size incompatible with dimension")
    return vec.reshape((dim, dim)).T


def build_liouvillian_superoperator(
    H: np.ndarray,
    jump_operators: list,
    gamma_rates: np.ndarray,
) -> np.ndarray:
    dim = H.shape[0]
    I = np.eye(dim, dtype=complex)
    L_sup = -1j / H_BAR * (np.kron(I, H) - np.kron(H.T, I))
    for Lk, gk in zip(jump_operators, gamma_rates):
        Ld = Lk.conj().T
        term1 = np.kron(Lk.T.conj(), Lk)
        term2 = 0.5 * (np.kron(I, Ld @ Lk) + np.kron((Lk.T @ Lk.conj()), I))
        L_sup += gk * (term1 - term2)
    return L_sup


def solve_steady_state(
    H: np.ndarray,
    jump_operators: list,
    gamma_rates: np.ndarray,
) -> np.ndarray:
    dim = H.shape[0]
    L_sup = build_liouvillian_superoperator(H, jump_operators, gamma_rates)

    trace_constraint = np.zeros(dim * dim, dtype=complex)
    for i in range(dim):
        trace_constraint[i * dim + i] = 1.0
    L_sup[-1, :] = trace_constraint
    b = np.zeros(dim * dim, dtype=complex)
    b[-1] = 1.0

    vec_rho, residuals, rank, s = np.linalg.lstsq(L_sup, b, rcond=None)
    rho_ss = unvectorize_density_matrix(vec_rho, dim)

    rho_ss = 0.5 * (rho_ss + rho_ss.conj().T)
    tr = np.trace(rho_ss)
    if abs(tr) > 1e-15:
        rho_ss /= tr
    return rho_ss


def midpoint_integration_ode(
    f: Callable[[np.ndarray], np.ndarray],
    y0: np.ndarray,
    t_span: Tuple[float, float],
    n_steps: int,
) -> Tuple[np.ndarray, np.ndarray]:
    y0 = validate_array_1d(y0, "y0")
    dim = y0.size
    dt = (t_span[1] - t_span[0]) / n_steps
    t_vals = np.linspace(t_span[0], t_span[1], n_steps + 1)
    y_vals = np.zeros((n_steps + 1, dim), dtype=complex)
    y_vals[0, :] = y0


    eps = 1e-8
    M = np.zeros((dim, dim), dtype=complex)
    for j in range(dim):
        e_j = np.zeros(dim, dtype=complex)
        e_j[j] = 1.0
        M[:, j] = (f(e_j * eps) - f(np.zeros(dim, dtype=complex))) / eps

    I = np.eye(dim, dtype=complex)
    LHS = I - 0.5 * dt * M
    RHS = I + 0.5 * dt * M
    try:
        inv_LHS = np.linalg.inv(LHS)
        propagator = inv_LHS @ RHS
    except np.linalg.LinAlgError:

        propagator = np.linalg.pinv(LHS) @ RHS

    for n in range(n_steps):
        y_vals[n + 1, :] = propagator @ y_vals[n, :]
    return t_vals, y_vals


def solve_master_equation_time_evolution(
    H: np.ndarray,
    jump_operators: list,
    gamma_rates: np.ndarray,
    rho0: np.ndarray,
    t_span: Tuple[float, float],
    n_steps: int,
) -> Dict[str, np.ndarray]:
    rho0 = validate_array_2d(rho0, "rho0")
    dim = H.shape[0]
    if rho0.shape != (dim, dim):
        raise ValueError("rho0 shape incompatible with H")

    def rhs_vec(y_vec: np.ndarray) -> np.ndarray:
        rho = unvectorize_density_matrix(y_vec, dim)
        drho = lindblad_master_equation_rhs(rho, H, jump_operators, gamma_rates)
        return vectorize_density_matrix(drho)

    y0 = vectorize_density_matrix(rho0)
    t_vals, y_traj = midpoint_integration_ode(rhs_vec, y0, t_span, n_steps)

    rho_traj = []
    for k in range(n_steps + 1):
        rho_k = unvectorize_density_matrix(y_traj[k, :], dim)

        rho_k = 0.5 * (rho_k + rho_k.conj().T)
        rho_traj.append(rho_k)

    return {
        "t": t_vals,
        "rho_traj": rho_traj,
    }


def excited_state_population(rho: np.ndarray) -> float:


    raise NotImplementedError("Hole 2: 请实现 excited_state_population 函数体")


def cavity_photon_number(rho: np.ndarray, n_cutoff: int) -> float:
    rho = validate_array_2d(rho, "rho")
    dim = rho.shape[0]
    if dim != 2 * n_cutoff:
        raise ValueError("Density matrix dimension mismatch with n_cutoff")
    n_avg = 0.0
    for n in range(n_cutoff):
        idx_g = 2 * n
        idx_e = 2 * n + 1
        n_avg += n * (np.real(rho[idx_g, idx_g]) + np.real(rho[idx_e, idx_e]))
    return float(n_avg)


def check_trace_conservation(rho_traj: list, tol: float = 1e-6) -> bool:
    for k, rho in enumerate(rho_traj):
        tr = np.trace(rho)
        if abs(tr - 1.0) > tol:
            return False
    return True
