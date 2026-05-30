
import numpy as np
from scipy.linalg import expm
from typing import Callable, Tuple






def velocity_verlet_propagator(H0: np.ndarray, dt: float, mass: float = 1.0) -> np.ndarray:
    if H0.shape[0] != H0.shape[1]:
        raise ValueError("H0 必须是方阵")
    if dt <= 0:
        raise ValueError("dt > 0 required")

    return expm(-1j * H0 * dt)


def driven_hubbard_evolution(H0: np.ndarray, drive_func: Callable, times: np.ndarray,
                              nsites: int) -> np.ndarray:
    dim = H0.shape[0]
    nt = len(times)
    if nt < 2:
        raise ValueError("times 长度必须 >= 2")
    rho = np.zeros((nt, dim, dim), dtype=np.complex128)

    evals, evecs = np.linalg.eigh(H0)
    gs = evecs[:, 0]
    rho[0] = np.outer(gs, gs.conj())
    for n in range(nt - 1):
        dt = times[n + 1] - times[n]
        t_mid = times[n] + dt * 0.5
        Vt = drive_func(t_mid)
        Ht = H0 + Vt
        U = expm(-1j * Ht * dt)
        rho[n + 1] = U @ rho[n] @ U.conj().T
    return rho


def sawtooth_wave(t: float, omega: float) -> float:
    if omega <= 0:
        raise ValueError("omega > 0 required")
    T = 2.0 * np.pi / omega
    val = (t % T) / T - 0.5
    return val


def sawtooth_drive_matrix(nsites: int, amplitude: float, omega: float, times: np.ndarray) -> list:
    V_list = []
    for t in times:
        V = np.diag([amplitude * sawtooth_wave(t, omega) * ((-1) ** i) for i in range(nsites)])
        V_list.append(V)
    return V_list






def reaction_twoway_ode_rhs(y: np.ndarray, k1: float, k2: float) -> np.ndarray:
    w1, w2 = y
    if w1 < 0 or w2 < 0:

        w1 = max(w1, 0.0)
        w2 = max(w2, 0.0)
    dw1dt = -k1 * w1 + k2 * w2
    dw2dt = +k1 * w1 - k2 * w2
    return np.array([dw1dt, dw2dt])


def solve_reaction_twoway(k1: float, k2: float, w0: np.ndarray, t_span: Tuple[float, float],
                           nt: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    if k1 < 0 or k2 < 0:
        raise ValueError("k1, k2 >= 0 required")
    if nt < 2:
        raise ValueError("nt >= 2 required")
    t = np.linspace(t_span[0], t_span[1], nt)
    dt = t[1] - t[0]
    y = np.zeros((nt, 2))
    y[0] = w0
    for i in range(nt - 1):
        k1_rk = reaction_twoway_ode_rhs(y[i], k1, k2)
        k2_rk = reaction_twoway_ode_rhs(y[i] + 0.5 * dt * k1_rk, k1, k2)
        k3_rk = reaction_twoway_ode_rhs(y[i] + 0.5 * dt * k2_rk, k1, k2)
        k4_rk = reaction_twoway_ode_rhs(y[i] + dt * k3_rk, k1, k2)
        y[i + 1] = y[i] + (dt / 6.0) * (k1_rk + 2 * k2_rk + 2 * k3_rk + k4_rk)

        y[i + 1] = np.maximum(y[i + 1], 0.0)
    return t, y


def doublon_dynamics_hubbard(U: float, t_hop: float, beta: float, t_max: float) -> Tuple[np.ndarray, np.ndarray]:
    if U <= 0:
        raise ValueError("U > 0 required")
    k1 = t_hop ** 2 / U
    k2 = np.exp(-beta * U) * t_hop
    w0 = np.array([0.1, 0.1])
    return solve_reaction_twoway(k1, k2, w0, (0.0, t_max), nt=500)


def energy_evolution(rho_t: np.ndarray, H0: np.ndarray) -> np.ndarray:
    nt = len(rho_t)
    E = np.zeros(nt)
    for i in range(nt):
        E[i] = float(np.trace(rho_t[i] @ H0).real)
    return E


if __name__ == "__main__":
    t, y = doublon_dynamics_hubbard(U=4.0, t_hop=1.0, beta=2.0, t_max=10.0)
    print(f"Doublon final: w1={y[-1,0]:.4f}, w2={y[-1,1]:.4f}")
