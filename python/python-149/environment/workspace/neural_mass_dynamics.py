
import numpy as np
from typing import Tuple, Callable, Optional




_DEFAULT_PARAMS = {
    "tau_e": 4.0,
    "tau_i": 12.0,
    "a_ee": 16.0,
    "a_ei": 12.0,
    "a_ie": 15.0,
    "a_ii": 3.0,
    "k_e": 1.3,
    "k_i": 2.0,
    "theta_e": 4.0,
    "theta_i": 3.7,
    "P": 1.0,
    "Q": 0.0,
    "sigma_e": 0.05,
    "sigma_i": 0.03,
    "t0": 0.0,
    "tstop": 200.0,
    "y0": np.array([0.1, 0.05]),
}


def get_neural_parameters(
    tau_e: Optional[float] = None,
    tau_i: Optional[float] = None,
    a_ee: Optional[float] = None,
    a_ei: Optional[float] = None,
    a_ie: Optional[float] = None,
    a_ii: Optional[float] = None,
    k_e: Optional[float] = None,
    k_i: Optional[float] = None,
    theta_e: Optional[float] = None,
    theta_i: Optional[float] = None,
    P: Optional[float] = None,
    Q: Optional[float] = None,
    sigma_e: Optional[float] = None,
    sigma_i: Optional[float] = None,
    t0: Optional[float] = None,
    tstop: Optional[float] = None,
    y0: Optional[np.ndarray] = None,
) -> Tuple:
    global _DEFAULT_PARAMS

    local_vars = locals()
    for key in _DEFAULT_PARAMS:
        if local_vars[key] is not None:
            _DEFAULT_PARAMS[key] = local_vars[key]

    return (
        _DEFAULT_PARAMS["tau_e"],
        _DEFAULT_PARAMS["tau_i"],
        _DEFAULT_PARAMS["a_ee"],
        _DEFAULT_PARAMS["a_ei"],
        _DEFAULT_PARAMS["a_ie"],
        _DEFAULT_PARAMS["a_ii"],
        _DEFAULT_PARAMS["k_e"],
        _DEFAULT_PARAMS["k_i"],
        _DEFAULT_PARAMS["theta_e"],
        _DEFAULT_PARAMS["theta_i"],
        _DEFAULT_PARAMS["P"],
        _DEFAULT_PARAMS["Q"],
        _DEFAULT_PARAMS["sigma_e"],
        _DEFAULT_PARAMS["sigma_i"],
        _DEFAULT_PARAMS["t0"],
        _DEFAULT_PARAMS["tstop"],
        _DEFAULT_PARAMS["y0"],
    )


def sigmoid_activation(x: float, k: float, theta: float) -> float:
    if not np.isfinite(x):
        return 0.0

    arg = -k * (x - theta)
    if arg > 700.0:
        return 0.0
    if arg < -700.0:
        return 1.0
    return 1.0 / (1.0 + np.exp(arg))



_sigmoid_vec = np.vectorize(sigmoid_activation, otypes=[float])


def neural_mass_deriv(
    t: float,
    y: np.ndarray,
    control_fn: Optional[Callable[[float, np.ndarray], float]] = None,
) -> np.ndarray:
    (
        tau_e, tau_i,
        a_ee, a_ei, a_ie, a_ii,
        k_e, k_i, theta_e, theta_i,
        P, Q,
        sigma_e, sigma_i,
        t0, tstop, y0,
    ) = get_neural_parameters()

    E, I = y[0], y[1]


    u = 0.0
    if control_fn is not None:
        try:
            u = float(control_fn(t, y))
        except Exception:
            u = 0.0

    u = np.clip(u, -5.0, 5.0)


    exc_input = a_ee * E - a_ei * I + P + u

    inh_input = a_ie * E - a_ii * I + Q


    Se = sigmoid_activation(exc_input, k_e, theta_e)
    Si = sigmoid_activation(inh_input, k_i, theta_i)


    Se = np.clip(Se, 0.0, 1.0)
    Si = np.clip(Si, 0.0, 1.0)




    pass


def neural_mass_jacobian(
    y: np.ndarray,
    control_fn: Optional[Callable[[float, np.ndarray], float]] = None,
) -> np.ndarray:
    (
        tau_e, tau_i,
        a_ee, a_ei, a_ie, a_ii,
        k_e, k_i, theta_e, theta_i,
        P, Q,
        sigma_e, sigma_i,
        t0, tstop, y0,
    ) = get_neural_parameters()

    E, I = y[0], y[1]

    u = 0.0
    if control_fn is not None:
        try:
            u = float(control_fn(0.0, y))
        except Exception:
            u = 0.0
    u = np.clip(u, -5.0, 5.0)

    exc_input = a_ee * E - a_ei * I + P + u
    inh_input = a_ie * E - a_ii * I + Q

    Se = sigmoid_activation(exc_input, k_e, theta_e)
    Si = sigmoid_activation(inh_input, k_i, theta_i)

    dSe = k_e * Se * (1.0 - Se)
    dSi = k_i * Si * (1.0 - Si)

    J = np.zeros((2, 2))
    J[0, 0] = (-1.0 + a_ee * dSe) / tau_e
    J[0, 1] = (-a_ei * dSe) / tau_e
    J[1, 0] = (a_ie * dSi) / tau_i
    J[1, 1] = (-1.0 - a_ii * dSi) / tau_i

    return J


def neural_oscillation_period(
    linearized: bool = True,
) -> float:
    (
        tau_e, tau_i,
        a_ee, a_ei, a_ie, a_ii,
        k_e, k_i, theta_e, theta_i,
        P, Q,
        sigma_e, sigma_i,
        t0, tstop, y0,
    ) = get_neural_parameters()

    if linearized:

        S0e = sigmoid_activation(P, k_e, theta_e)
        S0i = sigmoid_activation(Q, k_i, theta_i)
        dSe0 = k_e * S0e * (1.0 - S0e)
        dSi0 = k_i * S0i * (1.0 - S0i)

        A = np.array([
            [(-1.0 + a_ee * dSe0) / tau_e, (-a_ei * dSe0) / tau_e],
            [(a_ie * dSi0) / tau_i, (-1.0 - a_ii * dSi0) / tau_i],
        ])

        eigenvalues = np.linalg.eigvals(A)
        imag_parts = np.imag(eigenvalues)
        omega = np.max(np.abs(imag_parts))
        if omega < 1e-12:

            return 1e6
        period = 2.0 * np.pi / omega
    else:

        period = 2.0 * np.pi * np.sqrt(tau_e * tau_i)

    return float(period)


def compute_running_cost(
    y: np.ndarray,
    u: float,
    y_target: np.ndarray,
    Q_mat: Optional[np.ndarray] = None,
    R_scalar: float = 1.0,
) -> float:
    if Q_mat is None:
        Q_mat = np.eye(len(y))
    dy = y - y_target
    state_cost = float(dy @ Q_mat @ dy)
    control_cost = R_scalar * u * u
    return state_cost + control_cost


def compute_terminal_cost(
    y: np.ndarray,
    y_target: np.ndarray,
    P_mat: Optional[np.ndarray] = None,
) -> float:
    if P_mat is None:
        P_mat = np.eye(len(y))
    dy = y - y_target
    return float(dy @ P_mat @ dy)
