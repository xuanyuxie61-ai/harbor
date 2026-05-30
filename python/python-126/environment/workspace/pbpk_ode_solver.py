
import numpy as np
from typing import Callable, Tuple, List





def tough_deriv(t: float, y: np.ndarray) -> np.ndarray:
    y1, y2, y3, y4 = y
    if y1 <= 0.0:
        y1 = 1e-300
    dy = np.zeros(4)
    dy[0] = 2.0 * t * (y2 ** 0.2) * y4
    dy[1] = 10.0 * t * np.exp(5.0 * (y2 - 1.0)) * y4
    dy[2] = 2.0 * t * y4
    dy[3] = -2.0 * t * np.log(y1)
    return dy


def tough_exact(t: float) -> np.ndarray:
    s = np.sin(t * t)
    c = np.cos(t * t)
    return np.array([np.exp(s), np.exp(5.0 * s), s + 1.0, c])






def rk4_step(f: Callable, t: float, y: np.ndarray, h: float) -> np.ndarray:
    k1 = f(t, y)
    k2 = f(t + 0.5 * h, y + 0.5 * h * k1)
    k3 = f(t + 0.5 * h, y + 0.5 * h * k2)
    k4 = f(t + h, y + h * k3)
    return y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)






def implicit_trapezoidal_step(f: Callable, t: float, y: np.ndarray, h: float,
                               tol: float = 1e-10, max_iter: int = 50) -> np.ndarray:
    f_n = f(t, y)
    y_new = y + h * f_n
    for _ in range(max_iter):
        y_next = y + 0.5 * h * (f_n + f(t + h, y_new))
        if np.linalg.norm(y_next - y_new) < tol:
            return y_next
        y_new = y_next
    return y_new






def rosenbrock_step(f: Callable, t: float, y: np.ndarray, h: float,
                     J: Callable = None) -> np.ndarray:
    n = len(y)
    gamma = 1.0
    if J is None:

        eps = np.sqrt(np.finfo(float).eps)
        J_mat = np.zeros((n, n))
        f0 = f(t, y)
        for j in range(n):
            y_pert = y.copy()
            y_pert[j] += eps * max(1.0, abs(y[j]))
            J_mat[:, j] = (f(t, y_pert) - f0) / (y_pert[j] - y[j])
    else:
        J_mat = J(t, y)
        f0 = f(t, y)

    M = np.eye(n) - h * gamma * J_mat
    try:
        k = np.linalg.solve(M, f0)
    except np.linalg.LinAlgError:

        k = f0
    return y + h * k






def solve_ode(f: Callable, t_span: Tuple[float, float], y0: np.ndarray,
              method: str = "rk4", h_init: float = 0.01,
              rtol: float = 1e-6, atol: float = 1e-9,
              max_steps: int = 100000) -> Tuple[np.ndarray, np.ndarray]:
    t0, tf = t_span
    if t0 >= tf:
        raise ValueError("t_span must satisfy t0 < tf")
    if len(y0) < 1:
        raise ValueError("y0 must be non-empty")

    t = t0
    y = y0.astype(float).copy()
    ts = [t]
    ys = [y.copy()]
    h = h_init
    step = 0

    while t < tf and step < max_steps:
        h = min(h, tf - t)
        if method == "rk4":
            y_new = rk4_step(f, t, y, h)
        elif method == "implicit_trap":
            y_new = implicit_trapezoidal_step(f, t, y, h)
        elif method == "rosenbrock":
            y_new = rosenbrock_step(f, t, y, h)
        else:
            raise ValueError(f"Unknown method: {method}")


        if method == "rk4":

            y_half1 = rk4_step(f, t, y, h / 2.0)
            y_half2 = rk4_step(f, t + h / 2.0, y_half1, h / 2.0)
            err = np.linalg.norm(y_new - y_half2) / (atol + rtol * np.linalg.norm(y_new))
            if err > 1.0 and h > 1e-12:
                h *= max(0.5, 0.9 / np.sqrt(err))
                continue
            elif err < 0.5:
                h *= min(2.0, 0.9 / np.sqrt(max(err, 1e-10)))
        else:

            pass

        t += h
        y = y_new

        y = np.maximum(y, 0.0)
        ts.append(t)
        ys.append(y.copy())
        step += 1

    return np.array(ts), np.array(ys)






class PBPK_ODE_System:

    def __init__(self, params: dict = None):
        if params is None:
            params = self.default_params()
        self.params = params
        self.V = params["V"]
        self.Q = params["Q"]
        self.Kp = params["Kp"]
        self.CL = params["CL"]
        self.Vmax = params["Vmax"]
        self.Km = params["Km"]
        self.GFR = params["GFR"]
        self.fu = params["fu"]
        self.n_comp = 7

    @staticmethod
    def default_params() -> dict:
        return {
            "V": np.array([1.5, 1.5, 0.3, 30.0, 10.0, 0.5, 1.5]),
            "Q": np.array([6.0, 1.5, 1.2, 0.9, 0.2, 0.3, 6.0]),
            "Kp": np.array([1.0, 2.5, 1.2, 0.7, 5.0, 1.5, 1.0]),
            "CL": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            "Vmax": 5.0,
            "Km": 10.0,
            "GFR": 0.125,
            "fu": 0.1,
        }

    def rhs(self, t: float, C: np.ndarray) -> np.ndarray:
        if len(C) != self.n_comp:
            raise ValueError(f"C must have length {self.n_comp}")
        C = np.maximum(C, 0.0)
        dCdt = np.zeros(self.n_comp)
        C_art = C[0]
        C_ven = C[6]




        raise NotImplementedError("Hole 1: Liver Michaelis-Menten metabolism not implemented")




        raise NotImplementedError("Hole 1: Kidney GFR clearance not implemented")


        for i in [3, 4, 5]:
            dCdt[i] = self.Q[i] * (C_art - C[i] / self.Kp[i]) / self.V[i]



        dose_rate = self._oral_input(t)
        venous_return = sum(self.Q[i] * C[i] / self.Kp[i] for i in range(1, 6))
        dCdt[0] = (dose_rate + venous_return - self.Q[0] * C_art) / self.V[0]


        dCdt[6] = (self.Q[0] * C_art - self.Q[0] * C_ven) / self.V[6]

        dCdt[0] += self.Q[0] * (C_ven - C_art) / self.V[0]

        return dCdt

    def _oral_input(self, t: float) -> float:
        Dose = 100.0
        ka = 0.1
        return Dose * ka * np.exp(-ka * t) if t >= 0 else 0.0

    def jacobian(self, t: float, C: np.ndarray) -> np.ndarray:
        n = self.n_comp
        eps = np.sqrt(np.finfo(float).eps)
        J = np.zeros((n, n))
        f0 = self.rhs(t, C)
        for j in range(n):
            C_pert = C.copy()
            C_pert[j] += eps * max(1.0, abs(C[j]))
            J[:, j] = (self.rhs(t, C_pert) - f0) / (C_pert[j] - C[j])
        return J


def solve_pbpk_ode(t_span: Tuple[float, float], C0: np.ndarray = None,
                   method: str = "rosenbrock", h_init: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
    system = PBPK_ODE_System()
    if C0 is None:
        C0 = np.zeros(system.n_comp)
    f = system.rhs
    J = system.jacobian


    def step_wrapper(t, y, h):
        return rosenbrock_step(f, t, y, h, J)

    if method == "rosenbrock":

        t0, tf = t_span
        t = t0
        y = C0.astype(float).copy()
        ts = [t]
        ys = [y.copy()]
        h = h_init
        max_steps = 100000
        step = 0
        while t < tf and step < max_steps:
            h = min(h, tf - t)
            y_new = step_wrapper(t, y, h)
            y_new = np.maximum(y_new, 0.0)
            t += h
            y = y_new
            ts.append(t)
            ys.append(y.copy())
            step += 1
        return np.array(ts), np.array(ys)
    else:
        return solve_ode(f, t_span, C0, method=method, h_init=h_init)






if __name__ == "__main__":

    t_span = (0.0, 1.0)
    y0 = tough_exact(0.0)
    ts, ys = solve_ode(tough_deriv, t_span, y0, method="rk4", h_init=0.001)
    y_final_exact = tough_exact(ts[-1])
    y_final_num = ys[-1]
    print(f"Tough ODE final error: {np.linalg.norm(y_final_num - y_final_exact):.4e}")


    ts2, ys2 = solve_pbpk_ode((0.0, 120.0), method="rosenbrock", h_init=0.1)
    print(f"PBPK solved: {len(ts2)} steps, final concentrations: {ys2[-1]}")
    print(f"Cmax liver: {np.max(ys2[:, 1]):.3f} mg/L")
    print(f"Cmax tumor: {np.max(ys2[:, 5]):.3f} mg/L")
