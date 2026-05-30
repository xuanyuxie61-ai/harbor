
import numpy as np
from typing import Callable, Tuple, List


class StiffODESolver:
    
    def __init__(self, A: np.ndarray, g: Callable[[float], np.ndarray] = None):
        self.A = A
        self.dim = A.shape[0]
        self.g = g if g is not None else lambda t: np.zeros(self.dim)
    
    def solve(self, y0: np.ndarray, t_span: Tuple[float, float],
              n_steps: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        t0, tf = t_span
        h = (tf - t0) / n_steps
        t = np.linspace(t0, tf, n_steps + 1)
        
        y = np.zeros((n_steps + 1, self.dim))
        y[0] = y0
        

        I = np.eye(self.dim)
        M = I - 0.5 * h * self.A
        

        cond_num = np.linalg.cond(M)
        if cond_num > 1e12:

            M_inv = np.linalg.pinv(M)
        else:
            M_inv = np.linalg.inv(M)
        
        for n in range(n_steps):
            tn = t[n]
            tnp1 = t[n + 1]
            yn = y[n]
            
            gn = self.g(tn)
            gnp1 = self.g(tnp1)
            

            rhs = yn + 0.5 * h * (self.A @ yn + gn + gnp1)
            y[n + 1] = M_inv @ rhs
        
        return t, y


def transducer_ode_system(freq: float = 5e6, Q: float = 50.0,
                          m_eff: float = 1e-6) -> Tuple[np.ndarray, Callable]:
    omega0 = 2.0 * np.pi * freq
    c = m_eff * omega0 / Q
    k = m_eff * omega0 ** 2
    
    A = np.array([[0.0, 1.0],
                  [-k / m_eff, -c / m_eff]])
    

    def forcing(t: float) -> np.ndarray:
        pulse_width = 2.0 / freq
        envelope = np.exp(-(t - pulse_width / 2) ** 2 / (2 * (pulse_width / 4) ** 2))
        if t < 0 or t > pulse_width * 2:
            envelope = 0.0
        F_over_m = envelope * np.sin(omega0 * t) / m_eff
        return np.array([0.0, F_over_m])
    
    return A, forcing


def stiff_ode_exact(t: float, lam: float = -5.0) -> float:
    denom = 1.0 + lam**2
    return lam * (lam * np.cos(t) + np.sin(t)) / denom


def stiff_ode_derivative(t: float, y: float, lam: float = -5.0) -> float:
    return lam * (np.cos(t) - y)


def stiff_ode_exact_array(t_array: np.ndarray, lam: float = -5.0) -> np.ndarray:
    denom = 1.0 + lam**2
    return lam * (lam * np.cos(t_array) + np.sin(t_array)) / denom


def verify_stiff_solver(lam: float = -5.0,
                        t_span: Tuple[float, float] = (0.0, 1.0),
                        n_steps_list: List[int] = None) -> dict:
    if n_steps_list is None:
        n_steps_list = [50, 100, 200, 400, 800]
    
    y0 = np.array([stiff_ode_exact(0.0, lam)])
    A = np.array([[-lam]])
    
    def g_scalar(t: float) -> np.ndarray:
        return np.array([lam * np.cos(t)])
    
    errors_l2 = []
    errors_max = []
    
    for n_steps in n_steps_list:
        solver = StiffODESolver(A, g_scalar)
        t, y_num = solver.solve(y0, t_span, n_steps)
        y_exact = stiff_ode_exact_array(t, lam)
        
        diff = y_num[:, 0] - y_exact
        l2_error = np.sqrt(np.mean(diff ** 2))
        max_error = np.max(np.abs(diff))
        
        errors_l2.append(l2_error)
        errors_max.append(max_error)
    

    convergence_orders = []
    for i in range(1, len(n_steps_list)):
        ratio = errors_l2[i - 1] / errors_l2[i]
        order = np.log2(ratio)
        convergence_orders.append(float(order))
    
    return {
        'n_steps': n_steps_list,
        'l2_errors': errors_l2,
        'max_errors': errors_max,
        'convergence_orders': convergence_orders,
        'stiffness_ratio': abs(lam)
    }


def simulate_transducer_response(freq: float = 5e6, Q: float = 50.0,
                                 t_span: Tuple[float, float] = (0.0, 10e-6),
                                 n_steps: int = 2000) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    A, forcing = transducer_ode_system(freq, Q)
    
    y0 = np.array([0.0, 0.0])
    
    solver = StiffODESolver(A, forcing)
    t, y = solver.solve(y0, t_span, n_steps)
    
    displacement = y[:, 0]
    velocity = y[:, 1]
    
    return t, displacement, velocity



def verify_stiff_solver_fix(lam: float = -5.0,
                            t_span: Tuple[float, float] = (0.0, 1.0),
                            n_steps_list: List[int] = None) -> dict:
    if n_steps_list is None:
        n_steps_list = [50, 100, 200, 400, 800]
    
    y0 = np.array([stiff_ode_exact(0.0, lam)])
    A = np.array([[-lam]])
    
    def g_scalar(t: float) -> np.ndarray:
        return np.array([lam * np.cos(t)])
    
    errors_l2 = []
    errors_max = []
    
    for n_steps in n_steps_list:
        solver = StiffODESolver(A, g_scalar)
        t, y_num = solver.solve(y0, t_span, n_steps)
        y_exact = stiff_ode_exact_array(t, lam)
        
        diff = y_num[:, 0] - y_exact
        l2_error = np.sqrt(np.mean(diff ** 2))
        max_error = np.max(np.abs(diff))
        
        errors_l2.append(l2_error)
        errors_max.append(max_error)
    
    convergence_orders = []
    for i in range(1, len(n_steps_list)):
        ratio = errors_l2[i - 1] / errors_l2[i]
        order = np.log2(ratio)
        convergence_orders.append(float(order))
    
    return {
        'n_steps': n_steps_list,
        'l2_errors': errors_l2,
        'max_errors': errors_max,
        'convergence_orders': convergence_orders,
        'stiffness_ratio': abs(lam)
    }
