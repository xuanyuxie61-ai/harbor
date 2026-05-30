
import numpy as np
from typing import Tuple, Optional, Callable
from scipy.integrate import solve_ivp


class BoneRemodelingODE:

    def __init__(self, k_form: float = 0.05, k_res: float = 0.03,
                 U_ref: float = 0.5, rho_min: float = 0.01,
                 rho_max: float = 1.8):
        if k_form <= 0 or k_res <= 0:
            raise ValueError("Rate constants must be positive.")
        if U_ref <= 0:
            raise ValueError("Reference strain energy must be positive.")
        if rho_min >= rho_max:
            raise ValueError("rho_min must be less than rho_max.")

        self.k_form = k_form
        self.k_res = k_res
        self.U_ref = U_ref
        self.rho_min = rho_min
        self.rho_max = rho_max

    def remodeling_rate(self, rho: float, U: float) -> float:
        rho_clip = max(self.rho_min, min(self.rho_max, rho))


        formation = self.k_form * max(U - self.U_ref, 0.0)

        resorption = self.k_res * max(self.U_ref - U, 0.0) * rho_clip

        drhodt = formation - resorption


        if rho_clip <= self.rho_min and drhodt < 0:
            drhodt = 0.0
        if rho_clip >= self.rho_max and drhodt > 0:
            drhodt = 0.0

        return drhodt

    def steady_state_density(self, U: float) -> float:
        if U <= 0:
            return self.rho_min
        rho_ss = self.rho_max * (U / (U + self.U_ref))
        return max(self.rho_min, min(self.rho_max, rho_ss))

    def exact_solution_linear(self, t: np.ndarray, rho0: float,
                              A: float, B: float) -> np.ndarray:
        t = np.asarray(t)
        if B <= 0:
            raise ValueError("B must be positive for stable solution.")
        rho_inf = A / B
        return rho_inf + (rho0 - rho_inf) * np.exp(-B * t)

    def solve_time_dependent(self, rho0: np.ndarray,
                             strain_energy_field: np.ndarray,
                             t_span: Tuple[float, float] = (0.0, 365.0),
                             t_eval: Optional[np.ndarray] = None,
                             method: str = 'RK45') -> Tuple[np.ndarray, np.ndarray]:
        N = len(rho0)
        if len(strain_energy_field) != N:
            raise ValueError("Length mismatch between rho0 and strain_energy_field")

        if t_eval is None:
            t_eval = np.linspace(t_span[0], t_span[1], 50)

        def ode_func(t: float, rho: np.ndarray) -> np.ndarray:
            drhodt = np.zeros(N)
            for i in range(N):
                drhodt[i] = self.remodeling_rate(rho[i], strain_energy_field[i])
            return drhodt

        sol = solve_ivp(ode_func, t_span, rho0, t_eval=t_eval,
                        method=method, dense_output=True,
                        rtol=1e-6, atol=1e-9)

        if not sol.success:
            raise RuntimeError(f"ODE solver failed: {sol.message}")

        return sol.t, sol.y

    def conserved_quantity(self, rho: np.ndarray, volumes: np.ndarray) -> float:
        if len(rho) != len(volumes):
            raise ValueError("Length mismatch")
        return float(np.dot(rho, volumes))

    def check_mass_conservation(self, rho_history: np.ndarray,
                                volumes: np.ndarray,
                                tolerance: float = 1e-3) -> bool:
        M0 = self.conserved_quantity(rho_history[:, 0], volumes)
        M_final = self.conserved_quantity(rho_history[:, -1], volumes)
        relative_error = abs(M_final - M0) / max(abs(M0), 1e-14)
        return relative_error < tolerance





class CoupledBoneRemodelingODE:

    def __init__(self, k1: float = 0.02, k2: float = 0.03,
                 k3: float = 0.1, k4: float = 0.2,
                 k5: float = 0.1, k6: float = 0.15,
                 U_ref: float = 0.5):
        self.params = {
            'k1': k1, 'k2': k2, 'k3': k3,
            'k4': k4, 'k5': k5, 'k6': k6,
            'U_ref': U_ref
        }

    def deriv(self, t: float, y: np.ndarray, U: float) -> np.ndarray:
        rho, c_oc, c_ob = y
        p = self.params

        drhodt = p['k1'] * c_ob - p['k2'] * c_oc * rho
        doc_dt = p['k3'] * max(p['U_ref'] - U, 0.0) - p['k4'] * c_oc
        dob_dt = p['k5'] * max(U - p['U_ref'], 0.0) - p['k6'] * c_ob


        if rho <= 0.01 and drhodt < 0:
            drhodt = 0.0
        if c_oc < 0 and doc_dt < 0:
            doc_dt = 0.0
        if c_ob < 0 and dob_dt < 0:
            dob_dt = 0.0

        return np.array([drhodt, doc_dt, dob_dt])

    def solve(self, y0: np.ndarray, U_func: Callable[[float], float],
              t_span: Tuple[float, float] = (0.0, 365.0),
              t_eval: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        if t_eval is None:
            t_eval = np.linspace(t_span[0], t_span[1], 100)

        def ode_func(t: float, y: np.ndarray) -> np.ndarray:
            U = U_func(t)
            return self.deriv(t, y, U)

        sol = solve_ivp(ode_func, t_span, y0, t_eval=t_eval,
                        method='RK45', rtol=1e-6, atol=1e-9)
        if not sol.success:
            raise RuntimeError(f"Coupled ODE solver failed: {sol.message}")
        return sol.t, sol.y
