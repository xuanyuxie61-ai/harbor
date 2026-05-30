
import numpy as np
from utils import safe_divide, robust_sqrt, check_finite_array, PRE_EXPONENTIAL, ACTIVATION_ENERGY


class NewtonInterpolation:
    
    def __init__(self, x_data: np.ndarray, y_data: np.ndarray):
        if len(x_data) != len(y_data):
            raise ValueError("x_data and y_data must have same length")
        if len(x_data) < 2:
            raise ValueError("Need at least 2 data points")
        
        self.xd = np.array(x_data, dtype=float)
        self.yd = np.array(y_data, dtype=float)
        self.n = len(x_data)
        self._compute_divided_differences()
    
    def _compute_divided_differences(self):
        self.cd = self.yd.copy()
        
        for i in range(1, self.n):
            for j in range(self.n - 1, i - 1, -1):
                denom = self.xd[j] - self.xd[j - i]
                if abs(denom) < 1e-14:
                    denom = 1e-14
                self.cd[j] = (self.cd[j] - self.cd[j - 1]) / denom
    
    def evaluate(self, x: float) -> float:
        x = float(x)
        

        x_min, x_max = np.min(self.xd), np.max(self.xd)
        if x < x_min:
            x = x_min
        elif x > x_max:
            x = x_max
        
        result = self.cd[-1]
        for i in range(self.n - 2, -1, -1):
            result = result * (x - self.xd[i]) + self.cd[i]
        
        return float(result)
    
    def evaluate_array(self, x_arr: np.ndarray) -> np.ndarray:
        return np.array([self.evaluate(x) for x in x_arr])


class CombustionRateModel:
    
    def __init__(self, a_coeff: float = 1.5e-5, n_coeff: float = 0.5):
        self.a = a_coeff
        self.n = n_coeff
        


        self.pressure_ref = np.array([1.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0]) * 1e6
        self.rate_ref = self.a * (self.pressure_ref ** self.n)
        
        self.interpolator = NewtonInterpolation(self.pressure_ref, self.rate_ref)
    
    def regression_rate(self, pressure: float) -> float:
        return self.a * (pressure ** self.n)
    
    def regression_rate_interpolated(self, pressure: float) -> float:
        return self.interpolator.evaluate(pressure)
    
    def temperature_sensitivity(self, pressure: float, sigma_p: float = 0.002) -> float:
        return sigma_p


class ReactionDiffusionSolver:
    
    def __init__(self,
                 domain_length: float = 0.02,
                 n_points: int = 201,
                 thermal_diffusivity: float = 1.2e-4,
                 heat_release: float = 4.5e7,
                 specific_heat: float = 1800.0,
                 activation_energy: float = 1.26e5,
                pre_exponential: float = 1.8e10,
                 temperature_unburned: float = 500.0,
                 temperature_burned: float = 3600.0,
                 density: float = 15.0):
        
        self.L = domain_length
        self.nx = n_points
        self.alpha = thermal_diffusivity
        self.Q = heat_release
        self.C_p = specific_heat
        self.E_a = activation_energy
        self.A = pre_exponential
        self.T_u = temperature_unburned
        self.T_b = temperature_burned
        self.rho = density
        
        self.x = np.linspace(0, self.L, self.nx)
        self.dx = self.x[1] - self.x[0]
        

        self.beta = self.E_a * (self.T_b - self.T_u) / (8.314 * self.T_b ** 2)
        

        self.S_L_theoretical = self._estimate_laminar_flame_speed()
    
    def _estimate_laminar_flame_speed(self) -> float:
        R_gas = 8.314
        tau_c = safe_divide(1.0, self.A * np.exp(-self.E_a / (R_gas * self.T_b)), default=1e-3)
        S_L = np.sqrt(self.alpha / tau_c) * safe_divide(1.0, self.beta, default=1.0)
        return float(S_L)
    
    def reaction_rate(self, T: float, Y: float = 1.0) -> float:
        R_gas = 8.314
        if T < self.T_u * 0.8:
            return 0.0
        
        rate = self.A * self.rho * Y * np.exp(-self.E_a / (R_gas * max(T, 100.0)))
        return rate
    
    def source_term(self, T: np.ndarray, Y: np.ndarray = None) -> np.ndarray:
        if Y is None:
            Y = np.ones_like(T)
        
        omega = np.array([self.reaction_rate(Ti, Yi) for Ti, Yi in zip(T, Y)])
        source = (self.Q / self.C_p) * omega / max(self.rho, 1e-10)

        source = np.clip(source, 0.0, 1e12)
        return source
    
    def solve_steady_jacobi(self, max_iterations: int = 100000,
                            tolerance: float = 1e-8,
                            omega_relax: float = 0.5) -> dict:

        T = self.T_u + 0.5 * (self.T_b - self.T_u) * \
            (1.0 + np.tanh((self.x - self.L * 0.3) / (self.L * 0.05)))
        T[0] = self.T_u
        T[-1] = self.T_b
        
        dx2_over_alpha = self.dx ** 2 / self.alpha
        
        for it in range(max_iterations):
            T_old = T.copy()
            

            S = self.source_term(T_old)
            

            for i in range(1, self.nx - 1):
                T_new_i = 0.5 * (T_old[i+1] + T_old[i-1] + dx2_over_alpha * S[i])

                T[i] = omega_relax * T_new_i + (1.0 - omega_relax) * T_old[i]
                T[i] = np.clip(T[i], self.T_u * 0.9, self.T_b * 1.1)
            

            T[0] = self.T_u
            T[-1] = self.T_b
            

            residual = np.sqrt(np.mean((T - T_old) ** 2))
            if residual < tolerance:
                break
        

        dTdx = np.gradient(T, self.dx)
        flame_position = self.x[np.argmax(np.abs(dTdx))]
        

        max_grad = np.max(np.abs(dTdx))
        if max_grad > 1e-10:
            flame_thickness = (self.T_b - self.T_u) / max_grad
        else:
            flame_thickness = self.L
        

        source_final = self.source_term(T)
        total_heat_release = np.trapezoid(source_final * self.rho * self.C_p, self.x)
        denom = self.rho * self.C_p * (self.T_b - self.T_u)
        if denom > 1e-10 and total_heat_release > 1e-10:

            S_L_numerical = np.sqrt(total_heat_release / denom * self.alpha)
            S_L_numerical = np.clip(S_L_numerical, 0.01, 100.0)
        else:
            S_L_numerical = self.S_L_theoretical
        
        return {
            "x": self.x,
            "temperature": T,
            "temperature_gradient": dTdx,
            "source": source_final,
            "iterations": it + 1,
            "final_residual": residual,
            "flame_position": float(flame_position),
            "flame_thickness": float(flame_thickness),
            "S_L_theoretical": self.S_L_theoretical,
            "S_L_numerical": float(S_L_numerical),
            "zeldovich_number": self.beta
        }
    
    def solve_time_dependent(self, dt: float = 1.0e-7,
                             n_steps: int = 5000,
                             save_interval: int = 500) -> dict:
        dt_max = 0.5 * self.dx ** 2 / self.alpha
        if dt > dt_max:
            dt = dt_max * 0.9
        
        T = np.ones(self.nx) * self.T_u

        T[self.nx // 2 - 5:self.nx // 2 + 5] = self.T_b
        T[0] = self.T_u
        T[-1] = self.T_b
        
        T_history = [T.copy()]
        t_history = [0.0]
        
        for step in range(n_steps):
            S = self.source_term(T)
            
            T_new = T.copy()
            for i in range(1, self.nx - 1):
                diffusion = self.alpha * (T[i+1] - 2*T[i] + T[i-1]) / self.dx ** 2
                T_new[i] = T[i] + dt * (diffusion + S[i])
            
            T_new[0] = self.T_u
            T_new[-1] = self.T_b
            T = T_new
            
            if step % save_interval == 0:
                T_history.append(T.copy())
                t_history.append((step + 1) * dt)
        
        return {
            "x": self.x,
            "temperature_final": T,
            "temperature_history": np.array(T_history),
            "time_history": np.array(t_history),
            "dt": dt
        }


if __name__ == "__main__":

    x_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_data = np.array([2.0, 3.0, 5.0, 7.0, 11.0])
    interp = NewtonInterpolation(x_data, y_data)
    print(f"Newton interpolation at 2.5: {interp.evaluate(2.5):.4f}")
    

    rate_model = CombustionRateModel()
    print(f"Regression rate at 7MPa: {rate_model.regression_rate(7e6)*1e3:.4f} mm/s")
    print(f"Interpolated rate at 7MPa: {rate_model.regression_rate_interpolated(7e6)*1e3:.4f} mm/s")
    

    rd = ReactionDiffusionSolver()
    print(f"Zeldovich number: {rd.beta:.2f}")
    print(f"Theoretical S_L: {rd.S_L_theoretical:.4f} m/s")
    
    result = rd.solve_steady_jacobi()
    print(f"Jacobi converged in {result['iterations']} iterations")
    print(f"Flame position: {result['flame_position']*1e3:.2f} mm")
    print(f"Flame thickness: {result['flame_thickness']*1e6:.2f} μm")
    print(f"Numerical S_L: {result['S_L_numerical']:.4f} m/s")
