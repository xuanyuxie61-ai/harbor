
import numpy as np
from typing import Tuple, Optional, Callable
from numeric_utils import safe_divide


def vft_equation(T: float, A: float, B: float, T0: float) -> float:
    if T <= T0:

        return A * np.exp(B / max(T - T0, 0.01))
    return A * np.exp(B / (T - T0))


def vft_viscosity(T: float, eta_inf: float, B: float, T0: float) -> float:
    if T <= T0:
        return eta_inf * 1e10
    return eta_inf * np.exp(B / (T - T0))


def regula_falsi(
    f: Callable[[float], float],
    a: float,
    b: float,
    tol: float = 1e-6,
    max_iter: int = 100,
) -> Tuple[float, int]:
    fa = f(a)
    fb = f(b)
    

    if fa * fb > 0:

        raise ValueError("regula_falsi: f(a) 和 f(b) 必须异号")
    
    it = 0
    while abs(b - a) > tol:
        if it >= max_iter:
            break
        
        it += 1
        

        if abs(fb - fa) < 1e-15:
            break
        
        c = (a * fb - b * fa) / (fb - fa)
        fc = f(c)
        
        if abs(fc) < tol:
            return c, it
        
        if np.sign(fc) == np.sign(fa):
            a = c
            fa = fc
        else:
            b = c
            fb = fc
    
    return (a + b) / 2.0, it


class GlassTransitionAnalyzer:
    
    def __init__(self):
        self.temperatures = []
        self.specific_volumes = []
        self.energies = []
    
    def add_data_point(
        self,
        temperature: float,
        specific_volume: float,
        energy: Optional[float] = None,
    ):
        self.temperatures.append(temperature)
        self.specific_volumes.append(specific_volume)
        if energy is not None:
            self.energies.append(energy)
    
    def linear_fit(self, T_data: np.ndarray, v_data: np.ndarray) -> Tuple[float, float]:
        T_mean = np.mean(T_data)
        v_mean = np.mean(v_data)
        
        cov = np.mean((T_data - T_mean) * (v_data - v_mean))
        var = np.mean((T_data - T_mean) ** 2)
        
        if abs(var) < 1e-15:
            return v_mean, 0.0
        
        slope = cov / var
        intercept = v_mean - slope * T_mean
        
        return intercept, slope
    
    def find_tg_tangent_intersection(self) -> Tuple[float, float, float, float]:
        if len(self.temperatures) < 6:
            raise ValueError("数据点不足，至少需要 6 个点")
        
        T = np.array(self.temperatures)
        v = np.array(self.specific_volumes)
        

        sort_idx = np.argsort(T)
        T = T[sort_idx]
        v = v[sort_idx]
        
        n = len(T)
        n_split = n // 2
        

        T_high = T[n_split:]
        v_high = v[n_split:]
        a_high, b_high = self.linear_fit(T_high, v_high)
        

        T_low = T[:n_split]
        v_low = v[:n_split]
        a_low, b_low = self.linear_fit(T_low, v_low)
        

        if abs(b_high - b_low) < 1e-15:
            Tg = np.mean(T)
        else:
            Tg = (a_low - a_high) / (b_high - b_low)
        
        v_g = a_high + b_high * Tg
        
        return Tg, v_g, b_high, b_low
    
    def find_tg_regula_falsi(self) -> Tuple[float, float]:
        Tg_est, v_g, alpha_r, alpha_g = self.find_tg_tangent_intersection()
        
        T = np.array(self.temperatures)
        
        def f(T_test):
            v_rubber = v_g + alpha_r * (T_test - Tg_est)
            v_glass = v_g + alpha_g * (T_test - Tg_est)
            return v_rubber - v_glass
        
        T_min = np.min(T)
        T_max = np.max(T)
        

        a, b = T_min, T_max
        fa, fb = f(a), f(b)
        

        if fa * fb > 0:

            a = T_min - 0.5 * (T_max - T_min)
            b = T_max + 0.5 * (T_max - T_min)
            fa, fb = f(a), f(b)
            if fa * fb > 0:
                return Tg_est, 0
        
        Tg, it = regula_falsi(f, a, b, tol=1e-4, max_iter=100)
        return Tg, it
    
    def vft_fit(
        self,
        viscosity_data: Optional[np.ndarray] = None,
    ) -> Tuple[float, float, float]:
        T = np.array(self.temperatures)
        
        if viscosity_data is not None:
            eta = np.array(viscosity_data)
        else:

            v = np.array(self.specific_volumes)
            eta = 1.0 / np.maximum(v, 0.1)
        

        mask = (T > 0.05) & (eta > 0)
        T = T[mask]
        eta = eta[mask]
        
        if len(T) < 3:
            return 1.0, 1.0, 0.05
        
        log_eta = np.log(eta)
        

        T_min = np.min(T) * 0.5
        T_max = np.min(T) * 0.95
        T0_values = np.linspace(T_min, T_max, 50)
        
        best_residual = float('inf')
        best_params = (1.0, 1.0, T_min)
        
        for T0 in T0_values:
            inv_T_shifted = 1.0 / (T - T0)
            

            a, b = self.linear_fit(inv_T_shifted, log_eta)
            
            predicted = a + b * inv_T_shifted
            residual = np.mean((log_eta - predicted) ** 2)
            
            if residual < best_residual:
                best_residual = residual
                best_params = (np.exp(a), b, T0)
        
        return best_params
    
    def fragility_index(self, A: float, B: float, T0: float, Tg: float) -> float:
        if Tg <= T0:
            return 100.0
        
        m = (B * Tg) / (np.log(10) * (Tg - T0) ** 2)
        return float(m)
    
    def configurational_entropy(self, T: float, Tg: float, Delta_Cp: float = 1.0) -> float:
        if T <= 0.01:
            return 0.0
        return Delta_Cp * np.log(T / Tg) if T > Tg else 0.0
    
    def get_summary(self) -> dict:
        if len(self.temperatures) < 6:
            return {"error": "数据点不足"}
        
        Tg, v_g, alpha_r, alpha_g = self.find_tg_tangent_intersection()
        Tg_rf, it_rf = self.find_tg_regula_falsi()
        A, B, T0 = self.vft_fit()
        m = self.fragility_index(A, B, T0, Tg)
        
        return {
            "Tg_tangent": float(Tg),
            "Tg_regula_falsi": float(Tg_rf),
            "specific_volume_at_Tg": float(v_g),
            "alpha_rubber": float(alpha_r),
            "alpha_glass": float(alpha_g),
            "VFT_A": float(A),
            "VFT_B": float(B),
            "VFT_T0": float(T0),
            "fragility_index_m": float(m),
            "data_points": len(self.temperatures),
        }
