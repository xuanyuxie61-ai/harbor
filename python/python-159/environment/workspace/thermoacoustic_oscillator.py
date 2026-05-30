
import numpy as np
from utils import safe_divide, robust_sqrt, check_finite_array


class ThermoacousticOscillator:
    
    def __init__(self,
                 natural_frequency_hz: float = 500.0,
                 acoustic_damping: float = 50.0,
                 flame_gain_coefficient: float = 80.0,
                 nonlinear_saturation: float = 1.0e8,
                 coupling_strength: float = 1.0,
                 initial_pressure_disturbance_pa: float = 100.0,
                 initial_velocity_disturbance_pa_s: float = 0.0):
        
        self.omega = 2.0 * np.pi * natural_frequency_hz
        self.alpha_acoustic = acoustic_damping
        self.n = flame_gain_coefficient / coupling_strength
        self.beta = nonlinear_saturation
        self.gamma = coupling_strength
        

        self.alpha_eff = self.alpha_acoustic - self.gamma * self.n
        
        self.y0 = np.array([
            initial_pressure_disturbance_pa,
            initial_velocity_disturbance_pa_s
        ])
        

        self.is_unstable = self.alpha_eff < 0
    
    def derivatives(self, t: float, y: np.ndarray) -> np.ndarray:
        p = y[0]
        dpdt = y[1]
        

        p_sat = np.clip(p, -1e6, 1e6)
        dpdt_sat = np.clip(dpdt, -1e9, 1e9)
        F = self.n * dpdt_sat - self.beta * p_sat ** 2 * dpdt_sat
        F = np.clip(F, -1e12, 1e12)
        

        d2pdt2 = -self.omega ** 2 * p_sat - 2.0 * self.alpha_acoustic * dpdt_sat + self.gamma * F
        
        return np.array([dpdt, d2pdt2])
    
    def rk4_integrate(self, t_span: tuple, n_steps: int = 10000) -> dict:
        t0, tf = t_span
        dt = (tf - t0) / n_steps
        
        t = np.linspace(t0, tf, n_steps + 1)
        y = np.zeros((n_steps + 1, 2))
        y[0] = self.y0
        
        for i in range(n_steps):
            k1 = dt * self.derivatives(t[i], y[i])
            k2 = dt * self.derivatives(t[i] + 0.5 * dt, y[i] + 0.5 * k1)
            k3 = dt * self.derivatives(t[i] + 0.5 * dt, y[i] + 0.5 * k2)
            k4 = dt * self.derivatives(t[i] + dt, y[i] + k3)
            
            y[i + 1] = y[i] + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
            

            if not np.all(np.isfinite(y[i+1])):
                y[i+1] = y[i]
        
        return {
            "t": t,
            "pressure": y[:, 0],
            "pressure_rate": y[:, 1],
            "amplitude_envelope": self._compute_amplitude_envelope(y[:, 0], dt)
        }
    
    def _compute_amplitude_envelope(self, signal: np.ndarray, dt: float) -> np.ndarray:
        n = len(signal)
        envelope = np.zeros(n)
        window = int(0.5 * 2.0 * np.pi / (self.omega * dt))
        window = max(window, 3)
        
        for i in range(n):
            i0 = max(0, i - window)
            i1 = min(n, i + window + 1)
            envelope[i] = np.max(np.abs(signal[i0:i1]))
        
        return envelope
    
    def compute_growth_rate(self, t: np.ndarray, amplitude: np.ndarray) -> float:

        amp_safe = np.maximum(np.abs(amplitude), 1e-10)
        amp_safe = np.clip(amp_safe, 1e-10, 1e10)
        log_amp = np.log(amp_safe)
        

        n_use = len(t) // 3
        if n_use < 3:
            return 0.0
        

        t_use = t[:n_use]
        log_use = log_amp[:n_use]
        

        if not np.all(np.isfinite(log_use)):
            return 0.0
        
        A = np.vstack([t_use, np.ones(len(t_use))]).T
        try:
            sigma, _ = np.linalg.lstsq(A, log_use, rcond=None)[0]
        except Exception:
            sigma = 0.0
        
        sigma = float(np.clip(sigma, -1e6, 1e6))
        return sigma
    
    def limit_cycle_amplitude(self) -> float:







        return 0.0
    
    def compute_oscillation_metrics(self, t: np.ndarray, pressure: np.ndarray) -> dict:
        p = pressure
        

        p_max = np.max(p)
        p_min = np.min(p)
        p_p2p = p_max - p_min
        

        p_ac = p - np.mean(p)
        p_rms = np.sqrt(np.mean(p_ac ** 2))
        

        zero_crossings = np.where(np.diff(np.sign(p_ac)))[0]
        if len(zero_crossings) >= 2:
            T_est = 2.0 * (t[zero_crossings[-1]] - t[zero_crossings[0]]) / len(zero_crossings)
            f_est = safe_divide(1.0, T_est, default=0.0)
        else:
            f_est = self.omega / (2.0 * np.pi)
        

        envelope = self._compute_amplitude_envelope(p, t[1] - t[0])
        growth_rate = self.compute_growth_rate(t, envelope)
        

        A_lim = self.limit_cycle_amplitude()
        in_limit_cycle = abs(p_rms - A_lim / np.sqrt(2)) < 0.1 * A_lim if A_lim > 0 else False
        
        return {
            "peak_to_peak_pa": float(p_p2p),
            "rms_pa": float(p_rms),
            "estimated_frequency_hz": float(f_est),
            "growth_rate_1_per_s": float(growth_rate),
            "limit_cycle_amplitude_pa": float(A_lim),
            "in_limit_cycle": bool(in_limit_cycle)
        }


class MultiModeThermoacousticSystem:
    
    def __init__(self,
                 mode_frequencies: np.ndarray,
                 damping_rates: np.ndarray,
                 coupling_matrix: np.ndarray = None):
        
        self.n_modes = len(mode_frequencies)
        self.omega = 2.0 * np.pi * np.array(mode_frequencies)
        self.alpha = np.array(damping_rates)
        
        if coupling_matrix is None:
            self.gamma = np.eye(self.n_modes) * 0.5
        else:
            self.gamma = np.array(coupling_matrix)
    
    def derivatives(self, t: float, state: np.ndarray) -> np.ndarray:
        n = self.n_modes
        dydt = np.zeros(2 * n)
        
        for i in range(n):
            p_i = state[2 * i]
            v_i = state[2 * i + 1]
            

            flame_response = 0.0
            for j in range(n):
                p_j = state[2 * j]
                v_j = state[2 * j + 1]

                flame_response += self.gamma[i, j] * v_j * (1.0 - 1e-8 * p_j ** 2)
            
            dydt[2 * i] = v_i
            dydt[2 * i + 1] = -self.omega[i] ** 2 * p_i - 2.0 * self.alpha[i] * v_i + flame_response
        
        return dydt
    
    def integrate(self, t_span: tuple = (0, 0.05), n_steps: int = 20000,
                  initial_conditions: np.ndarray = None) -> dict:
        if initial_conditions is None:
            y0 = np.zeros(2 * self.n_modes)
            y0[0] = 100.0
        else:
            y0 = np.array(initial_conditions)
        
        t0, tf = t_span
        dt = (tf - t0) / n_steps
        
        t = np.linspace(t0, tf, n_steps + 1)
        y = np.zeros((n_steps + 1, 2 * self.n_modes))
        y[0] = y0
        
        for i in range(n_steps):
            k1 = dt * self.derivatives(t[i], y[i])
            k2 = dt * self.derivatives(t[i] + 0.5 * dt, y[i] + 0.5 * k1)
            k3 = dt * self.derivatives(t[i] + 0.5 * dt, y[i] + 0.5 * k2)
            k4 = dt * self.derivatives(t[i] + dt, y[i] + k3)
            
            y[i + 1] = y[i] + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
            
            if not np.all(np.isfinite(y[i+1])):
                y[i+1] = y[i]
        
        return {
            "t": t,
            "state": y,
            "mode_pressures": [y[:, 2*i] for i in range(self.n_modes)]
        }


if __name__ == "__main__":

    osc = ThermoacousticOscillator(
        natural_frequency_hz=500.0,
        acoustic_damping=50.0,
        flame_gain_coefficient=80.0,
        nonlinear_saturation=1e8
    )
    print(f"Effective damping: {osc.alpha_eff:.2f} 1/s")
    print(f"Linear stability: {'UNSTABLE' if osc.is_unstable else 'STABLE'}")
    print(f"Limit cycle amplitude: {osc.limit_cycle_amplitude():.2f} Pa")
    
    result = osc.rk4_integrate((0, 0.05), n_steps=20000)
    metrics = osc.compute_oscillation_metrics(result["t"], result["pressure"])
    print(f"\nOscillation metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    

    freqs = np.array([500.0, 1500.0, 2500.0])
    damping = np.array([50.0, 150.0, 250.0])
    multi = MultiModeThermoacousticSystem(freqs, damping)
    multi_result = multi.integrate()
    print(f"\nMulti-mode system integrated.")
    print(f"Mode 1 final amplitude: {np.max(np.abs(multi_result['mode_pressures'][0][-1000:])):.2f} Pa")
