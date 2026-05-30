
import numpy as np
from utils import safe_divide, robust_sqrt, check_finite_array


class ChebyshevNDInterpolation:
    
    def __init__(self, coefficients: np.ndarray, domains: list = None):
        self.coeffs = np.array(coefficients)
        self.ndim = self.coeffs.ndim
        
        if domains is None:
            self.domains = [(-1.0, 1.0)] * self.ndim
        else:
            self.domains = domains
    
    def _chebyshev_eval_1d(self, coeffs_1d: np.ndarray, x: float) -> float:
        n = len(coeffs_1d)
        if n < 1:
            return 0.0
        if n > 1000:
            raise ValueError("Too many coefficients")
        
        x = float(x)
        if x < -1.1 or x > 1.1:
            x = np.clip(x, -1.0, 1.0)
        
        b1 = 0.0
        b0 = 0.0
        
        for i in range(n - 1, -1, -1):
            b2 = b1
            b1 = b0
            b0 = 2.0 * x * b1 - b2 + coeffs_1d[i]
        
        return 0.5 * (b0 - b2)
    
    def _map_to_standard(self, x: float, dim: int) -> float:
        xmin, xmax = self.domains[dim]
        if abs(xmax - xmin) < 1e-14:
            return 0.0
        return 2.0 * (x - xmin) / (xmax - xmin) - 1.0
    
    def evaluate(self, x: np.ndarray) -> float:
        if len(x) != self.ndim:
            raise ValueError(f"Expected {self.ndim} dimensions, got {len(x)}")
        

        return self._evaluate_recursive(self.coeffs, x, 0)
    
    def _evaluate_recursive(self, coeffs: np.ndarray, x: np.ndarray, dim: int) -> float:
        if dim == self.ndim - 1:

            x_std = self._map_to_standard(x[dim], dim)
            return self._chebyshev_eval_1d(coeffs, x_std)
        

        result = 0.0
        x_std = self._map_to_standard(x[dim], dim)
        
        for k in range(coeffs.shape[0]):
            Tk = self._chebyshev_polynomial(k, x_std)
            result += Tk * self._evaluate_recursive(coeffs[k], x, dim + 1)
        
        return result
    
    def _chebyshev_polynomial(self, n: int, x: float) -> float:
        if n == 0:
            return 1.0
        if n == 1:
            return x
        
        T_prev2 = 1.0
        T_prev1 = x
        T_n = 0.0
        
        for k in range(2, n + 1):
            T_n = 2.0 * x * T_prev1 - T_prev2
            T_prev2 = T_prev1
            T_prev1 = T_n
        
        return T_n


class LebesgueStabilityAnalyzer:
    
    def __init__(self, interpolation_points: np.ndarray):
        self.x = np.array(interpolation_points)
        self.n = len(self.x)
        self.x = np.sort(self.x)
    
    def lagrange_basis(self, j: int, x_eval: float) -> float:
        result = 1.0
        for k in range(self.n):
            if k == j:
                continue
            denom = self.x[j] - self.x[k]
            if abs(denom) < 1e-14:
                return 0.0
            result *= (x_eval - self.x[k]) / denom
        return result
    
    def lebesgue_function(self, x_eval_points: np.ndarray) -> np.ndarray:
        x_eval = np.array(x_eval_points)
        L = np.zeros(len(x_eval))
        
        for i, xi in enumerate(x_eval):
            l_sum = 0.0
            for j in range(self.n):
                l_sum += abs(self.lagrange_basis(j, xi))
            L[i] = l_sum
        
        return L
    
    def lebesgue_constant(self, n_eval: int = 1000) -> float:

        x_min, x_max = self.x[0], self.x[-1]
        x_eval = np.linspace(x_min, x_max, n_eval)
        L = self.lebesgue_function(x_eval)
        return float(np.max(L))
    
    def chebyshev_nodes(self, n: int, a: float = -1.0, b: float = 1.0) -> np.ndarray:
        j = np.arange(n)
        nodes = np.cos((2.0 * j + 1.0) * np.pi / (2.0 * n))
        nodes = 0.5 * (a + b) + 0.5 * (b - a) * nodes
        return nodes


class FlameTransferFunction:
    
    def __init__(self,
                 interaction_index: float = 1.0,
                 time_delay_ms: float = 2.0,
                 chemical_time_ms: float = 0.5,
                 cutoff_frequency_hz: float = 1000.0):
        
        self.n = interaction_index
        self.tau = time_delay_ms * 1e-3
        self.tau_c = chemical_time_ms * 1e-3
        self.f_c = cutoff_frequency_hz
        

        self.freq_data = None
        self.ftf_data = None
    
    def analytical_ftf(self, frequency_hz: float) -> complex:
        f = float(frequency_hz)
        omega = 2.0 * np.pi * f
        

        delay = np.exp(-1j * omega * self.tau)
        

        if f < 1e-10:
            lowpass = 1.0
        else:
            lowpass = 1.0 / (1.0 + 1j * f / self.f_c)
        
        return self.n * delay * lowpass
    
    def generate_discrete_data(self,
                               freq_range_hz: tuple = (10.0, 5000.0),
                               n_points: int = 50) -> dict:
        stability = LebesgueStabilityAnalyzer(np.linspace(0, 1, n_points))
        

        f_cheb = stability.chebyshev_nodes(n_points, freq_range_hz[0], freq_range_hz[1])
        
        ftf_vals = np.array([self.analytical_ftf(f) for f in f_cheb])
        
        self.freq_data = f_cheb
        self.ftf_data = ftf_vals
        
        return {
            "frequencies": f_cheb,
            "ftf_real": np.real(ftf_vals),
            "ftf_imag": np.imag(ftf_vals),
            "ftf_magnitude": np.abs(ftf_vals),
            "ftf_phase": np.angle(ftf_vals)
        }
    
    def interpolate_ftf(self, frequency_hz: float) -> complex:
        if self.freq_data is None:
            self.generate_discrete_data()
        
        from combustion_wave import NewtonInterpolation
        

        interp_real = NewtonInterpolation(self.freq_data, np.real(self.ftf_data))

        interp_imag = NewtonInterpolation(self.freq_data, np.imag(self.ftf_data))
        
        real_part = interp_real.evaluate(frequency_hz)
        imag_part = interp_imag.evaluate(frequency_hz)
        
        return complex(real_part, imag_part)
    
    def nyquist_plot_data(self, n_points: int = 200) -> tuple:
        f = np.logspace(1, 4, n_points)
        ftf = np.array([self.analytical_ftf(fi) for fi in f])
        
        return np.real(ftf), np.imag(ftf)
    
    def compute_nyquist_stability_margin(self) -> dict:
        f = np.logspace(1, 4, 1000)
        ftf = np.array([self.analytical_ftf(fi) for fi in f])
        
        magnitude = np.abs(ftf)
        phase = np.angle(ftf)
        

        phase_cross_idx = np.where(np.diff(np.sign(phase + np.pi)))[0]
        gain_margin = np.inf
        if len(phase_cross_idx) > 0:
            idx = phase_cross_idx[0]
            mag_at_cross = magnitude[idx]
            if mag_at_cross > 0:
                gain_margin = 1.0 / mag_at_cross
        

        unity_idx = np.where(np.diff(np.sign(magnitude - 1.0)))[0]
        phase_margin = np.inf
        if len(unity_idx) > 0:
            idx = unity_idx[0]
            phase_at_unity = phase[idx]
            phase_margin = np.pi + phase_at_unity
        
        return {
            "gain_margin_db": 20.0 * np.log10(gain_margin) if gain_margin != np.inf else np.inf,
            "phase_margin_deg": np.degrees(phase_margin) if phase_margin != np.inf else np.inf,
            "critical_frequency_hz": f[phase_cross_idx[0]] if len(phase_cross_idx) > 0 else None
        }


if __name__ == "__main__":

    coeffs_2d = np.array([
        [1.0, 0.5, -0.2],
        [0.3, -0.1, 0.05],
        [-0.1, 0.02, 0.01]
    ])
    cheb = ChebyshevNDInterpolation(coeffs_2d, domains=[(0, 1), (0, 1)])
    val = cheb.evaluate(np.array([0.5, 0.5]))
    print(f"2D Chebyshev at (0.5, 0.5): {val:.6f}")
    

    equidistant = np.linspace(-1, 1, 10)
    leb_eq = LebesgueStabilityAnalyzer(equidistant)
    lambda_eq = leb_eq.lebesgue_constant()
    print(f"Lebesgue constant (equidistant, n=10): {lambda_eq:.4f}")
    
    cheb_nodes = leb_eq.chebyshev_nodes(10, -1, 1)
    leb_cheb = LebesgueStabilityAnalyzer(cheb_nodes)
    lambda_cheb = leb_cheb.lebesgue_constant()
    print(f"Lebesgue constant (Chebyshev, n=10): {lambda_cheb:.4f}")
    

    ftf = FlameTransferFunction()
    data = ftf.generate_discrete_data()
    print(f"\nFTF at 500 Hz: |F|={np.abs(ftf.analytical_ftf(500)):.4f}, "
          f"arg={np.degrees(np.angle(ftf.analytical_ftf(500))):.2f}°")
    
    stability = ftf.compute_nyquist_stability_margin()
    print(f"Gain margin: {stability['gain_margin_db']:.2f} dB")
    print(f"Phase margin: {stability['phase_margin_deg']:.2f}°")
