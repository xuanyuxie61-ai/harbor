"""
flame_response.py - 火焰传递函数与多维插值
==========================================
基于多维插值的火焰响应函数计算与插值稳定性分析。

原项目映射:
- 1214_test_interp_nd -> n维Chebyshev插值用于火焰传递函数
- 658_lebesgue        -> Lebesgue常数用于插值稳定性评估

科学背景:
=========
火焰传递函数 (Flame Transfer Function, FTF) 描述火焰对
速度/压力扰动的响应，是预测燃烧不稳定性的关键:

    F(ω) = q̃(ω) / ũ(ω)

其中 q̃ 为热释放率脉动, ũ 为入口速度脉动。

对于非预混燃烧 (火箭发动机典型工况):
    F(ω) = n·exp(-i·ω·τ_d) · I(ω·τ_c)

其中:
    n: 相互作用指数 (gain)
    τ_d: 对流延迟时间
    τ_c: 化学反应时间
    I(x): 低通滤波函数

本模块提供:
1. 火焰传递函数的解析模型
2. 基于多维Chebyshev插值的离散数据FTF重建
3. 插值稳定性评估 (Lebesgue常数)
"""

import numpy as np
from utils import safe_divide, robust_sqrt, check_finite_array


class ChebyshevNDInterpolation:
    """
    n维Chebyshev级数插值。
    
    原项目映射: 1214_test_interp_nd / csevl.m
    
    Chebyshev级数展开:
        f(x) ≈ Σ_{k=0}^{N-1} c_k · T_k(x)
    
    其中 T_k(x) 为第一类Chebyshev多项式:
        T_0(x) = 1
        T_1(x) = x
        T_{k+1}(x) = 2x·T_k(x) - T_{k-1}(x)
    
    Clenshaw递推求值:
        b_N = b_{N+1} = 0
        b_k = 2x·b_{k+1} - b_{k+2} + c_k
        f(x) = 0.5·(b_0 - b_2)
    
    多维推广:
        f(x_1,...,x_d) = Σ_{k_1}...Σ_{k_d} c_{k_1,...,k_d} · T_{k_1}(x_1)...T_{k_d}(x_d)
    """
    
    def __init__(self, coefficients: np.ndarray, domains: list = None):
        """
        参数:
            coefficients: Chebyshev系数数组 (d维)
            domains: 每个维度的定义域 [(xmin, xmax), ...]
        """
        self.coeffs = np.array(coefficients)
        self.ndim = self.coeffs.ndim
        
        if domains is None:
            self.domains = [(-1.0, 1.0)] * self.ndim
        else:
            self.domains = domains
    
    def _chebyshev_eval_1d(self, coeffs_1d: np.ndarray, x: float) -> float:
        """
        一维Chebyshev级数Clenshaw求值。
        
        原项目映射: csevl.m
        """
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
        """将物理坐标映射到标准区间[-1,1]。"""
        xmin, xmax = self.domains[dim]
        if abs(xmax - xmin) < 1e-14:
            return 0.0
        return 2.0 * (x - xmin) / (xmax - xmin) - 1.0
    
    def evaluate(self, x: np.ndarray) -> float:
        """
        在点x处求值多维Chebyshev级数。
        
        参数:
            x: 长度为ndim的数组
        """
        if len(x) != self.ndim:
            raise ValueError(f"Expected {self.ndim} dimensions, got {len(x)}")
        
        # 递归降维求值
        return self._evaluate_recursive(self.coeffs, x, 0)
    
    def _evaluate_recursive(self, coeffs: np.ndarray, x: np.ndarray, dim: int) -> float:
        """递归计算多维Chebyshev级数。"""
        if dim == self.ndim - 1:
            # 最后一维: 一维Chebyshev求值
            x_std = self._map_to_standard(x[dim], dim)
            return self._chebyshev_eval_1d(coeffs, x_std)
        
        # 对当前维度求和
        result = 0.0
        x_std = self._map_to_standard(x[dim], dim)
        
        for k in range(coeffs.shape[0]):
            Tk = self._chebyshev_polynomial(k, x_std)
            result += Tk * self._evaluate_recursive(coeffs[k], x, dim + 1)
        
        return result
    
    def _chebyshev_polynomial(self, n: int, x: float) -> float:
        """计算第n个Chebyshev多项式 T_n(x)。"""
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
    """
    插值稳定性分析器 (Lebesgue常数计算)。
    
    原项目映射: 658_lebesgue / lebesgue_constant.m
    
    Lebesgue函数:
        L(x) = Σ_{j=1}^{n} |l_j(x)|
    
    其中 l_j(x) 为Lagrange基函数:
        l_j(x) = Π_{k≠j} (x - x_k) / (x_j - x_k)
    
    Lebesgue常数:
        Λ = max_x L(x)
    
    插值误差界:
        ||f - P_n||_∞ ≤ (1 + Λ_n) · E_n(f)
    
    其中 E_n(f) 为最佳逼近误差。
    
    对于等距节点: Λ_n ~ 2^n/(n·ln(n))  (指数增长, 不稳定)
    对于Chebyshev节点: Λ_n ~ (2/π)·ln(n)  (对数增长, 稳定)
    """
    
    def __init__(self, interpolation_points: np.ndarray):
        self.x = np.array(interpolation_points)
        self.n = len(self.x)
        self.x = np.sort(self.x)
    
    def lagrange_basis(self, j: int, x_eval: float) -> float:
        """
        计算第j个Lagrange基函数在x_eval处的值。
        
        l_j(x) = Π_{k≠j} (x - x_k) / (x_j - x_k)
        """
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
        """
        计算Lebesgue函数在多个点的值。
        
        L(x) = Σ_{j=1}^{n} |l_j(x)|
        """
        x_eval = np.array(x_eval_points)
        L = np.zeros(len(x_eval))
        
        for i, xi in enumerate(x_eval):
            l_sum = 0.0
            for j in range(self.n):
                l_sum += abs(self.lagrange_basis(j, xi))
            L[i] = l_sum
        
        return L
    
    def lebesgue_constant(self, n_eval: int = 1000) -> float:
        """
        估计Lebesgue常数 (Lebesgue函数的最大值)。
        
        在插值节点之间的中点附近通常达到最大值。
        """
        # 在细网格上求最大值
        x_min, x_max = self.x[0], self.x[-1]
        x_eval = np.linspace(x_min, x_max, n_eval)
        L = self.lebesgue_function(x_eval)
        return float(np.max(L))
    
    def chebyshev_nodes(self, n: int, a: float = -1.0, b: float = 1.0) -> np.ndarray:
        """
        生成Chebyshev插值节点 (第一类)。
        
        x_j = cos((2j+1)π/(2n)), j=0,...,n-1
        
        映射到区间[a,b]:
            x_j = (a+b)/2 + (b-a)/2 · cos((2j+1)π/(2n))
        """
        j = np.arange(n)
        nodes = np.cos((2.0 * j + 1.0) * np.pi / (2.0 * n))
        nodes = 0.5 * (a + b) + 0.5 * (b - a) * nodes
        return nodes


class FlameTransferFunction:
    """
    火焰传递函数 (FTF) 模型。
    
    描述火焰对速度扰动的响应:
        F(ω) = n · exp(-i·ω·τ) · H(ω·τ_c)
    
    其中:
        n: 相互作用指数 (典型值0.5-2.0)
        τ: 对流延迟 (典型值1-5 ms)
        H(x): 低通滤波函数
    """
    
    def __init__(self,
                 interaction_index: float = 1.0,
                 time_delay_ms: float = 2.0,
                 chemical_time_ms: float = 0.5,
                 cutoff_frequency_hz: float = 1000.0):
        
        self.n = interaction_index
        self.tau = time_delay_ms * 1e-3
        self.tau_c = chemical_time_ms * 1e-3
        self.f_c = cutoff_frequency_hz
        
        # 构建离散频率点的FTF数据
        self.freq_data = None
        self.ftf_data = None
    
    def analytical_ftf(self, frequency_hz: float) -> complex:
        """
        计算解析火焰传递函数。
        
        F(f) = n · exp(-i·2π·f·τ) / (1 + i·f/f_c)
        
        其中:
            - exp(-i·2π·f·τ): 纯延迟项
            - 1/(1 + i·f/f_c): 一阶低通滤波
        """
        f = float(frequency_hz)
        omega = 2.0 * np.pi * f
        
        # 延迟项
        delay = np.exp(-1j * omega * self.tau)
        
        # 低通滤波
        if f < 1e-10:
            lowpass = 1.0
        else:
            lowpass = 1.0 / (1.0 + 1j * f / self.f_c)
        
        return self.n * delay * lowpass
    
    def generate_discrete_data(self,
                               freq_range_hz: tuple = (10.0, 5000.0),
                               n_points: int = 50) -> dict:
        """
        生成离散频率点的FTF数据。
        
        使用Chebyshev节点分布以获得最佳插值稳定性。
        """
        stability = LebesgueStabilityAnalyzer(np.linspace(0, 1, n_points))
        
        # 在频率域使用Chebyshev节点
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
        """
        使用Newton插值重建FTF。
        
        对实部和虚部分别进行1D插值。
        """
        if self.freq_data is None:
            self.generate_discrete_data()
        
        from combustion_wave import NewtonInterpolation
        
        # 实部插值
        interp_real = NewtonInterpolation(self.freq_data, np.real(self.ftf_data))
        # 虚部插值
        interp_imag = NewtonInterpolation(self.freq_data, np.imag(self.ftf_data))
        
        real_part = interp_real.evaluate(frequency_hz)
        imag_part = interp_imag.evaluate(frequency_hz)
        
        return complex(real_part, imag_part)
    
    def nyquist_plot_data(self, n_points: int = 200) -> tuple:
        """
        生成Nyquist图数据 (Re[F], Im[F])。
        
        Nyquist稳定性判据:
            若F(ω)的轨迹包围点(-1, 0)，则系统不稳定。
        """
        f = np.logspace(1, 4, n_points)
        ftf = np.array([self.analytical_ftf(fi) for fi in f])
        
        return np.real(ftf), np.imag(ftf)
    
    def compute_nyquist_stability_margin(self) -> dict:
        """
        计算Nyquist稳定性裕度。
        
        增益裕度: 当相位达到-π时的增益倒数
        相位裕度: 当增益为1时的相位与-π的距离
        """
        f = np.logspace(1, 4, 1000)
        ftf = np.array([self.analytical_ftf(fi) for fi in f])
        
        magnitude = np.abs(ftf)
        phase = np.angle(ftf)
        
        # 增益裕度
        phase_cross_idx = np.where(np.diff(np.sign(phase + np.pi)))[0]
        gain_margin = np.inf
        if len(phase_cross_idx) > 0:
            idx = phase_cross_idx[0]
            mag_at_cross = magnitude[idx]
            if mag_at_cross > 0:
                gain_margin = 1.0 / mag_at_cross
        
        # 相位裕度
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
    # 测试Chebyshev插值
    coeffs_2d = np.array([
        [1.0, 0.5, -0.2],
        [0.3, -0.1, 0.05],
        [-0.1, 0.02, 0.01]
    ])
    cheb = ChebyshevNDInterpolation(coeffs_2d, domains=[(0, 1), (0, 1)])
    val = cheb.evaluate(np.array([0.5, 0.5]))
    print(f"2D Chebyshev at (0.5, 0.5): {val:.6f}")
    
    # 测试Lebesgue常数
    equidistant = np.linspace(-1, 1, 10)
    leb_eq = LebesgueStabilityAnalyzer(equidistant)
    lambda_eq = leb_eq.lebesgue_constant()
    print(f"Lebesgue constant (equidistant, n=10): {lambda_eq:.4f}")
    
    cheb_nodes = leb_eq.chebyshev_nodes(10, -1, 1)
    leb_cheb = LebesgueStabilityAnalyzer(cheb_nodes)
    lambda_cheb = leb_cheb.lebesgue_constant()
    print(f"Lebesgue constant (Chebyshev, n=10): {lambda_cheb:.4f}")
    
    # 测试火焰传递函数
    ftf = FlameTransferFunction()
    data = ftf.generate_discrete_data()
    print(f"\nFTF at 500 Hz: |F|={np.abs(ftf.analytical_ftf(500)):.4f}, "
          f"arg={np.degrees(np.angle(ftf.analytical_ftf(500))):.2f}°")
    
    stability = ftf.compute_nyquist_stability_margin()
    print(f"Gain margin: {stability['gain_margin_db']:.2f} dB")
    print(f"Phase margin: {stability['phase_margin_deg']:.2f}°")
