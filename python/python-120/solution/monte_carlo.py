"""
monte_carlo.py
蒙特卡洛采样与数值积分模块

整合原项目:
  - 711_mandelbrot_area: 蒙特卡洛面积估计与收敛性分析
  - 944_quad_serial: 复合求积公式
  - 929_pwl_product_integral: 分段线性函数乘积积分

科学背景:
  表面催化反应中，以下物理量需要通过数值积分/采样计算:
  
  1. 吸附截面 (Adsorption Cross Section):
     σ_ads = ∫_Ω P_ads(E, θ, φ) dΩ
  
  2. 速率常数的热平均:
     k(T) = ∫_0^∞ σ(E) * v(E) * f_MB(E) dE
  
  3. 配分函数:
     Q = ∫ exp(-V(r) / (k_B T)) dr
  
  4. 反应路径上的隧道修正:
     κ_tunnel = ∫ exp(-2/ℏ ∫_{a(E)}^{b(E)} √(2m(V(x)-E)) dx) dE
"""

import numpy as np
from typing import Callable, Tuple, Optional


class MonteCarloSampler:
    """
    蒙特卡洛采样器
    
    整合原项目 711_mandelbrot_area 的收敛性测试思想
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def sample_uniform_box(self, n_samples: int, bounds: np.ndarray) -> np.ndarray:
        """
        在超矩形区域内均匀采样
        
        bounds: (ndim, 2) 每维的 [min, max]
        """
        bounds = np.asarray(bounds, dtype=float)
        ndim = bounds.shape[0]
        samples = np.zeros((n_samples, ndim))
        for d in range(ndim):
            samples[:, d] = self.rng.uniform(bounds[d, 0], bounds[d, 1], n_samples)
        return samples

    def estimate_integral(self, func: Callable[[np.ndarray], np.ndarray],
                          bounds: np.ndarray, n_samples: int) -> Tuple[float, float]:
        """
        蒙特卡洛积分估计
        
        公式:
          I ≈ V * (1/N) * Σ f(x_i)
          σ_I = V * sqrt(Var(f) / N)
        
        其中 V = Π (b_d - a_d) 为积分区域体积
        """
        bounds = np.asarray(bounds, dtype=float)
        volume = np.prod(bounds[:, 1] - bounds[:, 0])
        samples = self.sample_uniform_box(n_samples, bounds)
        f_vals = func(samples)
        mean_f = np.mean(f_vals)
        var_f = np.var(f_vals, ddof=1) if n_samples > 1 else 0.0
        integral = volume * mean_f
        error = volume * np.sqrt(var_f / n_samples)
        return float(integral), float(error)

    def adsorption_probability_monte_carlo(self,
                                            energy_func: Callable[[np.ndarray], np.ndarray],
                                            n_samples: int = 100000,
                                            bounds: Optional[np.ndarray] = None,
                                            temperature_k: float = 500.0) -> float:
        """
        使用蒙特卡洛方法计算热活化吸附概率
        
        公式:
          P_ads = ⟨exp(-ΔE / (k_B T))⟩
        
        其中 ΔE = max(0, E - E_threshold)
        """
        from utils import BOLTZMANN_KB
        if bounds is None:
            bounds = np.array([[-2e-10, 2e-10],
                               [-2e-10, 2e-10],
                               [0.5e-10, 4e-10]])
        samples = self.sample_uniform_box(n_samples, bounds)
        energies = energy_func(samples)
        e_threshold = np.min(energies) + 0.1  # 吸附阈值
        delta_e = np.maximum(0.0, energies - e_threshold)
        kb_t = BOLTZMANN_KB * temperature_k / 1.602176634e-19  # eV
        probs = np.exp(-delta_e / kb_t)
        return float(np.mean(probs))

    def convergence_test(self, func: Callable[[np.ndarray], np.ndarray],
                         bounds: np.ndarray,
                         sample_sizes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        蒙特卡洛积分收敛性测试
        
        整合原项目 711_mandelbrot_area 思想:
        在不同采样点数下计算积分，分析误差随 N^{-1/2} 衰减
        """
        estimates = np.zeros(len(sample_sizes))
        errors = np.zeros(len(sample_sizes))
        for idx, n in enumerate(sample_sizes):
            estimates[idx], errors[idx] = self.estimate_integral(func, bounds, n)
        return estimates, errors


class QuadratureIntegrator:
    """
    数值求积模块
    
    整合原项目 944_quad_serial:
    复合梯形/ Simpson 求积公式
    """

    @staticmethod
    def composite_trapezoidal(f: Callable[[np.ndarray], np.ndarray],
                              a: float, b: float, n: int) -> float:
        """
        复合梯形公式
        
        公式:
          ∫_a^b f(x) dx ≈ (b-a)/n * [0.5 f(a) + Σ_{i=1}^{n-1} f(x_i) + 0.5 f(b)]
        """
        if n < 1:
            raise ValueError("n >= 1")
        if b <= a:
            raise ValueError("b > a")
        h = (b - a) / n
        x = np.linspace(a, b, n + 1)
        y = f(x)
        return h * (0.5 * y[0] + np.sum(y[1:-1]) + 0.5 * y[-1])

    @staticmethod
    def composite_simpson(f: Callable[[np.ndarray], np.ndarray],
                          a: float, b: float, n: int) -> float:
        """
        复合 Simpson 公式 (n 必须为偶数)
        
        公式:
          ∫_a^b f(x) dx ≈ h/3 * [f_0 + 4 Σ f_{odd} + 2 Σ f_{even} + f_n]
        """
        if n % 2 != 0:
            n += 1
        if n < 2:
            raise ValueError("n >= 2")
        h = (b - a) / n
        x = np.linspace(a, b, n + 1)
        y = f(x)
        result = h / 3.0 * (y[0] + y[-1]
                           + 4.0 * np.sum(y[1:-1:2])
                           + 2.0 * np.sum(y[2:-1:2]))
        return float(result)

    @staticmethod
    def gauss_legendre_3point(f: Callable[[np.ndarray], np.ndarray],
                              a: float, b: float) -> float:
        """
        三点 Gauss-Legendre 求积 (代数精度 5)
        
        节点和权重:
          x_i = ±√(3/5), 0
          w_i = 5/9, 8/9, 5/9
        """
        nodes = np.array([-np.sqrt(3.0 / 5.0), 0.0, np.sqrt(3.0 / 5.0)])
        weights = np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0])
        # 变换到 [a, b]
        x_mapped = 0.5 * (b - a) * nodes + 0.5 * (a + b)
        jac = 0.5 * (b - a)
        return jac * np.sum(weights * f(x_mapped))


class PiecewiseLinearProductIntegral:
    """
    分段线性函数乘积积分
    
    整合原项目 929_pwl_product_integral:
    精确计算 ∫_a^b f(x) g(x) dx，其中 f 和 g 均为分段线性函数
    
    算法:
      1. 确定 f 和 g 的所有节点，合并为统一断点集
      2. 在每个子区间 [x_k, x_{k+1}] 上，f 和 g 均为线性
      3. 乘积 h(x) = f(x)g(x) 为二次多项式
      4. 解析积分: ∫_{x_k}^{x_{k+1}} h(x) dx
    """

    def integrate(self, f_x: np.ndarray, f_v: np.ndarray,
                  g_x: np.ndarray, g_v: np.ndarray,
                  a: float, b: float) -> float:
        """
        计算 ∫_a^b f(x) g(x) dx
        
        参数:
          f_x, f_v: f 的节点坐标和值
          g_x, g_v: g 的节点坐标和值
          a, b: 积分上下限
        """
        if len(f_x) < 2 or len(g_x) < 2:
            return 0.0
        if b <= a:
            return 0.0

        # 限制在有效范围内
        a_eff = max(a, f_x[0], g_x[0])
        b_eff = min(b, f_x[-1], g_x[-1])
        if b_eff <= a_eff:
            return 0.0

        # 合并断点
        all_breaks = np.sort(np.unique(np.concatenate([
            f_x[(f_x >= a_eff) & (f_x <= b_eff)],
            g_x[(g_x >= a_eff) & (g_x <= b_eff)]
        ])))
        if len(all_breaks) < 2:
            return 0.0

        total = 0.0
        for k in range(len(all_breaks) - 1):
            xl = all_breaks[k]
            xr = all_breaks[k + 1]
            if xr - xl < 1e-15:
                continue

            # f 在 [xl, xr] 上的线性插值
            fl = self._interp_linear(f_x, f_v, xl)
            fr = self._interp_linear(f_x, f_v, xr)
            # g 在 [xl, xr] 上的线性插值
            gl = self._interp_linear(g_x, g_v, xl)
            gr = self._interp_linear(g_x, g_v, xr)

            # f(x) = (fl * xr - fr * xl) / (xr - xl) + (fr - fl) / (xr - xl) * x
            # 写成 f(x) = α_f + β_f * x
            beta_f = (fr - fl) / (xr - xl)
            alpha_f = fl - beta_f * xl

            beta_g = (gr - gl) / (xr - xl)
            alpha_g = gl - beta_g * xl

            # h(x) = f(x) * g(x) = α_f α_g + (α_f β_g + α_g β_f) x + β_f β_g x²
            c0 = alpha_f * alpha_g
            c1 = alpha_f * beta_g + alpha_g * beta_f
            c2 = beta_f * beta_g

            # ∫ c0 + c1 x + c2 x² dx = c0 x + c1 x²/2 + c2 x³/3
            total += (c0 * (xr - xl)
                      + c1 * (xr ** 2 - xl ** 2) / 2.0
                      + c2 * (xr ** 3 - xl ** 3) / 3.0)

        return total

    @staticmethod
    def _interp_linear(x_nodes: np.ndarray, y_nodes: np.ndarray,
                       x_query: float) -> float:
        """线性插值"""
        if x_query <= x_nodes[0]:
            return y_nodes[0]
        if x_query >= x_nodes[-1]:
            return y_nodes[-1]
        idx = int(np.searchsorted(x_nodes, x_query)) - 1
        idx = max(0, min(idx, len(x_nodes) - 2))
        dx = x_nodes[idx + 1] - x_nodes[idx]
        if abs(dx) < 1e-15:
            return y_nodes[idx]
        t = (x_query - x_nodes[idx]) / dx
        return y_nodes[idx] + t * (y_nodes[idx + 1] - y_nodes[idx])

    def reaction_rate_integral(self, energy_grid_ev: np.ndarray,
                               cross_section: np.ndarray,
                               temperature_k: float) -> float:
        """
        计算热平均反应速率常数
        
        公式:
          k(T) = √(8/(π m (k_B T)³)) ∫_0^∞ σ(E) E exp(-E/(k_B T)) dE
        
        使用分段线性乘积积分精确计算
        """
        from utils import BOLTZMANN_KB
        kb_t_ev = BOLTZMANN_KB * temperature_k / 1.602176634e-19

        # Maxwell-Boltzmann 能量分布核
        mb_kernel = np.sqrt(energy_grid_ev) * np.exp(-energy_grid_ev / kb_t_ev)

        # 乘积积分
        return self.integrate(energy_grid_ev, cross_section,
                              energy_grid_ev, mb_kernel,
                              energy_grid_ev[0], energy_grid_ev[-1])
