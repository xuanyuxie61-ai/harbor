"""
glass_transition.py
玻璃化转变分析与 Tg 计算模块

融合原项目:
- 809_nonlin_regula: Regula Falsi 非线性求根方法
- 910_prime: 素数序列用于随机采样和统计分析

功能:
1. 比容-温度曲线分析
2. VFT (Vogel-Fulcher-Tammann) 方程拟合
3. Regula Falsi 求根法确定 Tg
4. 玻璃化转变的统计热力学判据
"""

import numpy as np
from typing import Tuple, Optional, Callable
from numeric_utils import safe_divide


def vft_equation(T: float, A: float, B: float, T0: float) -> float:
    """
    Vogel-Fulcher-Tammann (VFT) 方程。
    
    描述过冷液体的弛豫时间或粘度:
        τ(T) = A * exp[ B / (T - T0) ]
    
    其中:
        A: 前置因子
        B: 活化能参数（VFT 温度）
        T0: 理想玻璃化转变温度（Vogel 温度）
    
    参数:
        T: 温度
        A, B, T0: VFT 参数
    
    返回:
        τ(T) 值
    """
    if T <= T0:
        # 数值鲁棒性
        return A * np.exp(B / max(T - T0, 0.01))
    return A * np.exp(B / (T - T0))


def vft_viscosity(T: float, eta_inf: float, B: float, T0: float) -> float:
    """
    VFT 粘度方程。
    
    公式:
        η(T) = η_∞ * exp[ B / (T - T0) ]
    
    参数:
        T: 温度
        eta_inf: 高温极限粘度
        B: VFT 参数
        T0: Vogel 温度
    
    返回:
        粘度值
    """
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
    """
    Regula Falsi（试位法）求根。
    
    融合原项目 809_nonlin_regula:
        算法:
            1. 确保 f(a) 和 f(b) 异号
            2. c = (a * f(b) - b * f(a)) / (f(b) - f(a))
            3. 若 f(c) 与 f(a) 同号，则 a = c；否则 b = c
            4. 重复直到 |b-a| < tol
    
    参数:
        f: 目标函数
        a, b: 初始区间端点（必须满足 f(a)*f(b) < 0）
        tol: 容差
        max_iter: 最大迭代次数
    
    返回:
        (root, iterations)
    """
    fa = f(a)
    fb = f(b)
    
    # 检查符号变化
    if fa * fb > 0:
        # 无符号变化，尝试扩展区间
        raise ValueError("regula_falsi: f(a) 和 f(b) 必须异号")
    
    it = 0
    while abs(b - a) > tol:
        if it >= max_iter:
            break
        
        it += 1
        
        # 避免除零
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
    """
    玻璃化转变分析器。
    
    物理模型:
        比容 v(T) 在 T_g 附近出现斜率变化:
            v(T) = v_g + α_rubber * (T - T_g)   (T > T_g, 橡胶态)
            v(T) = v_g + α_glass * (T - T_g)    (T < T_g, 玻璃态)
        
        其中 α_rubber > α_glass 为热膨胀系数。
    
    T_g 的确定方法:
        1. 切线交点法: 高温切线与低温切线的交点
        2. 中点法: 转变区间的中间温度
        3. VFT 外推法: 通过粘度-温度曲线外推
    """
    
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
        """
        添加一个数据点。
        
        参数:
            temperature: 温度
            specific_volume: 比容（或密度倒数）
            energy: 可选的系统总能量
        """
        self.temperatures.append(temperature)
        self.specific_volumes.append(specific_volume)
        if energy is not None:
            self.energies.append(energy)
    
    def linear_fit(self, T_data: np.ndarray, v_data: np.ndarray) -> Tuple[float, float]:
        """
        线性拟合 v = a + b*T。
        
        使用最小二乘法:
            b = Cov(T,v) / Var(T)
            a = <v> - b * <T>
        
        参数:
            T_data: 温度数组
            v_data: 比容数组
        
        返回:
            (intercept, slope)
        """
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
        """
        使用切线交点法确定 T_g。
        
        方法:
            1. 对高温区数据线性拟合得到高温切线
            2. 对低温区数据线性拟合得到低温切线
            3. 求两条切线的交点即为 T_g
        
        返回:
            (Tg, v_g, alpha_rubber, alpha_glass)
        """
        if len(self.temperatures) < 6:
            raise ValueError("数据点不足，至少需要 6 个点")
        
        T = np.array(self.temperatures)
        v = np.array(self.specific_volumes)
        
        # 按温度排序
        sort_idx = np.argsort(T)
        T = T[sort_idx]
        v = v[sort_idx]
        
        n = len(T)
        n_split = n // 2
        
        # 高温区（后半段）
        T_high = T[n_split:]
        v_high = v[n_split:]
        a_high, b_high = self.linear_fit(T_high, v_high)
        
        # 低温区（前半段）
        T_low = T[:n_split]
        v_low = v[:n_split]
        a_low, b_low = self.linear_fit(T_low, v_low)
        
        # 求交点
        if abs(b_high - b_low) < 1e-15:
            Tg = np.mean(T)
        else:
            Tg = (a_low - a_high) / (b_high - b_low)
        
        v_g = a_high + b_high * Tg
        
        return Tg, v_g, b_high, b_low
    
    def find_tg_regula_falsi(self) -> Tuple[float, float]:
        """
        使用 Regula Falsi 求根法确定 T_g。
        
        构造目标函数:
            f(T) = v_rubber(T) - v_glass(T)
        
        其中 v_rubber 和 v_glass 分别为高温和低温外推的比容值。
        
        返回:
            (Tg, iterations)
        """
        Tg_est, v_g, alpha_r, alpha_g = self.find_tg_tangent_intersection()
        
        T = np.array(self.temperatures)
        
        def f(T_test):
            v_rubber = v_g + alpha_r * (T_test - Tg_est)
            v_glass = v_g + alpha_g * (T_test - Tg_est)
            return v_rubber - v_glass
        
        T_min = np.min(T)
        T_max = np.max(T)
        
        # 找到符号变化区间
        a, b = T_min, T_max
        fa, fb = f(a), f(b)
        
        # 若边界无符号变化，调整
        if fa * fb > 0:
            # 扩展到更宽区间
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
        """
        拟合 VFT 方程参数。
        
        使用 Arrhenius 型变换:
            ln(τ) = ln(A) + B / (T - T0)
        
        通过遍历 T0 寻找最优拟合。
        
        参数:
            viscosity_data: 粘度数据（若 None 则用弛豫时间近似）
        
        返回:
            (A, B, T0)
        """
        T = np.array(self.temperatures)
        
        if viscosity_data is not None:
            eta = np.array(viscosity_data)
        else:
            # 用比容的倒数近似粘度（定性关系）
            v = np.array(self.specific_volumes)
            eta = 1.0 / np.maximum(v, 0.1)
        
        # 筛选有效数据
        mask = (T > 0.05) & (eta > 0)
        T = T[mask]
        eta = eta[mask]
        
        if len(T) < 3:
            return 1.0, 1.0, 0.05
        
        log_eta = np.log(eta)
        
        # 搜索最优 T0
        T_min = np.min(T) * 0.5
        T_max = np.min(T) * 0.95
        T0_values = np.linspace(T_min, T_max, 50)
        
        best_residual = float('inf')
        best_params = (1.0, 1.0, T_min)
        
        for T0 in T0_values:
            inv_T_shifted = 1.0 / (T - T0)
            
            # 线性拟合 ln(eta) vs 1/(T-T0)
            a, b = self.linear_fit(inv_T_shifted, log_eta)
            
            predicted = a + b * inv_T_shifted
            residual = np.mean((log_eta - predicted) ** 2)
            
            if residual < best_residual:
                best_residual = residual
                best_params = (np.exp(a), b, T0)
        
        return best_params
    
    def fragility_index(self, A: float, B: float, T0: float, Tg: float) -> float:
        """
        计算脆性指数 m。
        
        公式:
            m = d(log τ) / d(Tg/T) |_{T=Tg}
              = (B * Tg) / [ln(10) * (Tg - T0)^2]
        
        参数:
            A, B, T0: VFT 参数
            Tg: 玻璃化转变温度
        
        返回:
            脆性指数 m
        """
        if Tg <= T0:
            return 100.0
        
        m = (B * Tg) / (np.log(10) * (Tg - T0) ** 2)
        return float(m)
    
    def configurational_entropy(self, T: float, Tg: float, Delta_Cp: float = 1.0) -> float:
        """
        计算构型熵（Adam-Gibbs 理论）。
        
        公式:
            S_c(T) = ΔC_p * ln(T / Tg)   (T > T_k)
            S_c(T) = 0                     (T <= T_k)
        
        其中 T_k = T0 为 Kauzmann 温度。
        
        参数:
            T: 温度
            Tg: 玻璃化转变温度
            Delta_Cp: 热容差
        
        返回:
            构型熵
        """
        if T <= 0.01:
            return 0.0
        return Delta_Cp * np.log(T / Tg) if T > Tg else 0.0
    
    def get_summary(self) -> dict:
        """
        获取玻璃化转变分析摘要。
        
        返回:
            包含 Tg、VFT 参数、脆性指数等的字典
        """
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
