"""
flexible_beam_modal.py
======================
柔性梁精确模态分析与位移场插值

本模块将以下种子项目的核心算法融入结构力学：
  - 316_doughnut_exact : 精确三角/双曲有理函数公式 → Euler-Bernoulli 梁精确模态函数
  - 595_interp_spline_data : 三次样条插值 → 梁变形场节点间平滑重构

核心物理模型：
  - Euler-Bernoulli 梁方程：
        ρA · ∂²w/∂t² + EI · ∂⁴w/∂x⁴ = q(x,t)
    其中 ρ 为密度，A 为截面积，E 为弹性模量，I 为截面惯性矩。
  
  - 分离变量 w(x,t) = W(x)·T(t) 后得到空间常微分方程：
        W''''(x) - β⁴ W(x) = 0,   β⁴ = ω² ρA / (EI)
    
  - 通解为精确三角-双曲组合：
        W(x) = C₁ sin(βx) + C₂ cos(βx) + C₃ sinh(βx) + C₄ cosh(βx)
    
  - 对悬臂梁（固支-自由），频率方程为：
        cos(βL) cosh(βL) = -1
    
  - 模态正交性：
        ∫₀ᴸ Wᵢ(x) Wⱼ(x) dx = δᵢⱼ
"""

import numpy as np
from scipy.optimize import brentq
from scipy.interpolate import CubicSpline
from typing import List, Tuple, Optional


class EulerBernoulliBeam:
    """
    均匀截面 Euler-Bernoulli 梁的精确模态分析器。
    """
    def __init__(self, L: float, E: float, I: float, rho: float, A: float,
                 boundary: str = "cantilever"):
        """
        参数
        ----
        L : 梁长度 [m]
        E : 弹性模量 [Pa]
        I : 截面惯性矩 [m⁴]
        rho : 材料密度 [kg/m³]
        A : 截面积 [m²]
        boundary : 边界条件类型
            "cantilever"    — 固支-自由 (clamped-free)
            "pinned_pinned" — 简支-简支
            "free_free"     — 自由-自由（用于漂浮构件）
            "clamped_clamped" — 固支-固支
        """
        if L <= 0 or E <= 0 or I <= 0 or rho <= 0 or A <= 0:
            raise ValueError("所有几何与材料参数必须为正")
        self.L = float(L)
        self.E = float(E)
        self.I = float(I)
        self.rho = float(rho)
        self.A = float(A)
        self.boundary = boundary
        self._beta_roots: Optional[np.ndarray] = None
        self._modes_cached: Optional[List] = None

    @property
    def c0(self) -> float:
        """波速参数 c₀ = √(EI / (ρA)) [m²/s]。"""
        return np.sqrt(self.E * self.I / (self.rho * self.A))

    def frequency_equation(self, beta_L: float) -> float:
        """
        频率方程残差 f(βL) = 0 的确定。
        对悬臂梁：cos(βL)cosh(βL) + 1 = 0
        """
        b = float(beta_L)
        if self.boundary == "cantilever":
            return np.cos(b) * np.cosh(b) + 1.0
        elif self.boundary == "pinned_pinned":
            return np.sin(b)
        elif self.boundary == "free_free":
            return np.cos(b) * np.cosh(b) - 1.0
        elif self.boundary == "clamped_clamped":
            return np.cos(b) * np.cosh(b) - 1.0
        else:
            raise ValueError(f"未知边界条件: {self.boundary}")

    def find_beta_roots(self, n_modes: int = 6) -> np.ndarray:
        """
        用 Brent 法在区间 [0.1, (n_modes+1)π] 内搜索前 n_modes 个正根 βL。
        
        物理意义：βₙ 决定第 n 阶固有频率 ωₙ = βₙ² √(EI/ρA)。
        """
        if n_modes < 1:
            raise ValueError("n_modes 必须 ≥ 1")
        roots = []
        upper = (n_modes + 2) * np.pi
        # 细分扫描区间寻找符号变化
        xs = np.linspace(0.1, upper, 20 * n_modes + 50)
        fs = np.array([self.frequency_equation(x) for x in xs])
        for i in range(len(xs) - 1):
            if fs[i] == 0.0:
                roots.append(xs[i])
            elif fs[i] * fs[i + 1] < 0:
                try:
                    r = brentq(self.frequency_equation, xs[i], xs[i + 1],
                               xtol=1e-14, maxiter=100)
                    if r > 1e-6:
                        roots.append(r)
                except ValueError:
                    pass
        roots = np.array(sorted(set(np.round(roots, 12))))
        if len(roots) < n_modes:
            # 对 pinned-pinned 等简单情况，根为 nπ
            if self.boundary == "pinned_pinned":
                roots = np.arange(1, n_modes + 1) * np.pi
            else:
                raise RuntimeError(f"仅找到 {len(roots)} 个根，需要 {n_modes} 个")
        self._beta_roots = roots[:n_modes]
        return self._beta_roots

    def modal_shape(self, x: np.ndarray, n: int) -> np.ndarray:
        """
        计算第 n 阶模态振型 W_n(x)。
        对悬臂梁，利用边界条件消去系数后得到：
            W(x) = cosh(βx) - cos(βx) - σ (sinh(βx) - sin(βx))
        其中 σ = (sin(βL) - sinh(βL)) / (cos(βL) + cosh(βL))
        
        返回的振型已按 ∫ W² dx = 1 归一化。
        """
        x = np.asarray(x, dtype=np.float64)
        if np.any(x < -1e-12) or np.any(x > self.L + 1e-12):
            raise ValueError("x 必须在 [0, L] 区间内")
        if self._beta_roots is None or n >= len(self._beta_roots):
            self.find_beta_roots(n_modes=max(n + 1, 6))
        b = self._beta_roots[n] / self.L  # β = (βL)/L
        bL = self._beta_roots[n]

        if self.boundary == "cantilever":
            sigma = (np.sin(bL) - np.sinh(bL)) / (np.cos(bL) + np.cosh(bL))
            W = (np.cosh(b * x) - np.cos(b * x)
                 - sigma * (np.sinh(b * x) - np.sin(b * x)))
        elif self.boundary == "pinned_pinned":
            W = np.sin(b * x)
        elif self.boundary in ("free_free", "clamped_clamped"):
            # 使用双曲-三角组合的一般形式，取标准归一化表达
            sigma = (np.sinh(bL) - np.sin(bL)) / (np.cosh(bL) - np.cos(bL))
            W = (np.cosh(b * x) + np.cos(b * x)
                 - sigma * (np.sinh(b * x) + np.sin(b * x)))
        else:
            raise ValueError(f"未知边界: {self.boundary}")
        # 归一化
        norm2 = np.trapezoid(W ** 2, x)
        if norm2 > 0:
            W /= np.sqrt(norm2)
        return W

    def natural_frequencies(self, n_modes: int = 6) -> np.ndarray:
        """
        计算前 n_modes 阶圆频率 [rad/s]。
        公式：ωₙ = βₙ² √(EI/ρA) = βₙ² · c₀
        """
        bL = self.find_beta_roots(n_modes)
        beta = bL / self.L
        omega = beta ** 2 * self.c0
        return omega

    def modal_stiffness_mass(self, n_modes: int = 6) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算模态坐标下的广义质量矩阵 M* 与广义刚度矩阵 K*。
        对归一化模态，M* = ρA · I（单位阵），K* = diag(EI · βₙ⁴)。
        """
        bL = self.find_beta_roots(n_modes)
        beta = bL / self.L
        # TODO: Hole 1 — 请根据 Euler-Bernoulli 梁模态理论实现 M* 与 K* 的计算
        raise NotImplementedError("Hole 1: modal_stiffness_mass 待实现")

    def displacement_field_spline(self, nodal_coords: np.ndarray,
                                   nodal_displacements: np.ndarray,
                                   num_eval: int = 101) -> Tuple[np.ndarray, np.ndarray]:
        """
        基于 595_interp_spline_data 的三次样条插值思想，将节点位移重构为连续梁变形场。
        
        使用 not-a-knot 边界条件的三次样条：
            w_spline(x) = Σ a_i (x - x_i)³ + b_i (x - x_i)² + c_i (x - x_i) + d_i
        在每一区间 [x_i, x_{i+1}] 上满足 C² 连续。
        
        参数
        ----
        nodal_coords : 节点轴向坐标，严格递增
        nodal_displacements : 对应挠度
        num_eval : 评估点数
        
        返回
        ----
        x_eval, w_eval : 连续挠度曲线采样
        """
        nodal_coords = np.asarray(nodal_coords, dtype=np.float64)
        nodal_displacements = np.asarray(nodal_displacements, dtype=np.float64)
        if len(nodal_coords) < 2:
            raise ValueError("至少需要两个节点才能插值")
        if not np.all(np.diff(nodal_coords) > 0):
            raise ValueError("节点坐标必须严格递增")
        if len(nodal_coords) != len(nodal_displacements):
            raise ValueError("坐标与位移长度不一致")
        cs = CubicSpline(nodal_coords, nodal_displacements, bc_type="not-a-knot")
        x_eval = np.linspace(nodal_coords[0], nodal_coords[-1], num_eval)
        w_eval = cs(x_eval)
        return x_eval, w_eval

    def exact_static_deflection(self, x: np.ndarray, load_type: str = "tip_force",
                                P: float = 1.0) -> np.ndarray:
        """
        计算悬臂梁在典型载荷下的精确静挠度曲线（解析解）。
        
        对端部集中力 P：
            w(x) = (P / (6EI)) · (3Lx² - x³)
        对均布载荷 q₀ = P/L：
            w(x) = (q₀ / (24EI)) · (x⁴ - 4Lx³ + 6L²x²)
        """
        x = np.asarray(x, dtype=np.float64)
        if np.any(x < 0) or np.any(x > self.L):
            raise ValueError("x 必须在 [0, L] 内")
        if load_type == "tip_force":
            w = (P / (6.0 * self.E * self.I)) * (3.0 * self.L * x ** 2 - x ** 3)
        elif load_type == "uniform":
            q0 = P / self.L
            w = (q0 / (24.0 * self.E * self.I)) * (x ** 4
                  - 4.0 * self.L * x ** 3 + 6.0 * self.L ** 2 * x ** 2)
        else:
            raise ValueError(f"未知载荷类型: {load_type}")
        return w
