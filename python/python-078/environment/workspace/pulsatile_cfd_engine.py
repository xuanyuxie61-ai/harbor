"""
pulsatile_cfd_engine.py
核心脉动流CFD引擎：Womersley流求解与壁面剪切应力（WSS）分析

融合来源:
- 966_r83t (通过linear_algebra_core): 三对角线性系统求解（Thomas算法/CG）
- 528+660 (通过quadrature_rules): 六边形/Legendre数值积分
- 119 (通过stochastic_diffusion): 有效扩散系数修正

科学背景:
Womersley流是轴对称圆管中受振荡压力梯度驱动的非定常层流。
控制方程（柱坐标，轴对称）:

    ρ ∂u/∂t = -∂p/∂z + μ [ ∂²u/∂r² + (1/r) ∂u/∂r ]

边界条件:
    u(R, t) = 0          (壁面无滑移)
    ∂u/∂r|_{r=0} = 0     (轴心对称)

压力梯度采用心动周期模型:
    -∂p/∂z = A_0 + A_1 cos(ωt) + A_2 cos(2ωt + φ_2)

Womersley精确解（对一个谐波分量）:
    u_n(r,t) = Re{ (i A_n / (ρ ω_n)) [ 1 - J_0(α_n i^{3/2} r/R) / J_0(α_n i^{3/2}) ] e^{i ω_n t} }

其中:
    J_0: 第一类零阶Bessel函数
    α_n = R sqrt(ω_n / ν): Womersley数
    i^{3/2} = (i-1)/√2

壁面剪切应力:
    τ_w(t) = μ ∂u/∂r|_{r=R}

时间平均WSS (TAWSS):
    TAWSS = (1/T) ∫_0^T |τ_w(t)| dt

振荡剪切指数 (OSI):
    OSI = 0.5 [ 1 - |∫ τ_w dt| / ∫ |τ_w| dt ]
    OSI ∈ [0, 0.5]，OSI > 0.15 提示动脉粥样硬化风险
"""

import numpy as np
from scipy.special import jv
from typing import Dict, Tuple

from linear_algebra_core import (
    R83TMatrix, build_womersley_tridiagonal, thomas_algorithm,
    r83t_cg_solve
)
from quadrature_rules import gauss_legendre_quadrature
from stochastic_diffusion import einstein_viscosity_correction


# ======================================================================
# Womersley脉动流求解器
# ======================================================================

class WomersleySolver:
    """
    Womersley轴对称脉动流数值求解器。

    使用隐式有限差分法离散，每个时间步求解三对角线性系统。
    """
    def __init__(self, radius: float = 0.005,
                 kinematic_viscosity: float = 3.3e-6,
                 blood_density: float = 1060.0,
                 n_radial: int = 100,
                 heart_rate_bpm: float = 72.0):
        """
        参数:
            radius: 血管半径 [m]
            kinematic_viscosity: 运动粘度 [m²/s]
            blood_density: 血液密度 [kg/m³]
            n_radial: 径向网格点数
            heart_rate_bpm: 心率 [次/分钟]
        """
        self.R = radius
        self.nu = kinematic_viscosity
        self.rho = blood_density
        self.n_r = n_radial
        self.HR = heart_rate_bpm

        # 径向网格（非均匀，壁面处加密）
        self.r = np.linspace(0, self.R, n_radial)
        self.dr = self.r[1] - self.r[0]

        # 角频率
        self.omega = 2.0 * np.pi * self.HR / 60.0
        self.alpha = self.R * np.sqrt(self.omega / self.nu)

        # WSS时间历史
        self.wss_history = []
        self.time_history = []

    def pressure_gradient(self, t: float,
                          base_gradient: float = 100.0,
                          pulsatile_amp: float = 80.0) -> float:
        """
        心动周期压力梯度模型。

        参数:
            t: 时间 [s]
            base_gradient: 平均压力梯度 [Pa/m]
            pulsatile_amp: 脉动幅度 [Pa/m]

        返回:
            -∂p/∂z [Pa/m]
        """
        T = 60.0 / self.HR
        phase = 2.0 * np.pi * t / T
        # 模拟收缩峰在t≈0.2T，舒张期在t≈0.5T-1.0T
        return base_gradient + pulsatile_amp * np.cos(phase) + 0.3 * pulsatile_amp * np.cos(2.0 * phase)

    def time_step(self, u_old: np.ndarray, dt: float, t: float,
                  use_thomas: bool = True) -> np.ndarray:
        """
        执行一个时间步的隐式求解。

        离散方程:
            (u^{n+1} - u^n)/Δt = RHS + ν [∇²_r u^{n+1}]

        整理为: A u^{n+1} = b
        """
        # TODO: 执行一个时间步的隐式求解
        # 提示：
        #   1. 调用 build_womersley_tridiagonal 构造三对角矩阵 A
        #   2. 组装右端项 b = u_old + (dt/rho) * pressure_gradient(t)
        #   3. 使用 thomas_algorithm 或 r83t_cg_solve 求解 A u_new = b
        #   4. 应用边界条件：壁面无滑移 (u_new[-1]=0)，轴心对称 (u_new[0]>=0)
        #   返回 u_new
        raise NotImplementedError("Hole 2: WomersleySolver.time_step 待实现")

    def solve_steady_state(self, max_iter: int = 10000,
                           dt: float = 1e-4,
                           tol: float = 1e-8) -> np.ndarray:
        """
        求解稳态Poiseuille速度剖面（用于验证）。

        精确解:
            u(r) = (R² - r²) / (4μ) · (-∂p/∂z)
        """
        # 使用恒定的正向压力梯度进行稳态验证
        dpdx = 100.0  # [Pa/m]
        u = np.zeros(self.n_r)
        mu = self.nu * self.rho
        for _ in range(max_iter):
            u_new = u.copy()
            # 内部点隐式更新
            for j in range(1, self.n_r - 1):
                r_j = self.r[j]
                # 简化的隐式格式
                laplacian = (u[j-1] - 2*u[j] + u[j+1]) / self.dr**2 + \
                            (u[j+1] - u[j-1]) / (2 * r_j * self.dr)
                u_new[j] = u[j] + dt * (dpdx / self.rho + self.nu * laplacian)
            # 边界条件
            u_new[0] = u_new[1]  # 轴心对称
            u_new[-1] = 0.0       # 壁面无滑移

            if np.linalg.norm(u_new - u, ord=np.inf) < tol:
                return u_new
            u = u_new
        return u

    def solve_pulsatile(self, n_cardiac_cycles: float = 2.0,
                        n_steps_per_cycle: int = 200,
                        dt: float = None) -> Dict:
        """
        求解多个心动周期的脉动流。

        参数:
            n_cardiac_cycles: 模拟心动周期数
            n_steps_per_cycle: 每周期时间步数
            dt: 时间步长（默认自动计算）

        返回:
            包含速度场、WSS历史、时间历史的字典
        """
        T = 60.0 / self.HR
        if dt is None:
            dt = T / n_steps_per_cycle

        total_steps = int(n_cardiac_cycles * n_steps_per_cycle)
        u = np.zeros(self.n_r)

        # 瞬态预热（避免初始条件影响）
        warmup_steps = n_steps_per_cycle
        for i in range(warmup_steps):
            t = i * dt
            u = self.time_step(u, dt, t)

        self.wss_history = []
        self.time_history = []
        velocity_snapshots = []

        for i in range(total_steps):
            t = i * dt
            u = self.time_step(u, dt, t)

            # 计算WSS: τ_w = μ (du/dr)|_{r=R}
            # 壁面处用后向差分，取绝对值以符合临床报告惯例
            wss = abs(self.nu * self.rho * (u[-1] - u[-2]) / self.dr)
            self.wss_history.append(float(wss))
            self.time_history.append(float(t))

            if i % (n_steps_per_cycle // 4) == 0:
                velocity_snapshots.append(u.copy())

        return {
            "velocity_final": u,
            "velocity_snapshots": velocity_snapshots,
            "wss_history": np.array(self.wss_history),
            "time_history": np.array(self.time_history),
            "radial_grid": self.r,
            "alpha": self.alpha
        }

    def womersley_exact_solution(self, t: float, n_harmonics: int = 3) -> np.ndarray:
        """
        计算Womersley精确解（Bessel函数形式），用于验证数值解。

        对于第n阶谐波:
            u_n(r,t) = Re{ (i A_n / (ρ n ω)) [1 - J_0(α_n i^{3/2} r/R) / J_0(α_n i^{3/2})] e^{i n ω t} }
        """
        u_exact = np.zeros(self.n_r, dtype=complex)
        A1 = -80.0  # 一阶谐波振幅

        for n in range(1, n_harmonics + 1):
            omega_n = n * self.omega
            alpha_n = self.R * np.sqrt(omega_n / self.nu)
            z = alpha_n * ((-1j) ** 1.5)
            z_wall = z

            # J_0在壁面处的值
            j0_wall = jv(0, z_wall)
            if abs(j0_wall) < 1e-15:
                j0_wall = 1e-15

            coeff = 1j * A1 / (self.rho * omega_n * n)
            for j, rj in enumerate(self.r):
                zr = z * rj / self.R
                j0_r = jv(0, zr)
                u_exact[j] += coeff * (1.0 - j0_r / j0_wall) * np.exp(1j * omega_n * t)

        return np.real(u_exact)


# ======================================================================
# WSS统计量计算
# ======================================================================

def compute_tawss(wss_history: np.ndarray, time_history: np.ndarray) -> float:
    """
    计算时间平均壁面剪切应力 (TAWSS)。

    TAWSS = (1/T) ∫_0^T |τ_w(t)| dt
    """
    if len(wss_history) < 2:
        return 0.0
    # 梯形法则
    tawss = np.trapezoid(np.abs(wss_history), time_history)
    T = time_history[-1] - time_history[0]
    return float(tawss / (T + 1e-15))


def compute_osi(wss_history: np.ndarray, time_history: np.ndarray) -> float:
    """
    计算振荡剪切指数 (Oscillatory Shear Index, OSI)。

    OSI = 0.5 [ 1 - |∫ τ_w dt| / ∫ |τ_w| dt ]

    临床意义:
        OSI < 0.05: 单向剪切，内皮保护
        OSI > 0.15: 强振荡，促动脉粥样硬化
    """
    # TODO: 计算振荡剪切指数 OSI
    # 提示：
    #   OSI = 0.5 * [1 - |∫ τ_w dt| / ∫ |τ_w| dt]
    #   使用梯形法则计算两个积分
    #   返回 clip 到 [0, 0.5] 的结果
    raise NotImplementedError("Hole 3: compute_osi 待实现")


def compute_wss_gradient(wss_history: np.ndarray, time_history: np.ndarray) -> float:
    """
    计算WSS时间梯度（WSSG）的均方根。

    WSSG = sqrt( (1/T) ∫ (dτ_w/dt)² dt )

    高WSSG提示内皮细胞经历快速力学环境变化。
    """
    if len(wss_history) < 3:
        return 0.0
    dt = np.diff(time_history)
    d_wss = np.diff(wss_history)
    d_tau_dt = d_wss / (dt + 1e-15)
    # 使用梯形法则近似积分
    T = time_history[-1] - time_history[0]
    wssg = np.sqrt(np.mean(d_tau_dt ** 2))
    return float(wssg)


def relative_resistance_index(wss_max: float, wss_min: float) -> float:
    """
    相对阻力指数 (Relative Resistance Index, RRI)。

    RRI = (WSS_max - WSS_min) / (WSS_max + WSS_min)

    用于评估脉动性对血管壁的影响。
    """
    denom = wss_max + wss_min + 1e-15
    return float((wss_max - wss_min) / denom)


# ======================================================================
# 综合WSS分析报告
# ======================================================================

def generate_wss_report(solver: WomersleySolver, result: Dict) -> Dict:
    """
    生成完整的WSS分析报告。
    """
    wss_hist = result["wss_history"]
    time_hist = result["time_history"]

    if len(wss_hist) == 0:
        return {"error": "No WSS data"}

    tawss = compute_tawss(wss_hist, time_hist)
    osi = compute_osi(wss_hist, time_hist)
    wssg = compute_wss_gradient(wss_hist, time_hist)
    wss_max = float(np.max(wss_hist))
    wss_min = float(np.min(wss_hist))
    rri = relative_resistance_index(wss_max, wss_min)

    # 使用Gauss-Legendre求积验证TAWSS（积分一个周期）
    def wss_abs_interp(t):
        # 线性插值
        return np.interp(t, time_hist, np.abs(wss_hist))

    T = time_hist[-1] - time_hist[0]
    if T > 0:
        tawss_gl = gauss_legendre_quadrature(wss_abs_interp, time_hist[0], time_hist[-1], n=64) / T
    else:
        tawss_gl = tawss

    return {
        "TAWSS_Pa": tawss,
        "TAWSS_GaussLegendre_Pa": float(tawss_gl),
        "OSI": osi,
        "WSSG_Pa_s": wssg,
        "WSS_max_Pa": wss_max,
        "WSS_min_Pa": wss_min,
        "RRI": rri,
        "Womersley_alpha": float(solver.alpha),
        "physiological_score": _overall_physiological_score(tawss, osi)
    }


def _overall_physiological_score(tawss: float, osi: float) -> float:
    """
    综合生理评分。
    """
    score = 1.0
    if tawss < 0.5:
        score -= 0.3
    elif tawss > 7.0:
        score -= 0.3
    if osi > 0.15:
        score -= 0.3
    return max(score, 0.0)
