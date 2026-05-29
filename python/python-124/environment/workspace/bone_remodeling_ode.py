"""
bone_remodeling_ode.py
骨重建生化动力学模型模块

融合来源：
- 090_biochemical_linear_ode: 生化反应线性ODE系统（守恒律、精确解析解）

科学背景：
骨重建（Bone Remodeling）是破骨细胞（Osteoclasts）吸收旧骨与
成骨细胞（Osteoblasts）形成新骨的耦合过程。

本项目基于力学调控的骨重建模型（Mechanostat Theory / Frost's Law）：

    dρ/dt = k_form * [U - U_ref]_+ - k_res * [U_ref - U]_+ * ρ

其中：
    ρ(x,t) : 局部骨密度
    U(x,t) : 应变能密度
    U_ref  : 参考应变能密度（设定点）
    k_form : 骨形成速率常数
    k_res  : 骨吸收速率常数
    [·]_+  : max(·, 0)

质量守恒约束：
    d/dt ∫_Ω ρ dΩ = 0  （在封闭系统中）

解析解（简化线性模型）：
    当 U 为常数时，方程退化为线性ODE：
        dρ/dt = A - B*ρ
    其解为：
        ρ(t) = ρ_∞ + (ρ_0 - ρ_∞) * exp(-B * t)
    其中 ρ_∞ = A / B 为稳态密度。
"""

import numpy as np
from typing import Tuple, Optional, Callable
from scipy.integrate import solve_ivp


class BoneRemodelingODE:
    """
    骨重建动力学ODE模型。
    """

    def __init__(self, k_form: float = 0.05, k_res: float = 0.03,
                 U_ref: float = 0.5, rho_min: float = 0.01,
                 rho_max: float = 1.8):
        """
        Parameters
        ----------
        k_form : float
            骨形成速率常数 (1/MPa·day)
        k_res : float
            骨吸收速率常数 (1/day)
        U_ref : float
            参考应变能密度 (MPa)
        rho_min : float
            最小骨密度 (g/cm³)
        rho_max : float
            最大骨密度 (g/cm³)
        """
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
        """
        计算骨密度变化率 dρ/dt。

        力学调控模型：
            当 U > U_ref 时，骨形成占优势
            当 U < U_ref 时，骨吸收占优势

        公式：
            dρ/dt = k_form * max(U - U_ref, 0) - k_res * max(U_ref - U, 0) * ρ

        Parameters
        ----------
        rho : float
            当前骨密度
        U : float
            当前应变能密度

        Returns
        -------
        float
            dρ/dt
        """
        # TODO HOLE_1: 实现骨重建速率公式（Mechanostat Theory）
        # 提示：需要考虑骨形成项、骨吸收项、边界限制
        raise NotImplementedError("Hole 1: 请实现 remodeling_rate 核心公式")


    def steady_state_density(self, U: float) -> float:
        """
        计算给定应变能密度下的稳态骨密度。

        令 dρ/dt = 0：
            k_form * (U - U_ref)_+ = k_res * (U_ref - U)_+ * ρ_∞

        当 U > U_ref 时：ρ_∞ = k_form * (U - U_ref) / (k_res * (U_ref - U)) → ∞
        实际上需考虑上限约束。

        更合理的稳态模型（混合形式）：
            ρ_∞ = rho_max * (U / (U + U_ref))
        """
        if U <= 0:
            return self.rho_min
        rho_ss = self.rho_max * (U / (U + self.U_ref))
        return max(self.rho_min, min(self.rho_max, rho_ss))

    def exact_solution_linear(self, t: np.ndarray, rho0: float,
                              A: float, B: float) -> np.ndarray:
        """
        线性ODE dρ/dt = A - B*ρ 的精确解析解。

        来自 090_biochemical_linear_ode 的 exact 解思想：
            ρ(t) = A/B + (ρ0 - A/B) * exp(-B * t)

        Parameters
        ----------
        t : np.ndarray
            时间数组
        rho0 : float
            初始密度
        A : float
            形成项常数
        B : float
            吸收项系数

        Returns
        -------
        np.ndarray
            ρ(t)
        """
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
        """
        求解时间依赖的骨密度演化。

        对每个节点独立积分（空间解耦近似）：
            dρ_i/dt = f(ρ_i, U_i),  i = 1,...,N

        Parameters
        ----------
        rho0 : np.ndarray, shape (N,)
            初始骨密度场
        strain_energy_field : np.ndarray, shape (N,)
            每个节点的应变能密度
        t_span : tuple
            时间区间 (day)
        t_eval : np.ndarray, optional
            输出时间点
        method : str
            ODE求解方法

        Returns
        -------
        t : np.ndarray
            时间点
        rho_history : np.ndarray, shape (len(t), N)
            密度演化历史
        """
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
        """
        计算总骨质量（守恒量）。

        来自 090_biochemical_linear_ode 的 conserved 思想。

        M_total = Σ_i ρ_i * V_i
        """
        if len(rho) != len(volumes):
            raise ValueError("Length mismatch")
        return float(np.dot(rho, volumes))

    def check_mass_conservation(self, rho_history: np.ndarray,
                                volumes: np.ndarray,
                                tolerance: float = 1e-3) -> bool:
        """
        检查质量守恒。
        """
        M0 = self.conserved_quantity(rho_history[:, 0], volumes)
        M_final = self.conserved_quantity(rho_history[:, -1], volumes)
        relative_error = abs(M_final - M0) / max(abs(M0), 1e-14)
        return relative_error < tolerance


# =====================================================================
# 扩展的力学-生化耦合ODE（多物种）
# =====================================================================
class CoupledBoneRemodelingODE:
    """
    多物种耦合骨重建模型。

    状态向量 y = [ρ, c_oc, c_ob]
    其中：
        ρ    : 骨密度
        c_oc : 破骨细胞浓度
        c_ob : 成骨细胞浓度

    方程组：
        dρ/dt    = k1 * c_ob - k2 * c_oc * ρ
        dc_oc/dt = k3 * [U_ref - U]_+ - k4 * c_oc
        dc_ob/dt = k5 * [U - U_ref]_+ - k6 * c_ob
    """

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
        """
        计算导数 dy/dt。

        Parameters
        ----------
        t : float
            时间
        y : np.ndarray, shape (3,)
            [ρ, c_oc, c_ob]
        U : float
            应变能密度

        Returns
        -------
        np.ndarray, shape (3,)
            导数
        """
        rho, c_oc, c_ob = y
        p = self.params

        drhodt = p['k1'] * c_ob - p['k2'] * c_oc * rho
        doc_dt = p['k3'] * max(p['U_ref'] - U, 0.0) - p['k4'] * c_oc
        dob_dt = p['k5'] * max(U - p['U_ref'], 0.0) - p['k6'] * c_ob

        # 边界处理
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
        """
        求解耦合ODE系统。
        """
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
