"""
pbpk_ode_solver.py
基于种子项目 1283_tough_ode

实现刚性 ODE 系统的数值积分，特别针对多 compartment PBPK 模型。
包含 Hairer-Nørsett-Wanner 经典 stiff ODE benchmark（4 方程组）
及其精确解，用于验证积分器精度。

在 PBPK 模型中用于：
- 多器官 compartment 药物浓度的瞬态动力学
- 肝脏代谢（Michaelis-Menten）与肾脏清除（GFR）的耦合刚性系统
- 快速平衡器官与慢速平衡器官的时间尺度分离问题
"""

import numpy as np
from typing import Callable, Tuple, List

# ---------------------------------------------------------------------------
# 经典 stiff ODE benchmark（用于验证）
# ---------------------------------------------------------------------------

def tough_deriv(t: float, y: np.ndarray) -> np.ndarray:
    """
    Hairer, Nørsett & Wanner 的 stiff ODE benchmark（4 方程）。
    dy1/dt = 2t * y2^0.2 * y4
    dy2/dt = 10t * exp(5(y2-1)) * y4
    dy3/dt = 2t * y4
    dy4/dt = -2t * ln(y1)
    """
    y1, y2, y3, y4 = y
    if y1 <= 0.0:
        y1 = 1e-300
    dy = np.zeros(4)
    dy[0] = 2.0 * t * (y2 ** 0.2) * y4
    dy[1] = 10.0 * t * np.exp(5.0 * (y2 - 1.0)) * y4
    dy[2] = 2.0 * t * y4
    dy[3] = -2.0 * t * np.log(y1)
    return dy


def tough_exact(t: float) -> np.ndarray:
    """
    上述 stiff ODE 的精确解：
    y1 = exp(sin(t^2))
    y2 = exp(5 sin(t^2))
    y3 = sin(t^2) + 1
    y4 = cos(t^2)
    """
    s = np.sin(t * t)
    c = np.cos(t * t)
    return np.array([np.exp(s), np.exp(5.0 * s), s + 1.0, c])


# ---------------------------------------------------------------------------
# 显式 Runge-Kutta 4 阶（非刚性）
# ---------------------------------------------------------------------------

def rk4_step(f: Callable, t: float, y: np.ndarray, h: float) -> np.ndarray:
    """单步 RK4。"""
    k1 = f(t, y)
    k2 = f(t + 0.5 * h, y + 0.5 * h * k1)
    k3 = f(t + 0.5 * h, y + 0.5 * h * k2)
    k4 = f(t + h, y + h * k3)
    return y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


# ---------------------------------------------------------------------------
# 隐式梯形法（A-稳定，适合刚性）
# ---------------------------------------------------------------------------

def implicit_trapezoidal_step(f: Callable, t: float, y: np.ndarray, h: float,
                               tol: float = 1e-10, max_iter: int = 50) -> np.ndarray:
    """
    隐式梯形法单步：y_{n+1} = y_n + h/2 * (f(t_n, y_n) + f(t_{n+1}, y_{n+1}))
    使用不动点迭代求解（适用于中小规模系统）。
    """
    f_n = f(t, y)
    y_new = y + h * f_n  # 初始猜测：Euler 前向
    for _ in range(max_iter):
        y_next = y + 0.5 * h * (f_n + f(t + h, y_new))
        if np.linalg.norm(y_next - y_new) < tol:
            return y_next
        y_new = y_next
    return y_new


# ---------------------------------------------------------------------------
# Rosenbrock 方法（线性隐式，适合中等刚性）
# ---------------------------------------------------------------------------

def rosenbrock_step(f: Callable, t: float, y: np.ndarray, h: float,
                     J: Callable = None) -> np.ndarray:
    """
    简化的 Rosenbrock 方法（1 阶，对角方法）。
    需要 Jacobian J(t,y) = df/dy。若未提供，使用数值差分近似。
    (I - h γ J) k = f(t, y)
    y_{new} = y + h k
    γ = 1.0
    """
    n = len(y)
    gamma = 1.0
    if J is None:
        # 数值 Jacobian
        eps = np.sqrt(np.finfo(float).eps)
        J_mat = np.zeros((n, n))
        f0 = f(t, y)
        for j in range(n):
            y_pert = y.copy()
            y_pert[j] += eps * max(1.0, abs(y[j]))
            J_mat[:, j] = (f(t, y_pert) - f0) / (y_pert[j] - y[j])
    else:
        J_mat = J(t, y)
        f0 = f(t, y)

    M = np.eye(n) - h * gamma * J_mat
    try:
        k = np.linalg.solve(M, f0)
    except np.linalg.LinAlgError:
        # 矩阵奇异时回退到前向 Euler
        k = f0
    return y + h * k


# ---------------------------------------------------------------------------
# 通用 ODE 求解器（自动步长控制）
# ---------------------------------------------------------------------------

def solve_ode(f: Callable, t_span: Tuple[float, float], y0: np.ndarray,
              method: str = "rk4", h_init: float = 0.01,
              rtol: float = 1e-6, atol: float = 1e-9,
              max_steps: int = 100000) -> Tuple[np.ndarray, np.ndarray]:
    """
    通用 ODE 求解器。
    method: "rk4" | "implicit_trap" | "rosenbrock"
    返回 (t_array, y_array)，其中 y_array 的每行对应一个时间点的状态向量。
    """
    t0, tf = t_span
    if t0 >= tf:
        raise ValueError("t_span must satisfy t0 < tf")
    if len(y0) < 1:
        raise ValueError("y0 must be non-empty")

    t = t0
    y = y0.astype(float).copy()
    ts = [t]
    ys = [y.copy()]
    h = h_init
    step = 0

    while t < tf and step < max_steps:
        h = min(h, tf - t)
        if method == "rk4":
            y_new = rk4_step(f, t, y, h)
        elif method == "implicit_trap":
            y_new = implicit_trapezoidal_step(f, t, y, h)
        elif method == "rosenbrock":
            y_new = rosenbrock_step(f, t, y, h)
        else:
            raise ValueError(f"Unknown method: {method}")

        # 简单的步长控制（基于相对误差估计）
        if method == "rk4":
            # 用半步 RK4 作为误差估计
            y_half1 = rk4_step(f, t, y, h / 2.0)
            y_half2 = rk4_step(f, t + h / 2.0, y_half1, h / 2.0)
            err = np.linalg.norm(y_new - y_half2) / (atol + rtol * np.linalg.norm(y_new))
            if err > 1.0 and h > 1e-12:
                h *= max(0.5, 0.9 / np.sqrt(err))
                continue
            elif err < 0.5:
                h *= min(2.0, 0.9 / np.sqrt(max(err, 1e-10)))
        else:
            # 隐式方法直接接受
            pass

        t += h
        y = y_new
        # 边界处理：保证浓度非负
        y = np.maximum(y, 0.0)
        ts.append(t)
        ys.append(y.copy())
        step += 1

    return np.array(ts), np.array(ys)


# ---------------------------------------------------------------------------
# PBPK 多 compartment 刚性系统
# ---------------------------------------------------------------------------

class PBPK_ODE_System:
    """
    构建一个 7-compartment PBPK ODE 系统：
    0: 动脉血 (Arterial Blood)
    1: 肝脏 (Liver)      — 含 Michaelis-Menten 代谢
    2: 肾脏 (Kidney)     — 含 GFR 清除
    3: 肌肉 (Muscle)     — 慢平衡
    4: 脂肪 (Adipose)    — 极慢平衡
    5: 肿瘤/靶组织 (Tumor)
    6: 静脉血 (Venous Blood)

    质量平衡方程（每 compartment）：
        V_i dC_i/dt = Q_i (C_art - C_i / Kp_i) - CL_i C_i + Input_i(t)
    其中 Kp_i 为组织-血分配系数，CL_i 为清除率。
    """

    def __init__(self, params: dict = None):
        if params is None:
            params = self.default_params()
        self.params = params
        self.V = params["V"]          # 分布容积 [L]
        self.Q = params["Q"]          # 血流 [L/min]
        self.Kp = params["Kp"]        # 分配系数
        self.CL = params["CL"]        # 清除率 [L/min]
        self.Vmax = params["Vmax"]    # Michaelis-Menten Vmax [mg/min]
        self.Km = params["Km"]        # Michaelis-Menten Km [mg/L]
        self.GFR = params["GFR"]      # 肾小球滤过率 [L/min]
        self.fu = params["fu"]        # 游离药物分数
        self.n_comp = 7

    @staticmethod
    def default_params() -> dict:
        """标准 70 kg 成年人的生理参数。"""
        return {
            "V": np.array([1.5, 1.5, 0.3, 30.0, 10.0, 0.5, 1.5]),   # L
            "Q": np.array([6.0, 1.5, 1.2, 0.9, 0.2, 0.3, 6.0]),      # L/min (arterial=total)
            "Kp": np.array([1.0, 2.5, 1.2, 0.7, 5.0, 1.5, 1.0]),     # 分配系数
            "CL": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),      # 基础清除
            "Vmax": 5.0,    # mg/min (肝脏代谢)
            "Km": 10.0,     # mg/L
            "GFR": 0.125,   # L/min
            "fu": 0.1,      # 游离分数
        }

    def rhs(self, t: float, C: np.ndarray) -> np.ndarray:
        """
        计算 dC/dt。
        C[0]=C_art, C[1]=C_liv, ..., C[6]=C_ven
        """
        if len(C) != self.n_comp:
            raise ValueError(f"C must have length {self.n_comp}")
        C = np.maximum(C, 0.0)  # 保证非负
        dCdt = np.zeros(self.n_comp)
        C_art = C[0]
        C_ven = C[6]

        # 肝脏：含 Michaelis-Menten 代谢
        # TODO: Hole 1 - 实现肝脏 compartment 的 Michaelis-Menten 代谢速率计算
        # 需要计算游离药物浓度 C_liv_free 和代谢速率 met_rate，然后更新 dCdt[1]
        raise NotImplementedError("Hole 1: Liver Michaelis-Menten metabolism not implemented")

        # 肾脏：含 GFR 清除
        # TODO: Hole 1 - 实现肾脏 compartment 的 GFR 清除速率计算
        # 需要计算游离药物浓度 C_kid_free 和清除速率 gfr_clear，然后更新 dCdt[2]
        raise NotImplementedError("Hole 1: Kidney GFR clearance not implemented")

        # 肌肉、脂肪、肿瘤
        for i in [3, 4, 5]:
            dCdt[i] = self.Q[i] * (C_art - C[i] / self.Kp[i]) / self.V[i]

        # 动脉血：接收静脉回流（假设即时混合 + 口服吸收）
        # 口服给药：一级吸收
        dose_rate = self._oral_input(t)
        venous_return = sum(self.Q[i] * C[i] / self.Kp[i] for i in range(1, 6))
        dCdt[0] = (dose_rate + venous_return - self.Q[0] * C_art) / self.V[0]

        # 静脉血
        dCdt[6] = (self.Q[0] * C_art - self.Q[0] * C_ven) / self.V[6]
        # 静脉与动脉的耦合修正（简化模型）
        dCdt[0] += self.Q[0] * (C_ven - C_art) / self.V[0]

        return dCdt

    def _oral_input(self, t: float) -> float:
        """一级吸收动力学：Dose * ka * exp(-ka * t)"""
        Dose = 100.0  # mg
        ka = 0.1      # 1/min
        return Dose * ka * np.exp(-ka * t) if t >= 0 else 0.0

    def jacobian(self, t: float, C: np.ndarray) -> np.ndarray:
        """数值 Jacobian（用于 Rosenbrock 方法）。"""
        n = self.n_comp
        eps = np.sqrt(np.finfo(float).eps)
        J = np.zeros((n, n))
        f0 = self.rhs(t, C)
        for j in range(n):
            C_pert = C.copy()
            C_pert[j] += eps * max(1.0, abs(C[j]))
            J[:, j] = (self.rhs(t, C_pert) - f0) / (C_pert[j] - C[j])
        return J


def solve_pbpk_ode(t_span: Tuple[float, float], C0: np.ndarray = None,
                   method: str = "rosenbrock", h_init: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
    """
    求解 PBPK ODE 系统。
    """
    system = PBPK_ODE_System()
    if C0 is None:
        C0 = np.zeros(system.n_comp)
    f = system.rhs
    J = system.jacobian

    # 包装 rosenbrock_step 以使用提供的 J
    def step_wrapper(t, y, h):
        return rosenbrock_step(f, t, y, h, J)

    if method == "rosenbrock":
        # 自定义步进
        t0, tf = t_span
        t = t0
        y = C0.astype(float).copy()
        ts = [t]
        ys = [y.copy()]
        h = h_init
        max_steps = 100000
        step = 0
        while t < tf and step < max_steps:
            h = min(h, tf - t)
            y_new = step_wrapper(t, y, h)
            y_new = np.maximum(y_new, 0.0)
            t += h
            y = y_new
            ts.append(t)
            ys.append(y.copy())
            step += 1
        return np.array(ts), np.array(ys)
    else:
        return solve_ode(f, t_span, C0, method=method, h_init=h_init)


# ---------------------------------------------------------------------------
# 模块自检
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # 验证 tough_ode
    t_span = (0.0, 1.0)
    y0 = tough_exact(0.0)
    ts, ys = solve_ode(tough_deriv, t_span, y0, method="rk4", h_init=0.001)
    y_final_exact = tough_exact(ts[-1])
    y_final_num = ys[-1]
    print(f"Tough ODE final error: {np.linalg.norm(y_final_num - y_final_exact):.4e}")

    # PBPK 求解
    ts2, ys2 = solve_pbpk_ode((0.0, 120.0), method="rosenbrock", h_init=0.1)
    print(f"PBPK solved: {len(ts2)} steps, final concentrations: {ys2[-1]}")
    print(f"Cmax liver: {np.max(ys2[:, 1]):.3f} mg/L")
    print(f"Cmax tumor: {np.max(ys2[:, 5]):.3f} mg/L")
