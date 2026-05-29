"""
压电超声换能器瞬态动力学与刚性ODE验证模块

基于种子项目 1164_stiff_ode 和 674_lindberg_exact 的核心算法，
为超声成像系统提供换能器振子的高精度瞬态响应模拟与数值验证。

物理模型:
压电换能器振子的运动方程可建模为阻尼受迫振动：
    m·ü + c·ů + k·u = F(t)
其中 m 为等效质量，c 为阻尼系数，k 为弹性刚度，F(t) 为压电驱动力。

在高Q值（低阻尼）换能器中，当驱动频率接近谐振频率时，系统呈现刚性特征，
需要隐式或半隐式数值方法以保证稳定性。

核心公式:
- 一阶ODE形式: y' = f(t, y)
- Lindberg刚性系统: 精确解 y(t) = exp(λt)·cos(ωt) + sin(ωt)
- 刚性比: S = |Re(λ_max)| / |Re(λ_min)|
"""

import numpy as np
from typing import Callable, Tuple, List


class StiffODESolver:
    """刚性ODE数值求解器，采用隐式梯形法（Crank-Nicolson）。
    
    隐式梯形法公式:
        y_{n+1} = y_n + (h/2) * [f(t_n, y_n) + f(t_{n+1}, y_{n+1})]
    
    对于线性问题 y' = A·y + g(t)，可显式求解：
        y_{n+1} = (I - hA/2)^{-1} · [y_n + (h/2)(A·y_n + g_n + g_{n+1})]
    
    该方法无条件稳定，适合刚性问题。
    """
    
    def __init__(self, A: np.ndarray, g: Callable[[float], np.ndarray] = None):
        """
        参数:
            A: 系统矩阵
            g: 非齐次项函数 g(t)，默认为零
        """
        self.A = A
        self.dim = A.shape[0]
        self.g = g if g is not None else lambda t: np.zeros(self.dim)
    
    def solve(self, y0: np.ndarray, t_span: Tuple[float, float],
              n_steps: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """求解ODE系统。
        
        参数:
            y0: 初始条件
            t_span: 时间区间 (t0, tf)
            n_steps: 时间步数
        
        返回:
            t: 时间网格
            y: 解矩阵 (n_steps+1, dim)
        """
        t0, tf = t_span
        h = (tf - t0) / n_steps
        t = np.linspace(t0, tf, n_steps + 1)
        
        y = np.zeros((n_steps + 1, self.dim))
        y[0] = y0
        
        # 预计算 (I - hA/2)^{-1}
        I = np.eye(self.dim)
        M = I - 0.5 * h * self.A
        
        # 数值鲁棒性：检查矩阵条件数
        cond_num = np.linalg.cond(M)
        if cond_num > 1e12:
            # 使用伪逆提高稳定性
            M_inv = np.linalg.pinv(M)
        else:
            M_inv = np.linalg.inv(M)
        
        for n in range(n_steps):
            tn = t[n]
            tnp1 = t[n + 1]
            yn = y[n]
            
            gn = self.g(tn)
            gnp1 = self.g(tnp1)
            
            # Crank-Nicolson步进
            rhs = yn + 0.5 * h * (self.A @ yn + gn + gnp1)
            y[n + 1] = M_inv @ rhs
        
        return t, y


def transducer_ode_system(freq: float = 5e6, Q: float = 50.0,
                          m_eff: float = 1e-6) -> Tuple[np.ndarray, Callable]:
    """构建压电换能器的等效电路/机械ODE系统矩阵。
    
    物理参数:
    - freq: 谐振频率 (Hz)，典型医学超声 1-15 MHz
    - Q: 品质因数，典型值 20-100
    - m_eff: 等效质量 (kg)
    
    谐振角频率: ω₀ = 2π·freq
    阻尼系数: c = m_eff·ω₀ / Q
    弹性刚度: k = m_eff·ω₀²
    
    状态空间表示 (u, v = ů):
        d/dt [u]   [  0      1  ] [u]   [    0    ]
             [v] = [ -k/m  -c/m ] [v] + [ F(t)/m ]
    
    返回:
        A: 2×2 系统矩阵
        forcing: 驱动力函数 F(t)/m
    """
    omega0 = 2.0 * np.pi * freq
    c = m_eff * omega0 / Q
    k = m_eff * omega0 ** 2
    
    A = np.array([[0.0, 1.0],
                  [-k / m_eff, -c / m_eff]])
    
    # 驱动力：压电脉冲激励
    def forcing(t: float) -> np.ndarray:
        pulse_width = 2.0 / freq  # 2个周期脉冲
        envelope = np.exp(-(t - pulse_width / 2) ** 2 / (2 * (pulse_width / 4) ** 2))
        if t < 0 or t > pulse_width * 2:
            envelope = 0.0
        F_over_m = envelope * np.sin(omega0 * t) / m_eff
        return np.array([0.0, F_over_m])
    
    return A, forcing


def stiff_ode_exact(t: float, lam: float = -5.0) -> float:
    """标准刚性ODE的精确解析解（Prothero-Robinson型）。
    
    模型方程: y' = λ(cos(t) - y)
    
    精确解（取初值 y(0) = λ²/(1+λ²) 消除瞬态发散项）:
        y(t) = λ·(λ·cos(t) + sin(t)) / (1 + λ²)
    
    此系统具有刚性特征：当 |λ| ≫ 1 时，隐式方法显著优于显式方法。
    精确解不含指数增长项，适合长时间积分验证。
    
    参数:
        t: 时间 (s)
        lam: 刚性参数 λ
    
    返回:
        y: 精确解
    """
    denom = 1.0 + lam**2
    return lam * (lam * np.cos(t) + np.sin(t)) / denom


def stiff_ode_derivative(t: float, y: float, lam: float = -5.0) -> float:
    """标准刚性ODE的右端项 f(t, y)。
    
    公式: f(t, y) = λ(cos(t) - y)
    """
    return lam * (np.cos(t) - y)


def stiff_ode_exact_array(t_array: np.ndarray, lam: float = -5.0) -> np.ndarray:
    """计算刚性ODE精确解在时间数组上的值。"""
    denom = 1.0 + lam**2
    return lam * (lam * np.cos(t_array) + np.sin(t_array)) / denom


def verify_stiff_solver(lam: float = -5.0,
                        t_span: Tuple[float, float] = (0.0, 1.0),
                        n_steps_list: List[int] = None) -> dict:
    """验证刚性ODE求解器对标准刚性系统的精度。
    
    模型: y' = λ(cos(t) - y)
    精确解: y(t) = λ·(λ·cos(t) + sin(t)) / (1 + λ²)
    
    通过比较数值解与精确解，计算L²误差和最大误差，
    验证隐式梯形法的收敛阶（理论为二阶）。
    
    返回:
        包含误差统计的字典
    """
    if n_steps_list is None:
        n_steps_list = [50, 100, 200, 400, 800]
    
    y0 = np.array([stiff_ode_exact(0.0, lam)])
    A = np.array([[-lam]])  # y' = -λ·y + λ·cos(t)
    
    def g_scalar(t: float) -> np.ndarray:
        return np.array([lam * np.cos(t)])
    
    errors_l2 = []
    errors_max = []
    
    for n_steps in n_steps_list:
        solver = StiffODESolver(A, g_scalar)
        t, y_num = solver.solve(y0, t_span, n_steps)
        y_exact = stiff_ode_exact_array(t, lam)
        
        diff = y_num[:, 0] - y_exact
        l2_error = np.sqrt(np.mean(diff ** 2))
        max_error = np.max(np.abs(diff))
        
        errors_l2.append(l2_error)
        errors_max.append(max_error)
    
    # 估计收敛阶
    convergence_orders = []
    for i in range(1, len(n_steps_list)):
        ratio = errors_l2[i - 1] / errors_l2[i]
        order = np.log2(ratio)
        convergence_orders.append(float(order))
    
    return {
        'n_steps': n_steps_list,
        'l2_errors': errors_l2,
        'max_errors': errors_max,
        'convergence_orders': convergence_orders,
        'stiffness_ratio': abs(lam)
    }


def simulate_transducer_response(freq: float = 5e6, Q: float = 50.0,
                                 t_span: Tuple[float, float] = (0.0, 10e-6),
                                 n_steps: int = 2000) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """模拟压电换能器的完整瞬态响应。
    
    返回:
        t: 时间数组 (s)
        displacement: 位移 u(t) (m)
        velocity: 速度 v(t) = ů(t) (m/s)
    """
    A, forcing = transducer_ode_system(freq, Q)
    
    y0 = np.array([0.0, 0.0])  # 零初始条件
    
    solver = StiffODESolver(A, forcing)
    t, y = solver.solve(y0, t_span, n_steps)
    
    displacement = y[:, 0]
    velocity = y[:, 1]
    
    return t, displacement, velocity


# 修复类型提示
def verify_stiff_solver_fix(lam: float = -5.0,
                            t_span: Tuple[float, float] = (0.0, 1.0),
                            n_steps_list: List[int] = None) -> dict:
    """验证刚性ODE求解器对标准刚性系统的精度。"""
    if n_steps_list is None:
        n_steps_list = [50, 100, 200, 400, 800]
    
    y0 = np.array([stiff_ode_exact(0.0, lam)])
    A = np.array([[-lam]])  # y' = -λ·y + λ·cos(t)
    
    def g_scalar(t: float) -> np.ndarray:
        return np.array([lam * np.cos(t)])
    
    errors_l2 = []
    errors_max = []
    
    for n_steps in n_steps_list:
        solver = StiffODESolver(A, g_scalar)
        t, y_num = solver.solve(y0, t_span, n_steps)
        y_exact = stiff_ode_exact_array(t, lam)
        
        diff = y_num[:, 0] - y_exact
        l2_error = np.sqrt(np.mean(diff ** 2))
        max_error = np.max(np.abs(diff))
        
        errors_l2.append(l2_error)
        errors_max.append(max_error)
    
    convergence_orders = []
    for i in range(1, len(n_steps_list)):
        ratio = errors_l2[i - 1] / errors_l2[i]
        order = np.log2(ratio)
        convergence_orders.append(float(order))
    
    return {
        'n_steps': n_steps_list,
        'l2_errors': errors_l2,
        'max_errors': errors_max,
        'convergence_orders': convergence_orders,
        'stiffness_ratio': abs(lam)
    }
