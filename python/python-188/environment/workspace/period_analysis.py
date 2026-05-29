"""
非线性振荡器周期分析模块：语义嵌入系统稳定性分析

原项目映射: 1387_vanderpol_ode_period, 909_predator_prey_ode_period

科学背景:
    1. Van der Pol 振荡器:
       x'' - mu*(1-x^2)*x' + x = 0
       描述自激振荡系统，在语义嵌入中可用于分析系统的周期性稳定行为。
       
       周期估计公式 (Urabe):
         p = (3 - 2*ln(2))*mu
             + 3*alpha/mu^(1/3)
             - (1/3)*ln(mu)/mu
             + (3*ln(2) - ln(3) - 1.5 + b0 - 2*d)/mu
         其中 alpha=2.338107, b0=0.1723, d=0.4889
    
    2. 捕食者-猎物 (Lotka-Volterra) 系统:
       du/dt =  alpha*u - beta*u*v
       dv/dt = -gamma*v + delta*u*v
       
       周期公式 (Shih):
         p = (1/(alpha*gamma)) * integral_0^E phi(s/gamma)*phi((E-s)/alpha) ds
         phi(s) = 1/(1+W_0(-exp(-1-s))) - 1/(1+W_{-1}(-exp(-1-s)))
         E = gamma*u0 - gamma*ln(u0) + alpha*v0 - alpha*ln(v0) - (alpha+gamma)

在NLP语义嵌入中的应用:
    将语义系统的演化视为非线性动力学系统，
    利用周期分析研究语义簇的振荡行为和长期稳定性。
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import lambertw
from scipy.integrate import quad


class VanDerPolSemanticOscillator:
    """
    Van der Pol 语义振荡器。
    
    模拟语义系统偏离平衡态后的自激振荡恢复过程。
    """

    def __init__(self, mu: float = 1.0):
        """
        Parameters
        ----------
        mu : float
            非线性阻尼系数，mu > 0。
        """
        if mu <= 0.0:
            raise ValueError(f"mu must be positive, got {mu}")
        self.mu = float(mu)

    def _derivative(self, t: float, y: np.ndarray) -> np.ndarray:
        """
        状态空间形式:
            y[0] = x, y[1] = x'
            dy[0]/dt = y[1]
            dy[1]/dt = mu*(1 - y[0]^2)*y[1] - y[0]
        """
        x, v = y[0], y[1]
        return np.array([
            v,
            self.mu * (1.0 - x ** 2) * v - x
        ])

    def period_estimate(self) -> float:
        """
        使用 Urabe 公式估计周期。
        
        公式:
            p = (3 - 2*ln(2))*mu
                + 3*alpha/mu^(1/3)
                - (1/3)*ln(mu)/mu
                + (3*ln(2) - ln(3) - 1.5 + b0 - 2*d)/mu
            
            alpha = 2.338107
            b0 = 0.1723
            d = 0.4889
        """
        mu = self.mu
        if mu == 0.0:
            return 2.0 * np.pi

        alpha = 2.338107
        b0 = 0.1723
        d = 0.4889

        p = ((3.0 - 2.0 * np.log(2.0)) * mu
             + 3.0 * alpha / (mu ** (1.0 / 3.0))
             - (1.0 / 3.0) * np.log(mu) / mu
             + (3.0 * np.log(2.0) - np.log(3.0) - 1.5 + b0 - 2.0 * d) / mu)
        return p

    def period_cartwright(self) -> float:
        """Cartwright 周期估计"""
        mu = self.mu
        if mu == 0.0:
            return 2.0 * np.pi
        return (3.0 - 2.0 * np.log(2.0)) * mu + 2.0 * np.pi / (mu ** (1.0 / 3.0))

    def period_cook(self) -> float:
        """Cook 周期估计"""
        mu = self.mu
        if mu == 0.0:
            return 2.0 * np.pi
        return (3.0 - 2.0 * np.log(2.0)) * mu

    def solve(self, t_span: tuple, y0: np.ndarray = None, num_points: int = 1000) -> tuple:
        """数值求解 Van der Pol 方程"""
        if y0 is None:
            y0 = np.array([2.0, 0.0])
        t_eval = np.linspace(t_span[0], t_span[1], num_points)
        sol = solve_ivp(
            fun=self._derivative,
            t_span=t_span,
            y0=y0,
            t_eval=t_eval,
            method='RK45',
            rtol=1e-9,
            atol=1e-12
        )
        return sol.t, sol.y

    def measure_period_numerical(self, t_span: tuple = (0.0, 50.0),
                                  num_points: int = 5000) -> float:
        """
        通过数值解测量周期。
        
        检测 x(t) 的零点穿越周期。
        """
        t, y = self.solve(t_span, num_points=num_points)
        x = y[0, :]

        # 找零点穿越 (从负到正)
        crossings = []
        for i in range(len(x) - 1):
            if x[i] < 0.0 and x[i + 1] >= 0.0:
                # 线性插值
                t_cross = t[i] + (t[i + 1] - t[i]) * (0.0 - x[i]) / (x[i + 1] - x[i])
                crossings.append(t_cross)

        if len(crossings) < 3:
            return float('nan')

        periods = np.diff(crossings[1:])  # 跳过第一个不完整周期
        return np.mean(periods) if len(periods) > 0 else float('nan')


class PredatorPreySemanticCycle:
    """
    捕食者-猎物语义周期分析器。
    
    将语义概念之间的此消彼长关系建模为 Lotka-Volterra 系统。
    """

    def __init__(self, alpha: float = 1.0, beta: float = 0.1,
                 gamma: float = 1.5, delta: float = 0.075,
                 u0: float = 10.0, v0: float = 5.0):
        """
        Parameters
        ----------
        alpha, beta, gamma, delta : float
            Lotka-Volterra 参数，必须为正。
        u0, v0 : float
            初始语义浓度 (猎物, 捕食者)。
        """
        for name, val in [('alpha', alpha), ('beta', beta), ('gamma', gamma), ('delta', delta)]:
            if val <= 0.0:
                raise ValueError(f"{name} must be positive, got {val}")
        for name, val in [('u0', u0), ('v0', v0)]:
            if val <= 0.0:
                raise ValueError(f"{name} must be positive, got {val}")

        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.delta = float(delta)
        self.u0 = float(u0)
        self.v0 = float(v0)

    def _derivative(self, t: float, y: np.ndarray) -> np.ndarray:
        u, v = y[0], y[1]
        return np.array([
            self.alpha * u - self.beta * u * v,
            -self.gamma * v + self.delta * u * v
        ])

    def conserved_energy(self, y: np.ndarray = None) -> float:
        """
        计算守恒量（Hamiltonian）。
        
        E = gamma*u - gamma*ln(u) + alpha*v - alpha*ln(v) - (alpha+gamma)
        """
        if y is None:
            y = np.array([self.u0, self.v0])
        u, v = y[0], y[1]
        return (self.gamma * u - self.gamma * np.log(u)
                + self.alpha * v - self.alpha * np.log(v)
                - (self.alpha + self.gamma))

    def _phi(self, s: float) -> float:
        """
        phi(s) 函数，用于周期积分。
        
        phi(s) = 1/(1+W_0(-exp(-1-s))) - 1/(1+W_{-1}(-exp(-1-s)))
        """
        z = -np.exp(-1.0 - s)
        # 处理 LambertW 的输入范围
        if z < -1.0 / np.e + 1e-10:
            return 0.0
        d1 = 1.0 + np.real(lambertw(0, z))
        d2 = 1.0 + np.real(lambertw(-1, z))
        if abs(d1) < 1e-15 or abs(d2) < 1e-15:
            return 0.0
        return float(1.0 / d1 - 1.0 / d2)

    def period_estimate(self) -> float:
        """
        使用 Shih 公式估计周期。
        
        p = (1/(alpha*gamma)) * integral_0^E phi(s/gamma)*phi((E-s)/alpha) ds
        """
        E = self.conserved_energy()
        if E <= 0.0:
            return float('nan')

        def integrand(s: float) -> float:
            return self._phi(s / self.gamma) * self._phi((E - s) / self.alpha)

        # 积分区间 [0, E]
        result, err = quad(integrand, 0.0, E, limit=100)
        period = result / (self.alpha * self.gamma)
        return period

    def solve(self, t_span: tuple = (0.0, 50.0), num_points: int = 1000) -> tuple:
        """数值求解 Lotka-Volterra 方程"""
        t_eval = np.linspace(t_span[0], t_span[1], num_points)
        y0 = np.array([self.u0, self.v0])
        sol = solve_ivp(
            fun=self._derivative,
            t_span=t_span,
            y0=y0,
            t_eval=t_eval,
            method='RK45',
            rtol=1e-9,
            atol=1e-12
        )
        return sol.t, sol.y

    def measure_period_numerical(self, t_span: tuple = (0.0, 100.0),
                                  num_points: int = 5000) -> float:
        """通过数值解测量 u(t) 的周期"""
        t, y = self.solve(t_span, num_points=num_points)
        u = y[0, :]

        # 找局部极大值
        peaks = []
        for i in range(1, len(u) - 1):
            if u[i - 1] < u[i] and u[i] > u[i + 1]:
                # 抛物线插值
                t_peak = t[i]
                peaks.append(t_peak)

        if len(peaks) < 3:
            return float('nan')

        periods = np.diff(peaks[1:])
        return np.mean(periods) if len(periods) > 0 else float('nan')


def demo():
    """模块功能演示"""
    print("=" * 60)
    print("语义系统非线性振荡器周期分析")
    print("=" * 60)

    # Van der Pol
    print("\n--- Van der Pol 语义振荡器 ---")
    for mu in [0.5, 1.0, 2.0, 5.0]:
        vdp = VanDerPolSemanticOscillator(mu=mu)
        p_est = vdp.period_estimate()
        p_cook = vdp.period_cook()
        p_cart = vdp.period_cartwright()
        print(f"\nmu = {mu}:")
        print(f"  Cook估计:     {p_cook:.6f}")
        print(f"  Cartwright估计: {p_cart:.6f}")
        print(f"  Urabe估计:    {p_est:.6f}")

        if mu <= 2.0:
            p_num = vdp.measure_period_numerical(t_span=(0.0, 100.0))
            print(f"  数值测量:     {p_num:.6f}")
            if not np.isnan(p_num):
                print(f"  相对偏差:     {abs(p_est - p_num) / p_num:.6e}")

    # Predator-Prey
    print("\n--- 捕食者-猎物语义周期分析 ---")
    pp = PredatorPreySemanticCycle(alpha=1.0, beta=0.1, gamma=1.5, delta=0.075,
                                   u0=10.0, v0=5.0)
    E = pp.conserved_energy()
    p_est = pp.period_estimate()
    p_num = pp.measure_period_numerical(t_span=(0.0, 100.0))

    print(f"\n参数: alpha={pp.alpha}, beta={pp.beta}, gamma={pp.gamma}, delta={pp.delta}")
    print(f"初始: u0={pp.u0}, v0={pp.v0}")
    print(f"守恒能量 E = {E:.6f}")
    print(f"Shih周期估计 = {p_est:.6f}")
    print(f"数值测量周期 = {p_num:.6f}")
    if not np.isnan(p_num) and p_num > 0:
        print(f"相对偏差 = {abs(p_est - p_num) / p_num:.6e}")

    print("\n模块运行完成")
    return vdp, pp


if __name__ == "__main__":
    demo()
