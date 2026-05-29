"""
phase_diagram.py
相图分析与过渡动力学模块

本模块分析脂质双分子层凝胶-液晶相变的相图，包括:
  - Newton-Maehly 根求法确定自洽过渡温度
  - Duffing 型非线性振荡器描述膜的集体厚度涨落

参考种子项目: 801_newton_maehly (多项式根求法)
                322_duffing_ode (Duffing 非线性振荡器)

物理背景:
    1. 自洽场过渡温度:
       取向序参数 S 满足自洽方程:
           S = I_1(β J S) / I_0(β J S)
       其中 I_n 为修正 Bessel 函数。
       定义 f(S; T) = S - I_1(β J S)/I_0(β J S) = 0。
       当温度 T 从低温升高时，非零根在 T=T_c 处消失。
       用 Newton-Maehly 方法同时追踪多个温度下的根。

    2. 膜厚度涨落（Duffing 型）:
       膜局部厚度 d(t) 满足受驱阻尼非线性振荡:
           d'' + δ d' + α d + β d³ = γ cos(ω t) + ξ(t)
       其中:
         - δ: 粘性阻尼（来自周围水环境）
         - α: 线性恢复力（膜张力）
         - β: 非线性硬化（脂质链排斥）
         - γ cos(ω t): 外部周期驱动（声波或电场）
         - ξ(t): 热噪声

       该方程 exhibits 丰富的动力学行为：
         - 弱非线性: 简谐振荡
         - 强非线性: 周期倍增、混沌
         - 在相变附近 α→0，系统进入软模（soft mode）区域
"""

import numpy as np
from scipy.special import i0e, i1e


class NewtonMaehlySolver:
    """
    Newton-Maehly 多项式根求法（受种子项目 801_newton_maehly 启发）。

    对于多项式 P(z) = c_0 + c_1 z + ... + c_d z^d，
    同时求全部 d 个根，利用已收敛根的偏移避免重根:
        z_i^{new} = z_i - P(z_i) / [P'(z_i) - P(z_i) * Σ_{j≠i} 1/(z_i - z_j)]
    """

    def __init__(self, coeffs, max_iter=100, tol=1e-12):
        """
        Parameters
        ----------
        coeffs : ndarray
            多项式系数，c[0] + c[1]z + ... + c[d]z^d。
        max_iter : int
            最大迭代次数。
        tol : float
            收敛容差。
        """
        self.coeffs = np.asarray(coeffs, dtype=complex)
        self.max_iter = max_iter
        self.tol = tol

    def poly_and_derivative(self, z):
        """
        用 Horner 法则计算 P(z) 和 P'(z)。
        """
        z = complex(z)
        p = self.coeffs[-1]
        dp = 0.0 + 0.0j
        for c in self.coeffs[-2::-1]:
            dp = dp * z + p
            p = p * z + c
        return p, dp

    def solve(self):
        """
        求解全部根。

        Returns
        -------
        roots : ndarray (complex)
            d 个根。
        """
        d = len(self.coeffs) - 1
        if d <= 0:
            return np.array([])

        # Cauchy 界: R = 1 + max|c_i/c_d|
        cd = self.coeffs[-1]
        if abs(cd) < 1e-15:
            raise ValueError("最高次项系数接近零。")
        radius = 1.0 + np.max(np.abs(self.coeffs[:-1] / cd))

        # 初始猜测: 根 of unity
        theta = np.linspace(0.0, 2.0 * np.pi, d, endpoint=False)
        roots = radius * np.exp(1j * theta)

        for iteration in range(self.max_iter):
            roots_old = roots.copy()
            for i in range(d):
                pz, dpz = self.poly_and_derivative(roots[i])
                s = 0.0 + 0.0j
                for j in range(d):
                    if j != i:
                        diff = roots[i] - roots[j]
                        if abs(diff) > 1e-15:
                            s += 1.0 / diff
                denom = dpz - pz * s
                if abs(denom) < 1e-15:
                    continue
                roots[i] = roots[i] - pz / denom

            max_change = np.max(np.abs(roots - roots_old))
            max_poly = np.max(np.abs([self.poly_and_derivative(r)[0] for r in roots]))
            if max_change < self.tol and max_poly < self.tol * 10:
                return roots

        return roots


class SelfConsistentTransition:
    """
    平均场自洽相变分析。
    """

    def __init__(self, J_coupling=2.5, kb=0.008314):
        self.J = J_coupling
        self.kb = kb

    def sc_equation(self, S, T):
        """
        自洽方程残差:
            f(S) = S - B(β J S)
        其中 B(x) = I_1(x)/I_0(x) 为 Langevin 型函数。
        利用缩放 Bessel 函数 i0e(x)=I_0(x)exp(-|x|), i1e(x)=I_1(x)exp(-|x|)
        避免溢出。
        """
        if T <= 0 or S < 0:
            return np.inf
        beta = 1.0 / (self.kb * T)
        x = beta * self.J * S
        # 使用缩放 Bessel: I1/I0 = i1e/i0e
        if abs(x) > 700:
            # 大参数渐近: I1/I0 → 1 - 1/(2x) + ...
            B = 1.0 - 1.0 / (2.0 * x)
        else:
            B = i1e(x) / i0e(x)
        return S - B

    def find_roots_vs_temperature(self, T_values):
        """
        对每个温度 T 求自洽方程 f(S)=0 的正根。

        策略:
          - 低温区: 存在非零根 S>0（有序相）
          - 高温区: 唯一根 S=0（无序相）
          - 在 T_c 附近，非零根趋于零

        逐点二分搜索，从 S=0.01 开始以避免零根。
        """
        roots = []
        for T in T_values:
            # 二分搜索正根，避开 S=0
            a, b = 0.01, 1.0
            fa = self.sc_equation(a, T)
            fb = self.sc_equation(b, T)
            if fa * fb > 0:
                # 无正根或唯一根在边界
                roots.append(0.0)
                continue
            for _ in range(80):
                c = (a + b) / 2.0
                fc = self.sc_equation(c, T)
                if abs(fc) < 1e-12:
                    break
                if fa * fc <= 0:
                    b = c
                    fb = fc
                else:
                    a = c
                    fa = fc
            roots.append((a + b) / 2.0)
        return np.array(roots)

    def critical_temperature(self):
        """
        临界温度解析估计 (2D Maier-Saupe 平均场):
            T_c = J / (2 k_B)
        由小 S 展开: S ≈ β J S / 2 ⇒ β_c J = 2
        """
        # TODO: 请补全 2D Maier-Saupe 平均场的临界温度解析公式
        # 提示: 在小 S 展开下，自洽方程线性化给出 β_c J = 2
        raise NotImplementedError("critical_temperature 方法需要补全")


class DuffingMembraneDynamics:
    """
    膜厚度涨落的 Duffing 型非线性动力学（受种子项目 322_duffing_ode 启发）。

    方程:
        d'' + δ d' + α d + β d³ = γ cos(ω t) + √(2D) ξ(t)
    状态变量: y = [d, v]
        dy1/dt = y2
        dy2/dt = -δ y2 - α y1 - β y1³ + γ cos(ω t) + noise
    """

    def __init__(self, delta=0.3, alpha=1.0, beta=-1.0, gamma=0.5,
                 omega=1.2, noise_amp=0.1, seed=None):
        self.delta = delta
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.omega = omega
        self.noise_amp = noise_amp
        self.rng = np.random.default_rng(seed)

    def deriv(self, t, y):
        """
        计算 dy/dt。
        """
        y1, y2 = y
        noise = self.noise_amp * self.rng.normal()
        dy1dt = y2
        dy2dt = (-self.delta * y2 - self.alpha * y1 -
                 self.beta * y1 ** 3 + self.gamma * np.cos(self.omega * t) +
                 noise)
        return np.array([dy1dt, dy2dt])

    def integrate_rk4(self, y0, t_span, n_steps=5000):
        """
        四阶 Runge-Kutta 积分。
        """
        t0, tf = t_span
        dt = (tf - t0) / n_steps
        t_values = np.linspace(t0, tf, n_steps + 1)
        y_values = np.zeros((n_steps + 1, 2))
        y_values[0] = y0
        y = np.array(y0, dtype=float)

        for i in range(n_steps):
            t = t_values[i]
            k1 = self.deriv(t, y)
            k2 = self.deriv(t + 0.5 * dt, y + 0.5 * dt * k1)
            k3 = self.deriv(t + 0.5 * dt, y + 0.5 * dt * k2)
            k4 = self.deriv(t + dt, y + dt * k3)
            y = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            y_values[i + 1] = y

        return t_values, y_values

    def lyapunov_exponent_estimate(self, y0, t_span, n_steps=5000, perturbation=1e-8):
        """
        最大 Lyapunov 指数估计（判断混沌）。

        λ_max ≈ (1/t) ln(|δy(t)| / |δy(0)|)
        """
        t, y = self.integrate_rk4(y0, t_span, n_steps)
        y_perturbed = y0 + np.array([perturbation, 0.0])
        yp = y_perturbed.copy()
        dt = (t_span[1] - t_span[0]) / n_steps

        for i in range(n_steps):
            t_i = t[i]
            k1 = self.deriv(t_i, yp)
            k2 = self.deriv(t_i + 0.5 * dt, yp + 0.5 * dt * k1)
            k3 = self.deriv(t_i + 0.5 * dt, yp + 0.5 * dt * k2)
            k4 = self.deriv(t_i + dt, yp + dt * k3)
            yp = yp + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        delta_init = perturbation
        delta_final = np.linalg.norm(yp - y[-1])
        if delta_final <= 0 or delta_init <= 0:
            return -np.inf
        lam = np.log(delta_final / delta_init) / (t_span[1] - t_span[0])
        return float(lam)


class PhaseDiagramBuilder:
    """
    构建脂质双分子层的相图 (T, S, P)。
    """

    def __init__(self, J=2.5, kb=0.008314):
        self.J = J
        self.kb = kb
        self.sc = SelfConsistentTransition(J, kb)

    def build_diagram(self, T_range=(250, 400), n_T=50):
        """
        计算温度-序参数相图。

        Returns
        -------
        T_vals : ndarray
        S_vals : ndarray
        P_vals : ndarray  （压力近似: P = -∂F/∂V）
        """
        T_vals = np.linspace(T_range[0], T_range[1], n_T)
        S_vals = self.sc.find_roots_vs_temperature(T_vals)

        # 近似压力（通过面积变化）
        # P ≈ k_B T / A - κ (A - A0) / A0
        A0 = 0.64
        kappa = 25.0
        A_vals = A0 * (1.0 + 0.1 * (1.0 - S_vals))
        P_vals = self.kb * T_vals / A_vals - kappa * (A_vals - A0) / A0

        return T_vals, S_vals, P_vals

    def latent_heat(self, Tc, S_gel, S_fluid):
        """
        相变潜热:
            ΔH = T_c * ΔS * (∂S/∂T)_{T_c}
        近似:
            ΔH ≈ J * (S_gel² - S_fluid²) / 2
        """
        return 0.5 * self.J * (S_gel ** 2 - S_fluid ** 2)
