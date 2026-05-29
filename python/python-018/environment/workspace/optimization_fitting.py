"""
optimization_fitting.py

基于种子项目 1266_toms178（Hooke-Jeeves直接搜索优化）和
1269_toms291（Gamma函数对数计算），实现拓扑超导实验数据的
非线性拟合与费米积分计算。

物理模型：
    1) 隧道电导谱拟合：
        利用BTK理论（Blonder-Tinkham-Klapwijk）模型拟合
        正常金属-超导界面的微分电导：

            G(V) = G_N * [1 + P_A(E=eV) - R_N(E=eV)]

        其中P_A为Andreev反射概率，R_N为正常反射概率。
        对于含有马约拉纳零能模的拓扑超导体，零偏压电导
        呈现量子化平台：
            G(0) = 2e^2/h

    2) 费米积分：
        在有限温度下，准粒子占据数由Fermi-Dirac分布给出：
            f(E) = 1 / (exp((E-μ)/k_B T) + 1)

        n阶费米积分定义为：
            F_n(η) = ∫_0^∞ x^n / (exp(x-η) + 1) dx

        其中η = μ/k_B T为约化化学势。
        费米积分与Gamma函数的关系：
            F_n(η) → Γ(n+1) * exp(η)  （η << -1，经典极限）
            F_n(η) → η^{n+1}/(n+1)    （η >> 1，简并极限）
"""

import numpy as np
from typing import Callable, Tuple, Optional


class GammaLogCalculator:
    """
    Gamma函数对数计算器（基于1269_toms291的alogam算法）。
    """

    @staticmethod
    def alogam(x: float) -> Tuple[float, int]:
        """
        计算ln(Γ(x))，使用Stirling展开修正。

        算法步骤：
            1) 若x<7，使用递推 Γ(x) = Γ(x+k)/(x(x+1)...(x+k-1))
               将参数提升到x+k≥7
            2) 对y≥7应用Stirling公式：
                ln Γ(y) ≈ (y-0.5)ln(y) - y + 0.918938533204673
                         + (-1/360y^3 + 1/1260y^5 - ...) / y
        """
        if x <= 0.0:
            return 0.0, 1

        ifault = 0
        y = float(x)

        if x < 7.0:
            f = 1.0
            z = y
            while z < 7.0:
                f *= z
                z += 1.0
            y = z
            f = -np.log(f)
        else:
            f = 0.0

        z = 1.0 / (y * y)
        value = (f + (y - 0.5) * np.log(y) - y + 0.918938533204673
                 + (((-0.000595238095238 * z
                      + 0.000793650793651) * z
                     - 0.002777777777778) * z
                    + 0.083333333333333) / y)

        return value, ifault

    @staticmethod
    def gamma(x: float) -> float:
        """
        计算Γ(x)。
        """
        log_g, err = GammaLogCalculator.alogam(x)
        if err != 0:
            return 0.0
        return np.exp(log_g)

    @staticmethod
    def factorial(n: int) -> float:
        """
        计算n! = Γ(n+1)。
        """
        if n < 0:
            return 0.0
        if n <= 1:
            return 1.0
        return GammaLogCalculator.gamma(n + 1.0)


class FermiIntegralCalculator:
    """
    费米积分计算器。
    """

    def __init__(self):
        self.gamma_calc = GammaLogCalculator()

    def fermi_integral(self, n: int, eta: float,
                        num_points: int = 10000) -> float:
        """
        数值计算n阶费米积分F_n(η)。

        采用Gauss-Laguerre或简单积分方法。
        对于η >> 0，被积函数在x≈η处有尖锐跃变。
        """
        if n < 0:
            raise ValueError("阶数n必须非负")

        # 积分上限自适应选择
        if eta > 0:
            x_max = max(eta + 20.0, 50.0)
        else:
            x_max = max(20.0, -eta + 20.0)

        x = np.linspace(0.0, x_max, num_points)
        dx = x[1] - x[0]

        # Fermi-Dirac分布
        denom = np.exp(x - eta) + 1.0
        # 避免溢出
        denom = np.where(denom > 1e300, 1e300, denom)
        denom = np.where(denom < 1e-300, 1e-300, denom)

        integrand = x ** n / denom
        integral = np.trapezoid(integrand, x)

        return float(integral)

    def asymptotic_classical(self, n: int, eta: float) -> float:
        """
        经典极限（η << -1）：
            F_n(η) ≈ Γ(n+1) exp(η)
        """
        gamma_val = self.gamma_calc.gamma(n + 1.0)
        return gamma_val * np.exp(eta)

    def asymptotic_degenerate(self, n: int, eta: float) -> float:
        """
        简并极限（η >> 1）：
            F_n(η) ≈ η^{n+1} / (n+1)
        """
        return eta ** (n + 1.0) / (n + 1.0)

    def sommerfeld_expansion(self, n: int, eta: float) -> float:
        """
        Sommerfeld展开（低温修正）：
            F_n(η) = η^{n+1}/(n+1) [1 + Σ_{k=1}^∞ c_k (πk_B T/μ)^{2k}]
        这里简化为只保留首项修正。
        """
        main = eta ** (n + 1.0) / (n + 1.0)
        correction = 1.0 + (n * (n + 1.0) * np.pi ** 2) / (6.0 * eta ** 2)
        return main * correction


class HookeJeevesOptimizer:
    """
    Hooke-Jeeves直接搜索优化算法（基于1266_toms178）。

    用于拟合实验数据（如隧道电导谱）到理论模型，
    无需目标函数的导数信息。
    """

    def __init__(self, rho: float = 0.5, eps: float = 1e-6,
                 itermax: int = 5000):
        """
        初始化优化器。

        Args:
            rho: 步长缩减因子 (0<rho<1)
            eps: 收敛阈值
            itermax: 最大迭代次数
        """
        if not (0.0 < rho < 1.0):
            raise ValueError("rho必须在(0,1)之间")
        self.rho = rho
        self.eps = eps
        self.itermax = itermax

    def _best_nearby(self, delta: np.ndarray, point: np.ndarray,
                     prev_best: float, nvars: int,
                     f: Callable, funevals: int) -> Tuple[float,
                                                           np.ndarray, int]:
        """
        沿各坐标轴寻找最优邻近点。
        """
        z = np.copy(point)
        best_f = prev_best

        for i in range(nvars):
            z[i] = point[i] + delta[i]
            ftmp = f(z)
            funevals += 1

            if ftmp < best_f:
                best_f = ftmp
            else:
                delta[i] = -delta[i]
                z[i] = point[i] + delta[i]
                ftmp = f(z)
                funevals += 1
                if ftmp < best_f:
                    best_f = ftmp
                else:
                    z[i] = point[i]

        return best_f, z, funevals

    def minimize(self, f: Callable, startpt: np.ndarray) -> Tuple[np.ndarray,
                                                                    float, int]:
        """
        最小化标量函数f。

        Returns:
            endpt: 最优参数点
            fmin: 最优函数值
            iters: 迭代次数
        """
        nvars = len(startpt)
        newx = np.copy(startpt)
        xbefore = np.copy(startpt)

        delta = np.zeros(nvars)
        for i in range(nvars):
            if abs(startpt[i]) < 1e-15:
                delta[i] = self.rho
            else:
                delta[i] = self.rho * abs(startpt[i])

        funevals = 0
        steplength = self.rho
        iters = 0
        fbefore = f(newx)
        funevals += 1
        newf = fbefore

        while iters < self.itermax and self.eps < steplength:
            iters += 1
            newx = np.copy(xbefore)
            newf, newx, funevals = self._best_nearby(
                delta, newx, fbefore, nvars, f, funevals)

            keep = True
            while newf < fbefore and keep:
                for i in range(nvars):
                    if newx[i] <= xbefore[i]:
                        delta[i] = -abs(delta[i])
                    else:
                        delta[i] = abs(delta[i])
                    tmp = xbefore[i]
                    xbefore[i] = newx[i]
                    newx[i] = newx[i] + newx[i] - tmp

                fbefore = newf
                newf, newx, funevals = self._best_nearby(
                    delta, newx, fbefore, nvars, f, funevals)

                if fbefore <= newf:
                    break

                keep = False
                for i in range(nvars):
                    if 0.5 * abs(delta[i]) < abs(newx[i] - xbefore[i]):
                        keep = True
                        break

            if self.eps <= steplength and fbefore <= newf:
                steplength *= self.rho
                delta *= self.rho

        endpt = np.copy(xbefore)
        return endpt, fbefore, iters


class BTKFittingModel:
    """
    BTK理论隧道电导谱模型。
    """

    def __init__(self, voltages: np.ndarray,
                 conductance_data: np.ndarray):
        """
        初始化实验数据。

        Args:
            voltages: 偏压数组V (meV)
            conductance_data: 微分电导G (2e^2/h为单位)
        """
        self.V = np.asarray(voltages, dtype=np.float64)
        self.G_data = np.asarray(conductance_data, dtype=np.float64)

    def btk_conductance(self, V: np.ndarray,
                         delta: float,
                         barrier_strength: float,
                         gamma: float) -> np.ndarray:
        """
        简化的BTK电导模型。

        参数：
            delta: 超导能隙 (meV)
            barrier_strength: 界面势垒强度Z
            gamma: 展宽参数 (meV，模拟非弹性散射)

        公式：
            G(V) = (1 + Z^2) / |A - B*Z^2|^2
            其中A,B为Andreev反射和正常反射振幅。

        对于含有马约拉纳的系统，额外添加零偏压峰：
            G_MZM(V) = G_0 * γ^2 / (V^2 + γ^2)
        """
        z2 = barrier_strength ** 2
        e = np.abs(V)

        # 标准BTK
        denom = np.sqrt(np.maximum(e ** 2 - delta ** 2, 0.0) + 1e-15)
        u2 = 0.5 * (1.0 + np.sqrt(np.maximum(e ** 2 - delta ** 2, 0.0))
                    / (np.abs(e) + 1e-15))
        u2 = np.clip(u2, 0.0, 1.0)
        v2 = 1.0 - u2

        a = u2
        b = v2
        g_btk = (1.0 + z2) / ((a - b * z2) ** 2 + 1e-15)
        g_btk = np.clip(g_btk, 0.0, 10.0)

        # 马约拉纳零偏压峰（拓扑特征）
        g_mzm = gamma ** 2 / (V ** 2 + gamma ** 2 + 1e-15)

        # 总电导
        g_total = g_btk + 0.5 * g_mzm
        return g_total

    def objective(self, params: np.ndarray) -> float:
        """
        最小二乘目标函数。
        """
        delta, z, gamma = params[0], params[1], params[2]

        # 参数边界
        if delta < 0.01 or delta > 5.0:
            return 1e10
        if z < 0.0 or z > 10.0:
            return 1e10
        if gamma < 0.001 or gamma > 1.0:
            return 1e10

        g_model = self.btk_conductance(self.V, delta, z, gamma)
        residual = self.G_data - g_model
        return float(np.sum(residual ** 2))

    def fit(self, initial_guess: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        拟合实验数据。

        Returns:
            params: [delta, Z, gamma]
            rmse: 均方根误差
        """
        optimizer = HookeJeevesOptimizer(rho=0.85, eps=1e-8,
                                          itermax=3000)
        best_params, fmin, iters = optimizer.minimize(
            self.objective, initial_guess)

        rmse = np.sqrt(fmin / len(self.V))
        return best_params, rmse


def demo():
    """演示优化拟合。"""
    # Gamma函数
    gl = GammaLogCalculator()
    print("ln Γ(5.5) =", gl.alogam(5.5)[0])
    print("Γ(5) =", gl.factorial(4))

    # 费米积分
    fi = FermiIntegralCalculator()
    for eta in [-5.0, 0.0, 5.0]:
        f_val = fi.fermi_integral(n=1, eta=eta)
        print(f"F_1({eta}) = {f_val:.4f}")

    # BTK拟合
    V = np.linspace(-2.0, 2.0, 81)
    # 模拟数据
    true_delta, true_z, true_gamma = 0.8, 0.5, 0.05
    model = BTKFittingModel(V, np.zeros_like(V))
    g_true = model.btk_conductance(V, true_delta, true_z, true_gamma)
    # 加噪声
    g_noisy = g_true + 0.02 * np.random.randn(len(V))

    model.G_data = g_noisy
    params, rmse = model.fit(initial_guess=np.array([0.6, 0.3, 0.03]))
    print(f"Fitted: delta={params[0]:.4f}, Z={params[1]:.4f}, gamma={params[2]:.4f}")
    print(f"RMSE: {rmse:.6f}")


if __name__ == "__main__":
    demo()
