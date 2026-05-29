"""
Autoregressive Time Series Predictor with Characteristic Root Analysis
======================================================================
源自种子项目 801_newton_maehly (Newton-Maehly 同时求多项式全部根)。

AR(p) 模型：
    x_t + a_1 x_{t-1} + a_2 x_{t-2} + ... + a_p x_{t-p} = e_t
其中 e_t ~ WN(0, sigma_e^2) 为白噪声。

特征多项式：
    P(z) = z^p + a_1 z^{p-1} + a_2 z^{p-2} + ... + a_p = 0
或等价地，用反向形式：
    A(z) = 1 + a_1 z^{-1} + ... + a_p z^{-p} = 0

稳定性与预测性：
- 若 A(z) 的所有根位于单位圆内（|z| < 1），则 AR 过程是因果平稳的。
- 根的模长倒数决定了系统的衰减/振荡模式：
    * 实根 r ∈ (0,1) 对应指数衰减 e^{-t/τ}, τ = -1/ln(r)
    * 复根对 re^{±iθ} 对应阻尼振荡，频率 f = θ/(2πΔt)，阻尼系数 ζ = -ln(r)/sqrt(ln(r)^2 + θ^2)

Newton-Maehly 算法：
    z_i^{(k+1)} = z_i^{(k)} - P(z_i^{(k)}) / [P'(z_i^{(k)}) - P(z_i^{(k)}) * sum_{j≠i} 1/(z_i^{(k)} - z_j^{(k)})]
该 deflation 项防止多个迭代点收敛到同一根。

Cauchy 界：所有根满足 |z| <= 1 + max(|a_1|, ..., |a_p|)。
"""

import numpy as np
from typing import Tuple


class ARPredictor:
    """
    基于 AR(p) 模型的时间序列预测器，集成 Newton-Maehly 根分析。
    """

    def __init__(self, order: int = 5, max_iter: int = 200, tol: float = 1e-12):
        if order < 1:
            raise ValueError("AR order must be >= 1.")
        self.p = order
        self.max_iter = max_iter
        self.tol = tol
        self.ar_coefs: np.ndarray | None = None
        self.sigma_e: float = 1.0
        self.roots: np.ndarray | None = None

    def fit(self, series: np.ndarray) -> "ARPredictor":
        """
        使用 Yule-Walker 方法估计 AR 参数。
        自相关通过有偏估计保证 Toeplitz 正定性。
        """
        if len(series) < 2 * self.p:
            raise ValueError(f"Series length {len(series)} too short for order {self.p}.")

        n = len(series)
        mean_val = np.mean(series)
        x = series - mean_val

        # 有偏自相关估计（保证正定性）
        autocorr = np.zeros(self.p + 1)
        autocorr[0] = np.dot(x, x) / n
        for lag in range(1, self.p + 1):
            autocorr[lag] = np.dot(x[:-lag], x[lag:]) / n

        if autocorr[0] <= 1e-15:
            raise ValueError("Series variance is zero or negative.")

        # 归一化自相关
        rho = autocorr / autocorr[0]

        # 使用 Levinson-Durbin 求解
        from toeplitz_solver import ToeplitzSolver
        solver = ToeplitzSolver()
        a, k = solver.solve_yule_walker(autocorr)
        self.ar_coefs = a.copy()

        # 估计噪声方差
        self.sigma_e = np.sqrt(max(autocorr[0] * (1.0 - np.dot(rho[1:], a)), 1e-15))

        # 特征多项式根分析
        # P(z) = z^p + a_1 z^{p-1} + ... + a_p
        poly = np.concatenate(([1.0], a))
        self.roots = self._newton_maehly_roots(poly)

        return self

    def predict(self, history: np.ndarray, steps: int = 1) -> np.ndarray:
        """
        多步递推预测。
        x_{t+h} = -sum_{i=1}^p a_i x_{t+h-i}
        其中对于 h-i <= 0，使用历史值；对于 h-i > 0，使用已预测值。
        """
        if self.ar_coefs is None:
            raise RuntimeError("Model not fitted yet.")
        if len(history) < self.p:
            raise ValueError(f"History length {len(history)} < AR order {self.p}.")

        pred = np.zeros(steps)
        buf = np.concatenate([history[-self.p:].copy(), pred])
        for h in range(steps):
            idx = self.p + h
            pred[h] = -np.dot(self.ar_coefs, buf[idx - self.p:idx][::-1])
            buf[idx] = pred[h]
        return pred

    def _poly_and_derivative(self, coefs: np.ndarray, z: np.complex128) -> Tuple[np.complex128, np.complex128]:
        """
        Horner 法则同时求多项式值与导数值。
        P(z) = c_0 z^n + c_1 z^{n-1} + ... + c_n
        反向递推：
            p = c_0; d = 0
            for k=1..n: d = z*d + p; p = z*p + c_k
        """
        p = coefs[0]
        d = 0.0 + 0.0j
        for c in coefs[1:]:
            d = z * d + p
            p = z * p + c
        return p, d

    def _newton_maehly_roots(self, coefs: np.ndarray) -> np.ndarray:
        """
        Newton-Maehly 算法同时求多项式全部复根。
        """
        p = len(coefs) - 1
        if p == 0:
            return np.array([])
        if p == 1:
            return np.array([-coefs[1] / coefs[0]])

        # Cauchy 界
        cauchy_bound = 1.0 + np.max(np.abs(coefs[1:])) / abs(coefs[0])

        # 初始猜测：单位根缩放
        roots = cauchy_bound * np.exp(2j * np.pi * np.arange(p) / p + 1j * np.pi / (2 * p))

        for _ in range(self.max_iter):
            max_diff = 0.0
            for i in range(p):
                z = roots[i]
                pz, dpz = self._poly_and_derivative(coefs, z)

                if abs(pz) < self.tol:
                    continue

                # deflation term
                deflate = 0.0 + 0.0j
                for j in range(p):
                    if i == j:
                        continue
                    diff = z - roots[j]
                    if abs(diff) < self.tol:
                        continue
                    deflate += 1.0 / diff

                denom = dpz - pz * deflate
                if abs(denom) < 1e-15:
                    continue

                dz = pz / denom
                roots[i] -= dz
                max_diff = max(max_diff, abs(dz))

            if max_diff < self.tol:
                break

        return roots

    def stability_analysis(self) -> dict:
        """
        基于特征根进行稳定性与模态分析。
        """
        if self.roots is None or len(self.roots) == 0:
            return {"stable": True, "modes": []}

        modes = []
        stable = True
        for r in self.roots:
            mod = abs(r)
            if mod > 1.0 + 1e-8:
                stable = False
            # 阻尼时间常数
            if mod > 0 and mod < 1.0:
                tau = -1.0 / np.log(mod) if mod > 0 else np.inf
            else:
                tau = np.inf
            # 振荡频率
            angle = np.angle(r)
            freq = abs(angle) / (2.0 * np.pi)
            modes.append({
                "root": r,
                "modulus": mod,
                "time_constant": tau,
                "frequency": freq,
                "damping_ratio": -np.log(mod) / np.sqrt(np.log(mod) ** 2 + angle ** 2) if mod > 0 else 1.0
            })
        return {"stable": stable, "modes": modes}

    def forecast_interval(self, history: np.ndarray, steps: int = 1, confidence: float = 0.95) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        基于正态假设的预测区间。
        x_{t+h} ~ N(\hat{x}_{t+h}, \sigma_h^2)
        方差通过脉冲响应累积：
            psi_0 = 1,  psi_j = -sum_{i=1}^{min(j,p)} a_i psi_{j-i}
            Var(e_{t+h}) = sigma_e^2 * sum_{j=0}^{h-1} psi_j^2
        """
        pred = self.predict(history, steps)
        # TODO: Hole_2 - 实现脉冲响应递推与预测方差计算
        # 要求：
        #   1. 计算脉冲响应序列 psi_0, ..., psi_{steps-1}
        #   2. 基于 psi 计算预测方差 Var(e_{t+h}) = sigma_e^2 * Σ psi_j^2
        #   3. 计算正态分位数 z 对应的置信区间 [pred - z*std, pred + z*std]
        # 提示：
        #   - psi_0 = 1
        #   - psi_j = -Σ_{i=1}^{min(j,p)} a_i * psi_{j-i}   (j >= 1)
        #   - 方差通过 cumsum(psi^2) * sigma_e^2 得到
        #   - z 值：confidence>=0.95 用 1.96，否则用 1.645
        raise NotImplementedError("Hole_2: 请实现脉冲响应递推与预测方差计算")
