"""
transition_state.py
过渡态搜索与验证模块

核心功能：
- Nudged Elastic Band (NEB) 反应路径优化
- 攀爬图像过渡态精确定位
- 过渡态 Hessian 分析（虚频验证）
- 隧道效应校正（Wigner 校正）
- 反应速率常数计算（过渡态理论）

科学背景：
过渡态理论（Eyring, 1935）给出反应速率：
    k = (k_B T / h) * (Q‡ / Q_R) * exp(-ΔG‡ / (k_B T))

其中：
    Q‡: 过渡态配分函数（含一个虚频模式）
    Q_R: 反应物配分函数
    ΔG‡ = G(ξ‡) - G(ξ_R): 活化自由能

过渡态的几何判据（一阶鞍点）：
    ∇V(x‡) = 0
    H(x‡) 恰有一个负特征值 λ_1 < 0

对应的虚频：
    ν_imag = √( |λ_1| ) / (2π)
    （注意：对于质量加权坐标，单位为 cm^{-1}）

隧道效应校正（Wigner 近似）：
    κ(T) = 1 + (1/24) * (h ν_imag / (k_B T))²

速率常数校正：
    k_corr = κ(T) * k_TST

---
NEB 方法：
    目标：找到连接反应物 R 和产物 P 的最小能量路径（MEP）
    离散化：N 个图像 {R_0, R_1, ..., R_{N-1}}，R_0 = R, R_{N-1} = P
    每个图像受力：
        F_i = F_i^⊥ + F_i^∥
        F_i^⊥ = -∇V(R_i) + (∇V(R_i)·τ̂_i) τ̂_i
        F_i^∥ = k (|R_{i+1} - R_i| - |R_i - R_{i-1}|) τ̂_i
    其中 τ̂_i 为路径单位切向量。

CI-NEB：
    最高能量图像改为沿梯度方向“攀爬”：
        F_i^{climb} = -∇V(R_i) + 2(∇V(R_i)·τ̂_i) τ̂_i
"""

import numpy as np
from sparse_operations import CRSMatrix, lanczos_eigenvalue_solver


class TransitionStateVerifier:
    """
    过渡态验证器
    """

    def __init__(self, gradient_func, hessian_func=None):
        """
        参数：
            gradient_func: ∇V(x) 梯度函数
            hessian_func: H(x) Hessian 函数（可选）
        """
        self.gradient_func = gradient_func
        self.hessian_func = hessian_func

    def verify_saddle_point(self, x_ts, grad_tol=1e-3):
        """
        验证鞍点条件

        判据：
            1. |∇V(x‡)| < ε_grad
            2. Hessian 恰有一个负特征值
        """
        grad = self.gradient_func(x_ts)
        grad_norm = np.linalg.norm(grad)

        result = {
            'gradient_norm': grad_norm,
            'is_stationary': grad_norm < grad_tol,
            'n_negative_modes': None,
            'is_transition_state': False,
            'imaginary_frequency': None
        }

        if self.hessian_func is not None:
            H = self.hessian_func(x_ts)
            eigenvalues = np.linalg.eigvalsh(H)
            n_neg = np.sum(eigenvalues < -1e-6)
            result['n_negative_modes'] = n_neg
            result['is_transition_state'] = (n_neg == 1)

            if n_neg >= 1:
                # 虚频（质量加权坐标，假设质量为 1 amu）
                # ν = sqrt(|λ|) / (2πc) [cm^{-1}]
                c_cm_fs = 2.998e-5  # cm/fs
                lam_neg = eigenvalues[eigenvalues < -1e-6][0]
                freq_cm = np.sqrt(abs(lam_neg)) / (2.0 * np.pi * c_cm_fs)
                result['imaginary_frequency'] = freq_cm
                result['eigenvalues'] = eigenvalues

        return result

    def wigner_correction(self, imaginary_freq_cm, temperature=300.0):
        """
        Wigner 隧道效应校正

        公式：
            κ(T) = 1 + (1/24) * (h c ν̃ / (k_B T))²
        """
        h = 6.626e-34  # J·s
        c = 2.998e10   # cm/s
        kB = 1.381e-23 # J/K

        x = h * c * abs(imaginary_freq_cm) / (kB * temperature)
        kappa = 1.0 + x ** 2 / 24.0
        return kappa

    def rate_constant_tst(self, delta_G, temperature=300.0, kappa=1.0):
        """
        过渡态理论速率常数

        公式：
            k = κ * (k_B T / h) * exp(-ΔG‡ / (k_B T))
        """
        # TODO(Hole_2): 实现过渡态理论速率常数计算
        # 使用 Eyring TST 公式: k = κ * (k_B T / h) * exp(-ΔG‡ / (k_B T))
        # 注意单位转换：ΔG 输入为 kcal/mol
        raise NotImplementedError("Hole_2: 请实现 rate_constant_tst 方法")


class NEBOptimizer:
    """
    Nudged Elastic Band 优化器（完整实现）
    """

    def __init__(self, energy_func, gradient_func, n_images=20,
                 spring_k=0.1, dt=0.01, max_iter=1000, tol=1e-4):
        self.energy_func = energy_func
        self.gradient_func = gradient_func
        self.n_images = n_images
        self.spring_k = spring_k
        self.dt = dt
        self.max_iter = max_iter
        self.tol = tol

    def _compute_tangent(self, path, energies, i):
        """计算第 i 个图像的路径切向量"""
        if energies[i + 1] > energies[i - 1]:
            tau = path[i + 1] - path[i]
        else:
            tau = path[i] - path[i - 1]
        norm = np.linalg.norm(tau)
        if norm > 1e-12:
            return tau / norm
        return np.zeros_like(tau)

    def optimize(self, x_reactant, x_product):
        """
        标准 NEB 优化
        """
        x_R = np.asarray(x_reactant, dtype=float)
        x_P = np.asarray(x_product, dtype=float)
        dim = len(x_R)

        # 线性插值初始化
        path = np.zeros((self.n_images, dim), dtype=float)
        for i in range(self.n_images):
            lam = i / (self.n_images - 1.0)
            path[i] = (1.0 - lam) * x_R + lam * x_P

        energies_history = []

        for it in range(self.max_iter):
            energies = np.array([self.energy_func(path[i]) for i in range(self.n_images)])
            gradients = np.array([self.gradient_func(path[i]) for i in range(self.n_images)])
            energies_history.append(energies.copy())

            forces = np.zeros_like(path)
            max_force = 0.0

            for i in range(1, self.n_images - 1):
                tau = self._compute_tangent(path, energies, i)
                grad = gradients[i]

                # 垂直力分量
                f_perp = grad - np.dot(grad, tau) * tau

                # 弹簧力（平行分量）
                f_spring = self.spring_k * (
                        np.linalg.norm(path[i + 1] - path[i]) -
                        np.linalg.norm(path[i] - path[i - 1])
                ) * tau

                forces[i] = -f_perp + f_spring
                max_force = max(max_force, np.linalg.norm(forces[i]))

            # 更新（保持端点固定）
            path[1:self.n_images - 1] += self.dt * forces[1:self.n_images - 1]

            if max_force < self.tol:
                break

        final_energies = np.array([self.energy_func(path[i]) for i in range(self.n_images)])
        return path, final_energies, energies_history

    def climbing_image(self, x_reactant, x_product, n_climb_steps=200):
        """
        攀爬图像 NEB
        """
        # 先运行标准 NEB
        path, energies, _ = self.optimize(x_reactant, x_product)

        # 确定最高能量图像
        ts_idx = np.argmax(energies[1:self.n_images - 1]) + 1

        # 局部攀爬优化
        for it in range(n_climb_steps):
            energies = np.array([self.energy_func(path[i]) for i in range(self.n_images)])
            gradients = np.array([self.gradient_func(path[i]) for i in range(self.n_images)])

            forces = np.zeros_like(path)
            max_force = 0.0

            for i in range(1, self.n_images - 1):
                tau = self._compute_tangent(path, energies, i)
                grad = gradients[i]

                if i == ts_idx:
                    # 攀爬图像
                    f_parallel = np.dot(grad, tau) * tau
                    forces[i] = -(grad - 2.0 * f_parallel)
                else:
                    f_perp = grad - np.dot(grad, tau) * tau
                    f_spring = self.spring_k * (
                            np.linalg.norm(path[i + 1] - path[i]) -
                            np.linalg.norm(path[i] - path[i - 1])
                    ) * tau
                    forces[i] = -f_perp + f_spring

                max_force = max(max_force, np.linalg.norm(forces[i]))

            path[1:self.n_images - 1] += self.dt * forces[1:self.n_images - 1]

            if max_force < self.tol:
                break

        final_energies = np.array([self.energy_func(path[i]) for i in range(self.n_images)])
        return path, final_energies, ts_idx


class ReactionPathAnalysis:
    """
    反应路径分析工具
    """

    @staticmethod
    def find_transition_state(path, energies):
        """
        从优化后的路径中确定过渡态位置
        """
        # 排除端点后找最大值
        ts_idx = np.argmax(energies[1:len(energies) - 1]) + 1
        return ts_idx, path[ts_idx], energies[ts_idx]

    @staticmethod
    def activation_energy(energies):
        """
        计算正/逆反应活化能
        """
        ts_idx = np.argmax(energies[1:len(energies) - 1]) + 1
        E_r = energies[0]
        E_p = energies[-1]
        E_ts = energies[ts_idx]
        Ea_forward = E_ts - E_r
        Ea_reverse = E_ts - E_p
        return Ea_forward, Ea_reverse, ts_idx

    @staticmethod
    def reaction_coordinate_values(path):
        """
        计算累积弧长作为反应坐标
        """
        s = np.zeros(len(path))
        for i in range(1, len(path)):
            s[i] = s[i - 1] + np.linalg.norm(path[i] - path[i - 1])
        return s

    @staticmethod
    def curvature_analysis(path, energies):
        """
        路径曲率分析

        曲率：
            κ(s) = |d²γ/ds²|
        在过渡态附近，曲率通常最大。
        """
        s = ReactionPathAnalysis.reaction_coordinate_values(path)
        if len(s) < 3:
            return np.zeros(len(s))

        # 数值二阶导数
        ds = np.gradient(s)
        path_flat = path.reshape(len(path), -1)
        d2gamma = np.zeros(len(path))

        for i in range(1, len(path) - 1):
            if ds[i] > 1e-12:
                d2 = (path_flat[i + 1] - 2 * path_flat[i] + path_flat[i - 1]) / (ds[i] ** 2)
                d2gamma[i] = np.linalg.norm(d2)

        return d2gamma
