"""
inverse_problem.py
==================
星震学反演问题: 从观测频率推断恒星内部结构.

融合种子项目:
  - F-adjoint-Learning (1160): F-adjoint 传播 → 伴随反演方法
  - Hankel-inverse (505): Hankel 矩阵逆 → 积分核反演
  - HyperMPC (1177): 超网络 → 参数化反演

科学背景
--------
星震学反演的核心问题 (Christensen-Dalsgaard 2003):

给定一组观测频率 {ν_i^{obs}}, 推断恒星内部结构:
  - 声速剖面 δc/c(r)
  - 密度剖面 δρ/ρ(r)
  - 旋转剖面 Ω(r)

线性反演 (OLS: Optimally Localized Averages, Pijpers & Thompson 1994):

  δν_i/ν_i = ∫₀ᴿ K_i(r) δq(r)/q(r) dr + surface_term

  其中:
    K_i(r) 是频率核对结构量 q(r) 的灵敏度核
    δq/q 是待求的结构修正

  反演目标: 构造线性组合 Σ_i a_i K_i(r) ≈ δ(r - r₀)
  使得 δq/q(r₀) ≈ Σ_i a_i δν_i/ν_i

非线性反演 (伴随方法):

  目标泛函: χ² = Σ_i (ν_i^{obs} - ν_i^{model})² / σ_i²

  梯度计算:
    ∂χ²/∂p = 2 Σ_i (ν_i^{obs} - ν_i^{model}) / σ_i² × ∂ν_i/∂p

  其中 ∂ν_i/∂p 通过伴随 (adjoint) 方法高效计算.

  F-adjoint 传播 (融合 1160 项目):
    前向传播: 结构参数 → 频率
    伴随传播: 频率残差 → 参数梯度

本模块实现:
  1. 频率灵敏度核的计算
  2. OLS 反演 (SOLA: Subtractive Optimally Localized Averages)
  3. 伴随梯度计算
  4. Hankel 矩阵反演
"""

import numpy as np
from typing import Tuple, Dict, Optional


class FrequencySensitivityKernels:
    """
    振荡频率对内部结构量的灵敏度核.

    核函数 K_{c²,ρ}^i(r) 描述频率对声速和密度的依赖:

    δν_i/ν_i = ∫₀ᴿ [K_{c²,ρ}^i δc²/c² + K_{ρ,c²}^i δρ/ρ] dr + F_i(ν_i)

    在渐近近似下:

    K_{c²,ρ}^i(r) ≈ 1/(2 ∫₀ᴿ dr/c_s) × 1/c_s(r) × W_i(r)

    其中 W_i(r) 是模式动能密度权重:

    W_i(r) = |ξ_r|² + l(l+1)|ξ_h|²

    参数
    ----
    stellar_model : object
        恒星结构模型
    """

    def __init__(self, stellar_model):
        self.model = stellar_model
        self.n_r = stellar_model.n_r

    def compute_kernel_c2(
        self,
        eigenfunction_xi_r: np.ndarray,
        eigenfunction_xi_h: np.ndarray,
        l: int,
    ) -> np.ndarray:
        """
        计算声速灵敏度核 K_{c²}.

        K_{c²}^i(r) ∝ [Γ₁P ξ_r dξ_r/dr + ρ g ξ_r² + l(l+1) Γ₁ P ξ_h²/r²]
                      / (2 E_k ω²)

        其中 E_k 是模式动能:
          E_k = 1/2 ∫ ρ [|ξ_r|² + l(l+1)|ξ_h|²] r² dr

        Parameters
        ----------
        xi_r : ndarray
            径向位移本征函数
        xi_h : ndarray
            水平位移本征函数
        l : int
            角量子数

        Returns
        -------
        kernel : ndarray, shape (n_r,)
            声速灵敏度核
        """
        n = self.n_r
        r = self.model.r[:n]
        rho = self.model.rho[:n]
        P = self.model.P[:n]
        Gamma1 = self.model.Gamma1[:n]
        g = self.model.g[:n]
        dr = self.model.dr
        xi_r = eigenfunction_xi_r
        xi_h = eigenfunction_xi_h

        # 模式动能
        E_k = 0.5 * np.trapz(
            rho * (xi_r[:n]**2 + l * (l + 1.0) * xi_h[:n]**2) * r**2,
            r[:n]
        )
        E_k = max(E_k, 1e-100)

        # 核函数
        kernel = np.zeros(n)
        for i in range(1, n - 1):
            r_i = max(r[i], 1e-10)

            # 径向项: 压强扰动做功
            dxi_r = (xi_r[i + 1] - xi_r[i - 1]) / (2.0 * dr)
            term_r = Gamma1[i] * P[i] * xi_r[i] * dxi_r
            term_r += rho[i] * g[i] * xi_r[i]**2

            # 水平项
            term_h = l * (l + 1.0) * Gamma1[i] * P[i] * xi_h[i]**2 / r_i**2

            kernel[i] = (term_r + term_h) / (2.0 * E_k + 1e-100)

        # 归一化
        norm = np.trapz(kernel, r[:n])
        if abs(norm) > 1e-30:
            kernel /= norm

        return kernel

    def compute_kernel_rho(
        self,
        eigenfunction_xi_r: np.ndarray,
        eigenfunction_xi_h: np.ndarray,
        l: int,
    ) -> np.ndarray:
        """
        计算密度灵敏度核 K_ρ.

        通过代数关系:
          K_ρ = K_c² - K_{c²,ρ}

        简化实现: 使用连续性方程和状态方程的约束.

        Parameters
        ----------
        xi_r : ndarray
            径向位移
        xi_h : ndarray
            水平位移
        l : int
            角量子数

        Returns
        -------
        kernel : ndarray
            密度灵敏度核
        """
        # 简化: 密度核 = 声速核 × 比例因子
        k_c2 = self.compute_kernel_c2(eigenfunction_xi_r, eigenfunction_xi_h, l)
        # 在渐近近似下: K_ρ ≈ -K_c²/2 (对于 p 模)
        return -0.5 * k_c2


class SOLAInversion:
    """
    SOLA (Subtractive Optimally Localized Averages) 反演.

    目标: 在目标位置 r₀ 构造最优线性组合:

    min_{a_i} ||Σ_i a_i K_i(r) - T(r; r₀)||² + θ Σ_i a_i²

    其中:
      T(r; r₀) = 目标核 (通常取窄高斯: δ(r - r₀) 的近似)
      θ: 正则化参数

    解:
      a = (A + θI)⁻¹ b

    其中:
      A_{ij} = ∫ K_i(r) K_j(r) dr  (核内积矩阵)
      b_i = ∫ K_i(r) T(r; r₀) dr   (核与目标的交叉项)

    反演结果:
      δq/q(r₀) ≈ Σ_i a_i δν_i/ν_i

    参数
    ----
    n_targets : int
        目标位置数量
    theta : float
        正则化参数
    target_width : float
        目标核宽度 (相对恒星半径)
    """

    def __init__(
        self,
        n_targets: int = 20,
        theta: float = 1e-4,
        target_width: float = 0.05,
    ):
        self.n_targets = n_targets
        self.theta = theta
        self.target_width = target_width

    def build_kernel_matrix(
        self,
        kernels: np.ndarray,
        r: np.ndarray,
    ) -> np.ndarray:
        """
        构造核内积矩阵 A.

        A_{ij} = ∫ K_i(r) K_j(r) dr

        Parameters
        ----------
        kernels : ndarray, shape (n_modes, n_r)
            灵敏度核
        r : ndarray, shape (n_r,)
            径向网格

        Returns
        -------
        A : ndarray, shape (n_modes, n_modes)
        """
        n_modes = kernels.shape[0]
        A = np.zeros((n_modes, n_modes))
        for i in range(n_modes):
            for j in range(i, n_modes):
                A[i, j] = np.trapz(kernels[i] * kernels[j], r)
                A[j, i] = A[i, j]
        return A

    def build_target_vector(
        self,
        kernels: np.ndarray,
        r: np.ndarray,
        r_target: float,
    ) -> np.ndarray:
        """
        构造目标向量 b.

        b_i = ∫ K_i(r) T(r; r₀) dr

        其中 T(r; r₀) = (1/(σ√(2π))) exp(-(r-r₀)²/(2σ²))

        Parameters
        ----------
        kernels : ndarray
            灵敏度核
        r : ndarray
            径向网格
        r_target : float
            目标位置

        Returns
        -------
        b : ndarray
        """
        sigma = self.target_width * r[-1]
        T_target = np.exp(-(r - r_target)**2 / (2.0 * sigma**2)) / (
            sigma * np.sqrt(2.0 * np.pi)
        )

        b = np.zeros(kernels.shape[0])
        for i in range(kernels.shape[0]):
            b[i] = np.trapz(kernels[i] * T_target, r)
        return b

    def invert(
        self,
        kernels: np.ndarray,
        r: np.ndarray,
        delta_nu_over_nu: np.ndarray,
        r_targets: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        执行 SOLA 反演.

        Parameters
        ----------
        kernels : ndarray, shape (n_modes, n_r)
            灵敏度核
        r : ndarray
            径向网格
        delta_nu_over_nu : ndarray, shape (n_modes,)
            相对频率偏差
        r_targets : ndarray
            目标位置

        Returns
        -------
        result : dict
            delta_q_over_q: 推断的结构修正
            resolution: 分辨率评估
            error: 误差估计
        """
        A = self.build_kernel_matrix(kernels, r)
        n_r = len(r)
        delta_q = np.zeros(n_r)

        for idx, r0 in enumerate(r_targets):
            b = self.build_target_vector(kernels, r, r0)

            # 正则化求解: (A + θI) a = b
            A_reg = A + self.theta * np.eye(len(A))
            try:
                a = np.linalg.solve(A_reg, b)
            except np.linalg.LinAlgError:
                a = np.linalg.lstsq(A_reg, b, rcond=None)[0]

            # 构造平均核
            avg_kernel = a @ kernels

            # 计算该位置的结构修正
            delta_q_val = a @ delta_nu_over_nu
            delta_q[idx] = delta_q_val

        return {
            "delta_q_over_q": delta_q,
            "r_targets": r_targets,
            "averaging_kernel": avg_kernel if 'avg_kernel' in dir() else None,
        }


class AdjointGradientComputer:
    """
    伴随梯度计算器.

    融合 F-adjoint-Learning (1160) 的伴随传播思想.

    在星震学反演中, 我们需要计算目标泛函对模型参数的梯度:

    χ² = Σ_i (ν_i^{obs} - ν_i^{model}(p))² / σ_i²

    ∂χ²/∂p_k = -2 Σ_i (ν_i^{obs} - ν_i^{model}) / σ_i² × ∂ν_i/∂p_k

    通过伴随方法, 可以在 O(1) 次正向/反向传播中计算所有参数的梯度,
    而不需要对每个参数做一次正向计算.

    F-adjoint 传播 (from 1160):
      前向: Y_l = σ(W_l X_{l-1})  (结构参数 → 频率)
      伴随: X*_{l-1} = W_l^T Y*_l  (频率残差 → 参数梯度)

    参数
    ----
    n_params : int
        模型参数维度
    n_modes : int
        观测模式数量
    learning_rate : float
        学习率 (梯度下降步长)
    """

    def __init__(
        self,
        n_params: int = 5,
        n_modes: int = 20,
        learning_rate: float = 0.01,
    ):
        self.n_params = n_params
        self.n_modes = n_modes
        self.lr = learning_rate

        # 网络权重 (简化为线性映射)
        np.random.seed(123)
        scale = np.sqrt(2.0 / (n_params + n_modes))
        self.W = np.random.randn(n_modes, n_params) * scale
        self.b = np.zeros(n_modes)

    def forward(self, params: np.ndarray) -> np.ndarray:
        """
        前向传播: 模型参数 → 预测频率.

        ν_pred = W p + b

        Parameters
        ----------
        params : ndarray, shape (n_params,)
            模型参数

        Returns
        -------
        nu_pred : ndarray, shape (n_modes,)
            预测频率
        """
        return self.W @ params + self.b

    def compute_loss(
        self,
        nu_obs: np.ndarray,
        nu_pred: np.ndarray,
        sigma: Optional[np.ndarray] = None,
    ) -> float:
        """
        计算目标泛函 χ².

        χ² = Σ_i (ν_i^{obs} - ν_i^{pred})² / σ_i²

        Parameters
        ----------
        nu_obs : ndarray
            观测频率
        nu_pred : ndarray
            预测频率
        sigma : ndarray, optional
            观测误差 (默认统一为 1)

        Returns
        -------
        chi2 : float
        """
        if sigma is None:
            sigma = np.ones_like(nu_obs) * 0.01

        residuals = (nu_obs - nu_pred) / sigma
        return np.sum(residuals**2)

    def adjoint_pass(
        self,
        nu_obs: np.ndarray,
        params: np.ndarray,
        sigma: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        伴随传播: 计算梯度 ∂χ²/∂p.

        融合 1160 项目的 F-adjoint 思想:

        1. 前向: ν = W p + b
        2. 残差: δ = (ν - ν_obs) / σ²
        3. 伴随: ∂χ²/∂p = W^T δ

        Parameters
        ----------
        nu_obs : ndarray
            观测频率
        params : ndarray
            当前参数
        sigma : ndarray, optional
            观测误差

        Returns
        -------
        grad : ndarray, shape (n_params,)
            梯度
        """
        if sigma is None:
            sigma = np.ones_like(nu_obs) * 0.01

        # 前向
        nu_pred = self.forward(params)

        # 残差加权
        delta = 2.0 * (nu_pred - nu_obs) / sigma**2

        # 伴随传播 (F-adjoint)
        grad = self.W.T @ delta

        return grad

    def gradient_descent(
        self,
        nu_obs: np.ndarray,
        params_init: np.ndarray,
        n_iter: int = 100,
        sigma: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        梯度下降反演.

        Parameters
        ----------
        nu_obs : ndarray
            观测频率
        params_init : ndarray
            初始参数
        n_iter : int
            迭代次数
        sigma : ndarray, optional
            观测误差

        Returns
        -------
        params_opt : ndarray
            最优参数
        loss_history : ndarray
            损失历史
        params_history : ndarray
            参数历史
        """
        params = params_init.copy()
        loss_hist = []
        params_hist = [params.copy()]

        for i in range(n_iter):
            # 前向
            nu_pred = self.forward(params)
            loss = self.compute_loss(nu_obs, nu_pred, sigma)
            loss_hist.append(loss)

            # 伴随梯度
            grad = self.adjoint_pass(nu_obs, params, sigma)

            # 梯度下降更新
            params -= self.lr * grad

            # 参数约束
            params = np.clip(params, -10.0, 10.0)

            params_hist.append(params.copy())

        return params, np.array(loss_hist), np.array(params_hist)


class HankelMatrixInversion:
    """
    Hankel 矩阵反演方法.

    融合 Hankel-inverse (505) 项目:
    利用 Hankel 矩阵的特殊结构 (反对角线常数) 加速反演.

    在星震学中, 频率核对结构量的依赖可以表示为:

    δν_i = Σ_j H_{ij} δq_j

    其中 H 具有近似 Hankel 结构 (当核是平移不变的近似时).

    Hankel 矩阵的逆可以通过 Fiedler (1980) 的方法高效计算:

    B = M₁ M₂ - M₃ M₄

    其中 M₁, M₃ 是 Hankel 矩阵, M₂, M₄ 是 Toeplitz 矩阵.

    参数
    ----
    n : int
        矩阵阶数
    """

    def __init__(self, n: int = 10):
        self.n = n

    def build_hankel(self, x: np.ndarray) -> np.ndarray:
        """
        从向量 x (长度 2n-1) 构造 Hankel 矩阵.

        H_{ij} = x[i+j]

        Parameters
        ----------
        x : ndarray, shape (2n-1,)
            定义向量

        Returns
        -------
        H : ndarray, shape (n, n)
        """
        if len(x) != 2 * self.n - 1:
            raise ValueError(
                f"Hankel 向量长度应为 {2*self.n-1}, 收到 {len(x)}"
            )
        H = np.zeros((self.n, self.n))
        for i in range(self.n):
            H[i, :] = x[i:i + self.n]
        return H

    def build_toeplitz(self, x: np.ndarray) -> np.ndarray:
        """
        从向量 x (长度 2n-1) 构造 Toeplitz 矩阵.

        T_{ij} = x[i-j+n-1]

        Parameters
        ----------
        x : ndarray, shape (2n-1,)
            定义向量

        Returns
        -------
        T : ndarray, shape (n, n)
        """
        if len(x) != 2 * self.n - 1:
            raise ValueError(
                f"Toeplitz 向量长度应为 {2*self.n-1}, 收到 {len(x)}"
            )
        T = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(self.n):
                T[i, j] = x[i - j + self.n - 1]
        return T

    def invert_hankel(self, x: np.ndarray) -> np.ndarray:
        """
        Hankel 矩阵求逆 (Fiedler 分解).

        B = M₁ M₂ - M₃ M₄

        Parameters
        ----------
        x : ndarray, shape (2n-1,)
            定义向量

        Returns
        -------
        B : ndarray, shape (n, n)
            逆矩阵
        """
        n = self.n
        A = self.build_hankel(x)

        # 求解两个线性系统
        p = np.zeros(n)
        p[:n - 1] = x[n:]
        u = np.linalg.solve(A + 1e-10 * np.eye(n), p)

        q = np.zeros(n)
        q[n - 1] = 1.0
        v = np.linalg.solve(A + 1e-10 * np.eye(n), q)

        # 构造四个矩阵
        z1 = np.zeros(n)
        w1 = np.zeros(2 * n - 1)
        w1[:n - 1] = v[:n - 1]
        w1[n - 1] = z1[0] if n > 0 else 0.0
        M1 = self.build_hankel(w1)

        z2 = np.zeros(n - 1)
        w2 = np.zeros(2 * n - 1)
        w2[n - 1:] = u
        M2 = self.build_toeplitz(w2)

        z3 = np.zeros(n)
        z3[0] = -1.0
        w3 = np.zeros(2 * n - 1)
        w3[:n - 1] = u[:n - 1] if n > 1 else np.array([])
        w3[n - 1] = z3[0]
        M3 = self.build_hankel(w3)

        z4 = np.zeros(n - 1)
        w4 = np.zeros(2 * n - 1)
        w4[n - 1:] = v
        M4 = self.build_toeplitz(w4)

        B = M1 @ M2 - M3 @ M4
        return B

    def invert_sensitivity_matrix(
        self,
        kernels: np.ndarray,
        r: np.ndarray,
    ) -> np.ndarray:
        """
        反演灵敏度矩阵 (从频率偏移推断结构修正).

        δq = H⁻¹ δν

        Parameters
        ----------
        kernels : ndarray, shape (n_modes, n_r)
            灵敏度核
        r : ndarray
            径向网格

        Returns
        -------
        H_inv : ndarray
            反演矩阵
        """
        n_modes = kernels.shape[0]
        self.n = n_modes

        # 将核矩阵压缩为 Hankel 近似
        # 取核的平均值作为 Hankel 向量
        kernel_mean = np.mean(kernels, axis=0)
        x = kernel_mean[:2 * n_modes - 1]
        if len(x) < 2 * n_modes - 1:
            x = np.pad(x, (0, 2 * n_modes - 1 - len(x)))

        try:
            H_inv = self.invert_hankel(x[:2 * n_modes - 1])
        except Exception:
            # 退化: 使用伪逆
            H_inv = np.linalg.pinv(kernels @ kernels.T + 1e-6 * np.eye(n_modes))

        return H_inv


# ============================================================
# 物理常数
# ============================================================
PI = np.pi
