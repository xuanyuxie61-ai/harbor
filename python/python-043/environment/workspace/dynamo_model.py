"""
dynamo_model.py — 地核运动学发电机磁感应方程核心求解模块

科学模型:
  本模块实现基于轴对称 α-Ω 运动学发电机的地核磁场演化模型。
  该模型是理解地磁场产生与极性反转的最小自洽框架之一。

物理方程:
  在球坐标 (r, θ, φ) 中，轴对称磁场分解为极型(poloidal)和环型(toroidal):
    B⃗ = B⃗_P + B_φ φ̂ = ∇×(A_φ φ̂) + B_φ φ̂

  引入流函数变量:
    S(r,θ,t) = r sinθ · A_φ   (极型势)
    T(r,θ,t) = r sinθ · B_φ   (环型场)

  磁感应方程化为耦合的标量PDE组:

    ∂S/∂t = η D²[S] - u⃗_P·∇S + α T                          ...(1)

    ∂T/∂t = η D²[T] - sinθ {S, Ω}                           ...(2)
            - (1/r) ∂/∂r(r α ∂S/∂r) - (1/r²) ∂/∂θ(α ∂S/∂θ)

  其中扩散算子:
    D²[f] = ∂²f/∂r² + (1/r²) ∂²f/∂θ² - (cotθ/r²) ∂f/∂θ

  泊松括号 (Jacobian):
    {S, Ω} = ∂S/∂r · ∂Ω/∂θ - ∂S/∂θ · ∂Ω/∂r

  α-Ω效应:
    Ω(r,θ) = C_Ω · f(r) · sin²θ          (差分旋转, 赤道加速)
    α(r,θ) = C_α · cosθ · g(r)           (α效应, 半球反对称)

  非线性α淬灭 (α-quenching):
    α_eff = α(r,θ) / [1 + (B_φ/B_eq)²]
          = α(r,θ) / [1 + (T/(r sinθ B_eq))²]

  该非线性机制是极性反转的关键：当场强饱和时，α效应被抑制，
  偶极子场衰减；随后扩散与对流重新建立场，可能以相反极性。

边界条件:
  - r = r_i (ICB):  S = 0,  T = 0   (完美导体/绝缘)
  - r = r_o (CMB):  S = 0,  T = 0   (绝缘边界, 匹配外部势场)
  - θ = 0, π (极点): S = 0, T = 0   (正则性条件)

无量纲化:
  长度尺度 L = r_o - r_i, 时间尺度 τ = L²/η
  磁感应强度尺度 B_0 = √(μ₀ ρ) U  (能量均分)
"""

import numpy as np
from typing import Tuple, Callable, Optional


class DynamoModel:
    """
    地核运动学发电机模型。
    """

    def __init__(
        self,
        r_inner: float,
        r_outer: float,
        nr: int,
        ntheta: int,
        eta: float,
        c_omega: float,
        c_alpha: float,
        b_eq: float,
    ):
        """
        初始化发电机模型。

        参数:
            r_inner, r_outer: 内外边界半径
            nr, ntheta: 网格点数
            eta: 磁扩散率
            c_omega: 差分旋转强度 C_Ω
            c_alpha: α效应强度 C_α
            b_eq: 平衡场强 B_eq
        """
        self.r_inner = r_inner
        self.r_outer = r_outer
        self.nr = nr
        self.ntheta = ntheta
        self.eta = eta
        self.c_omega = c_omega
        self.c_alpha = c_alpha
        self.b_eq = b_eq

        # 生成均匀网格
        self.r = np.linspace(r_inner, r_outer, nr)
        self.theta = np.linspace(0.0, np.pi, ntheta)
        self.dr = self.r[1] - self.r[0] if nr > 1 else 1.0
        self.dtheta = self.theta[1] - self.theta[0] if ntheta > 1 else 1.0

        # 预计算度规量
        self._precompute_metrics()

        # 初始化速度场
        self._setup_velocity_field()

    def _precompute_metrics(self):
        """预计算球坐标度规与三角函数。"""
        self.R, self.Theta = np.meshgrid(self.r, self.theta, indexing="ij")
        self.SinTheta = np.sin(self.Theta)
        self.CosTheta = np.cos(self.Theta)
        self.CotTheta = self.CosTheta / (self.SinTheta + 1e-30)
        self.R2 = self.R ** 2

    def _setup_velocity_field(self):
        """
        设置速度场分布。

        差分旋转 (solar-type):
          Ω(r,θ) = C_Ω · (r-r_i)(r_o-r) · sin²θ
        该形式在赤道面加速，在极区减速，在边界处为零。

        α效应 (hemispheric antisymmetric):
          α(r,θ) = C_α · cosθ · (r-r_i)(r_o-r) · exp(-(r-r_m)²/(2σ²))
        该形式在南北半球符号相反，在边界层处集中。
        """
        r_m = 0.5 * (self.r_inner + self.r_outer)
        sigma = 0.25 * (self.r_outer - self.r_inner)

        # 径向包络
        radial_envelope = (self.R - self.r_inner) * (self.r_outer - self.R)
        radial_envelope = np.maximum(radial_envelope, 0.0)

        # 差分旋转
        self.Omega = self.c_omega * radial_envelope * self.SinTheta ** 2

        # α效应 (未淬灭)
        gaussian = np.exp(-((self.R - r_m) ** 2) / (2.0 * sigma ** 2))
        self.alpha_base = self.c_alpha * self.CosTheta * radial_envelope * gaussian

        # 预计算梯度 (用于Jacobian)
        self._compute_velocity_gradients()

    def _compute_velocity_gradients(self):
        """预计算速度场的空间梯度。"""
        # dΩ/dr
        self.dOmega_dr = np.zeros_like(self.Omega)
        self.dOmega_dr[1:-1, :] = (self.Omega[2:, :] - self.Omega[:-2, :]) / (2.0 * self.dr)
        # dΩ/dθ
        self.dOmega_dtheta = np.zeros_like(self.Omega)
        self.dOmega_dtheta[:, 1:-1] = (self.Omega[:, 2:] - self.Omega[:, :-2]) / (2.0 * self.dtheta)

    def _diffusion_operator(self, F: np.ndarray) -> np.ndarray:
        """
        计算扩散算子 D²[F] 的离散近似。

        D²[F] = ∂²F/∂r² + (1/r²)∂²F/∂θ² - (cotθ/r²)∂F/∂θ
        """
        D2F = np.zeros_like(F)

        # 径向二阶导 (中心差分)
        d2F_dr2 = np.zeros_like(F)
        d2F_dr2[1:-1, :] = (F[2:, :] - 2.0 * F[1:-1, :] + F[:-2, :]) / (self.dr ** 2)

        # 角度二阶导
        d2F_dtheta2 = np.zeros_like(F)
        d2F_dtheta2[:, 1:-1] = (F[:, 2:] - 2.0 * F[:, 1:-1] + F[:, :-2]) / (self.dtheta ** 2)

        # 角度一阶导
        dF_dtheta = np.zeros_like(F)
        dF_dtheta[:, 1:-1] = (F[:, 2:] - F[:, :-2]) / (2.0 * self.dtheta)

        # 组合
        D2F = d2F_dr2 + (1.0 / self.R2) * d2F_dtheta2 - (self.CotTheta / self.R2) * dF_dtheta

        return D2F

    def _alpha_quenching(self, T: np.ndarray, t: float = 0.0) -> np.ndarray:
        """
        计算非线性α淬灭因子，含湍流涨落项。

        α_eff = α_base / [1 + (B_φ/B_eq)²] · [1 + ε·sin(ωt)·cosθ]

        其中第二项模拟地核湍流对流的时间依赖调制，
        是引发极性反转的关键物理机制之一。
        """
        denominator = 1.0 + (T / (self.R * self.SinTheta * self.b_eq + 1e-30)) ** 2
        alpha_eff = self.alpha_base / denominator
        # 湍流调制 (Meridional circulation / torsional oscillation proxy)
        modulation = 1.0 + 0.15 * np.sin(2.0 * np.pi * t / 3.0) * self.CosTheta
        return alpha_eff * modulation

    def _jacobian_bracket(self, S: np.ndarray) -> np.ndarray:
        """
        计算泊松括号 {S, Ω} = ∂S/∂r · ∂Ω/∂θ - ∂S/∂θ · ∂Ω/∂r。
        这是Ω效应的核心非线性耦合项。
        """
        dS_dr = np.zeros_like(S)
        dS_dr[1:-1, :] = (S[2:, :] - S[:-2, :]) / (2.0 * self.dr)

        dS_dtheta = np.zeros_like(S)
        dS_dtheta[:, 1:-1] = (S[:, 2:] - S[:, :-2]) / (2.0 * self.dtheta)

        jac = dS_dr * self.dOmega_dtheta - dS_dtheta * self.dOmega_dr
        return jac

    def _alpha_induction_terms(self, S: np.ndarray, alpha: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算α效应的感应项。

        对S方程: 源项 = α · T
        对T方程: 源项 = -(1/r)∂/∂r(r·α·∂S/∂r) - (1/r²)∂/∂θ(α·∂S/∂θ)
        """
        dS_dr = np.zeros_like(S)
        dS_dr[1:-1, :] = (S[2:, :] - S[:-2, :]) / (2.0 * self.dr)

        dS_dtheta = np.zeros_like(S)
        dS_dtheta[:, 1:-1] = (S[:, 2:] - S[:, :-2]) / (2.0 * self.dtheta)

        # T方程中的α感应项
        term_r = np.zeros_like(S)
        r_alpha_dSdr = self.R * alpha * dS_dr
        term_r[1:-1, :] = -(r_alpha_dSdr[2:, :] - r_alpha_dSdr[:-2, :]) / (2.0 * self.dr * self.R[1:-1, :])

        term_theta = np.zeros_like(S)
        alpha_dSdtheta = alpha * dS_dtheta
        term_theta[:, 1:-1] = -(alpha_dSdtheta[:, 2:] - alpha_dSdtheta[:, :-2]) / (
            2.0 * self.dtheta * self.R2[:, 1:-1]
        )

        return term_r + term_theta

    def rhs(self, t: float, state: np.ndarray) -> np.ndarray:
        """
        计算ODE右端项: d[state]/dt = rhs(t, state)。

        state = [S_flat; T_flat], 长度为 2·nr·ntheta
        """
        n = self.nr * self.ntheta
        S = state[:n].reshape((self.nr, self.ntheta))
        T = state[n:].reshape((self.nr, self.ntheta))

        # 非线性α淬灭
        alpha = self._alpha_quenching(T, t)

        # 扩散项
        D2S = self._diffusion_operator(S)
        D2T = self._diffusion_operator(T)

        # Ω效应 (Jacobian)
        jac = self._jacobian_bracket(S)

        # α感应项
        alpha_ind = self._alpha_induction_terms(S, alpha)

        # S方程
        dSdt = self.eta * D2S + alpha * T

        # T方程
        dTdt = self.eta * D2T - self.SinTheta * jac + alpha_ind

        # 边界条件: Dirichlet (强制为零)
        dSdt[0, :] = 0.0
        dSdt[-1, :] = 0.0
        dSdt[:, 0] = 0.0
        dSdt[:, -1] = 0.0

        dTdt[0, :] = 0.0
        dTdt[-1, :] = 0.0
        dTdt[:, 0] = 0.0
        dTdt[:, -1] = 0.0

        return np.concatenate([dSdt.reshape(-1), dTdt.reshape(-1)])

    def compute_magnetic_energy(self, S: np.ndarray, T: np.ndarray) -> float:
        """
        计算磁场的总能量 (无量纲)。

        E_m = (1/2) ∫ B² dV
            ≈ (1/2) Σ_{i,j} B²_{ij} · r_i² sinθ_j · Δr · Δθ · Δφ
        """
        # 从S和T重构B场分量 (简化)
        # B_r ~ (1/r²) dS/dθ, B_θ ~ -(1/r) dS/dr, B_φ = T/(r sinθ)
        dS_dr = np.zeros_like(S)
        dS_dr[1:-1, :] = (S[2:, :] - S[:-2, :]) / (2.0 * self.dr)

        dS_dtheta = np.zeros_like(S)
        dS_dtheta[:, 1:-1] = (S[:, 2:] - S[:, :-2]) / (2.0 * self.dtheta)

        Br = dS_dtheta / (self.R2 * self.SinTheta + 1e-30)
        Btheta = -dS_dr / (self.R + 1e-30)
        Bphi = T / (self.R * self.SinTheta + 1e-30)

        B2 = Br ** 2 + Btheta ** 2 + Bphi ** 2

        # 体积积分 (轴对称, Δφ = 2π)
        dV = 2.0 * np.pi * self.R2 * self.SinTheta * self.dr * self.dtheta
        energy = 0.5 * np.sum(B2 * dV)
        return float(energy)

    def compute_dipole_moment(self, S: np.ndarray) -> float:
        """
        计算轴向偶极矩。

        简化为CMB处S场的赤道-极地差值。
        """
        s_cmb = S[-1, :]
        mid = self.ntheta // 2
        return float(s_cmb[mid] - 0.5 * (s_cmb[0] + s_cmb[-1]))

    def initial_condition(self, seed: int = 42) -> np.ndarray:
        """
        生成小振幅随机初始扰动。
        """
        rng = np.random.default_rng(seed)
        S = np.zeros((self.nr, self.ntheta))
        T = np.zeros((self.nr, self.ntheta))

        # 在内部添加小振幅随机扰动
        amplitude = 0.01
        S[2:-2, 2:-2] = amplitude * rng.normal(size=(self.nr - 4, self.ntheta - 4))
        T[2:-2, 2:-2] = amplitude * rng.normal(size=(self.nr - 4, self.ntheta - 4))

        # 确保边界为零
        S[0, :] = S[-1, :] = S[:, 0] = S[:, -1] = 0.0
        T[0, :] = T[-1, :] = T[:, 0] = T[:, -1] = 0.0

        return np.concatenate([S.reshape(-1), T.reshape(-1)])

    def to_2d(self, state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """将状态向量还原为 (S, T) 二维场。"""
        n = self.nr * self.ntheta
        S = state[:n].reshape((self.nr, self.ntheta))
        T = state[n:].reshape((self.nr, self.ntheta))
        return S, T
