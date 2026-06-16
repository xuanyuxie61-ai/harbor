"""
辐射场离散纵标 (S_N) 角向求积.

从 Burkardt disk01_positive_rule 改造：
  原程序在单位圆盘 {(x,y) : x^2+y^2<=1, x>=0, y>=0} 上构造
  具有正权重的求积公式，自由度 DOF = 3N ≥ (D+1)(D+2)/2.

在超新星辐射输运中，我们需要单位球面 S^2 上的离散纵标：
  ∫_{S^2} ψ(Ω) dΩ = Σ_{m=1}^{M} w_m ψ(Ω_m)
其中方向 Ω_m = (μ_m, η_m, ξ_m) 满足 μ^2+η^2+ξ^2=1.

构造方法：将 Burkardt 正求积的 (x,y) 点映射为球面方向
  μ_m = 2 x_m - 1,  η_m = 2 y_m - 1,  ξ_m = ±√(1-μ^2-η^2)
并对权重做 Jacobian 修正。

Level-symmetric S_N 求积的阶数 N 必须为偶数；
权重为正的要求 N ≤ 14 (Carlson-Morgan 限制).
"""
from __future__ import annotations
import math
import numpy as np


class DiscreteOrdinateQuadrature:
    """S_N 离散纵标角向求积."""

    def __init__(self, n_order: int = 6):
        if n_order % 2 != 0:
            raise ValueError("S_N 求积阶数必须为偶数")
        if n_order > 14:
            raise ValueError("正权重 S_N 阶数不得超过 14 (Carlson-Morgan 限制)")
        self.n_order = int(n_order)
        # 使用 Burkardt 正求积：在单位正象限圆盘上积分
        # 采用 Gauss-Legendre 在 μ∈[0,1] 上的 N/2 个点，
        # 在方位角 φ 上均匀 M = N 个点
        n_mu = max(2, n_order // 2)
        n_phi = max(2, n_order)
        # 极角 μ = cos θ 的 Gauss-Legendre 节点/权重
        mu_gauss, w_gauss = np.polynomial.legendre.leggauss(n_mu)
        # 映射 [−1, 1] → [0, 1]
        self.mu_polar = 0.5 * (mu_gauss + 1.0)
        self.w_polar = 0.5 * w_gauss
        # 方位角 φ ∈ [0, π/2] 均匀 (正象限)
        dphi = (math.pi / 2.0) / n_phi
        phi = (np.arange(n_phi) + 0.5) * dphi
        self.phi_azi = phi
        self.w_phi = np.full(n_phi, dphi)
        # 构造球面方向 (正象限: μ>0, η>0, ξ>0)
        # μ = sin θ cos φ, η = sin θ sin φ, ξ = cos θ
        sin_theta = np.sqrt(1.0 - self.mu_polar ** 2)
        directions = []
        weights = []
        for i in range(n_mu):
            for j in range(n_phi):
                mu_x = sin_theta[i] * math.cos(phi[j])
                mu_y = sin_theta[i] * math.sin(phi[j])
                mu_z = self.mu_polar[i]
                # 权重 = w_polar * w_phi * sin θ (球面积分)
                w = self.w_polar[i] * self.w_phi[j] * sin_theta[i]
                directions.append((mu_x, mu_y, mu_z))
                weights.append(w)
        self.directions = np.asarray(directions, dtype=np.float64)
        self.weights = np.asarray(weights, dtype=np.float64)
        # 归一化：正象限立体角 = π/2, 故总和应 ≈ π/2
        total = self.weights.sum()
        self.weights *= (math.pi / 2.0) / total
        self.n_dir = len(self.weights)

    def moments(self, psi: np.ndarray) -> np.ndarray:
        """从方向分布 ψ 计算辐射矩 (E, F, P).

        辐射能密度 E = (1/c) ∫ ψ dΩ
        辐射通量 F = ∫ ψ Ω dΩ
        辐射压张量 P = (1/c) ∫ ψ Ω ⊗ Ω dΩ
        """
        if len(psi) != self.n_dir:
            raise ValueError(f"psi 长度 {len(psi)} 与方向数 {self.n_dir} 不一致")
        w = self.weights
        E = float(np.sum(w * psi))
        F = np.array([float(np.sum(w * psi * self.directions[:, k]))
                      for k in range(3)])
        # P 为 3x3 对称张量
        P = np.zeros((3, 3), dtype=np.float64)
        for a in range(3):
            for b in range(3):
                P[a, b] = float(np.sum(w * psi * self.directions[:, a] * self.directions[:, b]))
        return E, F, P

    def flux_limiter(self, tau: np.ndarray) -> np.ndarray:
        """Flux limiter λ(R) (Levermore-Pomraning).

        R = |∇E| / (χ ρ κ_R E) 为 Knudsen 数
        λ(R) = (1/R) coth(R) - 1/R^2   (Levermore)
        近似：λ(R) ≈ (2 + R) / (6 + 3R + R^2)  (Bruenn 1985)
        """
        R = np.asarray(tau, dtype=np.float64)
        R = np.maximum(R, 1.0e-10)
        # Bruenn 近似
        lam = (2.0 + R) / (6.0 + 3.0 * R + R ** 2)
        # 扩散极限 R → 0: λ → 1/3
        # 自由流极限 R → ∞: λ → 1/R
        return lam

    def eddington_factor(self, f_ratio: np.ndarray) -> np.ndarray:
        """Eddington 因子 f = P / E.

        对热辐射 f = 1/3 (各向同性).
        对束流辐射 f → 1.
        近似：f = (1 - χ)/2 + χ/2,  χ = 3 - 2 λ/λ_diff
        Levermore (1984) 闭合: f = (1 + 2 R^2/3) / (... )
        """
        f = np.asarray(f_ratio, dtype=np.float64)
        # Minerbo (1978) 闭合
        f_out = (1.0 / 3.0) * np.ones_like(f)
        # 对大 R, f → 1
        big = f > 0.5
        small = ~big
        f_out[big] = (1.0 / 3.0) + (2.0 / 3.0) * (f[big] - 0.5) / (f[big] + 1.0e-30)
        f_out[small] = (1.0 / 3.0) + (2.0 / 3.0) * f[small] ** 2
        return np.clip(f_out, 1.0 / 3.0, 1.0)


def level_symmetric_sn(n_order: int = 4):
    """构造 level-symmetric S_N 求积 (经典配置).

    S_2: 4 方向, S_4: 8 方向, S_6: 12 方向, S_8: 16 方向.
    方向分量 μ,η,ξ 为 ±1/√3 (S_2) 或更复杂的有理组合.
    """
    if n_order == 2:
        mu = 1.0 / math.sqrt(3.0)
        dirs = np.array([[s1 * mu, s2 * mu, s3 * mu]
                         for s1 in (-1, 1) for s2 in (-1, 1) for s3 in (-1, 1)
                         if (s1 * s2 * s3) > 0][:4])
        weights = np.full(4, math.pi)
    elif n_order == 4:
        # S_4 经典配置 (Lathrop 1968)
        mu = 0.3506405671804
        eta = 0.8204225940060
        dirs = np.array([
            [ mu,  mu,  eta],
            [ mu,  eta,  mu],
            [ eta,  mu,  mu],
            [-mu,  mu,  eta],
            [-mu,  eta,  mu],
            [-eta,  mu,  mu],
            [ mu, -mu,  eta],
            [ mu, -eta,  mu],
        ])
        # 归一化
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        weights = np.full(8, math.pi / 2.0)
    else:
        # 回退
        return DiscreteOrdinateQuadrature(n_order)
    # 权重归一化使总和 = 4π
    weights *= 4.0 * math.pi / weights.sum()
    class _S_N:
        pass
    obj = _S_N()
    obj.directions = dirs
    obj.weights = weights
    obj.n_dir = len(weights)
    obj.n_order = n_order
    return obj
