"""
adaptive_penalty.py — ADMM 自适应罚参数策略
============================================
来源项目: 1255_hyper-mpc_hypermpc_code (超参数预测/B样条插值)

本模块实现 ADMM 罚参数的自适应调整策略, 借鉴 HyperMPC 的
超参数预测思想, 使用 B 样条插值和在线学习来动态调节罚参数.

背景:
  ADMM 的收敛速度强烈依赖于罚参数 ρ 的选择:
    ρ 过大: 对偶收敛慢 (z 更新步长过小)
    ρ 过小: 原始收敛慢 (x 更新不够准确)
  最优 ρ 依赖于问题的条件数和谱结构.

策略:
  1. 残差均衡法 (Residual Balancing, Boyd 2011)
     ρ^{k+1} = ρ^k * sqrt(||r^k|| / ||s^k||)

  2. spectral 自适应 (Giselsson & Boyd 2016)
     基于残差对的谱估计:
     ρ^k = ||Δs|| / ||Δr||
     其中 Δr = r^k - r^{k-1}, Δs = s^k - s^{k-1}

  3. B 样条参数化 (来源: 1255 HyperMPC)
     将 ρ(t) 参数化为 B 样条:
     ρ(t) = Σ_j N_j(t) c_j
     其中 N_j 为 B 样条基函数, c_j 为控制系数.
     系数通过最小化累计残差来在线更新.

  4. PDE 条件数估计
     利用子域 PDE 的离散条件数来设置初始 ρ:
     ρ_0 ≈ sqrt(κ(A_i)) 其中 κ 为条件数
"""

import numpy as np
from typing import List, Tuple, Optional, Callable
from config import ADMMConfig, EPS_NUM


# ============================================================
#  残差均衡法 (来源: Boyd et al. 2011)
# ============================================================
class ResidualBalancingStrategy:
    """残差均衡自适应策略

    原理: 使原始残差和对偶残差以相同速率收敛.

    更新规则:
      若 ||r|| > μ ||s||:  ρ ← ρ * τ
      若 ||s|| > μ ||r||:  ρ ← ρ / τ
      否则: ρ 不变

    同时更新对偶变量以保持 scaled form 一致性:
      λ ← λ / τ  (当 ρ ← ρ * τ 时)
    """

    def __init__(self, mu: float = 10.0, tau: float = 2.0,
                 rho_min: float = 1e-4, rho_max: float = 1e6):
        self.mu = mu
        self.tau = tau
        self.rho_min = rho_min
        self.rho_max = rho_max

    def update(self, rho: float, primal_res: float, dual_res: float) -> Tuple[float, float]:
        """更新罚参数

        Returns:
            (new_rho, scale_factor): 新罚参数和缩放因子 (用于对偶变量更新)
        """
        scale = 1.0

        if primal_res > self.mu * dual_res and dual_res > EPS_NUM:
            # 原始残差过大 → 增大 ρ
            new_rho = min(rho * self.tau, self.rho_max)
            scale = self.tau
        elif dual_res > self.mu * primal_res and primal_res > EPS_NUM:
            # 对偶残差过大 → 减小 ρ
            new_rho = max(rho / self.tau, self.rho_min)
            scale = 1.0 / self.tau
        else:
            new_rho = rho

        return new_rho, scale


# ============================================================
#  Spectral 自适应 (来源: Giselsson & Boyd 2016)
# ============================================================
class SpectralAdaptiveStrategy:
    """谱自适应罚参数策略

    基于 ADR (Alternating Direction Relaxation) 的谱估计:
      ρ^k = ||Δs|| / ||Δr||

    其中:
      Δr = r^k - r^{k-1} (原始残差增量)
      Δs = s^k - s^{k-1} (对偶残差增量)

    理论依据:
      对于二次问题, 最优 ρ 与 Augmented Lagrangian 的
      特征值比率相关: ρ_opt ≈ sqrt(λ_max / λ_min)
    """

    def __init__(self, rho_min: float = 1e-4, rho_max: float = 1e6,
                 smoothing: float = 0.5):
        self.rho_min = rho_min
        self.rho_max = rho_max
        self.smoothing = smoothing  # 指数移动平均系数
        self.prev_primal_res: Optional[np.ndarray] = None
        self.prev_dual_res: Optional[np.ndarray] = None

    def update(self, rho: float, primal_res_vec: np.ndarray,
               dual_res_vec: np.ndarray) -> float:
        """更新罚参数

        Args:
            rho: 当前罚参数
            primal_res_vec: 原始残差向量
            dual_res_vec: 对偶残差向量 (未除以 ρ)

        Returns:
            new_rho: 新罚参数
        """
        if self.prev_primal_res is not None:
            delta_r = primal_res_vec - self.prev_primal_res
            delta_s = dual_res_vec - self.prev_dual_res

            dr_norm = np.linalg.norm(delta_r)
            ds_norm = np.linalg.norm(delta_s)

            if dr_norm > EPS_NUM and ds_norm > EPS_NUM:
                rho_spectral = ds_norm / dr_norm
                # 指数移动平均 (平滑)
                rho_new = self.smoothing * rho + (1.0 - self.smoothing) * rho_spectral
                rho_new = np.clip(rho_new, self.rho_min, self.rho_max)
            else:
                rho_new = rho
        else:
            rho_new = rho

        self.prev_primal_res = primal_res_vec.copy()
        self.prev_dual_res = dual_res_vec.copy()

        return rho_new


# ============================================================
#  B 样条参数化罚参数 (来源: 1255 HyperMPC)
# ============================================================
class BSplinePenaltyScheduler:
    """B 样条参数化罚参数调度器

    将罚参数 ρ(k) 参数化为关于迭代次数 k 的 B 样条:
      ρ(k) = exp(Σ_j N_j(k) c_j)
    其中 N_j 为均匀 B 样条基函数, c_j 为待学习系数.

    使用指数映射确保 ρ > 0.

    B 样条基函数 (de Boor 递推):
      N_{i,0}(t) = 1 if t_i ≤ t < t_{i+1}, else 0
      N_{i,p}(t) = (t - t_i)/(t_{i+p} - t_i) * N_{i,p-1}(t)
                  + (t_{i+p+1} - t)/(t_{i+p+1} - t_{i+1}) * N_{i+1,p-1}(t)

    在线学习 (梯度下降):
      c ← c - η ∇_c Σ_k ||r_k||² + ||s_k||²
    """

    def __init__(self, n_knots: int = 5, degree: int = 2,
                 total_iterations: int = 500,
                 learning_rate: float = 0.01):
        self.n_knots = n_knots
        self.degree = degree
        self.total_iterations = total_iterations
        self.learning_rate = learning_rate

        # 控制系数 (初始为 log(1) = 0, 即 ρ = exp(0) = 1)
        self.coefficients = np.zeros(n_knots)

        # 节点向量 (均匀)
        self.knots = np.linspace(0, 1, n_knots + degree + 1)

    def evaluate(self, iteration: int) -> float:
        """在当前迭代处评估 B 样条罚参数

        Returns:
            rho: 罚参数值 (> 0)
        """
        t = iteration / max(self.total_iterations, 1)
        t = np.clip(t, 0, 1)

        # 评估 B 样条
        basis_values = self._eval_bspline_basis(t)
        log_rho = np.dot(basis_values, self.coefficients)

        return np.exp(np.clip(log_rho, -10, 10))  # 限制范围

    def _eval_bspline_basis(self, t: float) -> np.ndarray:
        """评估 B 样条基函数 (de Boor 算法)"""
        n = self.n_knots
        p = self.degree
        N = np.zeros(n)

        # 0 阶
        for i in range(n + p):
            if i < len(self.knots) - 1:
                if self.knots[i] <= t < self.knots[i + 1]:
                    if i < n:
                        N[i] = 1.0

        # 递推到 p 阶
        for d in range(1, p + 1):
            N_new = np.zeros(n)
            for i in range(n):
                if i + d < len(self.knots) - 1:
                    left_span = self.knots[i + d] - self.knots[i]
                    right_span = self.knots[i + d + 1] - self.knots[i + 1]

                    val = 0.0
                    if left_span > EPS_NUM and i < n:
                        val += (t - self.knots[i]) / left_span * N[i]
                    if right_span > EPS_NUM and i + 1 < n:
                        val += (self.knots[i + d + 1] - t) / right_span * N[i + 1]
                    N_new[i] = val
            N = N_new

        # 归一化
        total = np.sum(N)
        if total > EPS_NUM:
            N /= total
        else:
            N = np.ones(n) / n

        return N

    def update_coefficients(self, gradient: np.ndarray):
        """更新 B 样条系数 (梯度下降)"""
        self.coefficients -= self.learning_rate * gradient

    def get_schedule(self) -> np.ndarray:
        """获取完整的罚参数调度曲线"""
        schedule = np.zeros(self.total_iterations)
        for k in range(self.total_iterations):
            schedule[k] = self.evaluate(k)
        return schedule


# ============================================================
#  PDE 条件数引导的初始罚参数
# ============================================================
def estimate_initial_rho_from_pDE(stiffness_matrix: np.ndarray,
                                   mass_matrix: np.ndarray) -> float:
    """基于 PDE 离散条件数估计初始罚参数

    对于椭圆 PDE: -∇·(κ∇u) = f
    离散系统: (K + M)u = b

    条件数: κ(K + M) = λ_max / λ_min

    建议初始罚参数: ρ_0 = sqrt(κ) (平衡原始和对偶收敛)

    Args:
        stiffness_matrix: 刚度矩阵 K
        mass_matrix: 质量矩阵 M

    Returns:
        rho_0: 建议的初始罚参数
    """
    A = stiffness_matrix + mass_matrix
    try:
        eigenvalues = np.linalg.eigvalsh(A)
        eigenvalues = eigenvalues[eigenvalues > EPS_NUM]
        if len(eigenvalues) > 0:
            kappa = eigenvalues[-1] / eigenvalues[0]
            return np.sqrt(kappa)
    except np.linalg.LinAlgError:
        pass
    return 1.0  # 默认值


# ============================================================
#  组合自适应策略
# ============================================================
class CombinedAdaptivePenalty:
    """组合自适应罚参数策略

    综合使用多种策略:
      1. 前 10 步: 使用 PDE 条件数估计
      2. 10-50 步: 残差均衡
      3. 50+ 步: Spectral 自适应 + B 样条平滑
    """

    def __init__(self, config: ADMMConfig, n_vars: int):
        self.config = config
        self.n_vars = n_vars
        self.balancing = ResidualBalancingStrategy(
            mu=config.mu, tau=config.tau_incr,
            rho_min=config.rho_min, rho_max=config.rho_max
        )
        self.spectral = SpectralAdaptiveStrategy(
            rho_min=config.rho_min, rho_max=config.rho_max
        )
        self.bspline = BSplinePenaltyScheduler(
            total_iterations=config.max_iter
        )
        self.rho = config.rho

    def update(self, iteration: int, primal_res: float, dual_res: float,
               primal_res_vec: Optional[np.ndarray] = None,
               dual_res_vec: Optional[np.ndarray] = None) -> float:
        """更新罚参数"""
        if iteration < 10:
            # 初始阶段: 残差均衡
            self.rho, _ = self.balancing.update(self.rho, primal_res, dual_res)
        elif iteration < 50:
            # 中期: 残差均衡 + 谱估计的过渡
            rho_balance, _ = self.balancing.update(self.rho, primal_res, dual_res)
            if primal_res_vec is not None and dual_res_vec is not None:
                rho_spectral = self.spectral.update(self.rho, primal_res_vec, dual_res_vec)
                # 混合
                alpha = (iteration - 10) / 40.0  # 从 0 到 1
                self.rho = (1 - alpha) * rho_balance + alpha * rho_spectral
            else:
                self.rho = rho_balance
        else:
            # 后期: 谱自适应
            if primal_res_vec is not None and dual_res_vec is not None:
                self.rho = self.spectral.update(self.rho, primal_res_vec, dual_res_vec)
            else:
                self.rho, _ = self.balancing.update(self.rho, primal_res, dual_res)

        return self.rho
