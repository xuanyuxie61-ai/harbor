"""
convergence.py — ADMM 收敛性分析与诊断
============================================
来源项目: 1259_EmperorAiphaton_MetricDistortion (LP 度量失真)

本模块实现 ADMM 收敛性的全面分析与诊断工具:
  1. 原始-对偶残差监控
  2. 目标函数下降曲线
  3. 收敛速率估计
  4. 子域间一致性度量
  5. 度量失真诊断 (来源: 1259)

ADMM 收敛理论:
  对于凸问题, ADMM 保证:
    1. 原始残差 r^k = Ax^k + Bz^k - c → 0
    2. 对偶残差 s^k = ρA^T B(z^k - z^{k-1}) → 0
    3. 目标值 f(x^k) + g(z^k) → p* (最优值)

  收敛速率:
    - 一般凸: O(1/k) (ergodic)
    - 强凸: O(1/k²) 或线性 (取决于条件数)

  度量失真 (Metric Distortion, 来源: 1259):
    D(z) = max_i ||x_i - z|| / min_i ||x_i - z||
    衡量子域间解的分散程度.
    当 D → 1 时, 所有子域达成一致.
"""

import numpy as np
from typing import List, Optional, Dict, Tuple
from config import EPS_NUM


# ============================================================
#  收敛诊断数据
# ============================================================
class ConvergenceDiagnostics:
    """ADMM 收敛诊断器

    跟踪和分析 ADMM 迭代的各种指标.
    """

    def __init__(self, n_subdomains: int, n_vars: int):
        self.n_subdomains = n_subdomains
        self.n_vars = n_vars

        self.primal_residuals: List[float] = []
        self.dual_residuals: List[float] = []
        self.objective_values: List[float] = []
        self.rho_values: List[float] = []
        self.consensus_errors: List[float] = []
        self.distortion_values: List[float] = []

    def record(self, iteration: int, primal_res: float, dual_res: float,
               objective: float, rho: float,
               x_locals: Optional[List[np.ndarray]] = None,
               z_consensus: Optional[np.ndarray] = None):
        """记录一次迭代的指标"""
        self.primal_residuals.append(primal_res)
        self.dual_residuals.append(dual_res)
        self.objective_values.append(objective)
        self.rho_values.append(rho)

        if x_locals is not None and z_consensus is not None:
            # 一致性误差: max_i ||x_i - z||
            cons_err = max(np.linalg.norm(x_i - z_consensus) for x_i in x_locals)
            self.consensus_errors.append(cons_err)

            # 度量失真
            norms = [np.linalg.norm(x_i - z_consensus) for x_i in x_locals]
            max_norm = max(norms)
            min_norm = min(norms) if min(norms) > EPS_NUM else EPS_NUM
            self.distortion_values.append(max_norm / min_norm)

    @property
    def n_iterations(self) -> int:
        return len(self.primal_residuals)

    # ---- 收敛速率估计 ----

    def estimate_primal_convergence_rate(self) -> float:
        """估计原始残差的收敛速率

        假设 ||r^k|| ≈ C * k^{-α}, 则:
          α ≈ -d log||r|| / d log(k)

        使用最近 20% 的迭代数据.
        """
        if len(self.primal_residuals) < 10:
            return 0.0

        n = len(self.primal_residuals)
        start = int(0.8 * n)
        logs = []
        for k in range(start, n):
            r = self.primal_residuals[k]
            if r > EPS_NUM:
                logs.append((np.log(k + 1), np.log(r)))

        if len(logs) < 3:
            return 0.0

        logs = np.array(logs)
        # 线性回归: log(r) = a * log(k) + b
        A_mat = np.column_stack([logs[:, 0], np.ones(len(logs))])
        result = np.linalg.lstsq(A_mat, logs[:, 1], rcond=None)
        return -result[0][0]  # α = -slope

    def estimate_dual_convergence_rate(self) -> float:
        """估计对偶残差的收敛速率"""
        if len(self.dual_residuals) < 10:
            return 0.0

        n = len(self.dual_residuals)
        start = int(0.8 * n)
        logs = []
        for k in range(start, n):
            s = self.dual_residuals[k]
            if s > EPS_NUM:
                logs.append((np.log(k + 1), np.log(s)))

        if len(logs) < 3:
            return 0.0

        logs = np.array(logs)
        A_mat = np.column_stack([logs[:, 0], np.ones(len(logs))])
        result = np.linalg.lstsq(A_mat, logs[:, 1], rcond=None)
        return -result[0][0]

    # ---- 最优性度量 ----

    def compute_duality_gap(self) -> float:
        """计算对偶间隙 (近似)

        对于 ADMM, 对偶间隙:
          gap = f(x^k) + g(z^k) - p*
        其中 p* 为最优值的下界估计.

        近似: p* ≈ min_k [f(x^k) + g(z^k)]
        """
        if not self.objective_values:
            return float('inf')
        best = min(self.objective_values)
        current = self.objective_values[-1]
        return current - best

    def compute_suboptimality(self) -> float:
        """计算次优性 (相对)

        subopt = (f(x^k) + g(z^k) - p*) / (|p*| + 1)
        """
        if not self.objective_values:
            return float('inf')
        best = min(self.objective_values)
        current = self.objective_values[-1]
        return abs(current - best) / (abs(best) + 1.0)

    # ---- 子域一致性分析 ----

    def compute_consensus_statistics(self) -> dict:
        """计算子域一致性的统计量

        Returns:
            dict: max_error, mean_error, distortion
        """
        if not self.consensus_errors:
            return {'max_error': 0.0, 'mean_error': 0.0, 'distortion': 1.0}

        return {
            'max_error': max(self.consensus_errors),
            'mean_error': np.mean(self.consensus_errors),
            'final_error': self.consensus_errors[-1],
            'distortion': self.distortion_values[-1] if self.distortion_values else 1.0,
        }

    # ---- 罚参数分析 ----

    def analyze_rho_evolution(self) -> dict:
        """分析罚参数的演化

        Returns:
            dict: initial, final, min, max, n_changes
        """
        if not self.rho_values:
            return {'initial': 1.0, 'final': 1.0, 'min': 1.0, 'max': 1.0, 'n_changes': 0}

        rhos = np.array(self.rho_values)
        n_changes = sum(1 for i in range(1, len(rhos)) if abs(rhos[i] - rhos[i-1]) > EPS_NUM)

        return {
            'initial': rhos[0],
            'final': rhos[-1],
            'min': np.min(rhos),
            'max': np.max(rhos),
            'mean': np.mean(rhos),
            'n_changes': n_changes,
        }


# ============================================================
#  收敛判据
# ============================================================
def check_admm_convergence(primal_res: float, dual_res: float,
                            x_locals: List[np.ndarray], z: np.ndarray,
                            u_duals: List[np.ndarray],
                            abs_tol: float = 1e-5,
                            rel_tol: float = 1e-3) -> Tuple[bool, dict]:
    """ADMM 收敛判据 (Boyd et al. 2011)

    停止条件:
      ||r^k|| ≤ ε_pri   (原始可行性)
      ||s^k|| ≤ ε_dual  (对偶可行性)

    其中:
      ε_pri = √n ε_abs + ε_rel max(||Ax^k||, ||Bz^k||, ||c||)
      ε_dual = √p ε_abs + ε_rel ||ρ A^T λ^k||

    Returns:
        (converged, info): 是否收敛及详细信息
    """
    N = len(x_locals)
    n = len(z)

    # 原始容差
    x_norms = [np.linalg.norm(x_i) for x_i in x_locals]
    z_norm = np.linalg.norm(z)
    eps_pri = np.sqrt(n * N) * abs_tol + rel_tol * max(max(x_norms), np.sqrt(N) * z_norm)

    # 对偶容差
    u_norms = [np.linalg.norm(u_i) for u_i in u_duals]
    eps_dual = np.sqrt(n) * abs_tol + rel_tol * sum(u_norms)

    converged = primal_res < eps_pri and dual_res < eps_dual

    info = {
        'primal_residual': primal_res,
        'dual_residual': dual_res,
        'eps_primal': eps_pri,
        'eps_dual': eps_dual,
        'primal_converged': primal_res < eps_pri,
        'dual_converged': dual_res < eps_dual,
    }

    return converged, info


# ============================================================
#  综合诊断报告
# ============================================================
def generate_convergence_report(diagnostics: ConvergenceDiagnostics) -> str:
    """生成收敛诊断报告

    Returns:
        str: 格式化的报告文本
    """
    lines = []
    lines.append("=" * 60)
    lines.append("  ADMM 收敛诊断报告")
    lines.append("=" * 60)

    n_iter = diagnostics.n_iterations
    lines.append(f"\n总迭代次数: {n_iter}")

    if n_iter == 0:
        lines.append("  (无迭代数据)")
        return "\n".join(lines)

    # 残差
    lines.append(f"\n--- 残差分析 ---")
    lines.append(f"  初始原始残差: {diagnostics.primal_residuals[0]:.6e}")
    lines.append(f"  最终原始残差: {diagnostics.primal_residuals[-1]:.6e}")
    lines.append(f"  初始对偶残差: {diagnostics.dual_residuals[0]:.6e}")
    lines.append(f"  最终对偶残差: {diagnostics.dual_residuals[-1]:.6e}")

    # 收敛速率
    alpha_p = diagnostics.estimate_primal_convergence_rate()
    alpha_d = diagnostics.estimate_dual_convergence_rate()
    lines.append(f"  原始收敛速率: O(k^(-{alpha_p:.3f}))")
    lines.append(f"  对偶收敛速率: O(k^(-{alpha_d:.3f}))")

    # 目标值
    if diagnostics.objective_values:
        lines.append(f"\n--- 目标函数 ---")
        lines.append(f"  初始目标值: {diagnostics.objective_values[0]:.6e}")
        lines.append(f"  最终目标值: {diagnostics.objective_values[-1]:.6e}")
        lines.append(f"  最优目标值: {min(diagnostics.objective_values):.6e}")
        lines.append(f"  对偶间隙:   {diagnostics.compute_duality_gap():.6e}")

    # 一致性
    cons_stats = diagnostics.compute_consensus_statistics()
    lines.append(f"\n--- 子域一致性 ---")
    lines.append(f"  最大一致性误差: {cons_stats['max_error']:.6e}")
    lines.append(f"  平均一致性误差: {cons_stats['mean_error']:.6e}")
    lines.append(f"  最终一致性误差: {cons_stats['final_error']:.6e}")
    lines.append(f"  度量失真:       {cons_stats['distortion']:.4f}")

    # 罚参数
    rho_stats = diagnostics.analyze_rho_evolution()
    lines.append(f"\n--- 罚参数演化 ---")
    lines.append(f"  初始 ρ: {rho_stats['initial']:.4e}")
    lines.append(f"  最终 ρ: {rho_stats['final']:.4e}")
    lines.append(f"  最小 ρ: {rho_stats['min']:.4e}")
    lines.append(f"  最大 ρ: {rho_stats['max']:.4e}")
    lines.append(f"  变化次数: {rho_stats['n_changes']}")

    lines.append("\n" + "=" * 60)
    return "\n".join(lines)
