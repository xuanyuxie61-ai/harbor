"""
全局灵敏度分析模块 (Global Sensitivity Analysis)
====================================================
基于多项式混沌展开的 Sobol 灵敏度指数计算。

Sobol 方差分解:
  Var[u] = Σ_{d=1}^D V_d + Σ_{d<d'} V_{dd'} + ... + V_{12...D}

  其中:
    V_d = Var_{ξ_d}[E[u|ξ_d]]  — 一阶效应
    V_{dd'} = Var_{ξ_d,ξ_{d'}}[E[u|ξ_d,ξ_{d'}]] - V_d - V_{d'}  — 二阶交互

Sobol 指数:
  一阶: S_d = V_d / Var[u]
  总效应: S_{T,d} = 1 - Var_{ξ_{~d}}[E[u|ξ_{~d}]] / Var[u]
         = (Var[u] - V_{~d}) / Var[u]

  其中 V_{~d} 是不包含 ξ_d 的方差分量。

基于 PC 的 Sobol 指数:
  利用 PC 正交性, 方差分解直接由系数给出:
    Var[u] = Σ_{α≠0} c_α² ||Ψ_α||²

  一阶 Sobol 指数:
    S_d = (Σ_{α: α_d>0, α_{j}=0 for j≠d} c_α² ||Ψ_α||²) / Var[u]

  总 Sobol 指数:
    S_{T,d} = (Σ_{α: α_d>0} c_α² ||Ψ_α||²) / Var[u]

参考文献:
  Sobol, I.M. (2001). Global sensitivity indices for nonlinear mathematical models.
  Sudret, B. (2008). Global sensitivity analysis using polynomial chaos expansions.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional


class SobolAnalyzer:
    """
    Sobol 全局灵敏度分析器。

    从多项式混沌系数计算一阶和总效应 Sobol 指数,
    无需额外的模型评估。

    性质:
      Σ_d S_d ≤ 1  (一阶指数之和 ≤ 1)
      Σ_d S_{T,d} ≥ 1  (总指数之和 ≥ 1)
      S_d ≤ S_{T,d}  (一阶 ≤ 总效应)
      S_d = S_{T,d} ⟺ 无交互效应
    """

    def __init__(
        self,
        pc_coefficients: np.ndarray,
        multi_indices: List[Tuple[int, ...]],
        basis_norms_sq: np.ndarray,
        dimension: int
    ):
        """
        参数:
            pc_coefficients: PC 系数 c_α, shape (n_basis,) 或 (n_basis, N_x)
            multi_indices: 多指标列表
            basis_norms_sq: 基函数范数平方, shape (n_basis,)
            dimension: 随机维度 D
        """
        self.coeffs = np.asarray(pc_coefficients, dtype=np.float64)
        if self.coeffs.ndim == 1:
            self.coeffs = self.coeffs[:, np.newaxis]  # (n_basis, 1)
        self.multi_indices = multi_indices
        self.norms_sq = basis_norms_sq
        self.D = dimension
        self.n_basis = len(multi_indices)

        if self.coeffs.shape[0] != self.n_basis:
            raise ValueError(
                f"Coefficients shape mismatch: {self.coeffs.shape[0]} != {self.n_basis}"
            )

    def total_variance(self) -> np.ndarray:
        """
        计算总方差:
          Var[u] = Σ_{α: |α|>0} c_α² ||Ψ_α||²

        注意: 常数项 (α=0) 对应均值, 不计入方差。

        返回: shape (N_x,)
        """
        var = np.zeros(self.coeffs.shape[1])
        for j in range(self.n_basis):
            alpha = self.multi_indices[j]
            if sum(alpha) > 0:  # 排除常数项
                var += self.coeffs[j] ** 2 * self.norms_sq[j]
        return var

    def first_order_sobol(self) -> Dict[int, np.ndarray]:
        """
        计算一阶 Sobol 指数:
          S_d = (Σ_{α: α_d>0, α_{j≠d}=0} c_α² ||Ψ_α||²) / Var[u]

        这捕获了第 d 个输入变量的单独 (非交互) 效应。

        返回:
            dict: {d: S_d array} for d = 0,...,D-1
        """
        var_total = self.total_variance()
        var_total = np.maximum(var_total, 1e-30)

        sobol = {}
        for d in range(self.D):
            var_d = np.zeros(self.coeffs.shape[1])
            for j in range(self.n_basis):
                alpha = self.multi_indices[j]
                # α_d > 0 且所有其他 α_j = 0
                if alpha[d] > 0 and sum(alpha) == alpha[d]:
                    var_d += self.coeffs[j] ** 2 * self.norms_sq[j]
            sobol[d] = var_d / var_total
        return sobol

    def total_order_sobol(self) -> Dict[int, np.ndarray]:
        """
        计算总效应 Sobol 指数:
          S_{T,d} = (Σ_{α: α_d>0} c_α² ||Ψ_α||²) / Var[u]
                  = 1 - (Σ_{α: α_d=0} c_α² ||Ψ_α||²) / Var[u]

        这捕获了第 d 个输入变量的全部效应 (包括所有交互)。

        等价公式:
          S_{T,d} = 1 - Var_{~d} / Var
          其中 Var_{~d} = Σ_{α: α_d=0} c_α² ||Ψ_α||²

        返回:
            dict: {d: S_T_d array} for d = 0,...,D-1
        """
        var_total = self.total_variance()
        var_total = np.maximum(var_total, 1e-30)

        sobol_T = {}
        for d in range(self.D):
            # Var_{~d}: 不包含 ξ_d 的方差
            var_excl = np.zeros(self.coeffs.shape[1])
            for j in range(self.n_basis):
                alpha = self.multi_indices[j]
                if alpha[d] == 0 and sum(alpha) > 0:
                    var_excl += self.coeffs[j] ** 2 * self.norms_sq[j]
            sobol_T[d] = 1.0 - var_excl / var_total
        return sobol_T

    def interaction_indices(self) -> Dict[Tuple[int, ...], np.ndarray]:
        """
        计算二阶交互 Sobol 指数:
          S_{dd'} = (Σ_{α: α_d>0, α_{d'}>0, α_{j}=0 for j≠d,d'} c_α² ||Ψ_α||²) / Var

        返回:
            dict: {(d,d'): S_{dd'} array}
        """
        var_total = self.total_variance()
        var_total = np.maximum(var_total, 1e-30)

        interactions = {}
        for d1 in range(self.D):
            for d2 in range(d1 + 1, self.D):
                var_int = np.zeros(self.coeffs.shape[1])
                for j in range(self.n_basis):
                    alpha = self.multi_indices[j]
                    # 仅 d1 和 d2 的指数 > 0
                    active = [dd for dd in range(self.D) if alpha[dd] > 0]
                    if set(active) == {d1, d2}:
                        var_int += self.coeffs[j] ** 2 * self.norms_sq[j]
                interactions[(d1, d2)] = var_int / var_total
        return interactions

    def effective_dimension(self) -> float:
        """
        计算有效维度 (Effective Dimension):
          d_eff = Σ_d S_{T,d}

        物理意义:
          d_eff = 1: 加性模型 (无交互)
          d_eff = D: 完全耦合 (所有维度同等重要)
          d_eff << D: 低维结构, 可用降维方法

        返回:
            有效维度 (标量)
        """
        sobol_T = self.total_order_sobol()
        # 对空间平均
        d_eff = 0.0
        for d in range(self.D):
            d_eff += float(np.mean(sobol_T[d]))
        return d_eff

    def important_dimensions(self, threshold: float = 0.01) -> List[int]:
        """
        识别重要维度 (基于总效应 Sobol 指数):
          维度 d 是重要的 if max(S_{T,d}) > threshold

        参数:
            threshold: 重要性阈值

        返回:
            重要维度的索引列表
        """
        sobol_T = self.total_order_sobol()
        important = []
        for d in range(self.D):
            if np.max(np.abs(sobol_T[d])) > threshold:
                important.append(d)
        return important

    def summary(self) -> Dict[str, object]:
        """
        生成灵敏度分析摘要。
        """
        var = self.total_variance()
        S1 = self.first_order_sobol()
        ST = self.total_order_sobol()

        result = {
            'total_variance_mean': float(np.mean(var)),
            'effective_dimension': self.effective_dimension(),
            'first_order': {},
            'total_order': {},
        }

        for d in range(self.D):
            result['first_order'][d] = {
                'mean': float(np.mean(S1[d])),
                'max': float(np.max(S1[d])),
            }
            result['total_order'][d] = {
                'mean': float(np.mean(ST[d])),
                'max': float(np.max(ST[d])),
            }

        return result
