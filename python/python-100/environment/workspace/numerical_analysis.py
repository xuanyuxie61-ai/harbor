"""
numerical_analysis.py
================================================================================
数值稳定性分析与误差估计

融合原项目:
  - 207_condition : 矩阵L1条件数估计（Hager/LINPACK算法）

核心科学内容:
  1. 有限元刚度矩阵的条件数估计:

     对于对称正定矩阵 K，条件数:

         κ(K) = λ_max / λ_min

     其中 λ_max, λ_min 分别为最大和最小特征值。

     条件数与数值误差的关系:
         ||δx|| / ||x|| ≤ κ(K) * ||δb|| / ||b||

     即右端项的相对误差放大 κ(K) 倍传递到解向量。

  2. Hager 条件数估计算法（O(n^2) 复杂度）:

     估计 ||A^{-1}||_1 的迭代过程:
         b = (1/n) * ones(n, 1)
         while not converged:
             x = A \ b
             c = ||x||_1
             b = sign(x)
             y = A^T \ b
             j = argmax |y_i|
             if y_j == c: break
             b = e_j

     cond_1(A) = c * ||A||_1

  3. LINPACK 条件数估计算法:

     通过求解辅助线性系统估计逆矩阵的范数:
         A z = y,  A^T w = v

     其中 y, v 的选择旨在产生最大局部增长。

  4. 有限元离散误差估计（先验估计）:

     对于 P1 有限元，H^1 误差满足:

         ||u - u_h||_{H^1(Ω)} ≤ C h |u|_{H^2(Ω)}

     其中 h 为最大单元尺寸，C 为与网格形状相关的常数。

  5. 网格 Peclet 数（对流-扩散稳定性）:

         Pe = |v| h / (2 D)

     当 Pe > 1 时，中心差分格式出现数值振荡，需采用迎风格式或
     人工扩散 stabilization。
================================================================================
"""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve


class ConditionEstimator:
    """
    矩阵条件数估计器，融合原项目 207_condition 的 Hager 与 LINPACK 算法。
    """

    @staticmethod
    def matrix_1norm(A: np.ndarray) -> float:
        """计算矩阵的 L1 范数: ||A||_1 = max_j Σ_i |A_{ij}|."""
        return np.max(np.sum(np.abs(A), axis=0))

    @classmethod
    def hager_estimate(cls, A: np.ndarray, max_iter: int = 5, tol: float = 1.0e-8) -> float:
        """
        Hager L1 条件数估计算法。

        估计 cond_1(A) = ||A||_1 * ||A^{-1}||_1.

        参考文献:
            William Hager, "Condition Estimates", SIAM J. Sci. Stat. Comput.,
            Vol. 5, No. 2, 1984, pp. 311-316.
        """
        n = A.shape[0]
        if A.shape[0] != A.shape[1]:
            raise ValueError("矩阵必须为方阵")

        anorm = cls.matrix_1norm(A)
        if anorm < 1.0e-30:
            return np.inf

        # 初始向量
        b = np.ones(n) / n
        c1 = 0.0
        i1 = -1

        for _ in range(max_iter):
            try:
                x = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                return np.inf

            c2 = np.sum(np.abs(x))
            b = np.sign(x)
            # 处理 x_i = 0 的情况
            b[b == 0.0] = 1.0

            try:
                y = np.linalg.solve(A.T, b)
            except np.linalg.LinAlgError:
                return np.inf

            i2 = np.argmax(np.abs(y))

            if i1 >= 0:
                if i1 == i2 or c2 <= c1 * (1.0 + tol):
                    break

            i1 = i2
            c1 = c2
            b = np.zeros(n)
            b[i1] = 1.0

        # 避免 c2 = 0 的退化情况
        if c2 < 1.0e-30:
            c2 = 1.0e-30

        return float(c2 * anorm)

    @classmethod
    def linpack_estimate(cls, A: np.ndarray) -> float:
        """
        LINPACK 风格条件数估计（简化版）。

        融合原项目 207_condition 的 condition_linpack 核心思想。
        """
        n = A.shape[0]
        anorm = cls.matrix_1norm(A)

        try:
            A_inv = np.linalg.inv(A)
        except np.linalg.LinAlgError:
            return np.inf

        ainv_norm = cls.matrix_1norm(A_inv)
        if ainv_norm < 1.0e-30:
            ainv_norm = 1.0e-30

        return float(anorm * ainv_norm)

    @classmethod
    def sparse_hager_estimate(cls, K: csr_matrix, max_iter: int = 5) -> float:
        """针对稀疏矩阵的 Hager 条件数估计."""
        n = K.shape[0]
        anorm = np.max(np.sum(np.abs(K.toarray()), axis=0))
        if anorm < 1.0e-30:
            return np.inf

        b = np.ones(n) / n
        c1 = 0.0
        i1 = -1

        for _ in range(max_iter):
            x = spsolve(K, b)
            if x is None:
                return np.inf
            c2 = np.sum(np.abs(x))
            b = np.sign(x)
            b[b == 0.0] = 1.0

            y = spsolve(K.T, b)
            if y is None:
                return np.inf
            i2 = np.argmax(np.abs(y))

            if i1 >= 0:
                if i1 == i2 or c2 <= c1:
                    break

            i1 = i2
            c1 = c2
            b = np.zeros(n)
            b[i1] = 1.0

        return float(c2 * anorm)


class FEMErrorEstimator:
    """
    有限元离散误差估计器。
    """

    @staticmethod
    def max_element_size(nodes: np.ndarray, elements: np.ndarray) -> float:
        """计算最大单元边长."""
        h_max = 0.0
        for e in range(elements.shape[0]):
            v = elements[e]
            for i in range(3):
                for j in range(i + 1, 3):
                    h = np.linalg.norm(nodes[v[i]] - nodes[v[j]])
                    h_max = max(h_max, h)
        return h_max

    @staticmethod
    def h1_error_estimate(h_max: float, u_h2_seminorm: float, C_interp: float = 1.0) -> float:
        """
        P1 有限元的先验 H^1 误差估计.

            ||u - u_h||_{H^1} ≤ C h |u|_{H^2}

        参数:
            h_max         : 最大单元尺寸
            u_h2_seminorm : |u|_{H^2} = sqrt( ∫ |D^2 u|^2 dΩ )
            C_interp      : 插值常数（对三角形 P1 元约 0.5-1.0）
        """
        return C_interp * h_max * u_h2_seminorm

    @staticmethod
    def peclet_number(velocity: float, h_max: float, diffusivity: float) -> float:
        """
        计算网格 Peclet 数.

            Pe = |v| h / (2 D)

        当 Pe > 1 时，标准 Galerkin 方法可能出现数值振荡。
        """
        if diffusivity <= 0.0:
            return np.inf
        return abs(velocity) * h_max / (2.0 * diffusivity)

    @staticmethod
    def check_stiffness_positive_definite(K_dense: np.ndarray) -> dict:
        """
        检查刚度矩阵的正定性。

        返回:
            {
                'is_spd': 是否对称正定,
                'min_eig': 最小特征值,
                'max_eig': 最大特征值,
                'cond_2': 谱条件数
            }
        """
        if not np.allclose(K_dense, K_dense.T, atol=1.0e-10):
            return {
                "is_spd": False,
                "min_eig": np.nan,
                "max_eig": np.nan,
                "cond_2": np.inf,
                "symmetry_error": float(np.max(np.abs(K_dense - K_dense.T))),
            }

        eigvals = np.linalg.eigvalsh(K_dense)
        min_eig = np.min(eigvals)
        max_eig = np.max(eigvals)
        cond_2 = max_eig / (abs(min_eig) + 1.0e-30)

        return {
            "is_spd": min_eig > 0.0,
            "min_eig": float(min_eig),
            "max_eig": float(max_eig),
            "cond_2": float(cond_2),
        }


class NumericalRobustness:
    """
    数值鲁棒性工具集：边界处理、病态检测、迭代收敛监控。
    """

    @staticmethod
    def safe_divide(a: np.ndarray, b: np.ndarray, eps: float = 1.0e-14) -> np.ndarray:
        """安全除法，防止除以零."""
        b_safe = np.where(np.abs(b) < eps, eps * np.sign(b + eps), b)
        return a / b_safe

    @staticmethod
    def clip_gradient(grad: np.ndarray, max_norm: float = 1.0e6) -> np.ndarray:
        """梯度裁剪，防止爆炸."""
        norm = np.linalg.norm(grad)
        if norm > max_norm:
            return grad * (max_norm / norm)
        return grad

    @staticmethod
    def check_convergence(residual_norm: float, tol: float, iter_idx: int, max_iter: int) -> bool:
        """检查迭代收敛性."""
        if residual_norm < tol:
            return True
        if iter_idx >= max_iter:
            return False
        if not np.isfinite(residual_norm):
            raise RuntimeError(f"迭代 {iter_idx}: 残差出现非有限值")
        return False

    @staticmethod
    def regularize_singular_matrix(A: np.ndarray, reg: float = 1.0e-12) -> np.ndarray:
        """通过添加正则项处理奇异矩阵."""
        A_reg = A.copy()
        diag_idx = np.diag_indices_from(A_reg)
        A_reg[diag_idx] += reg
        return A_reg
