"""
matrix_chain_optimizer.py
=========================
矩阵链乘法最优顺序的动态规划优化器。

融合种子项目：
  - 740_matrix_chain_dynamic : 矩阵链动态规划最优括号化

科学应用：
  在非线性声学的高维谱模拟中，涉及大量矩阵链运算（如 POD 模态投影、
  高阶张量收缩、多步预条件子应用）。本模块使用动态规划寻找最优计算
  顺序，最小化浮点运算次数 (FLOPs)。

  经典问题：给定矩阵链 :math:`A_1 \times A_2 \times \dots \times A_n`，
  其中 :math:`A_i` 的维度为 :math:`d_{i-1} \times d_i`，寻找括号化方案
  使得标量乘法次数最少。

  动态规划递推：
  .. math::
      m[i,j] = \min_{i \le k < j} \left\{
          m[i,k] + m[k+1,j] + d_{i-1} d_k d_j
      \right\}
"""

import numpy as np


def matrix_chain_optimal_order(dims):
    """
    矩阵链最优乘法顺序。

    原始算法来自 740_matrix_chain_dynamic/matrix_chain_dynamic.m。

    Parameters
    ----------
    dims : list or np.ndarray
        维度向量，长度为 n+1。
        矩阵 A_i 的维度为 dims[i] x dims[i+1]。

    Returns
    -------
    int
        最小标量乘法次数。
    np.ndarray, shape (n, n)
        最优分割点记录矩阵 s。
    """
    dims = np.asarray(dims, dtype=int)
    n = len(dims) - 1
    if n < 2:
        return 0, np.zeros((max(n, 1), max(n, 1)), dtype=int)
    if np.any(dims <= 0):
        raise ValueError("All dimensions must be positive.")

    m = np.full((n, n), np.inf, dtype=float)
    s = np.zeros((n, n), dtype=int)

    for i in range(n):
        m[i, i] = 0.0

    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            for k in range(i, j):
                cost = m[i, k] + m[k + 1, j] + dims[i] * dims[k + 1] * dims[j + 1]
                if cost < m[i, j]:
                    m[i, j] = cost
                    s[i, j] = k

    return int(m[0, n - 1]), s


def print_optimal_parens(s, i, j):
    """
    输出最优括号化方案。

    Parameters
    ----------
    s : np.ndarray
        分割点矩阵。
    i, j : int
        当前子链范围。

    Returns
    -------
    str
        括号化字符串。
    """
    if i == j:
        return f"A{i+1}"
    else:
        left = print_optimal_parens(s, i, s[i, j])
        right = print_optimal_parens(s, s[i, j] + 1, j)
        return f"({left} x {right})"


class TensorChainOptimizer:
    """
    张量链运算优化器，用于声学模拟中的高阶张量收缩。
    """

    def __init__(self, tensor_dims):
        """
        Parameters
        ----------
        tensor_dims : list of tuple
            每个张量的维度元组。
        """
        self.tensor_dims = tensor_dims

    def optimize_einsum_chain(self, contractions):
        """
        对 einsum 链进行运算顺序优化。

        Parameters
        ----------
        contractions : list of tuple
            每个元素为 ((i_left, i_right), shared_indices)。

        Returns
        -------
        list
            最优运算顺序。
        int
            最小FLOPs估计。
        """
        n = len(self.tensor_dims)
        # 简化为矩阵链问题：将每个张量展平为矩阵
        # 实际中需要更精细的指标分析，这里做概念性实现
        dims = [1] * (n + 1)
        for i, td in enumerate(self.tensor_dims):
            if len(td) >= 2:
                dims[i] = td[0]
                dims[i + 1] = td[-1]
            else:
                dims[i] = td[0]
                dims[i + 1] = 1

        cost, s = matrix_chain_optimal_order(dims)
        order_str = print_optimal_parens(s, 0, n - 1) if n > 0 else ""
        return order_str, cost


def apply_optimal_matrix_chain(matrices, s, i=None, j=None):
    """
    按照最优顺序实际执行矩阵链乘法。

    Parameters
    ----------
    matrices : list of np.ndarray
        矩阵列表 A_1, ..., A_n。
    s : np.ndarray
        最优分割矩阵。
    i, j : int or None
        当前范围。None 则计算整个链。

    Returns
    -------
    np.ndarray
        乘积结果。
    """
    n = len(matrices)
    if n == 0:
        raise ValueError("Empty matrix chain.")
    if n == 1:
        return matrices[0]
    if i is None:
        i = 0
    if j is None:
        j = n - 1

    if i == j:
        return matrices[i]

    k = s[i, j]
    left = apply_optimal_matrix_chain(matrices, s, i, k)
    right = apply_optimal_matrix_chain(matrices, s, k + 1, j)
    return left @ right


class AcousticOperatorChain:
    r"""
    声学算子链：封装谱模拟中的多步矩阵运算优化。

    典型算子链：
    .. math::
        u^{n+1} = D^{-1} M^{-1} K D u^n

    其中 D 为微分矩阵，M 为质量矩阵，K 为刚度矩阵。
    """

    def __init__(self, operators):
        """
        Parameters
        ----------
        operators : list of np.ndarray
            算子矩阵列表。
        """
        self.operators = operators
        self.dims = [op.shape[0] for op in operators]
        self.dims.append(operators[-1].shape[1])
        self._optimal_s = None
        self._optimal_cost = None

    def optimize(self):
        """
        优化算子链乘法顺序。
        """
        cost, s = matrix_chain_optimal_order(self.dims)
        self._optimal_cost = cost
        self._optimal_s = s
        return cost

    def apply(self, vector):
        """
        将优化后的算子链应用于向量。

        Parameters
        ----------
        vector : np.ndarray
            输入向量。

        Returns
        -------
        np.ndarray
            输出向量。
        """
        if self._optimal_s is None:
            self.optimize()

        # 将向量视为最后一个矩阵（对角）以纳入链优化
        # 简化：直接顺序相乘（向量在右侧）
        result = vector.copy()
        for op in reversed(self.operators):
            if result.ndim == 1:
                result = op @ result
            else:
                result = op @ result
        return result

    def flops_estimate(self):
        """
        返回最优顺序的 FLOPs 估计。
        """
        if self._optimal_cost is None:
            self.optimize()
        return self._optimal_cost
