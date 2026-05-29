"""
measurement_sampler.py
======================
量子测量统计采样与拒绝采样工具

原项目映射:
- 1021_rejection_sample: Chebyshev第二型分布的拒绝采样、CVT密度采样

科学功能:
本模块实现了VQE中所需的概率分布采样方法，包括量子Born规则
概率的有限样本估计、Chebyshev分布拒绝采样（用于某些特定
哈密顿量的谱密度采样），以及用于减少测量方差的CVT密度采样。
"""

import numpy as np
from typing import Tuple, Optional


def chebyshev2_sample(n: int) -> Tuple[np.ndarray, int]:
    """
    Chebyshev第二型分布的拒绝采样，对应 1021_rejection_sample/chebyshev2_sample。

    概率密度: rho(x) = (2/pi) * sqrt(1 - x^2),  x in [-1, 1]

    使用均匀分布作为提议分布，其最大值为 2/pi。
    拒绝采样步骤:
        1. 从 Uniform[-1,1] 采样 x
        2. 从 Uniform[0, 2/pi] 采样 y
        3. 若 y <= (2/pi)*sqrt(1-x^2)，接受 x

    返回:
        samples: (n,) 接受的样本
        n_trials: 总试验次数
    """
    pdfmax = 2.0 / np.pi
    samples = np.zeros(n)
    i = 0
    n_trials = 0
    while i < n:
        x = -1.0 + 2.0 * np.random.rand()
        y = pdfmax * np.random.rand()
        z = (2.0 / np.pi) * np.sqrt(max(1.0 - x ** 2, 0.0))
        n_trials += 1
        if y <= z:
            samples[i] = x
            i += 1
    return samples, n_trials


def cvt_density_sample(n: int, alpha: float = 1.0 / 6.0) -> np.ndarray:
    """
    CVT密度采样，对应 1021_rejection_sample/cvt_1d_sample。

    目标密度: rho(x) \propto 1 / (pi * sqrt(1-x^2))^{alpha}
    当 alpha = 1/6 时，对应将CVT节点映射到Chebyshev点的密度。

    使用拒绝采样从该密度中抽取样本。
    """
    samples = np.zeros(n)
    i = 0
    # pdfmax 在 x=0 处取得
    pdfmax = 1.0 / np.sqrt(np.pi) / (1.0 ** alpha)
    while i < n:
        x = 2.0 * np.random.rand() - 1.0
        y = pdfmax * np.random.rand()
        z = 1.0 / np.sqrt(np.pi) / (max(1.0 - x ** 2, 1e-10) ** alpha)
        if y <= z:
            samples[i] = x
            i += 1
    return samples


class QuantumMeasurementSampler:
    """
    量子测量采样器，模拟从量子态进行有限次投影测量。
    """
    def __init__(self, n_qubits: int, n_shots: int = 8192):
        self.n_qubits = n_qubits
        self.n_shots = n_shots
        self.dim = 2 ** n_qubits

    def sample_pauli_expectation(self, statevector: np.ndarray,
                                  pauli_string: str) -> float:
        """
        通过对Pauli基进行投影测量来估计期望值。
        对非Z的Pauli算符，先旋转到Z基再测量。

        测量方差: Var = (1 - <P>^2) / n_shots
        """
        statevector = np.asarray(statevector, dtype=complex)
        if statevector.shape[0] != self.dim:
            raise ValueError("态向量维度不匹配")

        # 对每个量子比特应用必要的基旋转
        psi = statevector.copy()
        for q, p in enumerate(pauli_string):
            if p == 'X':
                psi = self._apply_hadamard(q, psi)
            elif p == 'Y':
                psi = self._apply_hy(q, psi)

        # 计算Born概率
        probs = np.abs(psi) ** 2
        # 有限样本采样
        shots = np.random.multinomial(self.n_shots, probs)
        # 对测量结果分配 +/-1 本征值
        expectation = 0.0
        for outcome in range(self.dim):
            # 计算该结果对应的Pauli字符串的本征值
            eigenval = 1.0
            for q, p in enumerate(pauli_string):
                if p == 'I':
                    continue
                bit = (outcome >> q) & 1
                eigenval *= (-1.0) ** bit
            expectation += eigenval * shots[outcome]
        expectation /= self.n_shots
        return float(expectation)

    def _apply_hadamard(self, q: int, psi: np.ndarray) -> np.ndarray:
        """在量子比特q上应用Hadamard门。"""
        dim = psi.shape[0]
        psi_out = np.zeros(dim, dtype=complex)
        stride = 2 ** q
        for i in range(dim):
            partner = i ^ stride
            if i <= partner:
                psi_out[i] = (psi[i] + psi[partner]) / np.sqrt(2)
                psi_out[partner] = (psi[i] - psi[partner]) / np.sqrt(2)
        return psi_out

    def _apply_hy(self, q: int, psi: np.ndarray) -> np.ndarray:
        """在量子比特q上应用HS^{dagger}门，将Y基转换到Z基。"""
        dim = psi.shape[0]
        psi_out = np.zeros(dim, dtype=complex)
        stride = 2 ** q
        for i in range(dim):
            partner = i ^ stride
            if i <= partner:
                # |0> -> (|0> + i|1>)/sqrt(2), |1> -> (|0> - i|1>)/sqrt(2)
                psi_out[i] = (psi[i] + 1j * psi[partner]) / np.sqrt(2)
                psi_out[partner] = (psi[i] - 1j * psi[partner]) / np.sqrt(2)
        return psi_out

    def estimate_with_chebyshev_sampling(self, observable_func: callable,
                                          n_samples: int = 1000) -> Tuple[float, float]:
        """
        使用Chebyshev分布采样来估计期望值，降低方差。
        适用于谱密度具有半圆分布特征的系统（如随机矩阵理论中的哈密顿量）。
        """
        samples, trials = chebyshev2_sample(n_samples)
        vals = np.array([observable_func(x) for x in samples])
        mean_est = float(np.mean(vals))
        var_est = float(np.var(vals, ddof=1))
        stderr = np.sqrt(var_est / n_samples)
        return mean_est, stderr

    def sample_bitstrings(self, statevector: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        从量子态采样比特串及其经验频率。
        返回: bitstring数组 (0..2^n-1), 频率数组
        """
        probs = np.abs(statevector) ** 2
        probs = np.clip(probs, 0, 1)
        probs /= np.sum(probs)
        shots = np.random.multinomial(self.n_shots, probs)
        mask = shots > 0
        bitstrings = np.arange(self.dim)[mask]
        frequencies = shots[mask] / self.n_shots
        return bitstrings, frequencies
