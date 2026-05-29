r"""
entanglement_metrics.py
=======================
量子纠缠度量计算模块。

在双光子纠缠光源中，常用的纠缠度量包括：
1. **Concurrence**（并发度）：对于双体态，:math:`C = \sqrt{2(1-\text{Tr}\rho_r^2)}`。
2. **纠缠熵**（von Neumann entropy）：:math:`S = -\text{Tr}(\rho_r \log_2 \rho_r)`。
3. **Hong-Ou-Mandel (HOM) 干涉可见度**：

.. math::
    V = \frac{N_{\max} - N_{\min}}{N_{\max} + N_{\min}}

4. **CHSH 不等式 S 参数**：

.. math::
    S_{\text{CHSH}} = |E(a,b) - E(a,b') + E(a',b) + E(a',b')| \le 2\sqrt{2}

对于理想纠缠态 :math:`|\Psi^+\rangle = \frac{1}{\sqrt{2}}(|H V\rangle + |V H\rangle)`，
:math:`S_{\text{CHSH}} = 2\sqrt{2}`。

5. **态保真度**（State Fidelity）：

.. math::
    F = \langle \Psi_{\text{target}} | \rho | \Psi_{\text{target}} \rangle
"""

import numpy as np
from scipy.linalg import sqrtm
from typing import Tuple


def concurrence_from_purity(purity: float) -> float:
    r"""
    由约化密度矩阵纯度计算 concurrence。

    对于双光子纯态，:math:`\mathcal{P} = \text{Tr}(\rho_r^2)`，
    concurrence 为

    .. math::
        C = \sqrt{2(1 - \mathcal{P})}

    参数
    ----
    purity : float
        纯度，:math:`\in [0, 1]`。

    返回
    ----
    C : float
        Concurrence，:math:`\in [0, 1]`。
    """
    # TODO (Hole 2): 由纯度计算 Concurrence
    # 对于双光子纯态，C = sqrt(2(1 - P))，需做合法性截断
    raise NotImplementedError("Hole 2: 请实现 concurrence_from_purity")


def von_neumann_entropy_schmidt(lambdas: np.ndarray) -> float:
    r"""
    由 Schmidt 系数计算 von Neumann 纠缠熵。

    .. math::
        S = -\sum_n \lambda_n \log_2 \lambda_n

    参数
    ----
    lambdas : np.ndarray
        Schmidt 系数，和为 1。

    返回
    ----
    S : float
        纠缠熵，比特。
    """
    lambdas = np.asarray(lambdas)
    lambdas = lambdas[lambdas > 1e-15]
    S = -np.sum(lambdas * np.log2(lambdas))
    return max(0.0, S)


def hom_visibility(jsa_real: np.ndarray, jsa_imag: np.ndarray,
                   delay_grid: np.ndarray,
                   omega_s: np.ndarray,
                   omega_i: np.ndarray) -> Tuple[np.ndarray, float]:
    r"""
    计算 Hong-Ou-Mandel 干涉可见度随时间延迟的变化。

    双光子符合计数率

    .. math::
        R(\tau) \propto 1 - \int d\omega_s d\omega_i \,
        |f(\omega_s, \omega_i)|^2 \cos[ (\omega_s - \omega_i) \tau ]

    可见度

    .. math::
        V = \frac{R_{\max} - R_{\min}}{R_{\max} + R_{\min}}

    参数
    ----
    jsa_real, jsa_imag : np.ndarray, shape (n_s, n_i)
        JSA 的实部与虚部。
    delay_grid : np.ndarray
        时间延迟 :math:`\tau`，单位 s。

    返回
    ----
    R_tau : np.ndarray
        归一化符合计数率。
    V : float
        HOM 可见度。
    """
    jsa = jsa_real + 1j * jsa_imag
    n_s, n_i = jsa.shape
    # 频率轴假设为均匀网格
    # 计算 |f|^2
    f2 = np.abs(jsa) ** 2

    # 构建频率差矩阵
    Os, Oi = np.meshgrid(omega_s, omega_i, indexing='ij')
    dw = Os - Oi

    # 近似频率差积分
    R_tau = np.zeros_like(delay_grid, dtype=np.float64)
    for idx, tau in enumerate(delay_grid):
        cos_kernel = np.cos(dw * tau)
        R_tau[idx] = 1.0 - np.sum(f2 * cos_kernel) / np.sum(f2)

    R_max = np.max(R_tau)
    R_min = np.min(R_tau)
    denom = R_max + R_min
    V = (R_max - R_min) / denom if denom > 1e-15 else 0.0
    return R_tau, V


def state_fidelity_target(jsa: np.ndarray,
                          target_type: str = "singlet") -> float:
    r"""
    计算生成态与目标 Bell 态的保真度。

    对于频域连续变量，目标态投影到离散双光子子空间：

    .. math::
        F = |\langle \Psi_{\text{target}} | \Psi \rangle|^2
          = \left| \int d\omega_s d\omega_i \,
          f_{\text{target}}^*(\omega_s, \omega_i)
          f(\omega_s, \omega_i) \right|^2

    参数
    ----
    jsa : np.ndarray
        归一化 JSA。
    target_type : str
        "singlet" (:math:`|HV\rangle+|VH\rangle`) 或
        "triplet" (:math:`|HH\rangle+|VV\rangle`)。

    返回
    ----
    F : float
        保真度。
    """
    jsa = np.asarray(jsa)
    n_s, n_i = jsa.shape
    if n_s != n_i:
        # 非对称时取最小公共维度
        n = min(n_s, n_i)
        jsa = jsa[:n, :n]

    if target_type == "singlet":
        # 反对称目标：f_target(w_s, w_i) = (delta(w_s - w_0) delta(w_i - w_1)
        #                                 - delta(w_s - w_1) delta(w_i - w_0))/sqrt(2)
        # 离散近似
        target = np.zeros_like(jsa)
        target[0, 1] = 1.0 / np.sqrt(2.0)
        target[1, 0] = -1.0 / np.sqrt(2.0)
    elif target_type == "triplet":
        target = np.zeros_like(jsa)
        target[0, 0] = 1.0 / np.sqrt(2.0)
        target[1, 1] = 1.0 / np.sqrt(2.0)
    else:
        raise ValueError(f"未知目标类型: {target_type}")

    overlap = np.sum(np.conj(target) * jsa)
    F = np.abs(overlap) ** 2
    return np.clip(F, 0.0, 1.0)


def chsh_parameter(correlation_matrix: np.ndarray) -> float:
    r"""
    由关联矩阵计算 CHSH S 参数。

    关联函数

    .. math::
        E(a,b) = \frac{N_{++} + N_{--} - N_{+-} - N_{-+}}
                      {N_{++} + N_{--} + N_{+-} + N_{-+}}

    参数
    ----
    correlation_matrix : np.ndarray, shape (4, 4)
        测量设置 (:math:`a, a', b, b') 的联合计数。

    返回
    ----
    S : float
        CHSH 参数。
    """
    if correlation_matrix.shape != (4, 4):
        raise ValueError("correlation_matrix 必须为 4x4。")
    E = np.zeros((2, 2), dtype=np.float64)
    settings_a = [0, 2]  # a, a'
    settings_b = [0, 2]  # b, b'
    for i, ai in enumerate(settings_a):
        for j, bj in enumerate(settings_b):
            Npp = correlation_matrix[ai, bj]
            Nmm = correlation_matrix[ai + 1, bj + 1]
            Npm = correlation_matrix[ai, bj + 1]
            Nmp = correlation_matrix[ai + 1, bj]
            total = Npp + Nmm + Npm + Nmp
            if total > 1e-15:
                E[i, j] = (Npp + Nmm - Npm - Nmp) / total
            else:
                E[i, j] = 0.0

    S = abs(E[0, 0] - E[0, 1] + E[1, 0] + E[1, 1])
    return S
