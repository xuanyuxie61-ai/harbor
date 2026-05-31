"""
svd_rom.py
==========
基于 SVD 的降阶模型 (Reduced Order Model, ROM) 与数据压缩。

融合种子项目：
  - 1187_svd_fingerprint : SVD 分解、低秩近似、压缩比分析

科学应用：
  对高维非线性声学模拟产生的时空大数据进行 SVD 降阶，提取
  主导模态（POD模态），构建 Galerkin 投影降阶模型，显著
  降低多参数扫描的计算成本。

  核心公式：
  - 快照矩阵 :math:`S \in \mathbb{R}^{N_{space} \times N_{time}}`
  - SVD: :math:`S = U \Sigma V^T`
  - r 阶截断近似: :math:`S_r = U_r \Sigma_r V_r^T`
  - 相对误差: :math:`\epsilon_r = \sqrt{\sum_{i=r+1}^{R} \sigma_i^2} / \sqrt{\sum_{i=1}^{R} \sigma_i^2}`
"""

import numpy as np


class SVDRomCompressor:
    """
    SVD 降阶模型构建器。
    """

    def __init__(self, snapshot_matrix):
        """
        Parameters
        ----------
        snapshot_matrix : np.ndarray, shape (N_space, N_time)
            快照矩阵，每列为一个时间步的空间场。
        """
        self.S = np.asarray(snapshot_matrix, dtype=float)
        if self.S.ndim != 2:
            raise ValueError("snapshot_matrix must be 2D.")
        self.N_space, self.N_time = self.S.shape
        self.U = None
        self.Sigma = None
        self.Vt = None
        self.singular_values = None
        self._decomposed = False

    def decompose(self):
        """
        执行 SVD 分解：:math:`S = U \Sigma V^T`。

        Returns
        -------
        self
        """
        if self._decomposed:
            return self
        # 使用 numpy 的 SVD
        self.U, s, self.Vt = np.linalg.svd(self.S, full_matrices=False)
        self.Sigma = np.diag(s)
        self.singular_values = s
        self._decomposed = True
        return self

    def low_rank_approximation(self, rank):
        """
        计算 r 阶低秩近似。

        .. math::
            S_r = U_r \Sigma_r V_r^T = \sum_{i=1}^{r} \sigma_i u_i v_i^T

        Parameters
        ----------
        rank : int
            截断秩 r。

        Returns
        -------
        np.ndarray, shape (N_space, N_time)
            低秩近似矩阵。
        float
            近似相对误差。
        float
            压缩比。
        """
        if not self._decomposed:
            self.decompose()

        rank = int(rank)
        if rank < 1:
            raise ValueError("rank must be positive.")
        r_eff = min(rank, len(self.singular_values))

        Ur = self.U[:, :r_eff]
        Sr = np.diag(self.singular_values[:r_eff])
        Vtr = self.Vt[:r_eff, :]
        S_approx = Ur @ Sr @ Vtr

        # 相对 Frobenius 误差
        frob_original = np.linalg.norm(self.S, 'fro')
        if frob_original > 0.0:
            rel_error = np.linalg.norm(self.S - S_approx, 'fro') / frob_original
        else:
            rel_error = 0.0

        # 压缩比: 存储 S_r 所需数据 / 存储 S 所需数据
        compression = (self.N_space * r_eff + r_eff + r_eff * self.N_time) / (
            self.N_space * self.N_time)

        return S_approx, rel_error, compression

    def cumulative_energy(self):
        """
        计算累积能量比例。

        .. math::
            E_k = \frac{\sum_{i=1}^{k} \sigma_i^2}{\sum_{i=1}^{R} \sigma_i^2}

        Returns
        -------
        np.ndarray
            每个模态的累积能量比例。
        """
        if not self._decomposed:
            self.decompose()
        s2 = self.singular_values ** 2
        total = np.sum(s2)
        if total <= 0.0:
            return np.zeros_like(s2)
        cumsum = np.cumsum(s2) / total
        return cumsum

    def find_optimal_rank(self, threshold=0.99):
        """
        找到捕获 threshold 能量的最小秩。

        Parameters
        ----------
        threshold : float
            能量阈值 (0, 1]。

        Returns
        -------
        int
            最优秩。
        """
        if not (0.0 < threshold <= 1.0):
            raise ValueError("threshold must be in (0, 1].")
        cum_energy = self.cumulative_energy()
        rank = np.searchsorted(cum_energy, threshold, side='left') + 1
        return int(min(rank, len(self.singular_values)))

    def pod_modes(self, n_modes):
        """
        提取前 n 个 POD 模态（空间模态）。

        Parameters
        ----------
        n_modes : int

        Returns
        -------
        np.ndarray, shape (N_space, n_modes)
            POD 模态矩阵。
        np.ndarray, shape (n_modes,)
            对应的奇异值。
        """
        if not self._decomposed:
            self.decompose()
        n_modes = min(int(n_modes), self.U.shape[1])
        return self.U[:, :n_modes].copy(), self.singular_values[:n_modes].copy()

    def galerkin_projection_rhs(self, n_modes, full_rhs_func, current_state):
        r"""
        Galerkin 投影：将完整 RHS 投影到 POD 模态张成的子空间。

        .. math::
            \dot{a}_k = \langle f(\sum_j a_j \phi_j), \phi_k \rangle

        Parameters
        ----------
        n_modes : int
            模态数。
        full_rhs_func : callable
            完整 RHS 函数 full_rhs_func(state_vector) -> np.ndarray (N_space,)。
        current_state : np.ndarray, shape (n_modes,)
            当前模态系数。

        Returns
        -------
        np.ndarray, shape (n_modes,)
            模态系数的导数。
        """
        phi, _ = self.pod_modes(n_modes)
        # 重构完整状态
        full_state = phi @ current_state
        # 计算完整 RHS
        rhs_full = full_rhs_func(full_state)
        # 投影到模态空间: a_dot = phi^T @ rhs_full
        a_dot = phi.T @ rhs_full
        return a_dot

    def reconstruct_field(self, modal_coefficients):
        """
        从模态系数重构物理场。

        Parameters
        ----------
        modal_coefficients : np.ndarray, shape (n_modes,)

        Returns
        -------
        np.ndarray, shape (N_space,)
        """
        n_modes = modal_coefficients.size
        phi, _ = self.pod_modes(n_modes)
        return phi @ modal_coefficients


def svd_blackwhite_approx(m, n, rank, U, Sigma, Vt):
    """
    低秩近似重构，原始函数来自 1187_svd_fingerprint/svd_bw.m。

    Parameters
    ----------
    m, n : int
        矩阵维度。
    rank : int
        截断秩。
    U : np.ndarray
        左奇异向量。
    Sigma : np.ndarray
        奇异值矩阵或向量。
    Vt : np.ndarray
        右奇异向量转置。

    Returns
    -------
    np.ndarray, shape (m, n)
        低秩近似。
    """
    rank = int(rank)
    if np.ndim(Sigma) == 1:
        s = Sigma[:rank]
        approx = U[:, :rank] @ np.diag(s) @ Vt[:rank, :]
    else:
        approx = U[:, :rank] @ Sigma[:rank, :rank] @ Vt[:rank, :]
    return approx


class DynamicModeDecomposition:
    r"""
    动态模态分解 (DMD)，作为 SVD-ROM 的补充。

    DMD 提取系统的特征动力学：

    .. math::
        X_2 = A X_1 \approx U_r \tilde{A} U_r^T X_1

    其中 :math:`\tilde{A} = U_r^T A U_r`，通过 SVD 和特征值分析得到 DMD 模态。
    """

    def __init__(self, snapshot_matrix):
        self.S = np.asarray(snapshot_matrix, dtype=float)
        if self.S.ndim != 2:
            raise ValueError("snapshot_matrix must be 2D.")

    def compute_modes(self, rank=None):
        """
        计算 DMD 模态。

        Parameters
        ----------
        rank : int or None
            截断秩。None 则使用全部。

        Returns
        -------
        np.ndarray
            DMD 特征值。
        np.ndarray
            DMD 模态。
        """
        X1 = self.S[:, :-1]
        X2 = self.S[:, 1:]

        U, s, Vt = np.linalg.svd(X1, full_matrices=False)
        if rank is not None:
            rank = min(int(rank), len(s))
            U = U[:, :rank]
            s = s[:rank]
            Vt = Vt[:rank, :]

        # 伪逆构造 A_tilde
        S_inv = np.diag(1.0 / s)
        A_tilde = U.T @ X2 @ Vt.T @ S_inv

        eigenvalues, eigenvectors = np.linalg.eig(A_tilde)
        # DMD 模态
        dmd_modes = X2 @ Vt.T @ S_inv @ eigenvectors

        return eigenvalues, dmd_modes
