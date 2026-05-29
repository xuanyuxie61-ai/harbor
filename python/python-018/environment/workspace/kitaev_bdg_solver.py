"""
kitaev_bdg_solver.py

基于种子项目 964_r83p（周期三对角矩阵R83P分解）构建Kitaev链的
Bogoliubov-de Gennes (BdG) 哈密顿量，并利用周期三对角矩阵算法
高效求解本征值与本征矢。

核心物理模型：
    Kitaev一维p波超导链（周期边界条件）
    H = -μ Σ_i c_i† c_i - t Σ_i (c_i† c_{i+1} + h.c.)
        + Δ Σ_i (c_i c_{i+1} + h.c.)

在Nambu基 Ψ_k = (c_k, c_{-k}†)^T 下，BdG哈密顿量为 2N×2N 的
周期块三对角矩阵，其边缘耦合产生周期三对角子结构。
"""

import numpy as np
from typing import Tuple, Optional


class KitaevBdGSolver:
    """
    Kitaev链BdG哈密顿量求解器。

    将实空间费米子算符映射到马约拉纳算符：
        γ_{2j-1} = c_j + c_j†
        γ_{2j}   = i(c_j - c_j†)

    则哈密顿量可写为：
        H = (i/2) Σ_{j=1}^{N-1} [ -(t+Δ) γ_{2j-1} γ_{2j+2}
                                  + (t-Δ) γ_{2j} γ_{2j+1} ]
            + (i/2) Σ_{j=1}^{N} (-μ) γ_{2j-1} γ_{2j}

    对于周期边界条件，额外存在 (N,1) 和 (1,N) 的耦合项，
    构成周期三对角矩阵结构（R83P格式存储）。
    """

    def __init__(self, n_sites: int, mu: float, t: float, delta: float,
                 periodic: bool = True):
        """
        初始化Kitaev链参数。

        Args:
            n_sites: 晶格格点数 N (N >= 3)
            mu: 化学势 μ
            t: 最近邻跃迁强度 t
            delta: p波超导配对势 Δ
            periodic: 是否使用周期边界条件
        """
        if n_sites < 3:
            raise ValueError("晶格格点数 N 必须至少为 3")
        if abs(t) < 1e-14:
            raise ValueError("跃迁强度 t 不能为零（会导致退耦）")
        self.n = n_sites
        self.mu = float(mu)
        self.t = float(t)
        self.delta = float(delta)
        self.periodic = periodic

    def _build_r83p_matrix(self) -> np.ndarray:
        """
        构建R83P格式的周期三对角矩阵。

        R83P存储格式（3×N数组）：
            第0行: [A_{N,1}, A_{1,2}, A_{2,3}, ..., A_{N-1,N}]
            第1行: [A_{1,1}, A_{2,2}, A_{3,3}, ..., A_{N,N}]
            第2行: [A_{2,1}, A_{3,2}, A_{4,3}, ..., A_{1,N}]

        对于BdG哈密顿量的马约拉纳耦合部分，有效哈密顿量为
        2N×2N 的实反对称矩阵，其核心的周期耦合子块具有R83P结构。
        这里我们提取关键的周期三对角耦合矩阵用于稳定性分析。

        矩阵元素来源于最近邻马约拉纳耦合：
            H_{j,j}   = -μ/2
            H_{j,j+1} = (t - Δ)/2
            H_{j+1,j} = -(t + Δ)/2
            H_{1,N}   = 周期耦合 (t - Δ)/2
            H_{N,1}   = 周期耦合 -(t + Δ)/2
        """
        n = self.n
        a = np.zeros((3, n))

        # 上对角线（R83P第0行）
        # A(1,1) 在R83P中存储周期耦合 A(N,1)
        a[0, 0] = (self.t - self.delta) / 2.0
        for j in range(1, n):
            a[0, j] = (self.t - self.delta) / 2.0

        # 主对角线（R83P第1行）
        for j in range(n):
            a[1, j] = -self.mu / 2.0

        # 下对角线（R83P第2行）
        for j in range(n - 1):
            a[2, j] = -(self.t + self.delta) / 2.0
        # 周期耦合 A(1,N) 存储在 A(3,N) 即 a[2, n-1]
        a[2, n - 1] = -(self.t + self.delta) / 2.0

        return a

    def r83p_factorize(self, a: np.ndarray) -> Tuple[np.ndarray, np.ndarray,
                                                       np.ndarray, float, int]:
        """
        R83P周期三对角矩阵的边界带状分解（源自964_r83p核心算法）。

        将周期矩阵分解为：
            [ A1  A2 ]
            [ A3  A4 ]
        其中A1为(N-1)×(N-1)三对角矩阵，A2、A3为边界列/行，A4为标量。

        算法步骤：
            1) 对A1进行标准三对角分解（r83_np_fa）
            2) 计算 WORK2 = inv(A1) * A2
            3) 计算 WORK3 = inv(A1') * A3'
            4) 计算 Schur补 WORK4 = A4 - A3 * inv(A1) * A2

        Returns:
            a_lu: 分解信息数组
            work2, work3: 辅助向量
            work4: Schur补标量
            info: 0表示成功，非零表示失败
        """
        n = a.shape[1]
        if n < 3:
            raise ValueError("R83P分解要求矩阵阶数 N >= 3")

        a_lu = np.zeros((3, n))

        # 对A1（前N-1列的三对角部分）进行分解
        a1 = np.copy(a[:, :n - 1])
        a1_lu, info = self._r83_np_factorize(n - 1, a1)
        if info != 0:
            return a_lu, np.zeros(n - 1), np.zeros(n - 1), 0.0, info

        a_lu[:, :n - 1] = a1_lu
        a_lu[0, 0] = a[0, 0]
        a_lu[2, n - 2] = a[2, n - 2]
        a_lu[:, n - 1] = a[:, n - 1]

        # WORK2 := inv(A1) * A2
        work2 = np.zeros(n - 1)
        work2[0] = a[2, n - 1]   # A(N,1) 周期下耦合 -> 存储位置
        work2[n - 2] = a[0, n - 1]   # A(1,N) 周期上耦合
        work2 = self._r83_np_solve(n - 1, a1_lu, work2, job=0)

        # WORK3 := inv(A1') * A3'
        work3 = np.zeros(n - 1)
        work3[0] = a[0, 0]
        work3[n - 2] = a[2, n - 2]
        work3 = self._r83_np_solve(n - 1, a1_lu, work3, job=1)

        # A4 := A4 - A3 * inv(A1) * A2
        work4 = a[1, n - 1] - a[0, 0] * work2[0] - a[2, n - 2] * work2[n - 2]

        if abs(work4) < 1e-15:
            info = n
            return a_lu, work2, work3, work4, info

        return a_lu, work2, work3, work4, 0

    def _r83_np_factorize(self, n: int, a: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        非周期三对角矩阵R83的LU分解（Crout算法）。

        对于矩阵：
            [ d1 u1                    ]
            [ l2 d2 u2                 ]
            [    l3 d3 u3              ]
            [       ... ... ...        ]
            [          l_{N-1} d_{N-1} u_{N-1} ]
            [                l_N d_N   ]

        分解为 L*U 形式，其中L单位下三角，U上三角。
        """
        info = 0
        a_lu = np.copy(a)

        for i in range(n):
            if i > 0:
                a_lu[1, i] -= a_lu[2, i - 1] * a_lu[0, i - 1]
            if abs(a_lu[1, i]) < 1e-15:
                info = i + 1
                return a_lu, info
            if i < n - 1:
                a_lu[0, i] /= a_lu[1, i]

        return a_lu, 0

    def _r83_np_solve(self, n: int, a_lu: np.ndarray, b: np.ndarray,
                      job: int = 0) -> np.ndarray:
        """
        求解R83分解后的线性系统。

        Args:
            job: 0 求解 A*x = b, 1 求解 A'*x = b
        """
        x = np.copy(b)

        if job == 0:
            # 前代求解 L*y = b
            for i in range(1, n):
                x[i] -= a_lu[2, i - 1] * x[i - 1]
            # 回代求解 U*x = y
            for i in range(n):
                x[i] /= a_lu[1, i]
            for i in range(n - 2, -1, -1):
                x[i] -= a_lu[0, i] * x[i + 1]
        else:
            # 求解 A'*x = b，即 U'*L'*x = b
            for i in range(1, n):
                x[i] -= a_lu[0, i - 1] * x[i - 1]
            for i in range(n):
                x[i] /= a_lu[1, i]
            for i in range(n - 2, -1, -1):
                x[i] -= a_lu[2, i] * x[i + 1]

        return x

    def r83p_solve(self, a_lu: np.ndarray, work2: np.ndarray,
                   work3: np.ndarray, work4: float, b: np.ndarray) -> np.ndarray:
        """
        利用R83P分解结果求解线性系统 A*x = b。

        基于分块矩阵求逆：
            [ X1 ]   [ inv(A1)   -inv(A1)*A2*inv(S) ] [ B1 ]
            [ X2 ] = [ -inv(S)*A3*inv(A1)    inv(S) ] [ B2 ]
        其中 S = A4 - A3*inv(A1)*A2 为Schur补。
        """
        n = len(b)
        if n < 3:
            raise ValueError("R83P求解要求向量长度 N >= 3")

        x = np.zeros(n)
        n1 = n - 1

        # 求解 A1 * x1_temp = b1
        x1_temp = self._r83_np_solve(n1, a_lu[:, :n1], b[:n1], job=0)

        # X2 = (B2 - A3*x1_temp) / WORK4
        x[n - 1] = (b[n - 1] - a_lu[0, 0] * x1_temp[0]
                    - a_lu[2, n1 - 1] * x1_temp[n1 - 1]) / work4

        # X1 = x1_temp - WORK2 * X2
        x[:n1] = x1_temp - work2 * x[n - 1]

        return x

    def build_full_bdg_hamiltonian(self) -> np.ndarray:
        """
        构建完整的2N×2N BdG哈密顿量矩阵（实空间Nambu表示）。

        在Nambu基 (c_1, c_2, ..., c_N, c_1†, c_2†, ..., c_N†)^T 下：

            H_BdG = [  H_0       Δ      ]
                    [  Δ†     -H_0^T    ]

        其中 H_0 为单粒子哈密顿量：
            (H_0)_{i,i}   = -μ
            (H_0)_{i,i+1} = -t
            (H_0)_{i+1,i} = -t
        而 Δ 为配对矩阵（p波）：
            Δ_{i,i+1} = +Δ
            Δ_{i+1,i} = -Δ
        """
        n = self.n
        h_bdg = np.zeros((2 * n, 2 * n))

        # H_0 块 (单粒子部分)
        for i in range(n):
            h_bdg[i, i] = -self.mu
            if i < n - 1:
                h_bdg[i, i + 1] = -self.t
                h_bdg[i + 1, i] = -self.t
            elif self.periodic:
                h_bdg[i, 0] = -self.t
                h_bdg[0, i] = -self.t

        # -H_0^T 块 (空穴部分)
        for i in range(n):
            h_bdg[n + i, n + i] = self.mu
            if i < n - 1:
                h_bdg[n + i, n + i + 1] = self.t
                h_bdg[n + i + 1, n + i] = self.t
            elif self.periodic:
                h_bdg[n + i, n] = self.t
                h_bdg[n, n + i] = self.t

        # Δ 配对块: 右上=Δ(反对称), 左下=-Δ^* = -Δ (实数)
        for i in range(n):
            if i < n - 1:
                h_bdg[i, n + i + 1] = self.delta
                h_bdg[i + 1, n + i] = -self.delta
                h_bdg[n + i, i + 1] = -self.delta
                h_bdg[n + i + 1, i] = self.delta
            elif self.periodic:
                h_bdg[i, n] = self.delta
                h_bdg[0, n + i] = -self.delta
                h_bdg[n + i, 0] = -self.delta
                h_bdg[n, i] = self.delta

        return h_bdg

    def diagonalize(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        对角化BdG哈密顿量，返回本征值和本征矢。

        由于BdG哈密顿量具有粒子-空穴对称性：
            {H_BdG, C} = 0,  C = τ_x K
        其中K为复共轭，τ_x为Nambu空间的泡利矩阵。
        因此本征值成对出现：若E为特征值，则-E也是。

        对于Nambu基，波函数必须满足归一化：
            Σ_i (|u_i|^2 + |v_i|^2) = 1
        其中u为电子分量，v为空穴分量。
        """
        h_bdg = self.build_full_bdg_hamiltonian()

        # 利用BdG的厄米性进行对角化
        eigvals, eigvecs = np.linalg.eigh(h_bdg)

        # 按能量排序
        idx = np.argsort(eigvals)
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        # 数值鲁棒性：将接近零的本征值置零
        tol = 1e-12 * max(abs(self.t), abs(self.delta), abs(self.mu), 1.0)
        eigvals[np.abs(eigvals) < tol] = 0.0

        return eigvals, eigvecs

    def identify_majorana_zero_modes(self, eigvals: np.ndarray,
                                      eigvecs: np.ndarray,
                                      energy_tol: float = 1e-10
                                      ) -> Tuple[np.ndarray, np.ndarray]:
        """
        识别马约拉纳零能模（Majorana Zero Modes, MZMs）。

        在拓扑非平庸相中（|μ| < 2|t| 且 Δ ≠ 0），边界处会出现
        能量严格为零（E=0）的马约拉纳束缚态。

        马约拉纳条件：γ = γ†，即粒子是其自身的反粒子。
        在BdG框架下，零能模对应粒子-空穴对称的本征态：
            C ψ = ψ，即 u_i = v_i^*

        Returns:
            零能模的波函数（电子分量u和空穴分量v）
        """
        n = self.n
        mask = np.abs(eigvals) < energy_tol
        zero_indices = np.where(mask)[0]

        if len(zero_indices) == 0:
            return np.array([]), np.array([])

        # === HOLE 1 START ===
        # TODO: 实现马约拉纳零能模（MZM）的识别逻辑
        #
        # 科学背景：在拓扑非平庸相中（|μ| < 2|t| 且 Δ ≠ 0），开边界Kitaev链的
        # 两端会出现能量严格为零的马约拉纳束缚态。在BdG框架下，零能模对应
        # 粒子-空穴对称的本征态：C ψ = ψ，即 u_i = v_i^*
        #
        # 需要完成的任务：
        # 1. 遍历 zero_indices 中的每个零能本征态索引
        # 2. 提取该本征态的电子分量 u = psi[:n] 和空穴分量 v = psi[n:]
        # 3. 验证马约拉纳条件：计算 |u| 与 |v| 的重叠程度
        # 4. 将满足条件的模加入列表
        # 5. 返回字典格式：{'u_modes': modes_u, 'v_modes': modes_v, 'count': len(modes_u)}
        #
        # 注意：返回值已从元组 (modes_u, modes_v) 改为字典格式，
        # 调用方（main.py）需要同步适配。
        raise NotImplementedError("Hole 1: 请实现马约拉纳零能模识别逻辑")
        # === HOLE 1 END ===

    def compute_energy_gap(self, eigvals: np.ndarray) -> float:
        """
        计算BdG能谱的准粒子激发能隙。

        能隙定义为最低正能激发与零能之间的差：
            E_gap = min{ E_n | E_n > 0 }

        在拓扑相变点，能隙闭合；在拓扑相内，能隙打开。
        """
        positive = eigvals[eigvals > 1e-12]
        if len(positive) == 0:
            return 0.0
        return float(np.min(positive))

    def topological_phase_diagram(self, mu_vals: np.ndarray) -> np.ndarray:
        """
        计算拓扑相图：作为化学势μ的函数的能隙。

        Kitaev链的拓扑相边界由以下条件决定：
            |μ| = 2|t| （相变点，能隙闭合）

        拓扑非平庸相：|μ| < 2|t|
        拓扑平庸相：|μ| > 2|t|
        """
        gaps = np.zeros_like(mu_vals)
        original_mu = self.mu

        for i, mu in enumerate(mu_vals):
            self.mu = mu
            eigvals, _ = self.diagonalize()
            gaps[i] = self.compute_energy_gap(eigvals)

        self.mu = original_mu
        return gaps


def demo():
    """演示Kitaev链BdG求解。"""
    solver = KitaevBdGSolver(n_sites=20, mu=0.5, t=1.0, delta=0.8,
                             periodic=False)
    eigvals, eigvecs = solver.diagonalize()
    print("Kitaev Chain BdG Spectrum (first 10):")
    print(eigvals[:10])
    gap = solver.compute_energy_gap(eigvals)
    print(f"Energy gap: {gap:.6e}")

    modes_u, modes_v = solver.identify_majorana_zero_modes(eigvals, eigvecs)
    print(f"Number of MZMs detected: {len(modes_u)}")


if __name__ == "__main__":
    demo()
