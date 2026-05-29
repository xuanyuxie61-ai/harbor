"""
topology_optimize.py
====================
超构表面离散相位级数的优化与逆向设计。

本模块融合项目 156_change_dynamic（动态规划求解找零问题）与
206_compressed_solve（欠定系统的 QR 稀疏解）的核心算法，
应用于超构表面相位量化和逆向设计问题。

科学背景：
1. 离散相位量化：
   实际制造中，纳米柱几何参数只能取有限离散值（如 8 级或 16 级相位）。
   这相当于将连续相位目标 Φ_target 量化为离散集合 {2π m / N}_m=0^{N-1}。
   动态规划可用于寻找最优的离散相位分配，最小化衍射效率损失。

2. 压缩感知逆向设计：
   给定目标远场模式 E_target(θ,φ)，求解纳米柱排布使其产生的
   散射场与之匹配。这是一个欠定线性系统（纳米柱数量 >> 远场采样点数），
   可使用 QR 分解求最稀疏解（压缩感知思想）。

核心数学模型：

离散相位优化（动态规划）：
    状态：s[i][j] = 前 i 个单元使用离散相位 j 时的最小误差
    转移：s[i][j] = min_k { s[i-1][k] + cost(φ_j, φ_target_i) }
    其中 cost = |exp(jφ_j) - exp(jφ_target_i)|²

压缩感知逆向设计：
    min ||x||₀  s.t. A x = b
    松弛为：通过 QR 分解求最大稀疏解（与 206_compressed_solve 一致）
        A(:,P) = Q R,  x = zeros(n,1), x(P(1:r)) = R(1:r,1:r) \ (Q^T b)
"""

import numpy as np
from scipy.linalg import qr


class TopologyOptimizer:
    """
    超构表面拓扑优化器：离散相位量化 + 压缩感知逆向设计。
    """

    def __init__(self, n_levels=16):
        """
        n_levels : int
            离散相位级数（如 8 或 16）
        """
        self.n_levels = n_levels
        self.discrete_phases = np.linspace(0, 2 * np.pi, n_levels, endpoint=False)

    # ------------------------------------------------------------------
    # 离散相位优化（动态规划，源自 156_change_dynamic）
    # ------------------------------------------------------------------
    def quantize_phase_dp(self, target_phases, weights=None):
        """
        使用动态规划将连续目标相位最优地量化为离散级数。

        对于每个单元 i，需要从 n_levels 个离散相位中选择一个，
        使得总成本最小。这里允许相邻单元间有连续性惩罚。

        Parameters
        ----------
        target_phases : ndarray, shape (N,)
            目标连续相位 [rad]
        weights : ndarray, shape (N,), optional
            各单元权重（如单元面积）

        Returns
        -------
        quantized : ndarray, shape (N,)
            量化后的离散相位
        total_error : float
            总量化误差
        """
        N = len(target_phases)
        L = self.n_levels
        if weights is None:
            weights = np.ones(N)

        # 成本矩阵：cost[i][j] = 单元 i 选择离散相位 j 的局部成本
        cost_local = np.zeros((N, L), dtype=np.float64)
        for i in range(N):
            for j in range(L):
                dp = self.discrete_phases[j]
                # 复振幅距离
                cost_local[i, j] = weights[i] * abs(
                    np.exp(1.0j * dp) - np.exp(1.0j * target_phases[i])
                ) ** 2

        # 动态规划表
        # dp_table[i][j] = 前 i 个单元，第 i 个选相位 j 的最小总成本
        dp_table = np.full((N, L), np.inf)
        choice = np.zeros((N, L), dtype=np.int32)

        # 初始化第一个单元
        dp_table[0, :] = cost_local[0, :]

        # 转移
        for i in range(1, N):
            for j in range(L):
                # 寻找前一个单元的最优选择 k
                best_cost = np.inf
                best_k = 0
                for k in range(L):
                    # 加入连续性惩罚：相邻单元相位跳变不宜过大
                    continuity_penalty = 0.1 * abs(
                        np.exp(1.0j * self.discrete_phases[j]) -
                        np.exp(1.0j * self.discrete_phases[k])
                    ) ** 2
                    c = dp_table[i - 1, k] + cost_local[i, j] + continuity_penalty
                    if c < best_cost:
                        best_cost = c
                        best_k = k
                dp_table[i, j] = best_cost
                choice[i, j] = best_k

        # 回溯最优路径
        quantized = np.zeros(N, dtype=np.float64)
        min_final = np.inf
        best_j = 0
        for j in range(L):
            if dp_table[N - 1, j] < min_final:
                min_final = dp_table[N - 1, j]
                best_j = j
        quantized[N - 1] = self.discrete_phases[best_j]
        for i in range(N - 1, 0, -1):
            best_j = choice[i, best_j]
            quantized[i - 1] = self.discrete_phases[best_j]

        return quantized, min_final

    def optimize_pillar_geometry_dp(self, target_phases, param_table,
                                     weights=None):
        """
        将动态规划扩展为纳米柱几何参数的多维离散优化。

        param_table : list of dict
            每个离散相位级对应的几何参数：
            [{ 'phase': 0, 'height': h0, 'width': w0 }, ...]
        """
        N = len(target_phases)
        L = len(param_table)
        if weights is None:
            weights = np.ones(N)

        cost_local = np.zeros((N, L), dtype=np.float64)
        for i in range(N):
            for j in range(L):
                dp = param_table[j]['phase']
                cost_local[i, j] = weights[i] * abs(
                    np.exp(1.0j * dp) - np.exp(1.0j * target_phases[i])
                ) ** 2

        dp_table = np.full((N, L), np.inf)
        choice = np.zeros((N, L), dtype=np.int32)
        dp_table[0, :] = cost_local[0, :]

        for i in range(1, N):
            for j in range(L):
                best_cost = np.inf
                best_k = 0
                for k in range(L):
                    # 几何参数连续性惩罚
                    h_jump = abs(param_table[j]['height'] - param_table[k]['height'])
                    w_jump = abs(param_table[j]['width'] - param_table[k]['width'])
                    geo_penalty = 1e12 * (h_jump + w_jump)  # 惩罚剧烈变化
                    c = dp_table[i - 1, k] + cost_local[i, j] + geo_penalty
                    if c < best_cost:
                        best_cost = c
                        best_k = k
                dp_table[i, j] = best_cost
                choice[i, j] = best_k

        # 回溯
        params_opt = []
        min_final = np.inf
        best_j = 0
        for j in range(L):
            if dp_table[N - 1, j] < min_final:
                min_final = dp_table[N - 1, j]
                best_j = j
        params_opt.insert(0, param_table[best_j])
        for i in range(N - 1, 0, -1):
            best_j = choice[i, best_j]
            params_opt.insert(0, param_table[best_j])

        return params_opt, min_final

    # ------------------------------------------------------------------
    # 压缩感知逆向设计（源自 206_compressed_solve）
    # ------------------------------------------------------------------
    def compressed_inverse_design(self, A, b_target, sparsity_factor=0.3):
        """
        使用 QR 分解求欠定系统的最大稀疏解。

        问题设定：
            A x ≈ b_target
        其中 A 的列数远大于行数（纳米柱库 >> 目标模式采样点数）。
        我们希望选择尽可能少的纳米柱类型（稀疏解）。

        算法（与 206_compressed_solve 一致）：
            1. QR 分解：A(:,P) = Q R
            2. 秩 r = rank(A)
            3. x = zeros(n,1)
            4. x(P(1:r)) = R(1:r,1:r) \ (Q(:,1:r)^T b)

        Parameters
        ----------
        A : ndarray, shape (M, N)
            前向模型矩阵（每列对应一种纳米柱的远场贡献）
        b_target : ndarray, shape (M,)
            目标远场模式
        sparsity_factor : float
            期望稀疏度（非零元素比例上限）

        Returns
        -------
        x : ndarray, shape (N,)
            稀疏系数（选择各纳米柱类型的权重）
        residual : float
            拟合残差 ||A x - b_target||₂
        """
        M, N = A.shape
        # QR 分解（列主元）
        Q, R, p = qr(A, pivoting=True, mode='economic')
        # 确定数值秩
        tol = max(M, N) * np.finfo(float).eps * np.abs(R[0, 0])
        r = np.sum(np.abs(np.diag(R)) > tol)
        r = min(r, int(N * sparsity_factor))
        r = max(r, 1)

        y = Q[:, :r].T.conj() @ b_target
        x_reduced = np.linalg.solve(R[:r, :r], y)
        x = np.zeros(N, dtype=np.complex128)
        x[p[:r]] = x_reduced

        residual = np.linalg.norm(A @ x - b_target)
        return x, residual

    def greedy_pillar_selection(self, A, b_target, max_pillars=50):
        """
        贪心算法选择最优纳米柱组合（正交匹配追踪 OMP 思想）。
        """
        M, N = A.shape
        residual = b_target.copy()
        selected = []
        x = np.zeros(N, dtype=np.complex128)

        for _ in range(max_pillars):
            # 找与残差最相关的列
            correlations = np.abs(A.T.conj() @ residual)
            if np.max(correlations) < 1e-15:
                break
            idx = np.argmax(correlations)
            selected.append(idx)

            # 最小二乘求解已选列的系数
            A_sel = A[:, selected]
            x_sel, _, _, _ = np.linalg.lstsq(A_sel, b_target, rcond=None)
            x[selected] = x_sel
            residual = b_target - A_sel @ x_sel

        return x, np.linalg.norm(residual), selected

    def optimize_phase_gradient(self, target_field, x_coords, y_coords,
                                 k0, n_levels=8):
        """
        基于动态规划优化相位梯度分布，使远场辐射方向图逼近目标。

        使用局部相位-斜率关系：
            dΦ/dx ≈ -k0 n_eff sin(θ)
        """
        # 沿 x 方向的相位梯度目标
        dx = np.gradient(x_coords)
        dy = np.gradient(y_coords)
        # 简化：沿径向的相位梯度
        r = np.sqrt(x_coords ** 2 + y_coords ** 2)
        dr = np.gradient(r)
        # 目标相位
        target_phases = np.angle(target_field)
        # 使用 DP 量化
        quantized, error = self.quantize_phase_dp(target_phases)
        return quantized, error


def demo():
    """演示：相位量化与压缩感知逆向设计。"""
    opt = TopologyOptimizer(n_levels=8)

    # 1. 动态规划相位量化
    N = 100
    target = np.linspace(0, 4 * np.pi, N)
    weights = np.ones(N)
    quantized, err = opt.quantize_phase_dp(target, weights)
    print(f"[topology_optimize] DP 量化误差: {err:.4f}")
    print(f"[topology_optimize] 量化前 5 个相位: " +
          ", ".join(f"{q:.3f}" for q in quantized[:5]))

    # 2. 压缩感知逆向设计
    M = 50  # 远场采样点数
    N_lib = 200  # 纳米柱库大小
    np.random.seed(0)
    A = np.random.randn(M, N_lib) + 1.0j * np.random.randn(M, N_lib)
    # 构造稀疏真实解
    x_true = np.zeros(N_lib, dtype=np.complex128)
    x_true[np.random.choice(N_lib, 10, replace=False)] = np.random.randn(10) + 1.0j * np.random.randn(10)
    b_target = A @ x_true

    x_est, res = opt.compressed_inverse_design(A, b_target, sparsity_factor=0.3)
    nnz = np.sum(np.abs(x_est) > 1e-10)
    print(f"[topology_optimize] 压缩感知解: 非零元素={nnz}, 残差={res:.4e}")

    x_greedy, res_greedy, sel = opt.greedy_pillar_selection(A, b_target, max_pillars=15)
    print(f"[topology_optimize] 贪心选择: 选中 {len(sel)} 个, 残差={res_greedy:.4e}")
    return quantized, x_est


if __name__ == "__main__":
    demo()
