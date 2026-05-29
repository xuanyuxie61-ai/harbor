"""
dqmc_engine.py

基于 rnglib (1040_rnglib)、truncated_normal (1360_truncated_normal) 与 md_fast (745_md_fast)
的 Determinant Quantum Monte Carlo (DQMC) 引擎。

Hubbard-Stratonovich 变换:
    exp(-Δτ U n_{i↑} n_{i↓}) = (1/2) Σ_{s=±1} exp(λ s (n_{i↑} - n_{i↓}))
    其中 cosh(λ) = exp(Δτ U / 2)。

该方法将相互作用问题转化为在外场 {s_i(l)} 下的无相互作用问题，
通过 Monte Carlo 采样 HS 场构型。

md_fast 中的 Velocity-Verlet 思想被借鉴于构造"快速更新"算法：
当仅翻转单个 HS 场时，利用 Sherman-Morrison 公式在 O(N^2) 时间内更新格林函数。
"""

import numpy as np
from scipy.linalg import expm, solve, det
from typing import Tuple, Optional


class DQMCConfig:
    """DQMC 模拟参数配置。"""

    def __init__(self, nsites: int, beta: float, U: float, t: float, dtau: float):
        if nsites <= 0:
            raise ValueError("nsites > 0  required")
        if beta <= 0 or dtau <= 0:
            raise ValueError("beta, dtau > 0  required")
        if dtau > 0.5:
            raise ValueError("dtau 过大可能导致 Trotter 误差失控，建议 dtau <= 0.5")
        self.nsites = nsites
        self.beta = beta
        self.U = U
        self.t = t
        self.dtau = dtau
        self.L = int(np.ceil(beta / dtau))
        if self.L < 1:
            self.L = 1
        # HS 场参数 λ
        self.lambda_hs = np.arccosh(np.exp(dtau * U / 2.0))
        if np.isnan(self.lambda_hs):
            self.lambda_hs = 0.0


def build_kinetic_matrix(nsites: int, neighbors: list, t: float) -> np.ndarray:
    """
    构造动能矩阵 K_{ij}，其中 H_kin = Σ_{ij} c†_i K_{ij} c_j。
    对于 Hubbard 模型: K_{ij} = -t (若 i,j 近邻)，对角元为 0。
    """
    K = np.zeros((nsites, nsites), dtype=np.float64)
    for i in range(nsites):
        for j in neighbors[i]:
            K[i, j] = -t
    return K


def build_exp_kin(K: np.ndarray, dtau: float) -> np.ndarray:
    """计算 exp(-dtau * K) 的矩阵指数。"""
    return expm(-dtau * K)


def build_b_matrix(exp_kin: np.ndarray, hs_field: np.ndarray, lambda_hs: float, sigma: int) -> np.ndarray:
    """
    构造 B 矩阵: B_l(σ) = exp(-dtau K) * exp(σ λ diag(s(l)))。
    
    参数:
        exp_kin: exp(-dtau K)
        hs_field: 当前时间片的 HS 场，形状 (nsites,)
        lambda_hs: HS 变换参数
        sigma: 自旋 (+1 或 -1)
    """
    diag = np.exp(sigma * lambda_hs * hs_field)
    return exp_kin * diag[np.newaxis, :]


def compute_green_function(Bs: list, stabilize_every: int = 10) -> np.ndarray:
    """
    计算等时格林函数 G = (I + B_{L-1} ... B_0)^{-1}。
    采用周期性数值稳定化 (SVD stabilization) 防止浮点溢出。
    
    参数:
        Bs: B 矩阵列表，长度 L
        stabilize_every: 每隔多少层进行 SVD 稳定化
    
    返回:
        G: 形状 (nsites, nsites)
    """
    nsites = Bs[0].shape[0]
    # 使用 SVD 稳定化的乘积算法
    U = np.eye(nsites)
    D = np.ones(nsites)
    Vt = np.eye(nsites)
    for l, B in enumerate(Bs):
        U = B @ U
        if (l + 1) % stabilize_every == 0 or l == len(Bs) - 1:
            U, D, Vt = _svd_stabilize(U, D, Vt)
    # 构造 M = I + U @ diag(D) @ Vt
    # 使用 Woodbury 或直接从 SVD 求逆
    M = np.eye(nsites) + (U * D) @ Vt
    G = solve(M, np.eye(nsites), assume_a="gen")
    return G


def _svd_stabilize(U: np.ndarray, D: np.ndarray, Vt: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """SVD 稳定化: 对 U @ diag(D) @ Vt 重新分解。"""
    A = (U * D) @ Vt
    U_new, s, Vh_new = np.linalg.svd(A, full_matrices=False)
    # 截断极小奇异值
    s = np.where(s > 1e-12, s, 1e-12)
    return U_new, s, Vh_new


def compute_det_ratio(G: np.ndarray, i: int, delta: float) -> float:
    """
    计算翻转 HS 场 s_i -> -s_i 后的行列式比值。
    利用 Sherman-Morrison 公式:
        det(M') / det(M) = 1 + (1 - G_{ii}) * Δ
    其中 Δ = exp(2 σ λ s_i) - 1。
    """
    if not (0 <= i < G.shape[0]):
        raise IndexError("i 越界")
    gi = G[i, i]
    ratio = 1.0 + (1.0 - gi) * delta
    return ratio


def truncated_normal_ab_sample(mu: float, sigma: float, a: float, b: float, size: int = 1) -> np.ndarray:
    """
    基于 truncated_normal (1360) 的截断正态采样。
    用于生成 HS 场的初始扰动或热浴随机扰动。
    """
    if sigma <= 0:
        raise ValueError("sigma > 0  required")
    if a >= b:
        raise ValueError("a < b  required")
    alpha = (a - mu) / sigma
    beta = (b - mu) / sigma
    # 使用标准正态 CDF 的近似逆
    from scipy.stats import norm
    alpha_cdf = norm.cdf(alpha)
    beta_cdf = norm.cdf(beta)
    u = np.random.rand(size)
    xi_cdf = alpha_cdf + u * (beta_cdf - alpha_cdf)
    xi = norm.ppf(np.clip(xi_cdf, 1e-10, 1 - 1e-10))
    return mu + sigma * xi


def dqmc_sweep(nsites: int, L: int, hs_field: np.ndarray, B_up: list, B_dn: list,
               lambda_hs: float, exp_kin: np.ndarray, G_up: np.ndarray, G_dn: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    执行一次完整的 HS 场 Monte Carlo sweep。
    对每一个 (i, l) 尝试翻转 s_i(l)。
    
    返回:
        更新后的 hs_field, G_up, G_dn
    """
    for l in range(L):
        for i in range(nsites):
            s_old = hs_field[i, l]
            s_new = -s_old
            # 自旋↑的更新量
            delta_up = np.exp(2.0 * lambda_hs * s_new) - 1.0
            delta_dn = np.exp(-2.0 * lambda_hs * s_new) - 1.0
            # 行列式比值
            ratio_up = compute_det_ratio(G_up, i, delta_up)
            ratio_dn = compute_det_ratio(G_dn, i, delta_dn)
            ratio = ratio_up * ratio_dn
            # Metropolis 接受判据
            if ratio < 0:
                continue
            if np.random.rand() < min(1.0, ratio):
                # 接受翻转，更新格林函数 (Sherman-Morrison)
                hs_field[i, l] = s_new
                _update_green(G_up, i, delta_up)
                _update_green(G_dn, i, delta_dn)
                # 更新 B 矩阵
                B_up[l] = build_b_matrix(exp_kin, hs_field[:, l], lambda_hs, +1)
                B_dn[l] = build_b_matrix(exp_kin, hs_field[:, l], lambda_hs, -1)
    return hs_field, G_up, G_dn


def _update_green(G: np.ndarray, i: int, delta: float):
    """利用 Sherman-Morrison 公式原地更新格林函数。"""
    n = G.shape[0]
    u = np.zeros(n)
    u[i] = 1.0
    v = np.zeros(n)
    v[i] = delta
    # G' = G - G u v^T G / (1 + v^T G u)
    denom = 1.0 + v @ G @ u
    if abs(denom) < 1e-14:
        return
    G -= np.outer(G @ u, v @ G) / denom


def run_dqmc(config: DQMCConfig, neighbors: list, n_warmup: int = 100, n_measure: int = 200) -> dict:
    """
    运行 DQMC 模拟并测量物理量。
    
    返回字典包含:
        - double_occupancy: 双占据数
        - kinetic_energy: 动能
        - density: 平均粒子数密度
    """
    nsites = config.nsites
    L = config.L
    # 初始化 HS 场
    hs_field = np.random.choice([-1, 1], size=(nsites, L))
    # 动能矩阵
    K = build_kinetic_matrix(nsites, neighbors, config.t)
    exp_kin = build_exp_kin(K, config.dtau)
    B_up = [build_b_matrix(exp_kin, hs_field[:, l], config.lambda_hs, +1) for l in range(L)]
    B_dn = [build_b_matrix(exp_kin, hs_field[:, l], config.lambda_hs, -1) for l in range(L)]
    # 初始格林函数
    G_up = compute_green_function(B_up)
    G_dn = compute_green_function(B_dn)
    # 热化
    for _ in range(n_warmup):
        hs_field, G_up, G_dn = dqmc_sweep(nsites, L, hs_field, B_up, B_dn, config.lambda_hs, exp_kin, G_up, G_dn)
        # 周期性重算格林函数以保持数值稳定
        B_up = [build_b_matrix(exp_kin, hs_field[:, l], config.lambda_hs, +1) for l in range(L)]
        B_dn = [build_b_matrix(exp_kin, hs_field[:, l], config.lambda_hs, -1) for l in range(L)]
        G_up = compute_green_function(B_up)
        G_dn = compute_green_function(B_dn)
    # 测量
    d_occ_samples = []
    kin_samples = []
    for _ in range(n_measure):
        hs_field, G_up, G_dn = dqmc_sweep(nsites, L, hs_field, B_up, B_dn, config.lambda_hs, exp_kin, G_up, G_dn)
        B_up = [build_b_matrix(exp_kin, hs_field[:, l], config.lambda_hs, +1) for l in range(L)]
        B_dn = [build_b_matrix(exp_kin, hs_field[:, l], config.lambda_hs, -1) for l in range(L)]
        G_up = compute_green_function(B_up)
        G_dn = compute_green_function(B_dn)
        # 双占据: <n_{i↑} n_{i↓}> ≈ (1 - G_{ii}^{↑})(1 - G_{ii}^{↓})
        d_occ = np.mean((1.0 - np.diag(G_up)) * (1.0 - np.diag(G_dn)))
        d_occ_samples.append(d_occ)
        # 动能: -t <c†_i c_j> = t * G_{ij}
        kin = 0.0
        for i in range(nsites):
            for j in neighbors[i]:
                kin += G_up[i, j] + G_dn[i, j]
        kin_samples.append(config.t * kin / nsites)
    return {
        "double_occupancy": np.mean(d_occ_samples),
        "double_occupancy_err": np.std(d_occ_samples) / np.sqrt(len(d_occ_samples)) if len(d_occ_samples) > 1 else 0.0,
        "kinetic_energy": np.mean(kin_samples),
        "kinetic_energy_err": np.std(kin_samples) / np.sqrt(len(kin_samples)) if len(kin_samples) > 1 else 0.0,
    }


if __name__ == "__main__":
    cfg = DQMCConfig(nsites=4, beta=2.0, U=4.0, t=1.0, dtau=0.1)
    neighbors = [[1, 3], [0, 2], [1, 3], [0, 2]]
    res = run_dqmc(cfg, neighbors, n_warmup=20, n_measure=50)
    print(res)
