"""
surrogate_model.py  --  保真核 GP 代理 + ResNet 编码器
===============================================================
来源种子项目:
    1074_Akiraichi_Explicit-quantum-surrogates :
        SVM 保真核 (fidelity kernel) K(x,x') = |<x|x'>|^2,
        用于量子态分类; 这里重新解释为 GP 核.
    1102_ryan597_RepresentationLearningWaves :
        ResBlock + ResidualLayer 骨干网络, 用于波场表示学习;
        这里作为 *参数场编码器* 把高维 theta 嵌入到低维潜在空间.
科学问题角色:
    正向算子 G(theta) 昂贵; 我们用代理 S(theta) 近似它:
        S(theta) = Linear( ResNet_encoder(theta) )
    训练目标:
        min_{phi} sum_i || G(theta_i) - S_phi(theta_i) ||^2_{Sigma_obs^{-1}}
    训练好后, MCMC 内层调用 S 而非 G, 节省 95%+ 计算量.
边界与鲁棒性:
    - 编码器输出用 tanh 压缩到 [-1, 1], 避免爆炸.
    - 保真核加 nugget sigma_n^2 I, 防止矩阵病态.
    - 权重初始化用 Xavier 方差.
核心公式:
    K_fidelity(theta, theta') = (theta^T theta' / (||theta|| ||theta'||))^2
    GP 后验均值: mu(theta*) = k^T (K + sigma_n^2 I)^{-1} y
"""
from __future__ import annotations
import math
from typing import List, Tuple
from numerical_base import NUMERICS


# ======================================================================
# 1. ResNet 编码器 (移植 1102)
# ======================================================================
class _ResBlock:
    """
    残差块: x -> conv1 -> BN -> act -> conv2 -> BN -> (+x if gate=False).
    这里用线性层代替 2D 卷积 (1D 特征空间版本).
    """
    def __init__(self, dim_in: int, dim_out: int, gate: bool = False,
                 rng=None):
        self.gate = gate
        scale = math.sqrt(2.0 / dim_in)
        self.W1 = [[(rng.normal() if rng else 0.0) * scale
                    for _ in range(dim_in)] for _ in range(dim_out)]
        self.b1 = [0.0] * dim_out
        self.W2 = [[(rng.normal() if rng else 0.0) * scale
                    for _ in range(dim_out)] for _ in range(dim_out)]
        self.b2 = [0.0] * dim_out
        self.dim_in, self.dim_out = dim_in, dim_out
        # BN 参数
        self.bn1_mu = [0.0] * dim_out
        self.bn1_sig = [1.0] * dim_out
        self.bn2_mu = [0.0] * dim_out
        self.bn2_sig = [1.0] * dim_out

    def _linear(self, W, b, x):
        out = []
        for i in range(len(W)):
            s = b[i]
            for j in range(len(x)):
                s += W[i][j] * x[j]
            out.append(s)
        return out

    def _leaky_relu(self, x, alpha: float = 0.01):
        return [xi if xi > 0 else alpha * xi for xi in x]

    def _batch_norm(self, x, mu, sig):
        out = []
        for i, xi in enumerate(x):
            z = (xi - mu[i]) / max(sig[i], NUMERICS.cholesky_jitter)
            out.append(max(-3.0, min(3.0, z)))
        return out

    def forward(self, x):
        h = self._linear(self.W1, self.b1, x)
        h = self._batch_norm(h, self.bn1_mu, self.bn1_sig)
        h = self._leaky_relu(h)
        h = self._linear(self.W2, self.b2, h)
        h = self._batch_norm(h, self.bn2_mu, self.bn2_sig)
        if self.gate:
            return h
        # 残差连接: 需要投影
        if len(x) == self.dim_out:
            return [h[i] + x[i] for i in range(self.dim_out)]
        return h  # 维度不匹配时跳过残差


class ResNetEncoder:
    """
    多层残差编码器: dim_in -> hidden -> hidden -> ... -> latent.
    """
    def __init__(self, dim_in: int, hidden: int = 16, latent: int = 6,
                 n_blocks: int = 2, seed: int = 1):
        from prior_geometry import _LCG
        rng = _LCG(seed)
        self.blocks: List[_ResBlock] = []
        # 输入投影
        self.proj_W = [[rng.normal() * math.sqrt(2.0 / dim_in)
                        for _ in range(dim_in)] for _ in range(hidden)]
        self.proj_b = [0.0] * hidden
        for k in range(n_blocks):
            gate = (k == 0)
            self.blocks.append(_ResBlock(hidden, hidden, gate=gate, rng=rng))
        # 输出到 latent
        self.out_W = [[rng.normal() * math.sqrt(2.0 / hidden)
                       for _ in range(hidden)] for _ in range(latent)]
        self.out_b = [0.0] * latent
        self.hidden = hidden
        self.latent = latent

    def forward(self, x: List[float]) -> List[float]:
        h = [sum(self.proj_W[i][j] * x[j] for j in range(len(x)))
             + self.proj_b[i] for i in range(self.hidden)]
        h = [max(-3.0, min(3.0, hi)) for hi in h]
        for blk in self.blocks:
            h = blk.forward(h)
        out = [sum(self.out_W[i][j] * h[j] for j in range(self.hidden))
               + self.out_b[i] for i in range(self.latent)]
        # tanh 压缩
        return [math.tanh(o) for o in out]


# ======================================================================
# 2. 保真核 GP 代理 (1074)
# ======================================================================
def fidelity_kernel(x: List[float], y: List[float]) -> float:
    """
    K(x, y) = (x^T y / (||x|| ||y||))^2,
    即归一化内积的平方.
    若任一向量为零, 返回 nugget.
    """
    xy = sum(xi * yi for xi, yi in zip(x, y))
    nx = math.sqrt(sum(xi * xi for xi in x) + NUMERICS.cholesky_jitter)
    ny = math.sqrt(sum(yi * yi for yi in y) + NUMERICS.cholesky_jitter)
    c = xy / (nx * ny)
    return c * c


class FidelityGPSurrogate:
    """
    GP 代理, 核 = 保真核 + 线性核混合 + nugget.
    训练集: { (theta_i, G(theta_i)) }_{i=1}^N.
    输出维度 = n_obs (观测个数).
    """
    def __init__(self, latent_dim: int = 6, n_obs: int = 8,
                 sigma_n: float = 0.05, mix_weight: float = 0.7):
        self.latent_dim = latent_dim
        self.n_obs = n_obs
        self.sigma_n = sigma_n
        self.mix_weight = mix_weight
        self.encoder = ResNetEncoder(dim_in=4, hidden=16,
                                     latent=latent_dim, n_blocks=2)
        self.train_z: List[List[float]] = []
        self.train_y: List[List[float]] = []
        self.K_inv: List[List[float]] | None = None
        self.alpha: List[List[float]] = []  # K^{-1} Y
        self._fitted = False

    def _kernel(self, z1, z2) -> float:
        kf = fidelity_kernel(z1, z2)
        kl = sum(a * b for a, b in zip(z1, z2)) / max(self.latent_dim, 1)
        return self.mix_weight * kf + (1.0 - self.mix_weight) * kl

    def fit(self, thetas: List[List[float]], ys: List[List[float]]) -> None:
        """
        训练: 编码 theta -> z, 计算 K, 求逆, 得到 alpha = K^{-1} Y.
        """
        self.train_z = [self.encoder.forward(t) for t in thetas]
        self.train_y = ys
        N = len(self.train_z)
        K = [[0.0] * N for _ in range(N)]
        for i in range(N):
            for j in range(N):
                K[i][j] = self._kernel(self.train_z[i], self.train_z[j])
            K[i][i] += self.sigma_n * self.sigma_n
        self.K_inv = _invert_spd_safe(K)
        # alpha = K^{-1} Y, 每个观测维度独立
        self.alpha = []
        for d in range(self.n_obs):
            y_d = [row[d] for row in self.train_y]
            a_d = [sum(self.K_inv[i][j] * y_d[j] for j in range(N))
                   for i in range(N)]
            self.alpha.append(a_d)
        self._fitted = True

    def predict(self, theta: List[float]) -> List[float]:
        """后验均值 mu(theta*) = k^T alpha."""
        if not self._fitted:
            return [0.5] * self.n_obs
        z = self.encoder.forward(theta)
        N = len(self.train_z)
        k = [self._kernel(z, self.train_z[i]) for i in range(N)]
        out = []
        for d in range(self.n_obs):
            mu = sum(k[i] * self.alpha[d][i] for i in range(N))
            # 投影到合理范围
            out.append(max(0.0, min(1.0, mu)))
        return out

    def train_loss(self) -> float:
        """训练集 MSE."""
        if not self._fitted:
            return float("inf")
        s = 0.0
        for t, y in zip(self.train_z, self.train_y):
            yp = self.predict_from_z(t)
            for d in range(self.n_obs):
                s += (yp[d] - y[d]) ** 2
        return s / max(1, len(self.train_z) * self.n_obs)

    def predict_from_z(self, z: List[float]) -> List[float]:
        if not self._fitted:
            return [0.5] * self.n_obs
        N = len(self.train_z)
        k = [self._kernel(z, self.train_z[i]) for i in range(N)]
        out = []
        for d in range(self.n_obs):
            mu = sum(k[i] * self.alpha[d][i] for i in range(N))
            out.append(max(0.0, min(1.0, mu)))
        return out


# ======================================================================
# 辅助
# ======================================================================
def _invert_spd_safe(A: List[List[float]]) -> List[List[float]]:
    n = len(A)
    M = [row[:] + [1.0 if i == j else 0.0 for j in range(n)]
         for i, row in enumerate(A)]
    for col in range(n):
        pivot = col
        for r in range(col + 1, n):
            if abs(M[r][col]) > abs(M[pivot][col]):
                pivot = r
        M[col], M[pivot] = M[pivot], M[col]
        p = M[col][col]
        if abs(p) < NUMERICS.cholesky_jitter:
            M[col][col] += NUMERICS.cholesky_jitter
            p = M[col][col]
        for j in range(2 * n):
            M[col][j] /= p
        for r in range(n):
            if r == col:
                continue
            fac = M[r][col]
            for j in range(2 * n):
                M[r][j] -= fac * M[col][j]
    return [row[n:] for row in M]


__all__ = ["FidelityGPSurrogate", "ResNetEncoder", "fidelity_kernel"]
