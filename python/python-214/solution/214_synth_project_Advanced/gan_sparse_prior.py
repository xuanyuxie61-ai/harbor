"""
gan_sparse_prior.py — 生成式稀疏先验模块 (简化版)
==================================================
来源项目映射:
  - 1070_benstaf_ChemGAN-challenge → GAN 生成模型

科学背景:
  用轻量 GAN 学习稀疏支撑集的分布, 作为结构化先验指导 L1 恢复:
    - 生成器 G: z ↦ {0,1}^M  (候选支撑集)
    - 判别器 D: {0,1}^M ↦ [0,1]  (真实性评分)
  训练后用 G 的采样频率作为 L1 权重: 概率高的分量惩罚小.

  对抗训练:
    min_G max_D  E[log D(x)] + E[log(1 - D(G(z)))]
"""
import numpy as np
from l1_ista import lipschitz_constant


def _sigmoid(x):
    x = np.clip(x, -30, 30)
    return 1.0 / (1.0 + np.exp(-x))


def _relu(x):
    return np.maximum(0, x)


class SparsePriorGenerator:
    """轻量 GAN (numpy 实现)."""

    def __init__(self, M, dim_k=8, hidden=32, seed=0):
        self.M = M
        self.dim_k = dim_k
        rng = np.random.RandomState(seed)
        scale_g = np.sqrt(2.0 / dim_k)
        scale_d = np.sqrt(2.0 / M)
        # 生成器
        self.gW1 = scale_g * rng.randn(dim_k, hidden)
        self.gb1 = np.zeros(hidden)
        self.gW2 = np.sqrt(2.0 / hidden) * rng.randn(hidden, M)
        self.gb2 = np.zeros(M)
        # 判别器
        self.dW1 = scale_d * rng.randn(M, hidden)
        self.db1 = np.zeros(hidden)
        self.dW2 = np.sqrt(2.0 / hidden) * rng.randn(hidden, 1)
        self.db2 = np.zeros(1)
        self.rng = np.random.RandomState(seed + 100)

    def _forward_g(self, z):
        h = _relu(z @ self.gW1 + self.gb1)
        prob = _sigmoid(h @ self.gW2 + self.gb2)
        return h, prob

    def _forward_d(self, x):
        h = _relu(x @ self.dW1 + self.db1)
        return _sigmoid(h @ self.dW2 + self.db2).ravel()

    def generate(self, n, threshold=0.5):
        z = self.rng.randn(n, self.dim_k)
        _, prob = self._forward_g(z)
        samples = (prob > threshold).astype(float)
        return samples, prob

    def train_step(self, data_batch, lr=1e-3):
        batch = data_batch.shape[0]
        # ---- 判别器步 ----
        z = self.rng.randn(batch, self.dim_k)
        _, fake_prob = self._forward_g(z)
        fake = (fake_prob > 0.5).astype(float)

        h_real = _relu(data_batch @ self.dW1 + self.db1)
        d_real = _sigmoid(h_real @ self.dW2 + self.db2).ravel()
        h_fake = _relu(fake @ self.dW1 + self.db1)
        d_fake = _sigmoid(h_fake @ self.dW2 + self.db2).ravel()

        # 判别器 BCE 梯度 (对 logits)
        grad_d_out_real = (d_real - 1.0) / batch  # shape (batch,)
        grad_d_out_fake = d_fake / batch

        # 反向通过 dW2
        grad_dW2 = (h_real.T @ grad_d_out_real.reshape(-1, 1) +
                    h_fake.T @ grad_d_out_fake.reshape(-1, 1))
        grad_db2 = np.sum(grad_d_out_real).reshape(1) + np.sum(grad_d_out_fake).reshape(1)
        # 反向通过 h
        grad_h_real = grad_d_out_real.reshape(-1, 1) @ self.dW2.T
        grad_h_fake = grad_d_out_fake.reshape(-1, 1) @ self.dW2.T
        # relu
        grad_pre_real = grad_h_real * (h_real > 0).astype(float)
        grad_pre_fake = grad_h_fake * (h_fake > 0).astype(float)
        grad_dW1 = (data_batch.T @ grad_pre_real + fake.T @ grad_pre_fake) / batch
        grad_db1 = (np.sum(grad_pre_real, axis=0) +
                    np.sum(grad_pre_fake, axis=0)) / batch

        self.dW2 -= lr * grad_dW2
        self.db2 -= lr * grad_db2
        self.dW1 -= lr * np.clip(grad_dW1, -1, 1)
        self.db1 -= lr * np.clip(grad_db1, -1, 1)

        # ---- 生成器步 ----
        z = self.rng.randn(batch, self.dim_k)
        h_g, fake_prob = self._forward_g(z)
        fake2 = (fake_prob > 0.5).astype(float)
        h_d = _relu(fake2 @ self.dW1 + self.db1)
        d_fake2 = _sigmoid(h_d @ self.dW2 + self.db2).ravel()
        # G 损失: -log(d_fake2)
        grad_g_out = -(1.0 - d_fake2) / batch  # shape (batch,)
        # 通过 G 反向 (简化: 直接对 gW2 的梯度)
        # ∂(gW2)/∂L ≈ h_g^T @ [sigmoid_deriv(fake_prob) · (grad_g_out 广播)]
        grad_per_component = grad_g_out.reshape(-1, 1) * fake_prob * (1 - fake_prob)
        grad_gW2 = (h_g.T @ grad_per_component) / batch
        grad_gb2 = np.sum(grad_per_component, axis=0) / batch
        self.gW2 -= lr * np.clip(grad_gW2, -1, 1)
        self.gb2 -= lr * np.clip(grad_gb2, -1, 1)

        loss_d = -np.mean(np.log(d_real + 1e-8) + np.log(1 - d_fake2 + 1e-8))
        return float(loss_d)

    def generate_weight_vector(self, n_samples=200, threshold=0.5):
        _, probs = self.generate(n_samples, threshold)
        freq = np.mean(probs, axis=0)
        weights = 1.0 - freq
        weights = np.clip(weights, 0.1, 1.0)
        return weights


# ----------------------------------------------------------------------
# 训练数据构造
# ----------------------------------------------------------------------
def generate_training_supports(M, n_samples=500, decay_rate=0.8, seed=0):
    """合成训练支撑集 (模拟 PCE 系数的指数衰减稀疏结构)."""
    rng = np.random.RandomState(seed)
    supports = np.zeros((n_samples, M))
    for i in range(n_samples):
        max_order = rng.geometric(max(1 - decay_rate, 0.01))
        n_active = min(max_order, M)
        active_idx = rng.choice(M, size=n_active, replace=False)
        supports[i, active_idx] = 1.0
    return supports


def weighted_lasso_with_prior(A, y, lam, prior_weights, max_iter=1000, tol=1e-8):
    """加权 LASSO: min 0.5||Ax-y||² + λ ∑ w_i |x_i|."""
    m, n = A.shape
    x = np.zeros(n)
    L = lipschitz_constant(A)
    step = 1.0 / L
    AtA = A.T @ A
    Aty = A.T @ y
    for k in range(max_iter):
        grad = AtA @ x - Aty
        v = x - step * grad
        thresh = step * lam * prior_weights
        x_new = np.sign(v) * np.maximum(np.abs(v) - thresh, 0.0)
        dx = np.linalg.norm(x_new - x)
        x = x_new
        if dx < tol * (np.linalg.norm(x) + 1e-12):
            break
    return x
