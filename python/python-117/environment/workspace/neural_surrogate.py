
import numpy as np
from typing import Tuple, List


class NeuralSurrogate:

    def __init__(self, input_dim: int = 6,
                 hidden_dims: List[int] = None,
                 lr: float = 0.01,
                 lambda_reg: float = 1e-5,
                 seed: int = 42):
        if hidden_dims is None:
            hidden_dims = [32, 16, 8]
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.lr = lr
        self.lambda_reg = lambda_reg
        np.random.seed(seed)
        self._init_weights()
        self.x_mean = None
        self.x_std = None
        self.y_mean = None
        self.y_std = None

    def _init_weights(self):
        dims = [self.input_dim] + self.hidden_dims + [1]
        self.W = []
        self.b = []
        for i in range(len(dims) - 1):
            std = np.sqrt(2.0 / dims[i])
            self.W.append(np.random.randn(dims[i], dims[i + 1]) * std)
            self.b.append(np.zeros(dims[i + 1]))

        self.bn_gamma = []
        self.bn_beta = []
        self.bn_running_mean = []
        self.bn_running_var = []
        for h in self.hidden_dims:
            self.bn_gamma.append(np.ones(h))
            self.bn_beta.append(np.zeros(h))
            self.bn_running_mean.append(np.zeros(h))
            self.bn_running_var.append(np.ones(h))

    def _relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, x)

    def _relu_deriv(self, x: np.ndarray) -> np.ndarray:
        return (x > 0).astype(np.float64)

    def _batch_norm_forward(self, x: np.ndarray, idx: int,
                            training: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if training:
            mu = np.mean(x, axis=0)
            var = np.var(x, axis=0) + 1e-8

            self.bn_running_mean[idx] = 0.9 * self.bn_running_mean[idx] + 0.1 * mu
            self.bn_running_var[idx] = 0.9 * self.bn_running_var[idx] + 0.1 * var
        else:
            mu = self.bn_running_mean[idx]
            var = self.bn_running_var[idx] + 1e-8
        x_hat = (x - mu) / np.sqrt(var)
        y = self.bn_gamma[idx] * x_hat + self.bn_beta[idx]
        return y, mu, var

    def forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        self._cache_z = []
        self._cache_a = []
        self._cache_bn = []
        a = X
        for i in range(len(self.hidden_dims)):
            z = a @ self.W[i] + self.b[i]
            a, mu, var = self._batch_norm_forward(z, i, training)
            a = self._relu(a)
            self._cache_z.append(z)
            self._cache_a.append(a)
            self._cache_bn.append((mu, var))

        z_out = a @ self.W[-1] + self.b[-1]
        return z_out

    def backward(self, X: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray):
        batch_size = X.shape[0]

        dz = (y_pred - y_true) / batch_size

        grad_W = [None] * len(self.W)
        grad_b = [None] * len(self.b)
        for i in range(len(self.W) - 1, -1, -1):
            if i == 0:
                a_prev = X
            else:
                a_prev = self._cache_a[i - 1]
            grad_W[i] = a_prev.T @ dz + self.lambda_reg * self.W[i]
            grad_b[i] = np.sum(dz, axis=0)
            if i > 0:
                da_prev = dz @ self.W[i].T
                dz = da_prev * self._relu_deriv(self._cache_z[i - 1])

        for i in range(len(self.W)):
            self.W[i] -= self.lr * grad_W[i]
            self.b[i] -= self.lr * grad_b[i]

    def _normalize_X(self, X: np.ndarray) -> np.ndarray:
        if self.x_mean is None or self.x_std is None:
            return X
        return (X - self.x_mean) / (self.x_std + 1e-8)

    def _normalize_y(self, y: np.ndarray) -> np.ndarray:
        if self.y_mean is None or self.y_std is None:
            return y
        return (y - self.y_mean) / (self.y_std + 1e-8)

    def _denormalize_y(self, y_norm: np.ndarray) -> np.ndarray:
        if self.y_mean is None or self.y_std is None:
            return y_norm
        return y_norm * (self.y_std + 1e-8) + self.y_mean

    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 200,
              batch_size: int = 32, verbose: bool = False) -> List[float]:

        self.x_mean = np.mean(X, axis=0, keepdims=True)
        self.x_std = np.std(X, axis=0, keepdims=True)
        self.y_mean = np.mean(y, axis=0, keepdims=True)
        self.y_std = np.std(y, axis=0, keepdims=True)
        X_norm = self._normalize_X(X)
        y_norm = self._normalize_y(y)
        n = X.shape[0]
        losses = []
        for epoch in range(epochs):
            idx = np.random.permutation(n)
            X_shuf = X_norm[idx]
            y_shuf = y_norm[idx]
            epoch_loss = 0.0
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                Xb = X_shuf[start:end]
                yb = y_shuf[start:end]
                y_pred = self.forward(Xb, training=True)
                loss = np.mean((y_pred - yb) ** 2)
                epoch_loss += loss * (end - start)
                self.backward(Xb, yb, y_pred)
            epoch_loss /= n
            losses.append(float(epoch_loss * (self.y_std[0, 0] + 1e-8) ** 2))
            if verbose and epoch % 50 == 0:
                print(f"Epoch {epoch}, Loss={losses[-1]:.6f}")
        return losses

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_norm = self._normalize_X(X)
        y_pred_norm = self.forward(X_norm, training=False)
        return self._denormalize_y(y_pred_norm)






def generate_training_data(n_samples: int = 512,
                           seed: int = 123) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    z = rng.uniform(0.5, 8.0, n_samples)
    R_np = rng.uniform(1.0, 5.0, n_samples)
    zeta_np = rng.uniform(-0.1, 0.1, n_samples)
    zeta_mem = rng.uniform(-0.08, 0.0, n_samples)
    kappa_bend = rng.uniform(10.0, 40.0, n_samples)
    I_ionic = rng.uniform(0.01, 0.5, n_samples)

    lambda_D = 0.3 / np.sqrt(I_ionic + 1e-6)
    A_H = 40.0

    z_eff = np.maximum(z, 0.5 * R_np)

    dG_elec = 100.0 * R_np * zeta_np * zeta_mem * np.exp(-z_eff / lambda_D)
    dG_vdw = -A_H * R_np / (12.0 * z_eff + 1e-3)
    dG_bend = np.pi * kappa_bend * (R_np / (z_eff + 1e-3)) ** 2
    dG = dG_elec + dG_vdw + dG_bend

    dG += rng.normal(0.0, 2.0, n_samples)

    dG = np.clip(dG, -500.0, 500.0)
    X = np.column_stack((z, R_np, zeta_np, zeta_mem, kappa_bend, I_ionic))
    y = dG.reshape(-1, 1)
    return X, y
