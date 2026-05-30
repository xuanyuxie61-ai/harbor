
import numpy as np


class DenseLayer:
    def __init__(self, in_dim: int, out_dim: int):

        self.W = np.random.randn(in_dim, out_dim) * np.sqrt(2.0 / in_dim)
        self.b = np.zeros(out_dim)
        self.x = None
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        return x @ self.W + self.b

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        self.dW = self.x.T @ grad_out / self.x.shape[0]
        self.db = np.mean(grad_out, axis=0)
        return grad_out @ self.W.T

    def step(self, lr: float, momentum: float = 0.9):
        if not hasattr(self, 'vW'):
            self.vW = np.zeros_like(self.W)
            self.vb = np.zeros_like(self.b)
        self.vW = momentum * self.vW - lr * self.dW
        self.vb = momentum * self.vb - lr * self.db
        self.W += self.vW
        self.b += self.vb


class ReLU:
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        return np.maximum(0.0, x)

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        return grad_out * (self.x > 0.0).astype(float)


class LeakyReLU:
    def __init__(self, alpha: float = 0.2):
        self.alpha = alpha

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        return np.where(x > 0.0, x, self.alpha * x)

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        return grad_out * np.where(self.x > 0.0, 1.0, self.alpha)


class Sigmoid:
    def forward(self, x: np.ndarray) -> np.ndarray:

        self.x = x
        out = np.empty_like(x)
        pos_mask = x >= 0.0
        neg_mask = ~pos_mask
        out[pos_mask] = 1.0 / (1.0 + np.exp(-x[pos_mask]))
        exp_x = np.exp(x[neg_mask])
        out[neg_mask] = exp_x / (1.0 + exp_x)
        self.out = out
        return out

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        return grad_out * self.out * (1.0 - self.out)


class MSELoss:
    def forward(self, pred: np.ndarray, target: np.ndarray) -> float:
        self.pred = pred
        self.target = target
        self.diff = pred - target
        return float(np.mean(self.diff ** 2))

    def backward(self) -> np.ndarray:
        return 2.0 * self.diff / self.diff.size


class BCELoss:
    def forward(self, pred: np.ndarray, target: np.ndarray) -> float:
        self.pred = np.clip(pred, 1e-7, 1.0 - 1e-7)
        self.target = target
        return float(-np.mean(self.target * np.log(self.pred)
                              + (1.0 - self.target) * np.log(1.0 - self.pred)))

    def backward(self) -> np.ndarray:
        return -(self.target / self.pred - (1.0 - self.target) / (1.0 - self.pred)) / self.target.size


class Generator:
    def __init__(self, latent_dim: int = 8, coord_dim: int = 4,
                 hidden_dim: int = 32, output_dim: int = 4, seed: int = None):
        rng = np.random.default_rng(seed)
        np.random.seed(seed % (2**31) if seed is not None else None)
        in_dim = latent_dim + coord_dim
        self.fc1 = DenseLayer(in_dim, hidden_dim)
        self.relu1 = ReLU()
        self.fc2 = DenseLayer(hidden_dim, hidden_dim)
        self.relu2 = ReLU()
        self.fc3 = DenseLayer(hidden_dim, output_dim)
        self.layers = [self.fc1, self.relu1, self.fc2, self.relu2, self.fc3]
        self.latent_dim = latent_dim
        self.coord_dim = coord_dim

    def forward(self, z: np.ndarray, coords: np.ndarray) -> np.ndarray:
        x = np.concatenate([z, coords], axis=1)
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad_out: np.ndarray):
        for layer in reversed(self.layers):
            grad_out = layer.backward(grad_out)
        return grad_out

    def step(self, lr: float, momentum: float = 0.9):
        for layer in self.layers:
            if isinstance(layer, DenseLayer):
                layer.step(lr, momentum)

    def zero_grad(self):
        for layer in self.layers:
            if isinstance(layer, DenseLayer):
                layer.dW.fill(0.0)
                layer.db.fill(0.0)


class Discriminator:
    def __init__(self, input_dim: int = 8, hidden_dim: int = 32,
                 seed: int = None):
        np.random.seed(seed % (2**31) if seed is not None else None)
        self.fc1 = DenseLayer(input_dim, hidden_dim)
        self.lrelu1 = LeakyReLU(0.2)
        self.fc2 = DenseLayer(hidden_dim, 16)
        self.lrelu2 = LeakyReLU(0.2)
        self.fc3 = DenseLayer(16, 1)
        self.sigmoid = Sigmoid()
        self.layers = [self.fc1, self.lrelu1, self.fc2, self.lrelu2,
                       self.fc3, self.sigmoid]

    def forward(self, state: np.ndarray) -> np.ndarray:
        x = state
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad_out: np.ndarray):
        for layer in reversed(self.layers):
            grad_out = layer.backward(grad_out)
        return grad_out

    def step(self, lr: float, momentum: float = 0.9):
        for layer in self.layers:
            if isinstance(layer, DenseLayer):
                layer.step(lr, momentum)

    def zero_grad(self):
        for layer in self.layers:
            if isinstance(layer, DenseLayer):
                layer.dW.fill(0.0)
                layer.db.fill(0.0)


def compute_physics_loss_batch(gen: Generator, coords_batch: np.ndarray,
                               z_batch: np.ndarray, nx: int = 6, ny: int = 6,
                               nz: int = 6) -> float:
    batch = coords_batch.shape[0]
    if batch != nx * ny * nz:

        return 0.0

    z_single = z_batch[0:1, :]
    z_repeat = np.tile(z_single, (batch, 1))
    pred = gen.forward(z_repeat, coords_batch)

    u = pred[:, 0]
    v = pred[:, 1]
    w = pred[:, 2]
    p = pred[:, 3]

    x = coords_batch[:, 0]
    y = coords_batch[:, 1]
    z = coords_batch[:, 2]

    from navier_stokes_exact import ns_residual
    try:
        res = ns_residual(u, v, w, p, x, y, z, coords_batch[:, 3])
        return float(res["total"])
    except Exception:
        return 0.0
