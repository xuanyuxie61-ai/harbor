# -*- coding: utf-8 -*-

import numpy as np
from typing import List, Tuple, Optional


class SigmoidActivation:

    @staticmethod
    def f(z: np.ndarray) -> np.ndarray:

        z = np.clip(z, -500.0, 500.0)
        return 1.0 / (1.0 + np.exp(-z))

    @staticmethod
    def df_dz(a: np.ndarray) -> np.ndarray:
        return a * (1.0 - a)


class NeuralNetworkSurrogate:

    def __init__(self, layer_sizes: List[int], seed: Optional[int] = None):
        if len(layer_sizes) < 2:
            raise ValueError("至少需要输入层和输出层")
        self.layer_sizes = layer_sizes
        self.n_layers = len(layer_sizes)
        self.rng = np.random.default_rng(seed)
        self._init_weights()
        self.activation = SigmoidActivation()

    def _init_weights(self):
        self.W = []
        self.b = []
        for i in range(1, self.n_layers):
            n_in = self.layer_sizes[i - 1]
            n_out = self.layer_sizes[i]
            limit = np.sqrt(6.0 / (n_in + n_out))
            W = self.rng.uniform(-limit, limit, size=(n_out, n_in))
            b = np.zeros((n_out, 1))
            self.W.append(W)
            self.b.append(b)

    def forward(self, x: np.ndarray) -> List[np.ndarray]:
        a = [x]
        for l in range(self.n_layers - 1):
            z = self.W[l] @ a[-1] + self.b[l]
            a_next = self.activation.f(z)
            a.append(a_next)
        return a

    def predict(self, x: np.ndarray) -> np.ndarray:
        a = self.forward(x)
        return a[-1]

    def cost(self, X: np.ndarray, Y: np.ndarray) -> float:
        y_pred = self.predict(X)
        m = Y.shape[1]
        return float(np.sum((y_pred - Y) ** 2) / (2.0 * m))

    def train_sgd(self, X: np.ndarray, Y: np.ndarray,
                  eta: float = 0.5,
                  n_iterations: int = 5000,
                  print_every: int = 1000) -> List[float]:
        m = X.shape[1]
        costs = []
        for it in range(1, n_iterations + 1):

            idx = self.rng.integers(0, m)
            x_i = X[:, idx:idx + 1]
            y_i = Y[:, idx:idx + 1]


            a = self.forward(x_i)


            deltas = [None] * (self.n_layers - 1)

            deltas[-1] = (a[-1] - y_i) * self.activation.df_dz(a[-1])

            for l in range(self.n_layers - 3, -1, -1):
                deltas[l] = (self.W[l + 1].T @ deltas[l + 1]) * self.activation.df_dz(a[l + 1])


            for l in range(self.n_layers - 1):
                dW = deltas[l] @ a[l].T
                db = deltas[l]
                self.W[l] -= eta * dW
                self.b[l] -= eta * db

            if it % print_every == 0:
                c = self.cost(X, Y)
                costs.append(c)
        return costs

    def train_batch(self, X: np.ndarray, Y: np.ndarray,
                    eta: float = 0.1,
                    n_epochs: int = 2000,
                    batch_size: Optional[int] = None,
                    print_every: int = 500) -> List[float]:
        m = X.shape[1]
        if batch_size is None or batch_size > m:
            batch_size = m
        costs = []
        for epoch in range(1, n_epochs + 1):

            perm = self.rng.permutation(m)
            X_shuffled = X[:, perm]
            Y_shuffled = Y[:, perm]

            for start in range(0, m, batch_size):
                end = min(start + batch_size, m)
                X_batch = X_shuffled[:, start:end]
                Y_batch = Y_shuffled[:, start:end]

                a = self.forward(X_batch)
                deltas = [None] * (self.n_layers - 1)
                deltas[-1] = (a[-1] - Y_batch) * self.activation.df_dz(a[-1])
                for l in range(self.n_layers - 3, -1, -1):
                    deltas[l] = (self.W[l + 1].T @ deltas[l + 1]) * self.activation.df_dz(a[l + 1])

                batch_m = end - start
                for l in range(self.n_layers - 1):
                    dW = (deltas[l] @ a[l].T) / batch_m
                    db = np.mean(deltas[l], axis=1, keepdims=True)
                    self.W[l] -= eta * dW
                    self.b[l] -= eta * db

            if epoch % print_every == 0:
                c = self.cost(X, Y)
                costs.append(c)
        return costs

    def evaluate_metrics(self, X: np.ndarray, Y: np.ndarray) -> dict:
        y_pred = self.predict(X)
        mse = float(np.mean((y_pred - Y) ** 2))
        rmse = np.sqrt(mse)
        mae = float(np.mean(np.abs(y_pred - Y)))
        ss_res = np.sum((Y - y_pred) ** 2)
        ss_tot = np.sum((Y - np.mean(Y)) ** 2)
        r2 = 1.0 - ss_res / (ss_tot + 1e-30)
        return {"mse": mse, "rmse": float(rmse), "mae": mae, "r2": float(r2)}


class SensorResponseSurrogate:

    def __init__(self):
        self.nn = NeuralNetworkSurrogate(layer_sizes=[2, 12, 8, 1], seed=42)
        self._trained = False

    def generate_training_data(self, n_points: int = 200,
                                delta_T_range: Tuple[float, float] = (-10.0, 10.0),
                                n_env_range: Tuple[float, float] = (1.00, 1.05)) -> Tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(123)
        delta_T = rng.uniform(delta_T_range[0], delta_T_range[1], size=n_points)
        n_env = rng.uniform(n_env_range[0], n_env_range[1], size=n_points)


        lambda0 = 1550.0
        ng = 3.5
        dn_dT = 1.86e-4
        S = 100.0
        delta_n_env = n_env - 1.00

        delta_lambda = (lambda0 / ng) * dn_dT * delta_T + S * delta_n_env

        delta_lambda += rng.normal(0.0, 0.5, size=n_points)

        X = np.vstack([delta_T, n_env])
        Y = delta_lambda.reshape(1, -1)
        return X, Y

    def train(self, n_points: int = 200):
        X, Y = self.generate_training_data(n_points)

        self._y_min = float(np.min(Y))
        self._y_max = float(np.max(Y))
        self._y_range = self._y_max - self._y_min
        if self._y_range < 1e-12:
            self._y_range = 1.0
        Y_norm = (Y - self._y_min) / self._y_range
        self.nn.train_batch(X, Y_norm, eta=0.5, n_epochs=5000, batch_size=32)
        self._trained = True

    def predict(self, delta_T: float, n_env: float) -> float:
        if not self._trained:
            self.train()
        X = np.array([[delta_T], [n_env]])
        y_norm = float(self.nn.predict(X)[0, 0])

        y_raw = y_norm * self._y_range + self._y_min
        return y_raw
