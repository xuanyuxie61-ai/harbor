
import numpy as np
from typing import List, Tuple, Dict


class AdamOptimizer:

    def __init__(self, params: List[np.ndarray], lr: float = 1e-3,
                 beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, grads: List[np.ndarray]):
        self.t += 1
        for i, (p, g) in enumerate(zip(self.params, grads)):
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * (g ** 2)
            m_hat = self.m[i] / (1.0 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1.0 - self.beta2 ** self.t)
            self.params[i] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def zero_grad(self):
        pass


def compute_loss(model, graph, Z: np.ndarray, target: Dict[str, float],
                 lambda_physics: float = 0.01) -> float:
    out = model.forward(graph, Z)
    mse = 0.5 * ((out["total_energy"] - target["atomization_energy"]) ** 2 +
                 (out["gamma"] - target["homo_lumo_gap"]) ** 2)


    raise NotImplementedError("NIG loss computation and final loss combination are not implemented")


def spsa_gradient(model, graph, Z: np.ndarray, target: Dict[str, float],
                  lambda_physics: float = 0.01, c: float = 1e-2) -> List[np.ndarray]:
    params = model.parameters()
    delta = []
    for p in params:
        d = np.random.choice([-1.0, 1.0], size=p.shape)
        delta.append(d)


    for p, d in zip(params, delta):
        p += c * d
    loss_plus = compute_loss(model, graph, Z, target, lambda_physics)


    for p, d in zip(params, delta):
        p -= 2.0 * c * d
    loss_minus = compute_loss(model, graph, Z, target, lambda_physics)


    for p, d in zip(params, delta):
        p += c * d

    grads = []
    for d in delta:
        g = (loss_plus - loss_minus) / (2.0 * c) * d
        grads.append(g)
    return grads


def train_epoch(model, dataset, train_idx: List[int],
                optimizer: AdamOptimizer, lambda_physics: float = 0.01) -> float:
    total_loss = 0.0
    np.random.shuffle(train_idx)
    for idx in train_idx:
        graph, target, Z = dataset[idx]
        grads = spsa_gradient(model, graph, Z, target, lambda_physics, c=1e-2)
        optimizer.step(grads)
        loss = compute_loss(model, graph, Z, target, lambda_physics)
        total_loss += loss
    return total_loss / max(len(train_idx), 1)


def evaluate(model, dataset, test_idx: List[int]) -> Dict[str, float]:
    errors_energy = []
    errors_gap = []
    aleatoric_list = []
    epistemic_list = []

    for idx in test_idx:
        graph, target, Z = dataset[idx]
        out = model.forward(graph, Z)
        errors_energy.append(abs(out["total_energy"] - target["atomization_energy"]))
        errors_gap.append(abs(out["gamma"] - target["homo_lumo_gap"]))
        ale, epi = model.uncertainty_head.uncertainty(
            np.array([out["alpha"]]),
            np.array([out["beta"]]),
            np.array([out["nu"]])
        )
        aleatoric_list.append(float(ale[0]))
        epistemic_list.append(float(epi[0]))

    return {
        "mae_energy": float(np.mean(errors_energy)),
        "mae_gap": float(np.mean(errors_gap)),
        "mean_aleatoric": float(np.mean(aleatoric_list)),
        "mean_epistemic": float(np.mean(epistemic_list))
    }
