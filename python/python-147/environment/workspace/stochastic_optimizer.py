
import numpy as np


class SGDWithMomentum:

    def __init__(self, params_dim, lr=1e-3, momentum=0.9,
                 lr_decay=0.995, min_lr=1e-6):
        if params_dim < 1:
            raise ValueError("params_dim must be positive")
        self.params_dim = params_dim
        self.lr = float(lr)
        self.momentum = float(momentum)
        self.lr_decay = float(lr_decay)
        self.min_lr = float(min_lr)
        self.velocity = np.zeros(params_dim)
        self.iteration = 0

    def step(self, params, gradient):
        if len(params) != self.params_dim or len(gradient) != self.params_dim:
            raise ValueError("Dimension mismatch")

        self.iteration += 1
        eta = max(self.lr * (self.lr_decay ** self.iteration), self.min_lr)

        self.velocity = self.momentum * self.velocity - eta * gradient
        new_params = params + self.velocity
        return new_params

    def reset(self):
        self.velocity = np.zeros(self.params_dim)
        self.iteration = 0


class StochasticCoordinateDescent:

    def __init__(self, params_dim, block_size=8, lr=1e-4, lr_decay=0.99):
        if params_dim < 1:
            raise ValueError("params_dim must be positive")
        self.params_dim = params_dim
        self.block_size = min(block_size, params_dim)
        self.lr = float(lr)
        self.lr_decay = float(lr_decay)
        self.iteration = 0
        self.rng = np.random.default_rng(42)

    def step(self, params, gradient):
        if len(params) != self.params_dim or len(gradient) != self.params_dim:
            raise ValueError("Dimension mismatch")

        self.iteration += 1
        eta = self.lr * (self.lr_decay ** self.iteration)


        block = self.rng.choice(self.params_dim, size=self.block_size,
                                replace=False)
        new_params = params.copy()
        new_params[block] -= eta * gradient[block]
        return new_params

    def reset(self):
        self.iteration = 0


class CosineAnnealingScheduler:

    def __init__(self, eta_max, eta_min, T_max):
        self.eta_max = float(eta_max)
        self.eta_min = float(eta_min)
        self.T_max = int(T_max)
        if self.T_max < 1:
            raise ValueError("T_max must be positive")

    def get_lr(self, t):
        if t < 0:
            raise ValueError("t must be non-negative")
        if t >= self.T_max:
            return self.eta_min
        return self.eta_min + 0.5 * (self.eta_max - self.eta_min) * \
               (1.0 + np.cos(np.pi * t / self.T_max))


class CombinedOptimizer:

    def __init__(self, params_dim, sgd_lr=1e-3, sgd_momentum=0.9,
                 scd_lr=1e-4, scd_block_size=8,
                 switch_iteration=2000):
        self.sgd = SGDWithMomentum(params_dim, lr=sgd_lr, momentum=sgd_momentum)
        self.scd = StochasticCoordinateDescent(params_dim, block_size=scd_block_size, lr=scd_lr)
        self.switch_iteration = int(switch_iteration)
        self.iteration = 0

    def step(self, params, gradient):
        self.iteration += 1
        if self.iteration <= self.switch_iteration:
            return self.sgd.step(params, gradient)
        else:
            return self.scd.step(params, gradient)

    def reset(self):
        self.sgd.reset()
        self.scd.reset()
        self.iteration = 0
