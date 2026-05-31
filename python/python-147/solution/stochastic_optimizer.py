"""
stochastic_optimizer.py
=======================
Stochastic optimization routines for training the PINN.

In standard deep learning, stochastic gradient descent (SGD) samples a subset
of training data (mini-batch) to approximate the gradient.  Here we also
incorporate a stochastic Gauss-Seidel-style coordinate descent that updates
individual parameters or small blocks sequentially, which is analogous to the
stochastic Gauss-Seidel method for linear systems from seed project 453.

For the PINN loss:
    L(\theta) = L_pde(\theta) + \lambda_ic L_ic(\theta) + \lambda_bc L_bc(\theta)

we employ:
  1. Mini-batch SGD with momentum
  2. Stochastic coordinate descent (SCD) that randomly selects weight blocks
  3. Learning rate scheduling with cosine annealing

The combined optimizer first runs SGD for coarse convergence, then refines
with SCD for fine-tuning individual coordinate directions.
"""

import numpy as np


class SGDWithMomentum:
    """
    Stochastic Gradient Descent with Nesterov momentum.

    Update rule:
        v_{t+1} = \mu v_t - \eta_t g_t
        \theta_{t+1} = \theta_t + v_{t+1}

    where g_t is the gradient on a mini-batch, \mu is momentum, and
    \eta_t is the learning rate.
    """

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
        """
        Perform one optimization step.

        Parameters
        ----------
        params : ndarray, shape (params_dim,)
            Current parameter vector.
        gradient : ndarray, shape (params_dim,)
            Gradient of loss w.r.t. parameters.

        Returns
        -------
        new_params : ndarray
            Updated parameters.
        """
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
    """
    Stochastic coordinate descent inspired by Gauss-Seidel iteration.

    For linear systems Ax = b, Gauss-Seidel updates:
        x_i^{new} = x_i + (b_i - A_i^T x) / A_{ii}

    For nonlinear optimization, we randomly select a coordinate block and
    perform a line search along the negative gradient direction for that block.

    This mimics the stochastic Gauss-Seidel from seed project 453.
    """

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
        """
        Update a random block of coordinates.
        """
        if len(params) != self.params_dim or len(gradient) != self.params_dim:
            raise ValueError("Dimension mismatch")

        self.iteration += 1
        eta = self.lr * (self.lr_decay ** self.iteration)

        # Randomly select a block of coordinates without replacement
        block = self.rng.choice(self.params_dim, size=self.block_size,
                                replace=False)
        new_params = params.copy()
        new_params[block] -= eta * gradient[block]
        return new_params

    def reset(self):
        self.iteration = 0


class CosineAnnealingScheduler:
    """
    Cosine annealing learning rate scheduler:

        \eta_t = \eta_min + 0.5 * (\eta_max - \eta_min)
                 * (1 + cos( \pi * t / T_max ))

    where T_max is the maximum number of iterations.
    """

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
    """
    Combined training strategy: first SGD with momentum, then SCD refinement.
    """

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
