"""
physics_loss.py
===============
Physics-Informed Neural Network loss computation.

The total loss functional for the KS equation is:

    L_total(\theta) = \lambda_pde * L_pde(\theta)
                    + \lambda_ic  * L_ic(\theta)
                    + \lambda_bc  * L_bc(\theta)

where:

    L_pde = (1/N_f) \sum_{i=1}^{N_f} | r(t_i, x_i) |^2

    r(t,x) = \partial_t u_\theta + u_\theta \partial_x u_\theta
            + \partial_{xx} u_\theta + \partial_{xxxx} u_\theta

    L_ic = (1/N_ic) \sum_{j=1}^{N_ic} | u_\theta(0, x_j) - u_0(x_j) |^2

    L_bc = (1/N_bc) \sum_{k=1}^{N_bc} | u_\theta(t_k, 0) - u_\theta(t_k, L) |^2

The derivatives \partial_t, \partial_x, \partial_{xx}, \partial_{xxxx} are
computed via finite differences through the network forward pass.

For numerical stability, the fourth derivative uses a 5-point stencil with
sufficiently large step size to avoid catastrophic cancellation.
"""

import numpy as np


def compute_pde_residual(network, X_f):
    """
    Compute the KS PDE residual r(t,x) at collocation points X_f.

    Parameters
    ----------
    network : PINNNetwork
    X_f : ndarray, shape (N_f, 2)
        Collocation points with columns [t, x].

    Returns
    -------
    residual : ndarray, shape (N_f,)
        PDE residual at each point.
    """
    if X_f.ndim != 2 or X_f.shape[1] != 2:
        raise ValueError("X_f must have shape (N_f, 2)")

    # TODO (Hole 3): Compute the KS PDE residual using network predictions.
    # The Kuramoto-Sivashinsky equation is:
    #   u_t + u * u_x + u_xx + u_xxxx = 0
    # You need to:
    #   1. Get network prediction u = network.forward(X_f, store_cache=False).ravel()
    #   2. Compute u_t, u_x, u_xx, u_xxxx via network derivative methods
    #   3. Assemble the residual according to the KS equation
    raise NotImplementedError("Hole 3: compute_pde_residual not implemented")


def compute_ic_loss(network, X_ic, u_ic_target):
    """
    Compute initial condition loss.

    Parameters
    ----------
    network : PINNNetwork
    X_ic : ndarray, shape (N_ic, 2)
        Initial condition points [0, x_j].
    u_ic_target : ndarray, shape (N_ic,)
        Target initial values.

    Returns
    -------
    loss : float
        Mean squared IC error.
    """
    u_pred = network.forward(X_ic, store_cache=False).ravel()
    diff = u_pred - u_ic_target
    return np.mean(diff ** 2)


def compute_bc_loss(network, X_bc_0, X_bc_L):
    """
    Compute periodic boundary condition loss:
        u(t, 0) = u(t, L)  for all t

    Parameters
    ----------
    network : PINNNetwork
    X_bc_0 : ndarray, shape (N_bc, 2)
        Points at x = 0.
    X_bc_L : ndarray, shape (N_bc, 2)
        Points at x = L.

    Returns
    -------
    loss : float
        Mean squared BC error.
    """
    u_0 = network.forward(X_bc_0, store_cache=False).ravel()
    u_L = network.forward(X_bc_L, store_cache=False).ravel()
    diff = u_0 - u_L
    return np.mean(diff ** 2)


def compute_total_loss(network, X_f, X_ic, u_ic_target, X_bc_0, X_bc_L,
                       lambda_pde=1.0, lambda_ic=100.0, lambda_bc=10.0):
    """
    Compute the total physics-informed loss.

    Returns
    -------
    loss : float
    loss_dict : dict
        Breakdown of individual loss terms.
    """
    residual = compute_pde_residual(network, X_f)
    L_pde = np.mean(residual ** 2)
    L_ic = compute_ic_loss(network, X_ic, u_ic_target)
    L_bc = compute_bc_loss(network, X_bc_0, X_bc_L)

    total = lambda_pde * L_pde + lambda_ic * L_ic + lambda_bc * L_bc
    loss_dict = {
        'pde': L_pde,
        'ic': L_ic,
        'bc': L_bc,
        'total': total,
    }
    return total, loss_dict


def compute_loss_gradient(network, X_f, X_ic, u_ic_target, X_bc_0, X_bc_L,
                          lambda_pde=1.0, lambda_ic=100.0, lambda_bc=10.0):
    """
    Compute the gradient of the total loss with respect to all network
    parameters using numerical differentiation (central differences).

    For a network with P parameters, this requires 2P forward passes.
    To keep computation tractable, we use a small finite difference step.

    Returns
    -------
    gradient : ndarray, shape (P,)
    loss : float
    loss_dict : dict
    """
    params = network.get_params_flat()
    P = len(params)

    # Evaluate loss at current parameters
    loss, loss_dict = compute_total_loss(
        network, X_f, X_ic, u_ic_target, X_bc_0, X_bc_L,
        lambda_pde, lambda_ic, lambda_bc
    )

    h = 1e-6
    gradient = np.zeros(P)

    for i in range(P):
        params_plus = params.copy()
        params_minus = params.copy()
        params_plus[i] += h
        params_minus[i] -= h

        network.set_params_flat(params_plus)
        loss_plus, _ = compute_total_loss(
            network, X_f, X_ic, u_ic_target, X_bc_0, X_bc_L,
            lambda_pde, lambda_ic, lambda_bc
        )
        network.set_params_flat(params_minus)
        loss_minus, _ = compute_total_loss(
            network, X_f, X_ic, u_ic_target, X_bc_0, X_bc_L,
            lambda_pde, lambda_ic, lambda_bc
        )
        gradient[i] = (loss_plus - loss_minus) / (2.0 * h)

    # Restore original parameters
    network.set_params_flat(params)

    return gradient, loss, loss_dict


def compute_loss_gradient_minibatch(network, X_f, X_ic, u_ic_target,
                                    X_bc_0, X_bc_L,
                                    lambda_pde=1.0, lambda_ic=100.0,
                                    lambda_bc=10.0,
                                    batch_size_f=256, seed=42):
    """
    Compute an approximate gradient using a mini-batch of collocation points.
    This reduces the per-iteration cost for large collocation sets.
    """
    rng = np.random.default_rng(seed)
    N_f = X_f.shape[0]
    if batch_size_f < N_f:
        idx = rng.choice(N_f, size=batch_size_f, replace=False)
        X_f_batch = X_f[idx]
    else:
        X_f_batch = X_f

    return compute_loss_gradient(
        network, X_f_batch, X_ic, u_ic_target, X_bc_0, X_bc_L,
        lambda_pde, lambda_ic, lambda_bc
    )
