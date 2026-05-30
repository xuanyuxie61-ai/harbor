
import numpy as np


def compute_pde_residual(network, X_f):
    if X_f.ndim != 2 or X_f.shape[1] != 2:
        raise ValueError("X_f must have shape (N_f, 2)")








    raise NotImplementedError("Hole 3: compute_pde_residual not implemented")


def compute_ic_loss(network, X_ic, u_ic_target):
    u_pred = network.forward(X_ic, store_cache=False).ravel()
    diff = u_pred - u_ic_target
    return np.mean(diff ** 2)


def compute_bc_loss(network, X_bc_0, X_bc_L):
    u_0 = network.forward(X_bc_0, store_cache=False).ravel()
    u_L = network.forward(X_bc_L, store_cache=False).ravel()
    diff = u_0 - u_L
    return np.mean(diff ** 2)


def compute_total_loss(network, X_f, X_ic, u_ic_target, X_bc_0, X_bc_L,
                       lambda_pde=1.0, lambda_ic=100.0, lambda_bc=10.0):
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
    params = network.get_params_flat()
    P = len(params)


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


    network.set_params_flat(params)

    return gradient, loss, loss_dict


def compute_loss_gradient_minibatch(network, X_f, X_ic, u_ic_target,
                                    X_bc_0, X_bc_L,
                                    lambda_pde=1.0, lambda_ic=100.0,
                                    lambda_bc=10.0,
                                    batch_size_f=256, seed=42):
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
