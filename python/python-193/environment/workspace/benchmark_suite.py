
import numpy as np
import math


def logistic_reaction_diffusion_matrix(n, D=1.0, r=1.0, K=1.0, domain=1.0):
    h = domain / (n + 1)
    main_diag = 2.0 * D / (h * h) - r
    off_diag = -D / (h * h)

    A = np.zeros((n, n))
    for i in range(n):
        A[i, i] = main_diag
        if i > 0:
            A[i, i - 1] = off_diag
        if i < n - 1:
            A[i, i + 1] = off_diag
    return A


def feynman_kac_potential(a, x):
    return 2.0 * (x / (a * a)) ** 2 + 1.0 / (a * a)


def feynman_kac_exact(a, x):
    return np.exp((x / a) ** 2 - 1.0)


def feynman_kac_stochastic_solve(a=2.0, h=0.01, n_paths=5000, n_x=21):
    x_grid = np.linspace(-a, a, n_x)
    u_est = np.zeros(n_x)

    rng = np.random.default_rng(42)
    for ix, X0 in enumerate(x_grid):
        total = 0.0
        for _ in range(n_paths):
            X = X0
            integral_V = 0.0
            while abs(X) < a:

                dW = rng.normal(0.0, math.sqrt(h))
                V_old = feynman_kac_potential(a, X)
                X_new = X + dW

                V_new = feynman_kac_potential(a, X_new)
                integral_V += 0.5 * (V_old + V_new) * h
                X = X_new
            total += math.exp(-integral_V)
        u_est[ix] = total / n_paths

    return x_grid, u_est


def chirikov_map_jacobian_ensemble(n_ensemble=100, k=0.55, n_steps=50, seed=42):
    rng = np.random.default_rng(seed)
    J_sum = np.zeros((2, 2))
    for _ in range(n_ensemble):
        x = rng.random() * 2.0 * math.pi
        y = rng.random() * 2.0 * math.pi
        for _ in range(n_steps):
            J = np.array([[1.0, k * math.cos(x)],
                          [1.0, 1.0 + k * math.cos(x)]])
            J_sum += J
            y_new = y + k * math.sin(x)
            x_new = x + y_new
            x = x_new % (2.0 * math.pi)
            y = y_new % (2.0 * math.pi)
    J_avg = J_sum / n_ensemble
    return J_avg


def menger_sponge_hierarchical_matrix(level=3, epsilon=1e-3):
    if level == 0:
        return np.array([[1.0]])

    n_sub = 2 ** (level - 1)
    A_sub = menger_sponge_hierarchical_matrix(level - 1, epsilon)


    scale = epsilon ** (level - 1)
    B = scale * np.random.default_rng(42 + level).random((n_sub, n_sub))
    C = scale * np.random.default_rng(43 + level).random((n_sub, n_sub))

    n = 2 * n_sub
    A = np.zeros((n, n))
    A[:n_sub, :n_sub] = A_sub
    A[:n_sub, n_sub:] = B
    A[n_sub:, :n_sub] = C
    A[n_sub:, n_sub:] = A_sub


    A = 0.5 * (A + A.T)

    for i in range(n):
        A[i, i] += 1.0 + np.sum(np.abs(A[i, :])) - abs(A[i, i])
    return A


def candy_circulant_matrix(n, block_size=5, value=1.0):
    rng = np.random.default_rng(136)
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if (i + j) % block_size == 0:
                A[i, j] = value
            else:
                A[i, j] = 0.01 * rng.random()

    for i in range(n):
        row_sum = np.sum(np.abs(A[i, :])) - abs(A[i, i])
        A[i, i] = 1.0 + row_sum
    return A


def generate_all_benchmark_matrices():
    suite = {}


    n1 = 49
    A1 = logistic_reaction_diffusion_matrix(n1, D=0.1, r=2.0, K=1.0)
    b1 = np.ones(n1)
    suite['logistic_reaction_diffusion'] = (A1, b1)


    a = 2.0
    n2 = 49
    h = 2.0 * a / (n2 + 1)
    x_grid = np.linspace(-a + h, a - h, n2)
    A2 = np.zeros((n2, n2))
    for i in range(n2):
        V_i = feynman_kac_potential(a, x_grid[i])
        A2[i, i] = 1.0 / (h * h) + V_i
        if i > 0:
            A2[i, i - 1] = -0.5 / (h * h)
        if i < n2 - 1:
            A2[i, i + 1] = -0.5 / (h * h)

    b2 = np.zeros(n2)
    b2[0] += 0.5 / (h * h)
    b2[-1] += 0.5 / (h * h)
    suite['feynman_kac_fd'] = (A2, b2)


    J_avg = chirikov_map_jacobian_ensemble(n_ensemble=200, k=0.55, n_steps=30)
    n3 = 40
    A3 = np.zeros((n3, n3))
    for block in range(n3 // 2):
        base = 2 * block
        A3[base:base + 2, base:base + 2] = J_avg

    rng = np.random.default_rng(171)
    for i in range(n3):
        for j in range(n3):
            if i != j and abs(A3[i, j]) < 1e-15:
                A3[i, j] = 0.05 * rng.random()
    for i in range(n3):
        row_sum = np.sum(np.abs(A3[i, :])) - abs(A3[i, i])
        A3[i, i] = 2.0 + row_sum
    A3 = 0.5 * (A3 + A3.T)
    b3 = np.sin(np.linspace(0, 2 * math.pi, n3))
    suite['chirikov_tiled'] = (A3, b3)


    A4 = menger_sponge_hierarchical_matrix(level=4, epsilon=0.1)
    n4 = A4.shape[0]
    b4 = np.ones(n4)
    suite['menger_hierarchical'] = (A4, b4)


    A5 = candy_circulant_matrix(n=48, block_size=6, value=0.5)
    b5 = np.arange(48, dtype=float)
    suite['candy_circulant'] = (A5, b5)

    return suite
