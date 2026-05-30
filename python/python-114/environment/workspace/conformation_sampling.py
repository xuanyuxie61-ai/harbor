
import numpy as np


def generate_helix_conformations(n_points=30, radius=1.0, pitch=1.0):
    t = np.linspace(0, 4 * np.pi, n_points)
    X = np.zeros((n_points, 3))
    X[:, 0] = radius * np.cos(t)
    X[:, 1] = radius * np.sin(t)
    X[:, 2] = pitch * t / (2 * np.pi)
    return X


def generate_circle_conformations(n_points=20, radius=1.0, center=None):
    if center is None:
        center = np.zeros(2)
    theta = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    X = np.zeros((n_points, 2))
    X[:, 0] = center[0] + radius * np.cos(theta)
    X[:, 1] = center[1] + radius * np.sin(theta)
    return X


def simplex_vertices(n_dim):
    V = np.zeros((n_dim, n_dim + 1))
    for j in range(n_dim):
        V[j, j] = 1.0
    a = (1.0 - np.sqrt(1.0 + n_dim)) / n_dim
    V[:, n_dim] = a

    centroid = np.mean(V, axis=1, keepdims=True)
    V -= centroid

    s = np.linalg.norm(V[:, 0])
    if s > 0:
        V /= s
    return V


def sample_simplex_mixture(n_dim, n_points, std=0.2):
    V = simplex_vertices(n_dim)
    X = np.zeros((n_points, n_dim))
    labels = np.zeros(n_points, dtype=int)
    n_vertices = n_dim + 1
    for p in range(n_points):
        k = np.random.randint(0, n_vertices)
        X[p, :] = V[:, k] + std * np.random.randn(n_dim)
        labels[p] = k
    return X, labels


def sammon_mapping(X, n_components=2, max_iter=300, tol=1e-5, alpha=0.3):
    n_samples = X.shape[0]

    D_star = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            d = np.linalg.norm(X[i] - X[j])
            if d < 1e-10:
                d = 1e-10
            D_star[i, j] = d
            D_star[j, i] = d

    c = np.sum(D_star) / 2.0
    if c == 0:
        c = 1.0


    Y = np.random.randn(n_samples, n_components) * 0.01

    stds = np.std(X, axis=0)
    if len(stds) >= n_components:
        Y = X[:, :n_components] / (stds[:n_components] + 1e-10) * 0.1

    stress_history = []
    for it in range(max_iter):

        D = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                d = np.linalg.norm(Y[i] - Y[j])
                if d < 1e-10:
                    d = 1e-10
                D[i, j] = d
                D[j, i] = d


        stress = 0.0
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                stress += ((D_star[i, j] - D[i, j]) ** 2) / D_star[i, j]
        stress /= c
        stress_history.append(stress)

        if it > 0 and abs(stress_history[-1] - stress_history[-2]) < tol:
            break


        for i in range(n_samples):
            delta = np.zeros(n_components)
            for j in range(n_samples):
                if i == j:
                    continue
                diff = Y[i] - Y[j]
                denom = D[i, j] * D_star[i, j]
                if denom < 1e-14:
                    continue
                factor = (D_star[i, j] - D[i, j]) / denom
                delta += factor * diff
            Y[i] -= alpha * delta / c

    return Y, stress_history


def henon_orbit_trajectory(x0, y0, n_steps, alpha_angle=0.4):
    c = np.cos(alpha_angle)
    s = np.sin(alpha_angle)
    traj = np.zeros((n_steps, 2))
    x, y = x0, y0
    for k in range(n_steps):
        if abs(x) < 1.0 and abs(y) < 1.0:
            x_new = x * c - (y - x * x) * s
            y_new = x * s + (y - x * x) * c
            x, y = x_new, y_new
        traj[k] = [x, y]
    return traj


def lyapunov_exponent_henon(x0, y0, n_steps, alpha_angle=0.4, delta0=1e-8):
    c = np.cos(alpha_angle)
    s = np.sin(alpha_angle)


    x, y = x0, y0
    x_p, y_p = x0 + delta0, y0

    lyap_sum = 0.0
    count = 0
    for _ in range(n_steps):
        if abs(x) < 1.0 and abs(y) < 1.0:
            x_new = x * c - (y - x * x) * s
            y_new = x * s + (y - x * x) * c
            x, y = x_new, y_new

            x_p_new = x_p * c - (y_p - x_p * x_p) * s
            y_p_new = x_p * s + (y_p - x_p * x_p) * c
            x_p, y_p = x_p_new, y_p_new

            d = np.sqrt((x - x_p) ** 2 + (y - y_p) ** 2)
            if d > 0 and delta0 > 0:
                lyap_sum += np.log(d / delta0)
                count += 1

                x_p = x + delta0 * (x_p - x) / d
                y_p = y + delta0 * (y_p - y) / d

    if count == 0:
        return 0.0
    return lyap_sum / count
