
import numpy as np


def squircle_trajectory(s=4.0, t0=0.0, y0=None, tstop=20.0, n_points=1000):
    if s <= 1.0:
        raise ValueError("s must be > 1")
    if y0 is None:
        y0 = np.array([0.0, 1.0])
    else:
        y0 = np.array(y0, dtype=float)
    if y0.shape != (2,):
        raise ValueError("y0 must have shape (2,)")

    def rhs(t, y):
        u, v = y
        dudt = np.sign(v) * (np.abs(v) ** (s - 1.0))
        dvdt = -np.sign(u) * (np.abs(u) ** (s - 1.0))
        return np.array([dudt, dvdt])

    t = np.linspace(t0, tstop, n_points)
    dt = t[1] - t[0]
    y = np.zeros((n_points, 2))
    y[0] = y0

    for i in range(n_points - 1):
        k1 = rhs(t[i], y[i])
        k2 = rhs(t[i] + dt / 2, y[i] + dt * k1 / 2)
        k3 = rhs(t[i] + dt / 2, y[i] + dt * k2 / 2)
        k4 = rhs(t[i] + dt, y[i] + dt * k3)
        y[i + 1] = y[i] + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)

    return t, y


def squircle_activation_basis(x, s=4.0, n_modes=8):
    if n_modes < 1:
        raise ValueError("n_modes must be >= 1")
    tstop = 2.0 * np.pi
    t_base, y_base = squircle_trajectory(s=s, tstop=tstop, n_points=2048)
    u_base = y_base[:, 0]

    L = x.max() - x.min()
    if L <= 0:
        L = 1.0

    basis = np.zeros((len(x), n_modes))
    for k in range(n_modes):
        phase = k * np.pi / n_modes

        t_mapped = ((x - x.min()) / L * tstop + phase) % tstop
        indices = (t_mapped / tstop * (len(t_base) - 1)).astype(int)
        indices = np.clip(indices, 0, len(t_base) - 1)
        basis[:, k] = u_base[indices]

    return basis


def cross_chaos_ifs(n_points=5000, seed=42):
    if n_points < 1:
        raise ValueError("n_points must be >= 1")
    rng = np.random.default_rng(seed)

    A = np.array([[1.0 / 3.0, 0.0],
                  [0.0, 1.0 / 3.0]])
    b = np.array([
        [1.0 / 3.0, 0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0 / 3.0],
        [0.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 2.0 / 3.0]
    ])


    x = rng.random(2)
    burn_in = min(100, n_points)
    total = burn_in + n_points

    points = np.zeros((n_points, 2))
    count = 0
    for _ in range(total):
        x = A @ x
        j = rng.integers(0, 5)
        x = x + b[:, j]
        if _ >= burn_in:
            points[count] = x
            count += 1

    return points


def cellular_automaton_rule30(cell_num=256, step_num=128, seed_center=True):
    if cell_num < 3:
        raise ValueError("cell_num must be >= 3")
    if step_num < 1:
        raise ValueError("step_num must be >= 1")

    states = np.zeros((step_num, cell_num), dtype=int)
    if seed_center:
        mid = cell_num // 2
        states[0, mid] = 1
    else:
        rng = np.random.default_rng(42)
        states[0] = rng.integers(0, 2, size=cell_num)

    for i in range(1, step_num):
        for j in range(1, cell_num - 1):
            left = states[i - 1, j - 1]
            center = states[i - 1, j]
            right = states[i - 1, j + 1]

            pattern = (left << 2) | (center << 1) | right
            if pattern in [1, 2, 3, 4]:
                states[i, j] = 1
            else:
                states[i, j] = 0

    return states


def generate_chaotic_initial_condition(L_domain, nx, chaos_type='squircle',
                                       amplitude=1.0):
    x = np.linspace(0.0, L_domain, nx, endpoint=False)

    if chaos_type == 'squircle':
        basis = squircle_activation_basis(x, s=4.0, n_modes=4)
        coeffs = np.array([1.0, 0.5, -0.3, 0.2])
        u0 = basis @ coeffs
    elif chaos_type == 'cross_ifs':
        points = cross_chaos_ifs(n_points=nx * 10, seed=42)

        angles = np.arctan2(points[:, 1] - 0.5, points[:, 0] - 0.5)
        hist, bin_edges = np.histogram(angles, bins=nx, range=(-np.pi, np.pi))
        u0 = hist.astype(float)
        u0 = u0 / (np.max(np.abs(u0)) + 1e-12)
    elif chaos_type == 'ca_rule30':
        states = cellular_automaton_rule30(cell_num=nx, step_num=1)
        u0 = states[0].astype(float)
        u0 = u0 * 2.0 - 1.0
    else:
        raise ValueError(f"Unknown chaos_type: {chaos_type}")

    u0 = u0 * amplitude
    return u0, x
