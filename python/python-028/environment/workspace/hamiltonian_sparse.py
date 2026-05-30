
import numpy as np
from math import sqrt


HBARC = 197.3269804
M_NUCLEON = 939.0


def build_radial_hamiltonian_st(r_grid, potential_values, l, mass=M_NUCLEON):
    N = len(r_grid)
    if N < 3:
        raise ValueError("格点数至少为 3")
    dr = r_grid[1] - r_grid[0]
    if dr <= 0:
        raise ValueError("格点必须严格递增")

    kinetic_prefactor = HBARC ** 2 / (2.0 * mass * dr ** 2)

    ist = []
    jst = []
    Ast = []

    for i in range(N):
        r = r_grid[i]



        V_cent = 0.0
        H_ii = 2.0 * kinetic_prefactor + potential_values[i] + V_cent
        ist.append(i)
        jst.append(i)
        Ast.append(H_ii)


        if i < N - 1:
            H_off = -kinetic_prefactor
            ist.append(i)
            jst.append(i + 1)
            Ast.append(H_off)



            ist.append(i + 1)
            jst.append(i)
            Ast.append(H_off)

    return len(ist), np.array(ist), np.array(jst), np.array(Ast)


def st_to_ge(nst, ist, jst, Ast):
    if nst == 0:
        return np.zeros((0, 0))
    m = max(ist) + 1
    n = max(jst) + 1
    A = np.zeros((m, n))
    for k in range(nst):
        A[ist[k], jst[k]] = Ast[k]
    return A


def ge_to_crs(Age):
    m, n = Age.shape
    row = [0]
    col = []
    val = []
    nz = 0
    for i in range(m):
        for j in range(n):
            if abs(Age[i, j]) > 1e-15:
                col.append(j)
                val.append(Age[i, j])
                nz += 1
        row.append(nz)
    return m, n, nz, np.array(row), np.array(col), np.array(val)


def st_to_crs(nst, ist, jst, Ast):
    Age = st_to_ge(nst, ist, jst, Ast)
    return ge_to_crs(Age)


def crs_matvec(row, col, val, x):
    n = len(row) - 1
    y = np.zeros(n)
    for i in range(n):
        for idx in range(row[i], row[i + 1]):
            y[i] += val[idx] * x[col[idx]]
    return y


def shell_model_hamiltonian_sparse(n_particles, n_orbitals, interaction_strength,
                                   single_energies, max_particles_per_orbital=2):




    dim = n_orbitals * max_particles_per_orbital
    H = np.zeros((dim, dim))


    for i in range(n_orbitals):
        for spin in range(max_particles_per_orbital):
            idx = i * max_particles_per_orbital + spin
            H[idx, idx] = single_energies[i]



    G = interaction_strength
    for i in range(n_orbitals):
        idx_up = i * max_particles_per_orbital
        idx_down = i * max_particles_per_orbital + 1
        H[idx_up, idx_down] -= G
        H[idx_down, idx_up] -= G


    for i in range(n_orbitals):
        for j in range(i + 1, n_orbitals):
            coupling = -G * 0.5 / abs(single_energies[i] - single_energies[j] + 1.0)
            for s1 in range(max_particles_per_orbital):
                for s2 in range(max_particles_per_orbital):
                    idx_i = i * max_particles_per_orbital + s1
                    idx_j = j * max_particles_per_orbital + s2
                    H[idx_i, idx_j] += coupling
                    H[idx_j, idx_i] += coupling


    m, n, nz, row, col, val = ge_to_crs(H)
    return row, col, val, dim


def lanczos_iteration(row, col, val, dim, n_iter, v0=None):
    if v0 is None:
        v0 = np.random.randn(dim)
        v0 = v0 / np.linalg.norm(v0)

    alpha = np.zeros(n_iter)
    beta = np.zeros(n_iter + 1)

    v_prev = np.zeros(dim)
    v_curr = v0

    for k in range(n_iter):
        w = crs_matvec(row, col, val, v_curr)
        w = w - beta[k] * v_prev
        alpha[k] = np.dot(v_curr, w)
        w = w - alpha[k] * v_curr
        beta[k + 1] = np.linalg.norm(w)
        if beta[k + 1] < 1e-14:
            break
        v_prev = v_curr
        v_curr = w / beta[k + 1]


    T = np.diag(alpha) + np.diag(beta[1:n_iter], k=1) + np.diag(beta[1:n_iter], k=-1)
    eigenvalues = np.linalg.eigvalsh(T)
    return sorted(eigenvalues)
