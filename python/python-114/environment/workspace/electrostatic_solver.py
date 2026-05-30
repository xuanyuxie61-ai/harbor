
import numpy as np


def debye_huckel_parameter(ionic_strength_mol_L: float,
                           temperature_K: float = 298.15,
                           dielectric_water: float = 78.5) -> float:
    if ionic_strength_mol_L < 0:
        raise ValueError("ionic strength must be non-negative")

    kappa = 0.329 * np.sqrt(ionic_strength_mol_L)
    return kappa


def assemble_pb_jacobian(n: int, h: float, kappa: float,
                         boundary_type: str = 'neumann') -> tuple:
    numnodes = n * n
    A = np.zeros((numnodes, numnodes), dtype=float)
    b = np.zeros(numnodes, dtype=float)

    h2 = h * h
    k2h2 = kappa * kappa * h2


    rho = np.zeros((n, n))
    center = n // 2
    for i in range(n):
        for j in range(n):

            dx = (i - center) * h
            dy = (j - center) * h
            r2 = dx * dx + dy * dy
            rho[i, j] = -1.0 * np.exp(-r2 / (0.5 ** 2))



    for i in range(n):
        k = i
        if boundary_type == 'neumann':

            A[k, k] = -4.0 - k2h2
            if i > 0:
                A[k, k - 1] = 1.0
            if i < n - 1:
                A[k, k + 1] = 1.0
            A[k, k + n] = 2.0
        else:
            A[k, k] = 1.0
            b[k] = 0.0
            continue
        b[k] = -h2 * rho[i, 0]


    for j in range(1, n - 1):
        for i in range(n):
            k = j * n + i
            A[k, k] = -4.0 - k2h2
            if i > 0:
                A[k, k - 1] = 1.0
            if i < n - 1:
                A[k, k + 1] = 1.0
            A[k, k - n] = 1.0
            A[k, k + n] = 1.0
            b[k] = -h2 * rho[i, j]


    for i in range(n):
        k = (n - 1) * n + i
        if boundary_type == 'neumann':
            A[k, k] = -4.0 - k2h2
            if i > 0:
                A[k, k - 1] = 1.0
            if i < n - 1:
                A[k, k + 1] = 1.0
            A[k, k - n] = 2.0
        else:
            A[k, k] = 1.0
            b[k] = 0.0
            continue
        b[k] = -h2 * rho[i, n - 1]

    return A, b


def r8gb_fa_python(n: int, ml: int, mu: int, a: np.ndarray) -> tuple:
    if n <= 0:
        raise ValueError("n must be positive")
    if ml < 0 or mu < 0:
        raise ValueError("bandwidths must be non-negative")
    if ml >= n or mu >= n:
        raise ValueError("bandwidths must be less than n")

    alu = a.copy()
    m = ml + mu + 1
    info = 0
    pivot = np.zeros(n, dtype=int)


    j0 = mu + 2
    j1 = min(n, m) - 1
    for jz in range(j0, j1 + 1):
        i0 = m + 1 - jz
        if i0 <= ml:
            alu[i0:ml + 1, jz - 1] = 0.0

    jz = j1
    ju = 0

    for k in range(1, n):
        jz += 1
        if jz <= n:
            alu[0:ml, jz - 1] = 0.0

        lm = min(ml, n - k)
        l = m - 1


        for j in range(m, m + lm):
            if abs(alu[l, k - 1]) < abs(alu[j, k - 1]):
                l = j

        pivot[k - 1] = l + k - m + 1

        if alu[l, k - 1] == 0.0:
            info = k
            return alu, pivot, info


        if l != m - 1:
            temp = alu[l, k - 1]
            alu[l, k - 1] = alu[m - 1, k - 1]
            alu[m - 1, k - 1] = temp


        if lm > 0:
            alu[m:m + lm, k - 1] = -alu[m:m + lm, k - 1] / alu[m - 1, k - 1]


        ju = max(ju, mu + pivot[k - 1])
        ju = min(ju, n)

        for j in range(k + 1, ju + 1):
            l -= 1
            mm = m - 1 - (j - k)
            if l != mm and 0 <= l < alu.shape[0] and 0 <= mm < alu.shape[0]:
                temp = alu[l, j - 1]
                alu[l, j - 1] = alu[mm, j - 1]
                alu[mm, j - 1] = temp

            if lm > 0 and 0 <= mm < alu.shape[0]:
                alu[mm + 1:mm + 1 + lm, j - 1] += alu[mm, j - 1] * alu[m:m + lm, k - 1]

    pivot[n - 1] = n
    if alu[m - 1, n - 1] == 0.0:
        info = n

    return alu, pivot, info


def r8gb_sl_python(n: int, ml: int, mu: int, alu: np.ndarray,
                   pivot: np.ndarray, b: np.ndarray) -> np.ndarray:
    x = b.copy()
    m = ml + mu + 1


    for k in range(1, n + 1):
        lm = min(k - 1, ml)
        la = m - lm
        lb = k - lm

        for i in range(0, lm):
            x[k - 1] += alu[la + i - 1, k - 1] * x[lb + i - 1]

        l = pivot[k - 1]

        if l != k:
            temp = x[l - 1]
            x[l - 1] = x[k - 1]
            x[k - 1] = temp


    for k in range(n, 0, -1):
        x[k - 1] /= alu[m - 1, k - 1]
        lm = min(k - 1, ml)
        la = m - lm
        lb = k - lm

        for i in range(0, lm):
            x[lb + i - 1] -= alu[la + i - 1, k - 1] * x[k - 1]

    return x


def solve_pb_banded(n: int, h: float, kappa: float) -> np.ndarray:
    A_dense, b = assemble_pb_jacobian(n, h, kappa)
    numnodes = n * n






    try:
        phi = np.linalg.solve(A_dense, b)
    except np.linalg.LinAlgError:
        phi = np.linalg.lstsq(A_dense, b, rcond=None)[0]

    return phi.reshape((n, n))


def compute_electrostatic_binding_energy(phi: np.ndarray,
                                         binding_site_coords: np.ndarray,
                                         grid_origin: np.ndarray,
                                         h: float,
                                         charge: float = 1.0) -> float:
    energy = 0.0
    for coord in binding_site_coords:
        idx = ((coord - grid_origin) / h).astype(int)
        idx = np.clip(idx, 0, np.array(phi.shape) - 1)
        if phi.ndim == 2:
            i, j = idx[0], idx[1]
            if 0 <= i < phi.shape[0] and 0 <= j < phi.shape[1]:
                energy += charge * phi[i, j]
        elif phi.ndim == 3:
            i, j, k = idx[0], idx[1], idx[2]
            if 0 <= i < phi.shape[0] and 0 <= j < phi.shape[1] and 0 <= k < phi.shape[2]:
                energy += charge * phi[i, j, k]
    return energy
