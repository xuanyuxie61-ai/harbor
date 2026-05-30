
import numpy as np
from flamelet_core import scalar_dissipation_rate, thermal_diffusivity_ref


def solve_fd_scalar_dissipation(n, Z_nodes, chi_st, C_chi=2.0, omega_turb=100.0,
                                k_turb=10.0, tol=1.0e-10, max_iter=50):
    if n < 3:
        raise ValueError("节点数 n 必须 >= 3")
    if not np.all(np.diff(Z_nodes) > 0):
        raise ValueError("Z_nodes 必须严格单调递增")

    D_eff = thermal_diffusivity_ref()


    chi = scalar_dissipation_rate(Z_nodes, chi_st)


    chi_ox = float(chi[0])
    chi_fuel = float(chi[-1])

    for iteration in range(max_iter):
        chi_old = chi.copy()
        A = np.zeros((n, n))
        rhs = np.zeros(n)


        A[0, 0] = 1.0
        rhs[0] = chi_ox


        for i in range(1, n - 1):
            xm = Z_nodes[i]
            dxl = Z_nodes[i] - Z_nodes[i - 1]
            dxr = Z_nodes[i + 1] - Z_nodes[i]
            dx_total = Z_nodes[i + 1] - Z_nodes[i - 1]


            Dm = D_eff * (1.0 + 0.1 * chi_old[i] / max(chi_st, 1.0e-6))
            Dm = max(Dm, 1.0e-12)


            cm = C_chi * omega_turb / max(k_turb, 1.0e-6)


            fm = 0.5 * chi_st * np.exp(-((xm - 0.5) / 0.3) ** 2)


            coeff_l = -2.0 * Dm / (dxl * dx_total)
            coeff_r = -2.0 * Dm / (dxr * dx_total)
            coeff_c = 2.0 * Dm / (dxl * dxr) + cm

            A[i, i - 1] = coeff_l
            A[i, i] = coeff_c
            A[i, i + 1] = coeff_r
            rhs[i] = fm


        A[n - 1, n - 1] = 1.0
        rhs[n - 1] = chi_fuel


        chi = np.linalg.solve(A, rhs)


        chi = np.maximum(chi, 0.0)

        max_change = np.max(np.abs(chi - chi_old))
        if max_change < tol:
            return chi, iteration + 1

    return chi, max_iter
