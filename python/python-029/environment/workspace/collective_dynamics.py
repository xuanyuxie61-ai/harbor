
import numpy as np
from scipy.integrate import solve_ivp


def collective_mass_parameter(A, lam):
    m_nucleon = 931.5
    r0 = 1.2
    R0 = r0 * (A ** (1.0 / 3.0))
    B_lam = (3.0 / (4.0 * np.pi)) * A * m_nucleon * (R0 ** 2) / lam
    return B_lam


def restoring_force_parameter(A, lam):
    a_s = 17.8
    a_c = 0.711

    Z = A / 2.0
    C_lam = ((lam - 1.0) * (lam + 2.0) * a_s * (A ** (2.0 / 3.0))
             - (2.0 * lam - 1.0) * a_c * (Z ** 2) / (A ** (1.0 / 3.0)))
    return C_lam


def resonance_energy(A, lam):
    if lam == 1:
        return 31.2 * (A ** (-1.0 / 3.0)) + 20.6 * (A ** (-1.0 / 6.0))
    elif lam == 2:
        return 63.0 * (A ** (-1.0 / 3.0))
    elif lam == 3:
        return 110.0 * (A ** (-1.0 / 3.0))
    else:
        return 60.0 * (A ** (-1.0 / 3.0))


def damping_width(A, lam, E_exc, T_nucleus=0.0):
    if lam == 1:
        Gamma_0 = 4.0 + 0.05 * A
    elif lam == 2:
        Gamma_0 = 2.5 + 0.03 * A
    else:
        Gamma_0 = 1.5 + 0.02 * A


    thermal_broadening = 4.0 * (np.pi * T_nucleus) ** 2 / (E_exc + 1e-6)
    return Gamma_0 + 0.1 * E_exc + thermal_broadening


def forced_damped_oscillator(t_span, Q0, dQ0, B, C, Gamma, F_func, n_steps=1000):
    omega0_sq = C / B
    gamma_damp = Gamma / B
    F_scale = 1.0 / B

    def ode_system(t, y):
        Q, V = y
        dQdt = V
        dVdt = -gamma_damp * V - omega0_sq * Q + F_scale * F_func(t)
        return [dQdt, dVdt]

    t_eval = np.linspace(t_span[0], t_span[1], n_steps)
    sol = solve_ivp(ode_system, t_span, [Q0, dQ0], t_eval=t_eval, method='RK45', rtol=1e-9, atol=1e-12)
    return sol.t, sol.y[0, :], sol.y[1, :]


def giant_resonance_cross_section(E_gamma, A, lam=1):
    E_gamma = np.asarray(E_gamma, dtype=float)
    E_R = resonance_energy(A, lam)
    Gamma_R = damping_width(A, lam, E_R)


    if lam == 1:
        sigma_max = 60.0 * A / (2.0 * np.pi)
    elif lam == 2:
        sigma_max = 25.0 * A / (2.0 * np.pi)
    else:
        sigma_max = 10.0 * A / (2.0 * np.pi)

    sigma = sigma_max * ((Gamma_R / 2.0) ** 2) / ((E_gamma - E_R) ** 2 + (Gamma_R / 2.0) ** 2)
    return sigma


def energy_weighted_sum_rule(A, lam):
    hbar2_over_2m = 20.7
    r0 = 1.2
    R0 = r0 * (A ** (1.0 / 3.0))
    r2lam2 = (R0 ** (2.0 * lam - 2.0)) * (3.0 / (2.0 * lam + 1.0))
    S = (2.0 * lam + 1.0) * hbar2_over_2m * A * r2lam2 / (4.0 * np.pi)
    return S


def strength_function_integral(E_gamma, sigma, method='trapezoid'):
    if method == 'trapezoid':
        return np.trapezoid(sigma, E_gamma)
    else:
        from scipy.integrate import simpson
        return simpson(sigma, x=E_gamma)


def time_dependent_multipole_field(t, omega, E0, lam):
    tau = 50.0
    return E0 * np.cos(omega * t) * np.exp(-t / tau) * np.sqrt(2.0 * lam + 1.0)


if __name__ == "__main__":

    A = 56
    B2 = collective_mass_parameter(A, 2)
    C2 = restoring_force_parameter(A, 2)
    E2 = resonance_energy(A, 2)
    Gamma2 = damping_width(A, 2, E2)
    print(f"A={A}, λ=2: B={B2:.2f}, C={C2:.2f}, E_R={E2:.2f} MeV, Γ={Gamma2:.2f} MeV")


    omega = E2 / 197.3
    F = lambda t: time_dependent_multipole_field(t, omega, 1.0, 2)
    t, Q, dQ = forced_damped_oscillator((0, 200), 0.0, 0.0, B2, C2, Gamma2, F, n_steps=500)
    print(f"集体坐标最终振幅: {Q[-1]:.6f} fm")


    E_range = np.linspace(5, 25, 200)
    sigma = giant_resonance_cross_section(E_range, A, lam=2)
    S_int = strength_function_integral(E_range, sigma)
    print(f"GQR 强度积分: {S_int:.2f} mb·MeV")
    print(f"EWSR 上限: {energy_weighted_sum_rule(A, 2):.2f} mb·MeV (注意单位换算)")
