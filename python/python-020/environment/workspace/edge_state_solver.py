# -*- coding: utf-8 -*-
import numpy as np
from utils import cyclotron_frequency





def euler_integrate(dydt, tspan, y0, n_steps):
    y0 = np.atleast_1d(y0)
    t0, t_stop = tspan
    h = (t_stop - t0) / n_steps

    t = np.zeros(n_steps + 1)
    y = np.zeros((n_steps + 1, len(y0)))

    t[0] = t0
    y[0, :] = y0

    for i in range(n_steps):
        t[i + 1] = t[i] + h
        y[i + 1, :] = y[i, :] + h * np.asarray(dydt(t[i], y[i, :]))

    return t, y






def shooting_method_bvp(ode_rhs, a, b, ya, yb_target, alpha_guess1=0.0,
                        alpha_guess2=-1.0, max_iter=20, tol=1e-6, n_steps=1000):
    def F(alpha):
        y0 = np.array([ya, alpha])
        r_grid, y_sol = euler_integrate(ode_rhs, (a, b), y0, n_steps)
        yb_computed = y_sol[-1, 0]
        return yb_computed - yb_target

    alpha = alpha_guess1
    f_alpha = F(alpha)


    beta = None
    f_beta = None

    for iteration in range(max_iter):
        if iteration == 0:
            pass
        elif iteration == 1:
            beta = alpha
            f_beta = f_alpha
            alpha = alpha_guess2
            f_alpha = F(alpha)
        else:
            if abs(f_beta - f_alpha) < 1e-14:

                alpha_new = 0.5 * (alpha + beta)
            else:
                gamma = alpha - f_alpha * (beta - alpha) / (f_beta - f_alpha)
                alpha_new = gamma
            beta = alpha
            f_beta = f_alpha
            alpha = alpha_new
            f_alpha = F(alpha)

        if abs(f_alpha) < tol:
            y0 = np.array([ya, alpha])
            r_grid, y_sol = euler_integrate(ode_rhs, (a, b), y0, n_steps)
            return r_grid, y_sol[:, 0], True, iteration + 1


    y0 = np.array([ya, alpha])
    r_grid, y_sol = euler_integrate(ode_rhs, (a, b), y0, n_steps)
    return r_grid, y_sol[:, 0], False, max_iter






def edge_state_radial_ode(B, m_star, angular_m, E):
    omega_c = cyclotron_frequency(B, m_star)

    def ode_rhs(r, y):
        u, up = y
        if r < 1e-10:

            r = 1e-10



        k_sq = 2.0 * m_star * E
        V_r = (m_star * omega_c * r / (2.0)) ** 2 + (angular_m / r) ** 2
        upp = -(1.0 / r) * up - (k_sq - V_r) * u
        return np.array([up, upp])

    return ode_rhs






def chiral_luttinger_dispersion(k, v_F, g_factor=1.0):
    if g_factor <= 0:
        raise ValueError("Luttinger参数 g 必须为正")
    return k * v_F / g_factor


def edge_state_density_of_states(omega, v_F, L_edge, T=0.01):
    if v_F <= 0:
        raise ValueError("费米速度 v_F 必须为正")
    sigma = max(T, 1e-6)
    omega = np.asarray(omega, dtype=float)
    prefactor = L_edge / (2.0 * np.pi * v_F)

    dos = prefactor * 0.5 * (1.0 + np.tanh(omega / sigma))
    return dos






def blowup_ode_stabilized(t, y, blowup_threshold=1e6, stabilization=0.1):
    y = float(y)
    if abs(y) > blowup_threshold:

        return stabilization * blowup_threshold * np.sign(y)
    dydt = y ** 2 / (1.0 + (y / blowup_threshold) ** 2)
    return dydt





def test_edge_state_solver():
    print("=" * 60)
    print("[edge_state_solver.py] 边缘态BVP求解测试")
    print("=" * 60)


    print("\n1. Euler法测试 (dy/dt = -y, y(0)=1, 精确解 y=exp(-t)):")
    def decay_ode(t, y):
        return np.array([-y[0]])
    t, y = euler_integrate(decay_ode, (0.0, 2.0), np.array([1.0]), n_steps=200)
    exact = np.exp(-t)
    err = np.max(np.abs(y[:, 0] - exact))
    print(f"   最大误差: {err:.6e}")


    print("\n2. 打靶法测试 (u'' + u = 0, u(0)=0, u(π)=0, 精确解 u=sin(r)):")
    def harmonic_ode(r, y):
        u, up = y
        return np.array([up, -u])
    r_grid, u_sol, conv, nit = shooting_method_bvp(
        harmonic_ode, a=0.0, b=np.pi, ya=0.0, yb_target=0.0,
        alpha_guess1=0.5, alpha_guess2=1.5, max_iter=20, tol=1e-5, n_steps=500
    )
    u_exact = np.sin(r_grid)
    err = np.max(np.abs(u_sol - u_exact))
    print(f"   收敛: {conv}, 迭代次数: {nit}")
    print(f"   最大误差: {err:.6e}")


    print("\n3. 边缘态径向方程测试:")
    B = 10.0
    m_star = 1.0
    angular_m = 1
    E = 5.0
    ode_rhs = edge_state_radial_ode(B, m_star, angular_m, E)
    r_grid, u_sol, conv, nit = shooting_method_bvp(
        ode_rhs, a=1e-3, b=3.0, ya=0.0, yb_target=0.0,
        alpha_guess1=0.1, alpha_guess2=1.0, max_iter=15, tol=1e-4, n_steps=800
    )
    print(f"   收敛: {conv}, 迭代次数: {nit}")
    print(f"   解的范数: {np.linalg.norm(u_sol):.4f}")


    print("\n4. 手性Luttinger液体色散测试:")
    k = np.linspace(0.0, 5.0, 50)
    v_F = 1.0
    for g in [1.0, 0.5, 0.3]:
        eps = chiral_luttinger_dispersion(k[1:], v_F, g)
        print(f"   g={g}: ε(k=5) = {eps[-1]:.4f}")


    print("\n5. 爆炸ODE稳定化测试:")
    t, y = euler_integrate(
        lambda t, y: np.array([blowup_ode_stabilized(t, y[0])]),
        (0.0, 2.0), np.array([1.0]), n_steps=500
    )
    print(f"   y(0)={y[0,0]:.4f}, y(2)={y[-1,0]:.4f} (无稳定化时会爆炸)")

    print("\n[edge_state_solver.py] 测试完成。\n")


if __name__ == "__main__":
    test_edge_state_solver()
