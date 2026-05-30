# -*- coding: utf-8 -*-
import numpy as np
from utils import safe_exp





def fd1d_wave_solve(x_num, x1, x2, t_num, t1, t2, c, u_x1_fn, u_x2_fn,
                    u_t1_fn, ut_t1_fn):
    if x_num < 2:
        raise ValueError("x_num 必须 ≥ 2")
    if t_num < 1:
        raise ValueError("t_num 必须 ≥ 1")

    dt = (t2 - t1) / t_num
    dx = (x2 - x1) / x_num
    alpha = c * dt / dx

    if abs(alpha) > 1.0:

        dt_new = dx / (abs(c) + 1e-10)
        t_num_new = int(np.ceil((t2 - t1) / dt_new))
        dt = (t2 - t1) / t_num_new
        alpha = c * dt / dx

        u = np.zeros((t_num_new + 1, x_num + 1))
        t_num = t_num_new
    else:
        u = np.zeros((t_num + 1, x_num + 1))

    x = np.linspace(x1, x2, x_num + 1)


    times = np.linspace(t1, t2, t_num + 1)
    for n in range(t_num + 1):
        u[n, 0] = u_x1_fn(times[n])
        u[n, x_num] = u_x2_fn(times[n])


    u[0, :] = u_t1_fn(x)
    ut0 = ut_t1_fn(x)


    for j in range(1, x_num):
        u[1, j] = (0.5 * alpha ** 2) * u[0, j + 1] \
                  + (1.0 - alpha ** 2) * u[0, j] \
                  + (0.5 * alpha ** 2) * u[0, j - 1] \
                  + dt * ut0[j]


    for n in range(1, t_num):
        for j in range(1, x_num):
            u[n + 1, j] = (alpha ** 2) * u[n, j + 1] \
                          + 2.0 * (1.0 - alpha ** 2) * u[n, j] \
                          + (alpha ** 2) * u[n, j - 1] \
                          - u[n - 1, j]

    return u






def fisher_kpp_exact_solution(t, x, a=1.0, c=2.0, k=-1.0 / np.sqrt(2.0)):
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    z = x - c * t
    denom = 1.0 + a * np.exp(k * z)
    u = 1.0 / (denom ** 2)
    return u


def fisher_kpp_derivatives(t, x, a=1.0, c=2.0, k=-1.0 / np.sqrt(2.0)):
    z = x - c * t
    exp_kz = np.exp(k * z)
    denom1 = 1.0 + a * exp_kz
    denom3 = denom1 ** 3
    denom4 = denom1 ** 4

    ut = 2.0 * c * k * a * exp_kz / denom3
    ux = -2.0 * k * a * exp_kz / denom3
    uxx = 6.0 * (k ** 2) * (a ** 2) * np.exp(2.0 * k * z) / denom4 \
          - 2.0 * (k ** 2) * a * exp_kz / denom3
    return ut, ux, uxx


def fisher_kpp_fd_solve(x_num, x1, x2, t_num, t1, t2, D=1.0, r=1.0, K=1.0,
                         u0_fn=None):
    if x_num < 2 or t_num < 1:
        raise ValueError("网格数不足")

    dx = (x2 - x1) / x_num
    dt = (t2 - t1) / t_num


    lambda_val = D * dt / (dx ** 2)
    if lambda_val > 0.5:

        dt_new = 0.45 * dx ** 2 / (D + 1e-10)
        t_num_new = int(np.ceil((t2 - t1) / dt_new))
        dt = (t2 - t1) / t_num_new
        lambda_val = D * dt / (dx ** 2)
        u = np.zeros((t_num_new + 1, x_num + 1))
        t_num = t_num_new
    else:
        u = np.zeros((t_num + 1, x_num + 1))

    x = np.linspace(x1, x2, x_num + 1)


    if u0_fn is None:
        u[0, :] = np.exp(-x ** 2)
    else:
        u[0, :] = u0_fn(x)


    for n in range(t_num):
        for j in range(1, x_num):
            diffusion = D * (u[n, j + 1] - 2.0 * u[n, j] + u[n, j - 1]) / (dx ** 2)
            reaction = r * u[n, j] * (1.0 - u[n, j] / K)
            u[n + 1, j] = u[n, j] + dt * (diffusion + reaction)

        u[n + 1, 0] = u[n + 1, 1]
        u[n + 1, x_num] = u[n + 1, x_num - 1]

    return u






def density_matrix_evolution_lindblad(
    rho0, H, L_list, t_span, n_steps
):
    rho = np.asarray(rho0, dtype=complex)
    H = np.asarray(H, dtype=complex)
    t0, t_stop = t_span
    dt = (t_stop - t0) / n_steps

    times = np.linspace(t0, t_stop, n_steps + 1)
    rhos = [rho.copy()]

    for _ in range(n_steps):

        commutator = H @ rho - rho @ H
        d_rho = -1j * commutator


        for L in L_list:
            L = np.asarray(L, dtype=complex)
            L_dag = np.conj(L.T)
            d_rho += L @ rho @ L_dag - 0.5 * (L_dag @ L @ rho + rho @ L_dag @ L)


        rho = rho + dt * d_rho


        rho = 0.5 * (rho + np.conj(rho.T))
        trace = np.trace(rho)
        if abs(trace) > 1e-14:
            rho = rho / trace


        eigs, V = np.linalg.eigh(rho)
        eigs = np.maximum(eigs, 0.0)
        eigs = eigs / np.sum(eigs)
        rho = V @ np.diag(eigs) @ np.conj(V.T)

        rhos.append(rho.copy())

    return times, rhos





def test_density_evolution():
    print("=" * 60)
    print("[density_evolution.py] 密度演化测试")
    print("=" * 60)


    print("\n1. 一维波动方程测试 (c=1, 初始条件 sin(πx)):")
    def u_x1(t):
        return 0.0
    def u_x2(t):
        return 0.0
    def u_t1(x):
        return np.sin(np.pi * x)
    def ut_t1(x):
        return np.zeros_like(x)

    u = fd1d_wave_solve(50, 0.0, 1.0, 100, 0.0, 2.0, 1.0, u_x1, u_x2, u_t1, ut_t1)
    print(f"   解的形状: {u.shape}")
    print(f"   t=0 时最大振幅: {np.max(np.abs(u[0,:])):.6f}")
    print(f"   t=2 时最大振幅: {np.max(np.abs(u[-1,:])):.6f}")


    print("\n2. Fisher-KPP精确解测试:")
    t_test, x_test = 0.5, 0.0
    u_exact = fisher_kpp_exact_solution(t_test, x_test)
    ut, ux, uxx = fisher_kpp_derivatives(t_test, x_test)

    lhs = ut
    rhs = uxx + u_exact * (1.0 - u_exact)
    print(f"   u(0.5, 0.0) = {u_exact:.6f}")
    print(f"   u_t = {lhs:.6f}")
    print(f"   u_xx + u(1-u) = {rhs:.6f}")
    print(f"   残差 = {abs(lhs - rhs):.2e}")


    print("\n3. Fisher-KPP有限差分解测试:")
    u_fisher = fisher_kpp_fd_solve(80, -10.0, 10.0, 200, 0.0, 5.0, D=1.0, r=1.0, K=1.0)
    print(f"   解的形状: {u_fisher.shape}")
    print(f"   初始总密度: {np.sum(u_fisher[0,:]):.4f}")
    print(f"   最终总密度: {np.sum(u_fisher[-1,:]):.4f}")
    print(f"   最终最大密度: {np.max(u_fisher[-1,:]):.4f}")


    print("\n4. Lindblad密度矩阵演化测试:")
    H = np.array([[1.0, 0.5], [0.5, -1.0]], dtype=complex)
    rho0 = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=complex)
    L = np.array([[0.0, 0.1], [0.0, 0.0]], dtype=complex)
    times, rhos = density_matrix_evolution_lindblad(rho0, H, [L], (0.0, 2.0), 100)
    print(f"   时间步数: {len(times)}")
    print(f"   初始 ρ_11: {rhos[0][0,0].real:.4f}")
    print(f"   最终 ρ_11: {rhos[-1][0,0].real:.4f}")
    print(f"   最终迹: {np.trace(rhos[-1]).real:.6f}")

    print("\n[density_evolution.py] 测试完成。\n")


if __name__ == "__main__":
    test_density_evolution()
