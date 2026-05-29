# -*- coding: utf-8 -*-
"""
reaction_diffusion.py
核子反应-扩散动力学

本模块模拟中子星crust层核pasta相形成过程中的核子输运与β衰变动力学.
融入算法:
- fd_predator_prey (350): 有限差分反应扩散框架
- fe2d_predator_prey_fast (410): 有限元反应扩散, 稀疏矩阵, GMRES求解

核心物理公式:
1. 核子输运方程 (扩散+反应):
   drho_n/dt = D_n nabla^2 rho_n + lambda_beta+ rho_p - lambda_beta- rho_n
   drho_p/dt = D_p nabla^2 rho_p - lambda_beta+ rho_p + lambda_beta- rho_n
   
   其中:
   D_n, D_p: 中子/质子扩散系数 (fm^2/s)
   lambda_beta+: 正β衰变率 (p -> n + e+ + nu_e)
   lambda_beta-: 逆β衰变率 (n -> p + e- + anti-nu_e)
   
2. 化学平衡条件 (beta equilibrium):
   mu_n = mu_p + mu_e + mu_nu
   
3. 扩散系数 (Stokes-Einstein):
   D = k_B T / (6*pi*eta*R)
   其中eta为粘滞系数
   
4. 有限差分格式 (显式Euler):
   rho^{n+1} = rho^n + dt * (D * Laplacian(rho^n) + R(rho^n))
   
5. 有限元弱形式:
   (u^{n+1} - u^n)/dt, v) + D*(nabla u^{n+1}, nabla v) = (R(u^n), v)
   
6. 熵产生率:
   sigma = sum_k J_k . (-nabla mu_k / T)
"""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

# 物理常数
HBARC = 197.3269804  # MeV·fm
K_B = 8.617333262e-11  # MeV/K (玻尔兹曼常数)


def beta_decay_rates(temperature, rho_n, rho_p, electron_chemical_potential):
    """
    计算β衰变与逆β衰变率.
    
    简化模型 (Fermi theory):
    lambda_beta+ ~ G_F^2 * T^5 * f(Q/T)
    lambda_beta- ~ G_F^2 * T^5 * f(-Q/T)
    
    其中 Q = mu_n - mu_p - mu_e
    
    输入:
        temperature: 温度 (MeV)
        rho_n, rho_p: 中子/质子数密度 (fm^{-3})
        electron_chemical_potential: 电子化学势 (MeV)
    输出:
        lambda_plus, lambda_minus: 衰变率 (s^{-1})
    """
    if temperature <= 0.0:
        return 0.0, 0.0

    # 简化费米耦合常数 (自然单位)
    G_F = 1.1663787e-11  # MeV^{-2}

    # 核子化学势简化
    m_n = 939.565  # MeV
    m_p = 938.272  # MeV
    k_fn = (3.0 * np.pi**2 * rho_n)**(1.0/3.0)
    k_fp = (3.0 * np.pi**2 * rho_p)**(1.0/3.0)

    # 非相对论近似
    mu_n = m_n + k_fn**2 / (2.0 * m_n)
    mu_p = m_p + k_fp**2 / (2.0 * m_p)

    Q = mu_n - mu_p - electron_chemical_potential

    # 简化: 使用费米积分近似
    # lambda ~ T^5 * F_4(eta) 其中 eta = Q/T
    eta = Q / temperature

    # 简化近似
    if eta > 10.0:
        f_plus = eta**5 / 120.0
        f_minus = 0.0
    elif eta < -10.0:
        f_plus = 0.0
        f_minus = (-eta)**5 / 120.0
    else:
        f_plus = np.exp(eta) / (1.0 + np.exp(eta)) * (temperature / 1.0)**5
        f_minus = np.exp(-eta) / (1.0 + np.exp(-eta)) * (temperature / 1.0)**5

    prefactor = G_F**2 / (np.pi**3)
    lambda_plus = prefactor * f_plus * 1e42  # 单位转换
    lambda_minus = prefactor * f_minus * 1e42

    # 边界处理
    lambda_plus = max(0.0, min(lambda_plus, 1e20))
    lambda_minus = max(0.0, min(lambda_minus, 1e20))

    return lambda_plus, lambda_minus


def diffusion_coefficient(temperature, viscosity, radius):
    """
    Stokes-Einstein扩散系数.
    
    D = k_B T / (6 * pi * eta * R)
    
    输入:
        temperature: 温度 (MeV)
        viscosity: 粘滞系数 (MeV/fm^2·s)
        radius: 特征半径 (fm)
    输出:
        D: 扩散系数 (fm^2/s)
    """
    if viscosity <= 0.0 or radius <= 0.0:
        return 0.0
    D = K_B * temperature / (6.0 * np.pi * viscosity * radius)
    return max(0.0, D)


def fd_reaction_diffusion_1d(rho_n0, rho_p0, dx, dt, n_steps, D_n, D_p,
                              lambda_plus, lambda_minus, bc_type='neumann'):
    """
    一维有限差分反应扩散求解器 (来自350_fd_predator_prey).
    
    输入:
        rho_n0, rho_p0: 初始密度分布 (1D数组)
        dx: 空间步长
        dt: 时间步长
        n_steps: 时间步数
        D_n, D_p: 扩散系数
        lambda_plus, lambda_minus: 反应率 (可为常数或数组)
        bc_type: 边界条件类型
    输出:
        rho_n, rho_p: 最终密度分布
        history_n, history_p: 时间演化历史
    """
    N = len(rho_n0)
    if len(rho_p0) != N:
        raise ValueError("rho_n0和rho_p0长度必须相同")
    if dx <= 0.0 or dt <= 0.0:
        raise ValueError("dx和dt必须为正")

    rho_n = np.array(rho_n0, dtype=float)
    rho_p = np.array(rho_p0, dtype=float)

    # CFL条件检查
    cfl_n = D_n * dt / dx**2
    cfl_p = D_p * dt / dx**2
    if cfl_n > 0.5 or cfl_p > 0.5:
        # 自适应减小dt
        dt_new = 0.4 * dx**2 / max(D_n, D_p)
        n_steps = int(n_steps * dt / dt_new) + 1
        dt = dt_new

    history_n = [rho_n.copy()]
    history_p = [rho_p.copy()]

    lambda_plus = np.asarray(lambda_plus)
    lambda_minus = np.asarray(lambda_minus)
    if lambda_plus.ndim == 0:
        lambda_plus = np.full(N, lambda_plus)
    if lambda_minus.ndim == 0:
        lambda_minus = np.full(N, lambda_minus)

    for _ in range(n_steps):
        rho_n_new = rho_n.copy()
        rho_p_new = rho_p.copy()

        for i in range(1, N - 1):
            # 扩散项 (中心差分)
            lap_n = (rho_n[i + 1] - 2.0 * rho_n[i] + rho_n[i - 1]) / dx**2
            lap_p = (rho_p[i + 1] - 2.0 * rho_p[i] + rho_p[i - 1]) / dx**2

            # 反应项
            R_n = lambda_plus[i] * rho_p[i] - lambda_minus[i] * rho_n[i]
            R_p = -lambda_plus[i] * rho_p[i] + lambda_minus[i] * rho_n[i]

            # 显式Euler更新
            rho_n_new[i] = rho_n[i] + dt * (D_n * lap_n + R_n)
            rho_p_new[i] = rho_p[i] + dt * (D_p * lap_p + R_p)

        # 边界条件
        if bc_type == 'neumann':
            rho_n_new[0] = rho_n_new[1]
            rho_n_new[-1] = rho_n_new[-2]
            rho_p_new[0] = rho_p_new[1]
            rho_p_new[-1] = rho_p_new[-2]
        elif bc_type == 'dirichlet':
            rho_n_new[0] = rho_n[0]
            rho_n_new[-1] = rho_n[-1]
            rho_p_new[0] = rho_p[0]
            rho_p_new[-1] = rho_p[-1]
        elif bc_type == 'periodic':
            rho_n_new[0] = rho_n_new[-2]
            rho_n_new[-1] = rho_n_new[1]
            rho_p_new[0] = rho_p_new[-2]
            rho_p_new[-1] = rho_p_new[1]

        # 密度非负约束
        rho_n_new = np.maximum(rho_n_new, 0.0)
        rho_p_new = np.maximum(rho_p_new, 0.0)

        rho_n = rho_n_new
        rho_p = rho_p_new

        if len(history_n) < 1000:  # 限制历史记录大小
            history_n.append(rho_n.copy())
            history_p.append(rho_p.copy())

    return rho_n, rho_p, np.array(history_n), np.array(history_p)


def fe_reaction_diffusion_2d(nodes, elements, rho_n0, rho_p0, dt, n_steps,
                             D_n, D_p, lambda_plus, lambda_minus,
                             bc_nodes=None):
    """
    二维有限元反应扩散求解器 (来自410_fem2d_predator_prey_fast).
    
    简化实现: 使用质量阵集中(lumping)和显式时间积分.
    
    输入:
        nodes: (n_nodes, 2)
        elements: (n_elements, 3)
        rho_n0, rho_p0: 初始密度 (n_nodes,)
        dt: 时间步长
        n_steps: 步数
        D_n, D_p: 扩散系数
        lambda_plus, lambda_minus: 反应率
        bc_nodes: 边界节点索引
    输出:
        rho_n, rho_p: 最终密度
    """
    n_nodes = nodes.shape[0]
    rho_n = np.array(rho_n0, dtype=float)
    rho_p = np.array(rho_p0, dtype=float)

    # 组装质量阵和刚度阵 (简化: 使用集中质量)
    m_hat = np.zeros(n_nodes)
    K = csr_matrix((n_nodes, n_nodes))

    row_ind = []
    col_ind = []
    data_k = []

    for elem in range(elements.shape[0]):
        idx = elements[elem]
        xi, yi = nodes[idx[0]]
        xj, yj = nodes[idx[1]]
        xk, yk = nodes[idx[2]]

        area = abs((xj - xi) * (yk - yi) - (xk - xi) * (yj - yi)) / 2.0
        if area < 1e-15:
            continue

        # 集中质量
        m_i = area / 3.0
        for i in range(3):
            m_hat[idx[i]] += m_i

        # 刚度矩阵 (常数梯度近似)
        # h1, h2, h3 来自fe2d_predator_prey_fast
        h1 = (xi - xj) * (yk - yj) - (xk - xj) * (yi - yj)
        h2 = (xj - xk) * (yi - yk) - (xi - xk) * (yj - yk)
        h3 = (xk - xi) * (yj - yi) - (xj - xi) * (yk - yi)

        # 防止除零
        h1 = max(abs(h1), 1e-15) * np.sign(h1) if h1 != 0 else 1e-15
        h2 = max(abs(h2), 1e-15) * np.sign(h2) if h2 != 0 else 1e-15
        h3 = max(abs(h3), 1e-15) * np.sign(h3) if h3 != 0 else 1e-15

        s1 = (yj - yi) * (yk - yj) + (xi - xj) * (xj - xk)
        s2 = (yj - yi) * (yi - yk) + (xi - xj) * (xk - xi)
        s3 = (yk - yj) * (yi - yk) + (xj - xk) * (xk - xi)
        t1 = (yj - yi)**2 + (xi - xj)**2
        t2 = (yk - yj)**2 + (xj - xk)**2
        t3 = (yi - yk)**2 + (xk - xi)**2

        # 局部刚度矩阵元素
        local_k = {
            (idx[0], idx[0]): area * t2 / (h1 * h1),
            (idx[1], idx[1]): area * t3 / (h2 * h2),
            (idx[2], idx[2]): area * t1 / (h3 * h3),
            (idx[0], idx[1]): area * s3 / (h1 * h2),
            (idx[1], idx[0]): area * s3 / (h1 * h2),
            (idx[0], idx[2]): area * s2 / (h1 * h3),
            (idx[2], idx[0]): area * s2 / (h1 * h3),
            (idx[1], idx[2]): area * s1 / (h2 * h3),
            (idx[2], idx[1]): area * s1 / (h2 * h3),
        }

        for (i, j), val in local_k.items():
            row_ind.append(i)
            col_ind.append(j)
            data_k.append(val)

    K = csr_matrix((data_k, (row_ind, col_ind)), shape=(n_nodes, n_nodes))

    # 质量矩阵的逆 (集中)
    m_inv = 1.0 / (m_hat + 1e-15)

    # 构造 B = I + dt * M^{-1} * K
    # 使用显式Euler: rho^{n+1} = rho^n + dt * M^{-1} * (-K*rho^n + R)
    I = csr_matrix(np.eye(n_nodes))

    for step in range(n_steps):
        # 反应项
        R_n = lambda_plus * rho_p - lambda_minus * rho_n
        R_p = -lambda_plus * rho_p + lambda_minus * rho_n

        # 扩散项
        diff_n = -K.dot(rho_n)
        diff_p = -K.dot(rho_p)

        # 更新
        rho_n = rho_n + dt * m_inv * (diff_n + R_n)
        rho_p = rho_p + dt * m_inv * (diff_p + R_p)

        # 边界条件
        if bc_nodes is not None:
            rho_n[bc_nodes] = rho_n0[bc_nodes]
            rho_p[bc_nodes] = rho_p0[bc_nodes]

        # 非负约束
        rho_n = np.maximum(rho_n, 0.0)
        rho_p = np.maximum(rho_p, 0.0)

        # 检查NaN
        if not np.all(np.isfinite(rho_n)) or not np.all(np.isfinite(rho_p)):
            raise RuntimeError(f"FE求解在step={step}发散")

    return rho_n, rho_p


def entropy_production_rate(rho_n, rho_p, grad_mu_n, grad_mu_p, T, D_n, D_p):
    """
    计算熵产生率 (Onsager理论).
    
    sigma = sum_k J_k . X_k / T
    J_k = -D_k * rho_k * grad(mu_k) / T
    X_k = -grad(mu_k)
    
    输入:
        rho_n, rho_p: 密度
        grad_mu_n, grad_mu_p: 化学势梯度
        T: 温度
        D_n, D_p: 扩散系数
    输出:
        sigma: 熵产生率
    """
    if T <= 0.0:
        return 0.0
    J_n = -D_n * rho_n * grad_mu_n / T
    J_p = -D_p * rho_p * grad_mu_p / T
    X_n = -grad_mu_n
    X_p = -grad_mu_p
    sigma = J_n * X_n + J_p * X_p
    return sigma


if __name__ == '__main__':
    # 自测试: 一维扩散
    N = 100
    x = np.linspace(0, 10, N)
    rho_n0 = np.exp(-(x - 5)**2)
    rho_p0 = np.ones(N) * 0.1
    rho_n, rho_p, _, _ = fd_reaction_diffusion_1d(
        rho_n0, rho_p0, dx=x[1]-x[0], dt=0.001, n_steps=100,
        D_n=1.0, D_p=0.5, lambda_plus=0.1, lambda_minus=0.05
    )
    print(f"1D FD test: rho_n range = [{rho_n.min():.4f}, {rho_n.max():.4f}]")
