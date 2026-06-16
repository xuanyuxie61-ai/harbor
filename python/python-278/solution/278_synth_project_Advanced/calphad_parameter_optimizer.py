"""
calphad_parameter_optimizer.py
==============================
基于 MOEA/D (Multi-Objective Evolutionary Algorithm based on Decomposition)
的 CALPHAD 参数优化器。

种子项目 1219 MOEA/D-Container-Routing:
  - 自适应 PBI/Chebyshev 标量化切换
  - Monte Carlo 场景不确定性量化
  - 均值-方差双目标优化

映射到 CALPHAD 参数拟合:
  目标 1: 最小化相平衡预测与实验数据的偏差
  目标 2: 最小化参数的不确定性传播 (鲁棒性)

优化问题:
  min  f_1(theta) = sum_i [x_C^alpha,calc(T_i) - x_C^alpha,exp(T_i)]²
  min  f_2(theta) = max_T sigma[x_C^alpha](T)

  subject to: theta ∈ Theta_feasible (热力学一致性约束)

算法流程:
  1. 初始化均匀参考方向 (weight vectors)
  2. 对每个子问题, 选择邻域
  3. 使用 SBX 交叉 + 多项式变异生成新解
  4. 自适应选择 PBI 或 Chebyshev 标量化
  5. 更新参考点和理想点
"""

import numpy as np
from calphad_fec_constants import (
    MOEAD_POP_SIZE, MOEAD_N_GEN, MOEAD_N_OBJ,
    MOEAD_ETA, MOEAD_PBI_THETA,
    L_FCC_FE_C, L_BCC_FE_C, L_LIQUID_FE_C,
    R_GAS,
)


def generate_reference_directions(pop_size, n_obj):
    """
    生成均匀分布的参考方向向量 (Das-Dennis 方法)。

    对于 n_obj=2, 生成 pop_size 个在单位单纯形上的均匀点:
        w_i = (i/(N-1), 1 - i/(N-1)),  i = 0, ..., N-1

    Parameters
    ----------
    pop_size : int
        种群大小
    n_obj : int
        目标数

    Returns
    -------
    np.ndarray
        (pop_size, n_obj) 参考方向矩阵
    """
    if n_obj == 2:
        W = np.zeros((pop_size, 2))
        for i in range(pop_size):
            W[i, 0] = i / (pop_size - 1)
            W[i, 1] = 1.0 - W[i, 0]
        return W
    else:
        # 高维情况: 使用 simplex lattice
        H = np.arange(pop_size)
        W = np.zeros((pop_size, n_obj))
        for i in range(pop_size):
            W[i, 0] = i / (pop_size - 1)
            W[i, 1] = (1.0 - W[i, 0]) * 0.7
            W[i, 2] = 1.0 - W[i, 0] - W[i, 1] if n_obj > 2 else 0.0
        W = np.abs(W)
        row_sums = np.sum(W, axis=1, keepdims=True)
        row_sums = np.maximum(row_sums, 1e-30)
        W /= row_sums
        return W


def sbx_crossover(parent1, parent2, eta=20.0, bounds=None):
    """
    模拟二进制交叉 (SBX):

    对每一维 i:
        u ~ U(0,1)
        if u <= 0.5:
            beta_q = (2u)^{1/(eta+1)}
        else:
            beta_q = (1/(2(1-u)))^{1/(eta+1)}
        child1_i = 0.5 * ((1+beta_q)*p1_i + (1-beta_q)*p2_i)
        child2_i = 0.5 * ((1-beta_q)*p1_i + (1+beta_q)*p2_i)

    Parameters
    ----------
    parent1, parent2 : np.ndarray
        父代个体
    eta : float
        分布指数 (越大, 子代越接近父代)
    bounds : tuple, optional
        (lower, upper) 变量边界

    Returns
    -------
    tuple
        (child1, child2)
    """
    n = len(parent1)
    child1 = parent1.copy()
    child2 = parent2.copy()

    u = np.random.rand(n)
    beta = np.where(
        u <= 0.5,
        (2.0 * u) ** (1.0 / (eta + 1.0)),
        (1.0 / (2.0 * (1.0 - u + 1e-30))) ** (1.0 / (eta + 1.0))
    )

    child1 = 0.5 * ((1.0 + beta) * parent1 + (1.0 - beta) * parent2)
    child2 = 0.5 * ((1.0 - beta) * parent1 + (1.0 + beta) * parent2)

    if bounds is not None:
        lb, ub = bounds
        child1 = np.clip(child1, lb, ub)
        child2 = np.clip(child2, lb, ub)

    return child1, child2


def polynomial_mutation(individual, eta=20.0, bounds=None, prob=None):
    """
    多项式变异:

    delta_q = (2u)^{1/(eta+1)} - 1   if u < 0.5
            = 1 - (2(1-u))^{1/(eta+1)}  if u >= 0.5
    x'_i = x_i + delta_q * (ub_i - lb_i)

    Parameters
    ----------
    individual : np.ndarray
        个体
    eta : float
        分布指数
    bounds : tuple
        (lower, upper) 边界
    prob : float
        变异概率

    Returns
    -------
    np.ndarray
        变异后个体
    """
    n = len(individual)
    if prob is None:
        prob = 1.0 / n

    mutant = individual.copy()
    if bounds is None:
        return mutant

    lb, ub = bounds
    for i in range(n):
        if np.random.rand() > prob:
            continue
        u = np.random.rand()
        if u < 0.5:
            delta = (2.0 * u) ** (1.0 / (eta + 1.0)) - 1.0
        else:
            delta = 1.0 - (2.0 * (1.0 - u)) ** (1.0 / (eta + 1.0))
        mutant[i] += delta * (ub[i] - lb[i])
        mutant[i] = np.clip(mutant[i], lb[i], ub[i])

    return mutant


def pbi_scalarization(f, w, z_star, theta=5.0):
    """
    PBI (Penalty-based Boundary Intersection) 标量化:

    g(x|w,z*) = d1 + theta * d2

    d1 = ||(f(x) - z*)^T w / ||w||||   (沿参考方向的距离)
    d2 = ||f(x) - z* - d1 * w/||w||||  (垂直距离)

    Parameters
    ----------
    f : np.ndarray
        目标向量
    w : np.ndarray
        参考方向
    z_star : np.ndarray
        理想点
    theta : float
        罚参数

    Returns
    -------
    float
        标量化目标值
    """
    w_norm = np.linalg.norm(w)
    if w_norm < 1e-30:
        return np.sum(f)

    diff = f - z_star
    d1 = abs(np.dot(diff, w)) / w_norm
    proj = d1 * w / w_norm
    d2 = np.linalg.norm(diff - proj)

    return d1 + theta * d2


def chebyshev_scalarization(f, w, z_star):
    """
    Chebyshev 标量化 (Tchebycheff 方法):

    g(x|w,z*) = max_i { w_i * |f_i(x) - z*_i| }

    Parameters
    ----------
    f : np.ndarray
        目标向量
    w : np.ndarray
        参考方向
    z_star : np.ndarray
        理想点

    Returns
    -------
    float
        标量化目标值
    """
    w_safe = np.maximum(w, 1e-10)
    return float(np.max(w_safe * np.abs(f - z_star)))


def calphad_objective_functions(theta, T_data, x_exp_alpha, x_exp_beta,
                                phase_a='FCC', phase_b='BCC'):
    """
    CALPHAD 参数拟合的双目标函数:

    f_1: 拟合误差 (均方根)
        f_1 = sqrt(1/N * sum [x_calc(T_i) - x_exp(T_i)]²)

    f_2: 鲁棒性目标 (参数灵敏度)
        f_2 = max_i |df_1/dtheta| * sigma_theta

    Parameters
    ----------
    theta : np.ndarray
        CALPHAD 参数向量 (L0_FCC, L1_FCC, L2_FCC, L0_BCC, L1_BCC, ...)
    T_data : np.ndarray
        实验温度数据 (K)
    x_exp_alpha : np.ndarray
        实验 alpha 相成分
    x_exp_beta : np.ndarray
        实验 beta 相成分
    phase_a : str
        alpha 相名称
    phase_b : str
        beta 相名称

    Returns
    -------
    np.ndarray
        [f_1, f_2] 目标向量
    """
    import calphad_fec_constants as const
    from newton_maehly_equilibrium import newton_maehly_solve

    # 保存原始参数
    orig_fcc = const.L_FCC_FE_C[:]
    orig_bcc = const.L_BCC_FE_C[:]

    # 设置新参数
    n_fcc = min(3, len(theta))
    n_bcc = min(2, max(0, len(theta) - 3))
    const.L_FCC_FE_C = list(theta[:n_fcc])
    if n_bcc > 0:
        const.L_BCC_FE_C = list(theta[3:3 + n_bcc])

    # 计算拟合误差
    n_data = len(T_data)
    errors = []
    for i in range(n_data):
        try:
            result = newton_maehly_solve(float(T_data[i]), phase_a, phase_b)
            if result['converged']:
                err_a = (result['x_C_alpha'] - x_exp_alpha[i]) ** 2
                err_b = (result['x_C_beta'] - x_exp_beta[i]) ** 2
                errors.append(err_a + err_b)
            else:
                errors.append(1.0)  # 大惩罚
        except Exception:
            errors.append(1.0)

    f1 = np.sqrt(np.mean(errors)) if errors else 1.0

    # 鲁棒性: 参数扰动后的误差变化 (小规模实验)
    sigma_theta = 0.05 * np.abs(theta) + 1e-6
    f1_pert = []
    rng = np.random.RandomState(278)
    for _ in range(3):  # 小规模: 仅 3 次扰动
        theta_p = theta + sigma_theta * rng.randn(len(theta))
        const.L_FCC_FE_C = list(theta_p[:n_fcc])
        if n_bcc > 0:
            const.L_BCC_FE_C = list(theta_p[3:3 + n_bcc])
        errs_p = []
        for i in range(min(3, n_data)):
            try:
                res = newton_maehly_solve(float(T_data[i]), phase_a, phase_b)
                if res['converged']:
                    errs_p.append((res['x_C_alpha'] - x_exp_alpha[i]) ** 2
                                  + (res['x_C_beta'] - x_exp_beta[i]) ** 2)
                else:
                    errs_p.append(1.0)
            except Exception:
                errs_p.append(1.0)
        f1_pert.append(np.sqrt(np.mean(errs_p)) if errs_p else 1.0)

    f2 = np.std(f1_pert) if len(f1_pert) > 1 else 0.0

    # 恢复参数
    const.L_FCC_FE_C = orig_fcc
    const.L_BCC_FE_C = orig_bcc

    return np.array([f1, f2])


def moead_calphad_optimize(T_data=None, x_exp_alpha=None, x_exp_beta=None):
    """
    MOEA/D 优化 CALPHAD 参数。

    Parameters
    ----------
    T_data : np.ndarray
        实验温度 (K)
    x_exp_alpha : np.ndarray
        实验 alpha 成分
    x_exp_beta : np.ndarray
        实验 beta 成分

    Returns
    -------
    dict
        {
            'pareto_front': np.ndarray,
            'best_theta': np.ndarray,
            'population': np.ndarray,
            'objective_values': np.ndarray,
        }
    """
    # 默认合成实验数据 (Fe-C FCC/BCC 平衡, 近似值)
    if T_data is None:
        T_data = np.array([1000.0, 1100.0, 1200.0])  # 小规模实验: 3 个温度点
    if x_exp_alpha is None:
        x_exp_alpha = np.array([0.0008, 0.0018, 0.0042])
    if x_exp_beta is None:
        x_exp_beta = np.array([0.004, 0.009, 0.018])

    n_dim = 5  # 优化 5 个参数: L0_FCC, L1_FCC, L2_FCC, L0_BCC, L1_BCC
    N = MOEAD_POP_SIZE
    n_gen = MOEAD_N_GEN

    # 变量边界
    lb = np.array([-50000.0, -30000.0, -10000.0, 10000.0, 0.0])
    ub = np.array([50000.0, 30000.0, 10000.0, 100000.0, 50000.0])

    # 参考方向
    W = generate_reference_directions(N, MOEAD_N_OBJ)

    # 初始化种群
    rng = np.random.RandomState(278)
    population = np.zeros((N, n_dim))
    for i in range(N):
        population[i] = lb + rng.rand(n_dim) * (ub - lb)

    # 评估初始种群
    objectives = np.zeros((N, MOEAD_N_OBJ))
    for i in range(N):
        objectives[i] = calphad_objective_functions(
            population[i], T_data, x_exp_alpha, x_exp_beta
        )

    # 理想点
    z_star = np.min(objectives, axis=0)

    # 邻域大小
    T_size = max(5, N // 5)
    # 计算参考方向间的距离
    W_dist = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            W_dist[i, j] = np.linalg.norm(W[i] - W[j])
    neighbors = np.argsort(W_dist, axis=1)[:, :T_size]

    # 进化
    for gen in range(n_gen):
        for i in range(N):
            # 选择邻域中的两个父代
            mating = neighbors[i]
            idx = rng.choice(mating, 2, replace=False)

            # SBX 交叉
            child1, child2 = sbx_crossover(
                population[idx[0]], population[idx[1]],
                eta=MOEAD_ETA, bounds=(lb, ub)
            )

            # 多项式变异
            child1 = polynomial_mutation(child1, eta=MOEAD_ETA,
                                         bounds=(lb, ub))

            # 评估
            f_child = calphad_objective_functions(
                child1, T_data, x_exp_alpha, x_exp_beta
            )

            # 更新理想点
            z_star = np.minimum(z_star, f_child)

            # 更新邻域
            for j in mating:
                # 自适应标量化: PBI 或 Chebyshev
                use_pbi = (rng.rand() < 0.5)
                if use_pbi:
                    g_old = pbi_scalarization(objectives[j], W[j], z_star,
                                              MOEAD_PBI_THETA)
                    g_new = pbi_scalarization(f_child, W[j], z_star,
                                              MOEAD_PBI_THETA)
                else:
                    g_old = chebyshev_scalarization(objectives[j], W[j], z_star)
                    g_new = chebyshev_scalarization(f_child, W[j], z_star)

                if g_new < g_old:
                    population[j] = child1.copy()
                    objectives[j] = f_child.copy()

    # 提取 Pareto 前沿
    is_pareto = np.ones(N, dtype=bool)
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            if np.all(objectives[j] <= objectives[i]) and \
               np.any(objectives[j] < objectives[i]):
                is_pareto[i] = False
                break

    pareto_front = objectives[is_pareto]
    pareto_pop = population[is_pareto]

    # 最佳折中解 (最接近 utopia 点)
    if len(pareto_front) > 0:
        # 归一化
        f_range = np.ptp(pareto_front, axis=0)
        f_range = np.maximum(f_range, 1e-30)
        f_norm = (pareto_front - np.min(pareto_front, axis=0)) / f_range
        dists = np.sqrt(np.sum(f_norm ** 2, axis=1))
        best_idx = np.argmin(dists)
        best_theta = pareto_pop[best_idx]
    else:
        best_theta = population[0]

    return {
        'pareto_front': pareto_front,
        'best_theta': best_theta,
        'population': population,
        'objective_values': objectives,
    }
