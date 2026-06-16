"""
diagnostics_bayesian.py - 等离子体诊断贝叶斯参数估计

本模块使用贝叶斯推断方法从仿真诊断数据中估计等离子体参数。

贝叶斯推断框架:

  后验分布:
      p(theta | D) = p(D | theta) * p(theta) / p(D)

  其中:
      theta = (omega_p, T_e, n_hot, ...)  待估参数
      D = 诊断数据 (场能量、粒子谱、密度等)
      p(D | theta) = 似然函数
      p(theta) = 先验分布
      p(D) = 边际似然 (证据)

  似然函数 (Gaussian):
      ln L = -0.5 * sum_i ((D_model_i(theta) - D_obs_i) / sigma_i)^2

  MCMC 采样 (Metropolis-Hastings):
      1. 从当前状态 theta 提议新状态 theta' ~ q(theta' | theta)
      2. 计算接受率: alpha = min(1, p(D|theta')*p(theta') / (p(D|theta)*p(theta)))
      3. 以概率 alpha 接受 theta' -> theta

  仿射不变系综采样 (emcee 风格):
      使用 n_walkers 个行走器组成的系综,
      提议分布基于系综中其他行走器的位置:
          theta' = theta + z * (theta_random - theta)
      其中 z ~ g(z) 为缩放因子分布。

物理应用:
  从仿真输出的激光吸收率、超热电子温度、反射率等诊断量
  反推等离子体的密度、温度和密度梯度尺度长度。
"""

import numpy as np


def gaussian_log_likelihood(model, data, sigma):
    """Gaussian 对数似然函数。

    ln L = -0.5 * sum_i ((model_i - data_i) / sigma_i)^2
           - 0.5 * sum_i ln(2*pi*sigma_i^2)

    Parameters
    ----------
    model : ndarray
        模型预测值
    data : ndarray
        观测数据
    sigma : ndarray
        观测不确定度

    Returns
    -------
    ln_L : float
        对数似然值
    """
    residuals = (data - model) / sigma
    ln_L = -0.5 * np.sum(residuals ** 2)
    ln_L -= 0.5 * np.sum(np.log(2.0 * np.pi * sigma ** 2))
    return ln_L


def uniform_log_prior(params, bounds):
    """均匀先验的对数概率。

    p(theta) = 1/(upper - lower)  for lower <= theta <= upper
    ln p(theta) = -ln(upper - lower)  for lower <= theta <= upper
                  = -inf                otherwise

    Parameters
    ----------
    params : ndarray
        参数值
    bounds : list of (lower, upper)
        参数边界

    Returns
    -------
    ln_prior : float
        对数先验概率
    """
    for p, (lo, hi) in zip(params, bounds):
        if p < lo or p > hi:
            return -np.inf
    return 0.0  # 均匀先验 (常数, 省略归一化)


def forward_model(theta, config_template):
    """正向模型: 从等离子体参数预测诊断量。

    使用简化的解析模型代替完整的 Vlasov 仿真:

    参数 theta = (omega_p, T_e, L_n):
        omega_p: 等离子体频率
        T_e: 电子温度
        L_n: 密度梯度尺度长度

    预测量:
        absorption = 1 - exp(-alpha_IB * L_n)  逆韧致吸收率
        T_hot = T_e * (1 + a0^2 * omega_p^2 / (4*omega_L^2))  超热电子温度
        reflectivity = ((1 - sqrt(1 - omega_p^2/omega_L^2)) /
                        (1 + sqrt(1 - omega_p^2/omega_L^2)))^2

    Parameters
    ----------
    theta : ndarray
        参数向量 (omega_p, T_e, L_n)
    config_template : SimulationConfig
        配置模板 (用于获取激光参数)

    Returns
    -------
    predictions : ndarray
        预测的诊断量
    """
    omega_p, T_e, L_n = theta
    omega_L = config_template.omega_0
    a0 = config_template.a0

    # 逆韧致吸收率 (简化模型)
    nu_eff = 0.01 * omega_p / (T_e ** 1.5 + 0.01)  # 有效碰撞频率
    alpha_IB = nu_eff * omega_p ** 2 / (omega_L ** 2 + 0.01)
    absorption = 1.0 - np.exp(-alpha_IB * max(L_n, 0.01))
    absorption = np.clip(absorption, 0, 1)

    # 超热电子温度
    T_hot = T_e * (1.0 + a0 ** 2 * omega_p ** 2 / (4.0 * omega_L ** 2 + 0.01))

    # 反射率
    ratio_sq = (omega_p / omega_L) ** 2
    if ratio_sq < 1.0:
        sqrt_term = np.sqrt(1.0 - ratio_sq)
        reflectivity = ((1.0 - sqrt_term) / (1.0 + sqrt_term)) ** 2
    else:
        reflectivity = 1.0

    # 透射率
    transmission = max(0, 1.0 - absorption - reflectivity)

    return np.array([absorption, T_hot, reflectivity, transmission])


def metropolis_hastings(log_posterior, theta0, n_steps, proposal_scale, seed=42):
    """Metropolis-Hastings MCMC 采样。

    使用 Gaussian 提议分布:
        theta' = theta + sigma_prop * N(0, I)

    接受率:
        alpha = min(1, exp(ln_p(theta') - ln_p(theta)))

    Parameters
    ----------
    log_posterior : callable
        对数后验概率函数
    theta0 : ndarray
        初始参数
    n_steps : int
        采样步数
    proposal_scale : float
        提议分布尺度
    seed : int
        随机种子

    Returns
    -------
    chain : ndarray (n_steps, n_params)
        MCMC 链
    acceptance_rate : float
        接受率
    """
    rng = np.random.RandomState(seed)
    n_params = len(theta0)
    chain = np.zeros((n_steps, n_params))
    ln_p_current = log_posterior(theta0)

    theta = theta0.copy()
    n_accepted = 0

    for i in range(n_steps):
        # 提议
        theta_prop = theta + proposal_scale * rng.randn(n_params)

        # 评估后验
        ln_p_prop = log_posterior(theta_prop)

        # 接受/拒绝
        if np.isfinite(ln_p_prop):
            ln_alpha = ln_p_prop - ln_p_current
            if ln_alpha > 0 or np.log(rng.uniform()) < ln_alpha:
                theta = theta_prop
                ln_p_current = ln_p_prop
                n_accepted += 1

        chain[i] = theta

    acceptance_rate = n_accepted / n_steps
    return chain, acceptance_rate


def ensemble_sampler(log_posterior, n_walkers, theta0, n_steps, seed=42):
    """简化版仿射不变系综采样器。

    使用 stretch move (Goodman & Weare 2010):
        theta' = theta + z * (theta_other - theta)
        z ~ U(0, 1) 的变换: z = ((1+s)*u)^2 / (1+s) - 1) / s
        其中 s = 2, u ~ U(0, 1)

    Parameters
    ----------
    log_posterior : callable
        对数后验
    n_walkers : int
        行走器数量
    theta0 : ndarray
        初始参数 (所有行走器的中心)
    n_steps : int
        采样步数
    seed : int
        随机种子

    Returns
    -------
    chains : ndarray (n_walkers, n_steps, n_params)
        所有行走器的链
    acceptance_rates : ndarray (n_walkers,)
        各行走器的接受率
    """
    rng = np.random.RandomState(seed)
    n_params = len(theta0)
    chains = np.zeros((n_walkers, n_steps, n_params))
    n_accepted = np.zeros(n_walkers)

    # 初始化行走器 (在小范围内散布)
    ensemble = theta0[np.newaxis, :] + 0.1 * np.abs(theta0[np.newaxis, :]) * rng.randn(n_walkers, n_params)
    ln_p_ensemble = np.array([log_posterior(ensemble[k]) for k in range(n_walkers)])

    s = 2.0  # stretch move 参数

    for step in range(n_steps):
        for k in range(n_walkers):
            # 选择互补行走器
            j = k
            while j == k:
                j = rng.randint(n_walkers)

            # stretch move
            u = rng.uniform()
            z = ((1.0 + s) * u) ** 2 / (1.0 + s) - 1.0
            z = z / s + 1.0 / (1.0 + s)  # 修正
            z = max(0.01, min(z, 100.0))  # 限制范围

            # 提议
            theta_prop = ensemble[k] + z * (ensemble[j] - ensemble[k])

            # 评估
            ln_p_prop = log_posterior(theta_prop)

            # 接受准则 (含 Jacobian)
            n_dim = n_params
            ln_alpha = (n_dim - 1) * np.log(z) + ln_p_prop - ln_p_ensemble[k]

            if np.isfinite(ln_p_prop) and (ln_alpha > 0 or np.log(rng.uniform()) < ln_alpha):
                ensemble[k] = theta_prop
                ln_p_ensemble[k] = ln_p_prop
                n_accepted[k] += 1

            chains[k, step] = ensemble[k]

    acceptance_rates = n_accepted / n_steps
    return chains, acceptance_rates


def bayesian_plasma_diagnostics(simulation_results, config):
    """执行贝叶斯等离子体诊断。

    从 Vlasov-Maxwell 仿真结果中提取诊断量,
    然后用 MCMC 推断等离子体参数。

    Parameters
    ----------
    simulation_results : dict
        Vlasov-Maxwell 仿真结果
    config : SimulationConfig
        仿真配置

    Returns
    -------
    diagnostics : dict
        诊断结果
    """
    # 从仿真结果提取 "观测" 数据
    if 'field_energies' in simulation_results and len(simulation_results['field_energies']) > 0:
        fe = simulation_results['field_energies']
        ke = simulation_results['kinetic_energies']
        total_init = fe[0] + ke[0] if len(fe) > 0 else 1.0
        if total_init < 1e-30:
            total_init = 1.0

        # 合成 "观测" 数据 (带噪声)
        absorption_obs = 1.0 - fe[-1] / total_init if len(fe) > 0 else 0.1
        T_hot_obs = 1.2  # 假设超热电子温度
        reflectivity_obs = 0.05

        data = np.array([absorption_obs, T_hot_obs, reflectivity_obs, 1.0 - absorption_obs - reflectivity_obs])
        sigma = np.array([0.05, 0.2, 0.02, 0.05])
    else:
        data = np.array([0.1, 1.0, 0.05, 0.85])
        sigma = np.array([0.05, 0.2, 0.02, 0.05])

    # 参数边界
    bounds = [
        (0.1, 3.0),     # omega_p
        (0.1, 5.0),     # T_e
        (0.1, 10.0),    # L_n
    ]

    # 对数后验
    def log_posterior(theta):
        ln_prior = uniform_log_prior(theta, bounds)
        if not np.isfinite(ln_prior):
            return -np.inf
        model = forward_model(theta, config)
        ln_like = gaussian_log_likelihood(model, data, sigma)
        return ln_prior + ln_like

    # MCMC 采样 (减少步数以保证效率)
    n_walkers = min(config.n_mcmc_walkers, 16)
    n_burn = min(config.n_mcmc_burn, 200)
    n_prod = min(config.n_mcmc_prod, 500)

    theta0 = np.array([1.0, 1.0, 3.0])  # 初始猜测
    chains, acc_rates = ensemble_sampler(
        log_posterior, n_walkers, theta0, n_burn + n_prod, seed=config.mcmc_seed
    )

    # 去除燃烧期
    production = chains[:, n_burn:, :]

    # 统计
    all_samples = production.reshape(-1, 3)
    param_names = ['omega_p', 'T_e', 'L_n']

    estimates = {}
    for i, name in enumerate(param_names):
        estimates[name] = {
            'mean': np.mean(all_samples[:, i]),
            'std': np.std(all_samples[:, i]),
            'median': np.median(all_samples[:, i]),
            'q16': np.percentile(all_samples[:, i], 16),
            'q84': np.percentile(all_samples[:, i], 84),
        }

    diagnostics = {
        'data': data,
        'sigma': sigma,
        'param_estimates': estimates,
        'acceptance_rates': acc_rates,
        'mean_acceptance': np.mean(acc_rates),
        'n_walkers': n_walkers,
        'n_burn': n_burn,
        'n_prod': n_prod,
        'chains_shape': production.shape,
    }

    return diagnostics
