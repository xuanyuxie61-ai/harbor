"""
mcmc_inversion.py
DREAM MCMC参数反演模块（对应种子项目 319_dream）

在光纤非线性光学中，非线性系数γ、色散系数β₂/β₃、损耗α等参数
往往难以直接精确测量。本模块实现DREAM（Differential Evolution Adaptive
Metropolis）MCMC算法，从观测到的脉冲频谱/时域数据中反演这些参数。

核心物理模型:
  观测数据 y_obs = GNLSE(θ; z, t) + ε
  其中θ = [γ, β₂, β₃, α, T_R]为待反演参数，ε为观测噪声。

  后验分布:
    p(θ|y_obs) ∝ p(y_obs|θ) p(θ)

  似然函数（高斯噪声假设）:
    log L(θ) = -1/2 Σ_i [ (y_obs,i - y_model,i)² / σ_i² ] - 1/2 Σ_i log(2πσ_i²)

DREAM算法核心：
  候选生成（差分进化）:
    θ_p = θ_chain + (1+e) * [Σ_{j=1}^{δ} (θ_{a_j} - θ_{b_j})] + ε
    其中e~U(-c,c), ε~N(0, 小方差)

  Metropolis接受准则:
    α = min(1, L(θ_p)p(θ_p) / [L(θ_old)p(θ_old)])
"""

import numpy as np


def r8_uniform_01_sample():
    return np.random.rand()


def cr_init(cr_num):
    """初始化交叉概率CR的值和概率分布。"""
    cr = np.linspace(0.0, 1.0, cr_num)
    cr_dis = np.ones(cr_num)
    cr_prob = np.ones(cr_num) / cr_num
    cr_ups = np.zeros(cr_num)
    return cr, cr_dis, cr_prob, cr_ups


def cr_index_choose(cr_num, cr_prob):
    """按概率选择CR索引。"""
    r = np.random.rand()
    cumsum = np.cumsum(cr_prob)
    for i in range(cr_num):
        if r <= cumsum[i]:
            return i
    return cr_num - 1


def cr_dis_update(chain_index, chain_num, cr_dis, cr_index, cr_num, cr_ups, gen_index, gen_num, par_num, z):
    """更新CR距离统计。"""
    diff = 0.0
    for j in range(par_num):
        diff += (z[j, chain_index, gen_index] - z[j, chain_index, gen_index - 1]) ** 2
    cr_dis[cr_index] += diff
    cr_ups[cr_index] += 1
    return cr_dis, cr_ups


def cr_prob_update(cr_dis, cr_num, cr_ups):
    """更新CR概率分布。"""
    cr_prob = np.zeros(cr_num)
    for i in range(cr_num):
        if cr_ups[i] > 0:
            cr_prob[i] = cr_dis[i] / cr_ups[i]
    total = np.sum(cr_prob)
    if total > 1e-15:
        cr_prob = cr_prob / total
    else:
        cr_prob = np.ones(cr_num) / cr_num
    return cr_prob


def sample_candidate(chain_index, chain_num, cr, cr_index, cr_num, gen_index, gen_num, jumprate_table, jumpstep, limits, pair_num, par_num, z):
    """
    利用差分进化生成候选参数。
    """
    zp = z[:, chain_index, gen_index - 1].copy()
    delta = np.zeros(par_num)

    # 随机选择pair_num对链
    for _ in range(pair_num):
        while True:
            a = np.random.randint(0, chain_num)
            if a != chain_index:
                break
        while True:
            b = np.random.randint(0, chain_num)
            if b != chain_index and b != a:
                break
        delta += z[:, a, gen_index - 1] - z[:, b, gen_index - 1]

    # CR交叉
    r_cr = cr[cr_index]
    gamma_scale = 2.38 / np.sqrt(2.0 * pair_num * par_num)

    # 偶尔长跳跃
    if gen_index % max(jumpstep, 1) == 0:
        gamma_scale = 1.0

    for j in range(par_num):
        if np.random.rand() < r_cr or j == par_num - 1:
            noise = np.random.randn() * 1e-6
            zp[j] += gamma_scale * delta[j] + noise

    # 边界处理
    for j in range(par_num):
        zp[j] = np.clip(zp[j], limits[0, j], limits[1, j])

    return zp


def gr_compute(chain_num, gen_index, gen_num, gr_count, gr_num, gr_threshold, par_num, z):
    """
    计算Gelman-Rubin R统计量以评估收敛。

    R ≈ √(V̂/W)，其中W为链内方差，V̂为总方差估计。
    R < threshold 表示收敛。
    """
    gr = np.zeros((par_num, gr_num))
    gr_conv = False

    # 计算每条链的均值和方差
    chain_mean = np.zeros((par_num, chain_num))
    chain_var = np.zeros((par_num, chain_num))

    for j in range(par_num):
        for c in range(chain_num):
            chain_mean[j, c] = np.mean(z[j, c, :gen_index])
            chain_var[j, c] = np.var(z[j, c, :gen_index], ddof=1) if gen_index > 1 else 1e-10

    overall_mean = np.mean(chain_mean, axis=1)
    B = gen_index * np.var(chain_mean, axis=1, ddof=1)
    W = np.mean(chain_var, axis=1)
    V_hat = ((gen_index - 1.0) / gen_index) * W + B / gen_index

    R_stat = np.sqrt(V_hat / np.where(W > 1e-30, W, 1e-30))

    if gr_count < gr_num:
        gr[:, gr_count] = R_stat
    gr_count += 1

    if np.all(R_stat < gr_threshold):
        gr_conv = True

    return gr, gr_conv, gr_count


def dream_mcmc(log_likelihood_fn, log_prior_fn, par_num, chain_num, gen_num, limits, pair_num=3, cr_num=3, jumpstep=5, gr_threshold=1.2, printstep=100, seed=None):
    """
    DREAM MCMC主函数。

    参数:
        log_likelihood_fn: callable, fn(params) -> float
        log_prior_fn: callable, fn(params) -> float
        par_num: int, 参数维数
        chain_num: int, 链数
        gen_num: int, 每链代数
        limits: ndarray shape (2, par_num), 参数上下界
        pair_num: int, 差分进化对数
        cr_num: int, CR值个数
        jumpstep: int, 长跳跃间隔
        gr_threshold: float, Gelman-Rubin收敛阈值
        printstep: int, 打印间隔
        seed: int or None

    返回:
        z: ndarray shape (par_num, chain_num, gen_num), 样本
        fit: ndarray shape (chain_num, gen_num), 似然值
        acceptance_rate: float
    """
    if seed is not None:
        np.random.seed(seed)

    # 初始化链
    z = np.zeros((par_num, chain_num, gen_num))
    fit = np.zeros((chain_num, gen_num))

    for c in range(chain_num):
        for j in range(par_num):
            z[j, c, 0] = np.random.uniform(limits[0, j], limits[1, j])
        fit[c, 0] = log_likelihood_fn(z[:, c, 0])

    # 初始化CR
    cr, cr_dis, cr_prob, cr_ups = cr_init(cr_num)
    jumprate_table = np.ones(par_num)

    gr = np.zeros((par_num, gen_num))
    gr_conv = False
    gr_count = 0

    zp_count = 0
    zp_accept = 0

    for gen_index in range(1, gen_num):
        for chain_index in range(chain_num):
            cr_index = cr_index_choose(cr_num, cr_prob)
            zp = sample_candidate(chain_index, chain_num, cr, cr_index, cr_num, gen_index, gen_num, jumprate_table, jumpstep, limits, pair_num, par_num, z)

            zp_count += 1
            zp_fit = log_likelihood_fn(zp)
            zp_old = z[:, chain_index, gen_index - 1]
            zp_old_fit = fit[chain_index, gen_index - 1]

            pd1 = log_prior_fn(zp)
            pd2 = log_prior_fn(zp_old)

            # 防止数值溢出
            log_ratio = (zp_fit + pd1) - (zp_old_fit + pd2)
            log_ratio = min(log_ratio, 0.0)
            zp_ratio = np.exp(log_ratio)

            if r8_uniform_01_sample() <= zp_ratio:
                z[:, chain_index, gen_index] = zp
                fit[chain_index, gen_index] = zp_fit
                zp_accept += 1
            else:
                z[:, chain_index, gen_index] = zp_old
                fit[chain_index, gen_index] = zp_old_fit

            if not gr_conv and cr_num > 1:
                cr_dis, cr_ups = cr_dis_update(chain_index, chain_num, cr_dis, cr_index, cr_num, cr_ups, gen_index, gen_num, par_num, z)

        if not gr_conv and cr_num > 1 and gen_index % 10 == 0:
            cr_prob = cr_prob_update(cr_dis, cr_num, cr_ups)

        if gen_index % printstep == 0:
            gr, gr_conv, gr_count = gr_compute(chain_num, gen_index, gen_num, gr_count, gr_num, gr_threshold, par_num, z)

    acceptance_rate = zp_accept / zp_count if zp_count > 0 else 0.0
    return z, fit, acceptance_rate
