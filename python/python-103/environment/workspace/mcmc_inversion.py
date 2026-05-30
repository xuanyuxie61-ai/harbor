
import numpy as np


def r8_uniform_01_sample():
    return np.random.rand()


def cr_init(cr_num):
    cr = np.linspace(0.0, 1.0, cr_num)
    cr_dis = np.ones(cr_num)
    cr_prob = np.ones(cr_num) / cr_num
    cr_ups = np.zeros(cr_num)
    return cr, cr_dis, cr_prob, cr_ups


def cr_index_choose(cr_num, cr_prob):
    r = np.random.rand()
    cumsum = np.cumsum(cr_prob)
    for i in range(cr_num):
        if r <= cumsum[i]:
            return i
    return cr_num - 1


def cr_dis_update(chain_index, chain_num, cr_dis, cr_index, cr_num, cr_ups, gen_index, gen_num, par_num, z):
    diff = 0.0
    for j in range(par_num):
        diff += (z[j, chain_index, gen_index] - z[j, chain_index, gen_index - 1]) ** 2
    cr_dis[cr_index] += diff
    cr_ups[cr_index] += 1
    return cr_dis, cr_ups


def cr_prob_update(cr_dis, cr_num, cr_ups):
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
    zp = z[:, chain_index, gen_index - 1].copy()
    delta = np.zeros(par_num)


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


    r_cr = cr[cr_index]
    gamma_scale = 2.38 / np.sqrt(2.0 * pair_num * par_num)


    if gen_index % max(jumpstep, 1) == 0:
        gamma_scale = 1.0

    for j in range(par_num):
        if np.random.rand() < r_cr or j == par_num - 1:
            noise = np.random.randn() * 1e-6
            zp[j] += gamma_scale * delta[j] + noise


    for j in range(par_num):
        zp[j] = np.clip(zp[j], limits[0, j], limits[1, j])

    return zp


def gr_compute(chain_num, gen_index, gen_num, gr_count, gr_num, gr_threshold, par_num, z):
    gr = np.zeros((par_num, gr_num))
    gr_conv = False


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
    if seed is not None:
        np.random.seed(seed)


    z = np.zeros((par_num, chain_num, gen_num))
    fit = np.zeros((chain_num, gen_num))

    for c in range(chain_num):
        for j in range(par_num):
            z[j, c, 0] = np.random.uniform(limits[0, j], limits[1, j])
        fit[c, 0] = log_likelihood_fn(z[:, c, 0])


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
