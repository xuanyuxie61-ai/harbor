# -*- coding: utf-8 -*-

import numpy as np


def cvtm_1d(g_num, it_num, s_num, seed=None):
    if g_num < 1:
        raise ValueError("g_num 必须 >= 1。")
    if it_num < 0:
        it_num = 0
    if s_num < 1:
        s_num = 1
    if seed is not None:
        np.random.seed(seed)


    g = np.sort(np.random.rand(g_num))
    energy_history = np.zeros(it_num)
    motion_history = np.zeros(it_num)

    for it in range(it_num):
        s = np.random.rand(s_num)

        sa = -s
        sb = 2.0 - s




        d_real = np.abs(s[:, np.newaxis] - g[np.newaxis, :])
        d_left = np.abs(sa[:, np.newaxis] - g[np.newaxis, :])
        d_right = np.abs(sb[:, np.newaxis] - g[np.newaxis, :])

        d_all = np.stack([d_real, d_left, d_right], axis=2)
        min_idx = np.argmin(d_all, axis=2)
        min_val = np.min(d_all, axis=2)


        s_eff = s.copy()
        mask_left = min_idx[:, 0] == 1
        mask_right = min_idx[:, 0] == 2
        s_eff[mask_left] = 0.0
        s_eff[mask_right] = 1.0


        d_eff = np.abs(s_eff[:, np.newaxis] - g[np.newaxis, :])
        nearest = np.argmin(d_eff, axis=1)


        g_new = np.zeros(g_num)
        w_new = np.zeros(g_num)
        e_new = np.zeros(g_num)

        for i in range(g_num):
            mask = nearest == i
            if np.any(mask):
                g_new[i] = np.sum(s_eff[mask])
                w_new[i] = np.count_nonzero(mask)
                e_new[i] = np.sum(min_val[mask] ** 2)
            else:
                g_new[i] = g[i]
                w_new[i] = 0
                e_new[i] = 0.0


        with np.errstate(divide='ignore', invalid='ignore'):
            g_new = np.where(w_new > 0, g_new / w_new, g)
        g_new = np.clip(g_new, 0.0, 1.0)
        g_new = np.sort(g_new)

        motion = np.mean((g_new - g) ** 2)
        energy = np.sum(e_new)
        energy_history[it] = energy
        motion_history[it] = motion
        g = g_new

    return g, energy_history, motion_history


def optimal_k_path_sampling(k_min, k_max, n_points, it_num=50, s_num=5000):
    if k_min >= k_max:
        raise ValueError("k_min 必须 < k_max。")
    if n_points < 1:
        raise ValueError("n_points 必须 >= 1。")
    generators, _, _ = cvtm_1d(n_points, it_num, s_num)
    return k_min + generators * (k_max - k_min)
