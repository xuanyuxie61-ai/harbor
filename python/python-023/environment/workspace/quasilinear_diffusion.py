#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.special import jv


def compute_ql_diffusion_coefficients(v_parallel, v_perp, omega_solutions, params):











    
    nv = len(v_parallel)
    D_par = np.zeros((nv, nv), dtype=np.float64)
    D_perp = np.zeros((nv, nv), dtype=np.float64)
    D_cross = np.zeros((nv, nv), dtype=np.float64)
    
    return D_par, D_perp, D_cross


def assemble_ql_diffusion_matrix(v_parallel, v_perp, omega_solutions, params,
                                  n_stochastic=2, p_degree=2):
    nv = len(v_parallel)
    

    from pce_expansion import enumerate_multi_indices
    indices = enumerate_multi_indices(n_stochastic, p_degree)
    M_pce = len(indices)
    

    N_total = nv * nv * M_pce
    


    D_par, D_perp, D_cross = compute_ql_diffusion_coefficients(
        v_parallel, v_perp, omega_solutions, params
    )
    
    dv_par = v_parallel[1] - v_parallel[0]
    dv_perp = v_perp[1] - v_perp[0]
    










    
    N = nv * nv
    A = np.zeros((N, N))
    rhs = np.zeros(N)
    
    return A, rhs
