
import numpy as np






def hypersphere01_area(m):
    from math import gamma, pi
    if m < 1:
        raise ValueError("维度 m 必须 ≥ 1")
    return 2.0 * pi**(m / 2.0) / gamma(m / 2.0)


def hypersphere01_sample(m, n, seed=None):
    if seed is not None:
        np.random.seed(seed)
    x = np.random.randn(m, n)
    norms = np.linalg.norm(x, axis=0)
    norms = np.where(norms < 1e-15, 1.0, norms)
    return x / norms


def hypersphere_monomial_integral(m, exponents):
    from math import gamma
    exponents = np.asarray(exponents)
    
    if np.any(exponents % 2 != 0):
        return 0.0
    
    numerator = 1.0
    for e in exponents:
        numerator *= gamma((e + 1.0) / 2.0)
    
    denominator = gamma(0.5 * np.sum(exponents + 1.0))
    return 2.0 * numerator / denominator






def propagate_uncertainty(model_func, base_params, param_scales,
                          n_samples=1000, seed=None):
    m = len(base_params)
    samples_dir = hypersphere01_sample(m, n_samples, seed)
    
    outputs = []
    params_samples = []
    
    for i in range(n_samples):
        perturbed = base_params + param_scales * samples_dir[:, i]

        perturbed = np.maximum(perturbed, 1e-8)
        params_samples.append(perturbed)
        
        try:
            val = model_func(perturbed)
        except Exception:
            val = np.nan
        outputs.append(val)
    
    outputs = np.array(outputs)
    valid = outputs[~np.isnan(outputs)]
    
    return {
        'mean': np.mean(valid) if len(valid) > 0 else np.nan,
        'std': np.std(valid) if len(valid) > 0 else np.nan,
        'min': np.min(valid) if len(valid) > 0 else np.nan,
        'max': np.max(valid) if len(valid) > 0 else np.nan,
        'cv': np.std(valid) / np.mean(valid) if len(valid) > 0 and np.mean(valid) != 0 else np.nan,
        'samples': outputs,
        'params_samples': np.array(params_samples),
    }


def sobol_first_order_index(model_func, base_params, param_scales, param_index,
                            n_samples=1000, seed=None):
    m = len(base_params)
    

    result_total = propagate_uncertainty(model_func, base_params, param_scales,
                                         n_samples=n_samples, seed=seed)
    var_total = result_total['std']**2
    
    if var_total < 1e-15 or np.isnan(var_total):
        return 0.0
    

    def conditional_model(other_params):
        full_params = base_params.copy()

        idx = 0
        for i in range(m):
            if i != param_index:
                full_params[i] = other_params[idx]
                idx += 1
        return model_func(full_params)
    

    other_base = np.delete(base_params, param_index)
    other_scales = np.delete(param_scales, param_index)
    
    result_cond = propagate_uncertainty(conditional_model, other_base, other_scales,
                                        n_samples=n_samples, seed=(seed+1 if seed else None))
    var_cond = result_cond['std']**2
    

    S_i = 1.0 - var_cond / var_total
    S_i = max(0.0, min(1.0, S_i))
    return S_i


def full_sensitivity_analysis(model_func, base_params, param_names, param_scales,
                               n_samples=500, seed=None):
    m = len(base_params)
    sobol = {}
    
    for i in range(m):
        S_i = sobol_first_order_index(model_func, base_params, param_scales, i,
                                      n_samples=n_samples, seed=seed)
        name = param_names[i] if i < len(param_names) else f"param_{i}"
        sobol[name] = S_i
    
    uncertainty = propagate_uncertainty(model_func, base_params, param_scales,
                                        n_samples=n_samples, seed=seed)
    
    return {
        'sobol_indices': sobol,
        'uncertainty': uncertainty,
    }






def carbon_cycle_uncertainty_analysis(DIC_surf, TA_surf, T, S,
                                       n_samples=500, seed=None):
    from carbonate_chemistry import solve_carbonate_system
    
    base_params = np.array([DIC_surf, TA_surf, T, S, 1.0, 1.0])
    param_scales = np.array([
        DIC_surf * 0.02,
        TA_surf * 0.015,
        1.5,
        0.35,
        0.05,
        0.05
    ])
    param_names = ['DIC', 'TA', 'T', 'S', 'K1_scale', 'K2_scale']
    
    def model_pH(params):
        D, A, Tp, Sp, k1s, k2s = params
        try:
            res = solve_carbonate_system(D * 1e-6, A * 1e-6, Tp, Sp)

            res['pH'] += np.log10(k1s) * 0.1
            return res['pH']
        except Exception:
            return np.nan
    
    def model_omega(params):
        D, A, Tp, Sp, k1s, k2s = params
        try:
            res = solve_carbonate_system(D * 1e-6, A * 1e-6, Tp, Sp)
            return res['Omega_aragonite']
        except Exception:
            return np.nan
    
    result_pH = propagate_uncertainty(model_pH, base_params, param_scales,
                                      n_samples=n_samples, seed=seed)
    result_omega = propagate_uncertainty(model_omega, base_params, param_scales,
                                         n_samples=n_samples, seed=(seed+1 if seed else None))
    

    sobol_pH = {}
    for i in range(len(param_names)):
        S_i = sobol_first_order_index(model_pH, base_params, param_scales, i,
                                      n_samples=min(n_samples, 200), seed=seed)
        sobol_pH[param_names[i]] = S_i
    
    return {
        'pH_uncertainty': result_pH,
        'omega_uncertainty': result_omega,
        'sobol_pH': sobol_pH,
        'param_names': param_names,
    }
