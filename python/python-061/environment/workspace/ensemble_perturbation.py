
import numpy as np


def sphere01_sample(n):
    x = np.random.randn(3, n)
    norms = np.sqrt(np.sum(x**2, axis=0))
    norms = np.where(norms < 1e-12, 1.0, norms)
    x = x / norms[np.newaxis, :]
    return x


def sphere01_monomial_integral(e):
    if np.any(e < 0):
        return 0.0
    if np.any(e % 2 == 1):
        return 0.0
    
    def double_factorial(k):
        if k < 0:
            return 1.0
        result = 1.0
        for i in range(k, 0, -2):
            result *= i
        return result
    
    total_sum = np.sum(e)
    numerator = 1.0
    for i in range(3):
        numerator *= double_factorial(e[i] - 1)
    denominator = double_factorial(total_sum + 1)
    
    return 4.0 * np.pi * numerator / denominator


def generate_ensemble_perturbations(n_ens=20, state_dim=4, amplitude=2.0):

    sigma_b = np.array([0.5, 0.3, 5.0, 10.0])
    


    raw_pert = np.random.randn(n_ens, state_dim)
    

    perturbations = np.zeros((n_ens, state_dim))
    for i in range(n_ens):
        v = raw_pert[i, :].copy()
        for j in range(i):
            proj = np.dot(perturbations[j, :], v) / (np.dot(perturbations[j, :], perturbations[j, :]) + 1e-12)
            v = v - proj * perturbations[j, :]
        norm_v = np.linalg.norm(v)
        if norm_v > 1e-12:
            perturbations[i, :] = amplitude * (v / norm_v) * sigma_b
        else:
            perturbations[i, :] = 0.0
    
    return perturbations


class EnsembleStatistics:
    def __init__(self, ensemble_states):
        self.states = np.array(ensemble_states)
        self.n_ens = self.states.shape[0]
    
    def ensemble_mean(self):
        return np.mean(self.states, axis=0)
    
    def ensemble_spread(self):
        mean = self.ensemble_mean()
        if self.states.ndim == 3:

            diff = self.states - mean[np.newaxis, :, :]
        else:
            diff = self.states - mean[np.newaxis, :]
        return np.sqrt(np.mean(diff**2, axis=0))
    
    def probability_in_interval(self, variable_idx, lower, upper):
        if self.states.ndim == 3:
            vals = self.states[:, :, variable_idx]
        else:
            vals = self.states[:, variable_idx]
        
        count = np.sum((vals >= lower) & (vals <= upper), axis=0)
        return count / self.n_ens
    
    def confidence_interval(self, variable_idx, level=0.95):
        alpha = (1.0 - level) / 2.0
        if self.states.ndim == 3:
            vals = self.states[:, :, variable_idx]
        else:
            vals = self.states[:, variable_idx]
        
        lower = np.percentile(vals, alpha * 100.0, axis=0)
        upper = np.percentile(vals, (1.0 - alpha) * 100.0, axis=0)
        return lower, upper
    
    def group_by_intensity(self, pmin_idx=2, thresholds=(980, 960, 940)):
        if self.states.ndim == 3:

            pmin = self.states[:, -1, pmin_idx]
        else:
            pmin = self.states[:, pmin_idx]
        
        groups = {
            'TD': [],
            'TS': [],
            'TY': [],
            'STY': []
        }
        
        for i in range(self.n_ens):
            p = pmin[i]
            if p > thresholds[0]:
                groups['TD'].append(i)
            elif p > thresholds[1]:
                groups['TS'].append(i)
            elif p > thresholds[2]:
                groups['TY'].append(i)
            else:
                groups['STY'].append(i)
        
        return groups
    
    def summarize_groups(self, groups):
        summary = {}
        total = self.n_ens
        for name, members in groups.items():
            count = len(members)
            summary[name] = {
                'count': count,
                'percentage': count / total * 100.0 if total > 0 else 0.0
            }
        return summary


def run_ensemble_forecast(vortex_solver_class, n_ens=20, t_span=(0.0, 72.0), n_steps=720):
    from typhoon_vortex_ode import TyphoonVortexODE, TyphoonVortexParameters
    



    ensemble_states = np.zeros((n_ens, n_steps + 1, 4))
    stats = EnsembleStatistics(ensemble_states)
    t_arr = np.linspace(t_span[0], t_span[1], n_steps + 1)

    
    return ensemble_states, stats, t_arr
