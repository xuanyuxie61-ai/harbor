
import numpy as np
from numpy.polynomial.legendre import leggauss, legval

from special_functions_qcd import legendre_poly_vals


class LegendrePCE:
    
    def __init__(self, order=5):
        self.order = order
        self.coeffs = np.zeros(order + 1)
    
    def fit_projection(self, f, n_quad=None):
        if n_quad is None:
            n_quad = self.order + 5
        
        xi, w = leggauss(n_quad)
        f_vals = np.array([f(x) for x in xi])
        

        poly_vals = legendre_poly_vals(self.order, xi)
        
        for k in range(self.order + 1):

            norm = 2.0 / (2.0 * k + 1.0)
            integrand = f_vals * poly_vals[:, k]
            self.coeffs[k] = np.sum(w * integrand) / norm
    
    def fit_collocation(self, xi_samples, f_samples):
        xi_samples = np.asarray(xi_samples)
        f_samples = np.asarray(f_samples)
        poly_vals = legendre_poly_vals(self.order, xi_samples)
        
        for k in range(self.order + 1):
            norm = np.sum(poly_vals[:, k]**2)
            if norm < 1e-15:
                self.coeffs[k] = 0.0
            else:
                self.coeffs[k] = np.sum(f_samples * poly_vals[:, k]) / norm
    
    def evaluate(self, xi):
        xi = np.asarray(xi)
        poly_vals = legendre_poly_vals(self.order, xi)
        return poly_vals @ self.coeffs
    
    def mean(self):
        return float(self.coeffs[0])
    
    def variance(self):
        var = 0.0
        for k in range(1, self.order + 1):
            var += self.coeffs[k]**2 / (2.0 * k + 1.0)
        return var
    
    def std(self):
        return float(np.sqrt(self.variance()))
    
    def sobol_first_order(self):
        var = self.variance()
        return 1.0 if var > 0 else 0.0


def pce_jet_mass_uncertainty(shower_func, alpha_s_range=(0.10, 0.14),
                             order=4, n_samples=200, seed=42):
    rng = np.random.default_rng(seed)
    a_min, a_max = alpha_s_range
    

    xi_samples = np.linspace(-0.95, 0.95, n_samples)
    rng.shuffle(xi_samples)
    
    alpha_samples = 0.5 * (a_max - a_min) * xi_samples + 0.5 * (a_max + a_min)
    mass_samples = np.array([shower_func(a) for a in alpha_samples])
    
    pce = LegendrePCE(order=order)
    pce.fit_collocation(xi_samples, mass_samples)
    
    samples = {
        'alpha_s': alpha_samples,
        'jet_mass': mass_samples,
        'xi': xi_samples
    }
    return pce, samples


def pce_pdf_uncertainty(pdf_func, x_value, param_range=(0.25, 0.35), order=4):
    lam_min, lam_max = param_range
    
    def f(xi):
        lam = 0.5 * (lam_max - lam_min) * xi + 0.5 * (lam_max + lam_min)
        return pdf_func(x_value, lam)
    
    pce = LegendrePCE(order=order)
    pce.fit_projection(f, n_quad=order + 8)
    return pce


def global_sensitivity_analysis(model_func, param_ranges, order=3, n_mc=5000, seed=42):
    rng = np.random.default_rng(seed)
    names = list(param_ranges.keys())
    dim = len(names)
    

    xi_mc = rng.uniform(-1.0, 1.0, size=(n_mc, dim))
    

    y_vals = np.zeros(n_mc)
    for i in range(n_mc):
        params = {}
        for j, name in enumerate(names):
            low, high = param_ranges[name]
            params[name] = 0.5 * (high - low) * xi_mc[i, j] + 0.5 * (high + low)
        y_vals[i] = model_func(params)
    
    mean_val = float(np.mean(y_vals))
    var_val = float(np.var(y_vals, ddof=1))
    std_val = float(np.std(y_vals, ddof=1))
    

    sobol = {}
    for j, name in enumerate(names):
        corr = np.corrcoef(xi_mc[:, j], y_vals)[0, 1]
        sobol[name] = float(corr**2)
    
    results = {
        'mean': mean_val,
        'variance': var_val,
        'std': std_val,
        'sobol_indices': sobol,
        'n_samples': n_mc
    }
    return results


def test_pce():

    pce = LegendrePCE(order=5)
    pce.fit_projection(lambda xi: xi**2, n_quad=12)
    assert abs(pce.mean() - 1.0/3.0) < 1e-6, f"PCE mean error: {pce.mean()}"
    assert abs(pce.variance() - 4.0/45.0) < 1e-5, f"PCE var error: {pce.variance()}"
    

    pce2 = LegendrePCE(order=8)
    pce2.fit_projection(lambda xi: np.exp(xi), n_quad=16)
    for test_xi in [-0.5, 0.0, 0.5]:
        approx = pce2.evaluate(test_xi)
        exact = np.exp(test_xi)
        assert abs(approx - exact) < 0.01, f"PCE eval error at {test_xi}"
    
    return True


if __name__ == "__main__":
    test_pce()
    print("PCE uncertainty quantification tests passed.")
