
import numpy as np
from typing import Callable, List, Optional


class MonteCarloEngine:

    def __init__(self, model_func: Callable[[np.ndarray], float],
                 param_sampler: Callable[[np.ndarray], np.ndarray],
                 n_params: int):
        self.model_func = model_func
        self.param_sampler = param_sampler
        self.n_params = n_params

    def run_mc(self, N: int, seed: int = 42) -> dict:
        if N < 1:
            raise ValueError("N 必须 ≥ 1")
        rng = np.random.default_rng(seed)
        outputs = np.zeros(N)
        params = np.zeros((N, self.n_params))

        for i in range(N):
            u = rng.random(self.n_params)
            beta = self.param_sampler(u)
            params[i] = beta
            outputs[i] = self.model_func(beta)

        return self._summarize(outputs, params)

    def run_qmc(self, N: int, seed: int = 42) -> dict:
        if N < 1:
            raise ValueError("N 必须 ≥ 1")
        try:
            from scipy.stats import qmc
            sampler = qmc.Sobol(d=self.n_params, scramble=True, seed=seed)
            u_samples = sampler.random(n=N)
        except Exception:

            u_samples = self._simple_latin_hypercube(N, seed)

        outputs = np.zeros(N)
        params = np.zeros((N, self.n_params))
        for i in range(N):
            beta = self.param_sampler(u_samples[i])
            params[i] = beta
            outputs[i] = self.model_func(beta)

        return self._summarize(outputs, params)

    def _simple_latin_hypercube(self, N: int, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        samples = np.zeros((N, self.n_params))
        for d in range(self.n_params):
            bins = np.arange(N) / N
            offsets = rng.random(N) / N
            samples[:, d] = bins + offsets
        return samples

    def _summarize(self, outputs: np.ndarray, params: np.ndarray) -> dict:
        return {
            "mean": float(np.mean(outputs)),
            "variance": float(np.var(outputs, ddof=1)),
            "std": float(np.std(outputs, ddof=1)),
            "min": float(np.min(outputs)),
            "max": float(np.max(outputs)),
            "median": float(np.median(outputs)),
            "samples_output": outputs.copy(),
            "samples_params": params.copy(),
        }

    def estimate_exceedance_probability(self, outputs: np.ndarray,
                                        threshold: float) -> dict:
        N = len(outputs)
        if N == 0:
            raise ValueError("输出样本不能为空")
        indicator = (outputs > threshold).astype(float)
        P = float(np.mean(indicator))
        se = np.sqrt(P * (1.0 - P) / max(N, 1))
        ci_lower = max(0.0, P - 1.96 * se)
        ci_upper = min(1.0, P + 1.96 * se)
        return {
            "threshold": threshold,
            "exceedance_probability": P,
            "ci_95_lower": ci_lower,
            "ci_95_upper": ci_upper,
            "standard_error": float(se),
        }

    def global_sensitivity_sobol(self, outputs_A: np.ndarray,
                                  outputs_B: np.ndarray,
                                  outputs_AB: np.ndarray) -> dict:
        N = len(outputs_A)
        if N == 0:
            raise ValueError("样本不能为空")
        var_Y = np.var(np.concatenate([outputs_A, outputs_B]), ddof=1)
        if var_Y < 1e-15:
            return {"S1": np.array([0.0]), "total_variance": 0.0}

        S1 = np.mean(outputs_A * (outputs_AB - outputs_B)) / var_Y
        return {
            "S1": float(S1),
            "total_variance": float(var_Y),
            "N": N,
        }


def convergence_analysis(model_func: Callable[[np.ndarray], float],
                         param_sampler: Callable[[np.ndarray], np.ndarray],
                         n_params: int,
                         sample_sizes: List[int] = [50, 100, 200, 500, 1000]) -> dict:
    results = []
    for N in sample_sizes:
        engine = MonteCarloEngine(model_func, param_sampler, n_params)
        res = engine.run_mc(N, seed=42)
        results.append({
            "N": N,
            "mean": res["mean"],
            "std_err": res["std"] / np.sqrt(N),
        })
    return {"convergence": results}


if __name__ == "__main__":

    def model(beta):
        return beta[0] + beta[1] * 2.0

    def sampler(u):

        from math import sqrt
        return np.array([1.0 + sqrt(0.2) * (u[0] - 0.5) * 2 * sqrt(3),
                         0.5 + sqrt(0.1) * (u[1] - 0.5) * 2 * sqrt(3)])

    engine = MonteCarloEngine(model, sampler, n_params=2)
    res = engine.run_mc(1000)
    assert res["mean"] > 0

    exceed = engine.estimate_exceedance_probability(res["samples_output"], 2.0)
    assert 0.0 <= exceed["exceedance_probability"] <= 1.0
    print("uncertainty_engine: 自测试通过")
