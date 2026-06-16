"""
benchmark.py
============
Benchmark suite for convergence and accuracy tests of the FD schemes
and the ETDRK4 RG flow solver.

Implements convergence studies (seed 1169_SysBioUAB_docking_benchmark):
  1. FD order convergence: verify O(h^{2m}) scaling for d^2/dx^2 of
     a known test function (e.g., f(x) = exp(-x^2)).
  2. ETDRK4 vs RK4 step-size scaling for RG flow.
  3. Condition-number stability of the Fisher matrix vs number of channels.

For each benchmark, compute:
    observed_order = log(err_1/err_2) / log(h_1/h_2)
and compare with theoretical order.

Likelihood profile benchmark (Feynman-Kac inspired, seed 423):
  Use a Feynman-Kac path-integral representation of the profile likelihood:
    L_eff(mu) = E_W[exp(-Integral V(W(s), mu) ds)]
  where W is a Brownian path. For Gaussian L this reproduces the exact
  profile likelihood. We use this as a cross-check.
"""
from __future__ import annotations
import numpy as np
from finite_difference import HighOrderFD


class BenchmarkRunner:
    """
    Benchmark suite for FD convergence and RG flow accuracy.
    (seed 1169_SysBioUAB_docking_benchmark)
    """

    def __init__(self, fd: HighOrderFD | None = None, fitter=None) -> None:
        self.fd = fd if fd is not None else HighOrderFD()
        self.fitter = fitter

    # ------------------------------------------------------------------ #
    #            1. FD order convergence on test function                #
    # ------------------------------------------------------------------ #
    def fd_convergence_test(
        self,
        f=None,
        f_second_deriv=None,
        x0: float = 0.5,
        orders: tuple = (2, 4, 6, 8),
        h_vals: tuple = (1e-1, 5e-2, 2.5e-2, 1.25e-2, 6.25e-3),
    ) -> dict:
        """
        Verify O(h^{2m}) convergence for each order 2m of the FD 2nd derivative.
        Returns dict with errors and observed orders.

        Test function default: f(x) = exp(-x^2), f''(x) = (4x^2 - 2) exp(-x^2).
        """
        if f is None:
            f = lambda x: np.exp(-x ** 2)
        if f_second_deriv is None:
            f_second_deriv = lambda x: (4.0 * x ** 2 - 2.0) * np.exp(-x ** 2)

        exact = f_second_deriv(x0)
        results = {}
        for order in orders:
            errors = []
            for h in h_vals:
                approx = self.fd.d2(f, x0, h=h, order=order)
                errors.append(abs(approx - exact))
            errors = np.array(errors)
            obs_orders = []
            for i in range(len(errors) - 1):
                if errors[i + 1] > 1e-15 and h_vals[i] > h_vals[i + 1]:
                    obs_orders.append(
                        np.log(errors[i] / max(errors[i + 1], 1e-15))
                        / np.log(h_vals[i] / h_vals[i + 1])
                    )
            results[order] = {
                "errors": errors,
                "observed_orders": obs_orders,
                "mean_order": float(np.mean(obs_orders)) if obs_orders else 0.0,
            }
        return results

    # ------------------------------------------------------------------ #
    #      2. Convergence rates summary (used by main)                   #
    # ------------------------------------------------------------------ #
    def convergence_rates(self) -> dict:
        """Run FD convergence tests and return summary."""
        results = self.fd_convergence_test()
        summary = {}
        for order, data in results.items():
            summary[f"order_{order}"] = data["mean_order"]
        vals = [d["mean_order"] for d in results.values() if d["mean_order"] > 0]
        summary["observed_order"] = float(np.mean(vals)) if vals else 0.0
        return summary

    # ------------------------------------------------------------------ #
    #      3. Fisher matrix condition vs channel count                   #
    # ------------------------------------------------------------------ #
    def fisher_condition_vs_channels(self, n_channels_list=(2, 3, 4, 5)) -> dict:
        """
        Study how the condition number of the Fisher matrix scales
        with the number of channels in the fit.
        """
        if self.fitter is None:
            return {}
        from fisher_information import FisherMatrix

        # Build a lightweight fitter proxy that is consistent with the subset
        class _FitterSubset:
            def __init__(self, channels, parent):
                self.channels = list(channels)
                self.s_SM = {ch: parent.s_SM[ch] for ch in self.channels}
                self.b = {ch: parent.b[ch] for ch in self.channels}
                self.syst_frac = {ch: parent.syst_frac[ch] for ch in self.channels}

            def expected_signal(self, mu_vec, theta_vec=None):
                mu_vec = np.asarray(mu_vec, dtype=float)
                s = np.array([self.s_SM[ch] for ch in self.channels])
                return mu_vec * s

            def expected_background(self, theta_vec):
                theta_vec = np.asarray(theta_vec, dtype=float)
                b = np.array([self.b[ch] for ch in self.channels])
                syst = np.array([self.syst_frac[ch] for ch in self.channels])
                return b * np.exp(theta_vec * syst)

        results = {}
        for n in n_channels_list:
            sub_channels = self.fitter.channels[:n]
            sub = _FitterSubset(sub_channels, self.fitter)
            fisher = FisherMatrix(sub)
            F = fisher.asimov_fisher()
            ev = np.linalg.eigvalsh(F)
            ev_abs = np.abs(ev)
            cond = float(ev_abs.max() / max(ev_abs.min(), 1e-15))
            results[n] = {
                "condition_number": cond,
                "min_eigenvalue": float(ev.min()),
                "max_eigenvalue": float(ev.max()),
            }
        return results

    # ------------------------------------------------------------------ #
    #      4. Feynman-Kac cross-check of profile likelihood              #
    # ------------------------------------------------------------------ #
    def feynman_kac_likelihood_check(
        self,
        mu_center: float = 1.0,
        sigma: float = 0.2,
        n_paths: int = 500,
        n_steps: int = 100,
        T: float = 1.0,
        mu_eval: float = 1.0,
    ) -> dict:
        """
        Feynman-Kac representation of the profile likelihood (seed 423):
            L_eff(mu) = E_W [ exp(-Integral_0^T V(W(s), mu) ds) ]

        For a Gaussian profile likelihood:
            -ln L(mu) = (mu - mu_center)^2 / (2 sigma^2) + const

        Choose V(x, mu) = (x - mu)^2 / (2 sigma^2).

        We simulate n_paths Brownian paths W(t) starting at mu and
        compute the empirical average of exp(-Integral V ds) to
        cross-check the FD-computed likelihood curvature.
        """
        rng = np.random.default_rng(423)
        dt = T / n_steps
        dW_std = np.sqrt(dt)
        path_integrals = np.zeros(n_paths)

        for p in range(n_paths):
            W = mu_eval
            integral = 0.0
            for _ in range(n_steps):
                V = (W - mu_center) ** 2 / (2.0 * sigma ** 2)
                integral += V * dt
                W += rng.normal(0.0, dW_std)
            path_integrals[p] = np.exp(-integral)

        mean_val = float(np.mean(path_integrals))
        analytical = np.exp(-(mu_eval - mu_center) ** 2 / (2.0 * sigma ** 2) * T)
        return {
            "fk_mean": mean_val,
            "analytical_small_T": analytical,
            "ratio": mean_val / max(analytical, 1e-300),
        }
