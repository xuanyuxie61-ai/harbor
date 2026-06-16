"""
channel_selector.py
===================
Decision pipeline for Higgs decay-channel selection and referral.

For each channel i, we compute a figure of merit:
    S_i = mu_i / sigma(mu_i)         (significance)
    epsilon_i = sigma(mu_i) / mu_i   (relative precision)

Decision rules (seed 1175_vara-ai_decision-referral):
    - ACCEPT if S_i > 5 and epsilon_i < 0.20
    - REVIEW if 3 < S_i <= 5 or 0.20 <= epsilon_i < 0.50
    - REFER  if S_i <= 3 or epsilon_i >= 0.50
    - REFERRED channels are queued for additional analysis (more data,
      different production mode, combined fit).

The referral decision function is:
    D(ch) = argmax_{a in {accept, review, refer}} U(a, ch)
where U is a utility function combining sensitivity and cost.

Channel combination (serial vs parallel):
  - Serial (AND): S_combined = sqrt(Sum S_i^2)
  - Parallel (OR): 1 - (1-p_1)(1-p_2)...
"""
from __future__ import annotations
import numpy as np


class ChannelSelector:
    """
    Decision-referral pipeline for Higgs decay channels.
    (seed 1175_vara-ai_decision-referral)
    """

    # Decision thresholds
    S_STRONG = 5.0         # 5-sigma "discovery" threshold
    S_MEDIUM = 3.0         # 3-sigma "evidence" threshold
    EPS_PRECISE = 0.20
    EPS_MARGINAL = 0.50

    ACTIONS = ("accept", "review", "refer")

    def __init__(self, channels: list) -> None:
        self.channels = list(channels)

    # ------------------------------------------------------------------ #
    #              Figure-of-merit computation                           #
    # ------------------------------------------------------------------ #
    def significance(self, mu_hat: np.ndarray, sigma_mu: np.ndarray) -> np.ndarray:
        """Significance S_i = mu_i / sigma(mu_i)."""
        return np.asarray(mu_hat) / np.maximum(np.asarray(sigma_mu), 1e-10)

    def relative_precision(self, mu_hat: np.ndarray, sigma_mu: np.ndarray) -> np.ndarray:
        """Relative precision epsilon_i = sigma(mu_i) / |mu_i|."""
        return np.abs(np.asarray(sigma_mu)) / np.maximum(np.abs(np.asarray(mu_hat)), 1e-10)

    # ------------------------------------------------------------------ #
    #                  Decision rule per channel                         #
    # ------------------------------------------------------------------ #
    def decide_channel(self, S: float, epsilon: float) -> str:
        """
        Decision: accept / review / refer for one channel.
        (seed 1175_vara-ai_decision-referral)
        """
        if S > self.S_STRONG and epsilon < self.EPS_PRECISE:
            return "accept"
        if S > self.S_MEDIUM and epsilon < self.EPS_MARGINAL:
            return "review"
        return "refer"

    # ------------------------------------------------------------------ #
    #                  Utility for referral decision                     #
    # ------------------------------------------------------------------ #
    def utility(self, action: str, S: float, epsilon: float, cost: float = 1.0) -> float:
        """
        Utility of taking action a for a channel with significance S,
        precision epsilon, and cost (for referral).

            U(accept, S, eps) = S - 5 eps
            U(review,  S, eps) = 0.5 S - 2 eps
            U(refer,   S, eps) = - cost + 0.1 S
        """
        if action == "accept":
            return S - 5.0 * epsilon
        if action == "review":
            return 0.5 * S - 2.0 * epsilon
        if action == "refer":
            return -cost + 0.1 * S
        raise ValueError(f"Unknown action {action}")

    def optimal_action(self, S: float, epsilon: float, cost: float = 1.0) -> str:
        """Select action maximizing expected utility."""
        return max(self.ACTIONS, key=lambda a: self.utility(a, S, epsilon, cost))

    # ------------------------------------------------------------------ #
    #                  Full ranking pipeline                             #
    # ------------------------------------------------------------------ #
    def rank_by_sensitivity(self, fit_result, fisher_uncertainties: np.ndarray | None = None) -> list:
        """
        Rank channels by significance. Returns list of dicts.
        """
        mu_hat = fit_result.mu_hat
        if fisher_uncertainties is None:
            # Fall back to 10% of mu as uncertainty
            fisher_uncertainties = 0.1 * np.abs(mu_hat)
        S = self.significance(mu_hat, fisher_uncertainties)
        eps = self.relative_precision(mu_hat, fisher_uncertainties)

        ranking = []
        for i, ch in enumerate(self.channels):
            decision = self.decide_channel(float(S[i]), float(eps[i]))
            ranking.append({
                "channel": ch,
                "mu_hat": float(mu_hat[i]),
                "sigma_mu": float(fisher_uncertainties[i]),
                "significance": float(S[i]),
                "relative_precision": float(eps[i]),
                "decision": decision,
            })
        # Sort by significance descending
        ranking.sort(key=lambda d: -d["significance"])
        return ranking

    # ------------------------------------------------------------------ #
    #                  Combined sensitivity                              #
    # ------------------------------------------------------------------ #
    @staticmethod
    def combined_significance(S_array: np.ndarray, mode: str = "serial") -> float:
        """
        Combine per-channel significances.
        serial  (AND): S_comb = sqrt(Sum S_i^2)
        parallel (OR): S_comb = max(S_i)  (simple approximation)
        """
        S = np.asarray(S_array, dtype=float)
        if mode == "serial":
            return float(np.sqrt(np.sum(S ** 2)))
        if mode == "parallel":
            return float(np.max(S))
        raise ValueError(f"Unknown combination mode {mode}")
